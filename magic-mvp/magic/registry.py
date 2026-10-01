"""The installed, curated registry is the execution allowlist."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import re
from importlib.resources import files
from types import MappingProxyType
from typing import Any

from .errors import UnsafeExecutionError
from .models import FunctionDescriptor


def normalize(text: str) -> str:
    return re.sub(r"[\W_]+", " ", text.casefold()).strip()


_STOPWORDS = frozenset("a an the of this these using with to into for on in and from by".split())


def content_tokens(text: str) -> set[str]:
    return set(normalize(text).split()) - _STOPWORDS


class Registry:
    def __init__(self) -> None:
        raw = files("magic").joinpath("data/functions.json").read_text(encoding="utf-8")
        descriptors = [FunctionDescriptor.from_dict(item) for item in json.loads(raw)]
        if len({d.id for d in descriptors}) != len(descriptors):
            raise ValueError("Duplicate function IDs in curated registry")
        for d in descriptors:
            names = {p.name for p in d.parameters}
            if not set(d.parameter_aliases.values()) <= names:
                raise ValueError(f"Invalid parameter alias in {d.id}")
            _ = d.signature  # Validate metadata without importing or executing targets.
        self._functions = MappingProxyType(
            {d.id: d for d in sorted(descriptors, key=lambda d: d.id)}
        )
        self.fingerprint = hashlib.sha256(raw.encode()).hexdigest()
        self._installed: dict[str, bool] = {}
        self._callables: dict[str, Any] = {}

    @property
    def functions(self) -> tuple[FunctionDescriptor, ...]:
        return tuple(self._functions.values())

    def get(self, function_id: str) -> FunctionDescriptor:
        try:
            return self._functions[function_id]
        except KeyError as e:
            raise UnsafeExecutionError(f"Function is not allowlisted: {function_id}") from e

    def exact_matches(self, intent: str) -> tuple[FunctionDescriptor, ...]:
        text = normalize(intent)
        return tuple(
            d
            for d in self.functions
            if text in {normalize(d.id), normalize(d.function), *(normalize(a) for a in d.aliases)}
        )

    def alias_matches(self, intent: str) -> dict[str, tuple[str, int, bool]]:
        """Prefer specific curated phrases; small grammar words and word order are immaterial."""
        normalized = normalize(intent)
        query_tokens = content_tokens(intent)
        result = {}
        for d in self.functions:
            for alias in (d.id, d.function, *d.aliases):
                tokens = content_tokens(alias)
                exact = normalize(alias) == normalized
                # A single word inside a sentence ("average squared residuals") is too
                # broad to anchor an operation. Exact single-word calls remain supported.
                if tokens and tokens <= query_tokens and (exact or len(tokens) >= 2):
                    specificity = len(tokens) + (100 if exact else 0)
                    if d.id not in result or specificity > result[d.id][1]:
                        result[d.id] = (alias, specificity, exact)
        return result

    def installed(self, package: str) -> bool:
        if package not in self._installed:
            try:
                self._installed[package] = importlib.util.find_spec(package) is not None
            except (ImportError, ValueError):
                self._installed[package] = False
        return self._installed[package]

    def load_callable(self, function_id: str) -> Any:
        d = self.get(function_id)
        if function_id in self._callables:
            return self._callables[function_id]
        if not self.installed(d.package):
            raise ImportError(f"Install package {d.package!r} to execute {d.id}")
        target = importlib.import_module(d.module)
        for part in d.function.split("."):
            if not part.isidentifier() or part.startswith("_"):
                raise UnsafeExecutionError(f"Unsafe curated callable path: {d.id}")
            target = getattr(target, part)
        if not callable(target):
            raise UnsafeExecutionError(f"Registered target is not callable: {d.id}")
        self._callables[function_id] = target
        return target
