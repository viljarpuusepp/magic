"""Registry-driven argument mapping; candidates are never executed here."""

from __future__ import annotations

import inspect
import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .models import FunctionDescriptor
from .registry import normalize


@dataclass(frozen=True)
class MappedArguments:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    mapping: dict[str, str]
    missing: tuple[str, ...]
    errors: tuple[str, ...]
    argument_match: float
    type_match: float


def type_score(value: Any, hint: str) -> float:
    """Small conservative type rules. Unknown types are not guessed by the transformer."""
    if value is None or hint == "any":
        return 1.0
    cls = type(value)
    is_series = cls.__module__.startswith("pandas.") and cls.__name__ == "Series"
    is_frame = cls.__module__.startswith("pandas.") and cls.__name__ == "DataFrame"
    is_array = isinstance(value, np.ndarray)
    is_sequence = isinstance(value, (list, tuple, range))
    is_real = isinstance(value, numbers.Real) and not isinstance(value, bool)
    if hint == "array_like":
        if is_series or is_frame:
            return 0.8
        return 1.0 if is_array or is_sequence or is_real else 0.0
    if hint == "iterable":
        if is_array or is_series:
            return 0.75
        return 1.0 if is_sequence else 0.0
    if hint == "series":
        return 1.0 if is_series else 0.0
    if hint == "real_or_array":
        return 1.0 if is_real or is_array or is_sequence or is_series else 0.0
    if hint == "real":
        return 1.0 if is_real else 0.0
    if hint == "integer":
        return 1.0 if isinstance(value, numbers.Integral) and not isinstance(value, bool) else 0.0
    if hint == "size":
        if isinstance(value, numbers.Integral) and not isinstance(value, bool):
            return 1.0 if value >= 0 else 0.0
        if isinstance(value, tuple):
            return float(all(isinstance(v, numbers.Integral) and v >= 0 for v in value))
        return 0.0
    if hint == "bool":
        return float(isinstance(value, (bool, np.bool_)))
    if hint == "path":
        return float(isinstance(value, (str, Path)) or callable(getattr(value, "read", None)))
    if hint == "mapping_or_sequence":
        return float(isinstance(value, (dict, list, tuple)))
    return 1.0


def map_arguments(
    descriptor: FunctionDescriptor,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    runtime_signature: inspect.Signature | None = None,
) -> MappedArguments:
    signature = descriptor.signature
    names = {p.name for p in descriptor.parameters}
    aliases = {normalize(k): v for k, v in descriptor.parameter_aliases.items()}
    canonical = {normalize(k): k for k in names}
    mapped: dict[str, Any] = {}
    mapping: dict[str, str] = {}
    errors: list[str] = []
    for name, value in kwargs.items():
        target = canonical.get(normalize(name)) or aliases.get(normalize(name))
        if target is None:
            errors.append(f"Unsupported parameter: {name}")
            continue
        if target in mapped:
            errors.append(f"Multiple arguments map to {target}: {name}")
            continue
        mapped[target] = value
        mapping[name] = target
    try:
        bound = signature.bind_partial(*args, **mapped)
        values = bound.arguments
    except TypeError as e:
        errors.append(str(e))
        values = {}
    # inspect.signature() supplements, but cannot replace, the curated subset.
    if runtime_signature is not None and not errors:
        try:
            runtime_signature.bind_partial(*args, **mapped)
        except TypeError as e:
            errors.append(f"Runtime signature: {e}")
    missing = tuple(p.name for p in descriptor.parameters if p.required and p.name not in values)
    matches = []
    for p in descriptor.parameters:
        if p.name in values:
            match = type_score(values[p.name], p.type_hint)
            matches.append(match)
            if match == 0:
                errors.append(
                    f"{p.name} expects {p.type_hint}; got {type(values[p.name]).__name__}"
                )
    count = len(args) + len(kwargs)
    argument_match = 1.0 if count == 0 else max(0.0, 1.0 - len(errors) / count)
    if count and missing:
        argument_match *= 0.75
    return MappedArguments(
        args,
        mapped,
        mapping,
        missing,
        tuple(errors),
        argument_match,
        sum(matches) / len(matches) if matches else 0.5,
    )
