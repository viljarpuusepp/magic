"""Curated metadata, including safe signature fallbacks for C and SciPy callables."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    description: str = ""
    type_hint: str = "any"
    required: bool = False
    default: Any = None
    kind: str = "positional_or_keyword"

    def as_parameter(self) -> inspect.Parameter:
        kinds = {
            "positional_or_keyword": inspect.Parameter.POSITIONAL_OR_KEYWORD,
            "positional_only": inspect.Parameter.POSITIONAL_ONLY,
            "keyword_only": inspect.Parameter.KEYWORD_ONLY,
        }
        return inspect.Parameter(
            self.name,
            kinds[self.kind],
            default=inspect.Parameter.empty if self.required else self.default,
        )


@dataclass(frozen=True)
class FunctionDescriptor:
    id: str
    package: str
    module: str
    function: str
    description: str
    semantic_group: str
    parameters: tuple[ParameterSpec, ...]
    aliases: tuple[str, ...] = ()
    parameter_aliases: dict[str, str] = field(default_factory=dict)
    prior: float = 0.8
    effect: str = "compute"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FunctionDescriptor:
        data = dict(data)
        data["parameters"] = tuple(ParameterSpec(**p) for p in data["parameters"])
        data["aliases"] = tuple(data.get("aliases", ()))
        return cls(**data)

    @property
    def signature(self) -> inspect.Signature:
        # Never eval a signature string. Only build from explicit parameter metadata.
        return inspect.Signature([p.as_parameter() for p in self.parameters])

    @property
    def embedding_text(self) -> str:
        meanings = "; ".join(f"{p.name}: {p.description}" for p in self.parameters)
        return (
            f"{self.id}. {self.description}\n"
            f"Aliases: {', '.join(self.aliases)}\n"
            f"Parameters: {meanings}\nPackage: {self.package}"
        )
