from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class Candidate:
    function_id: str
    description: str
    semantic_group: str
    score: float
    semantic_similarity: float
    intent_match: float
    argument_match: float
    type_match: float
    installed_package_score: float
    package_prior: float
    exact_alias: bool
    matched_alias: str | None
    eligible: bool
    mapped_args: tuple[Any, ...] = field(repr=False)
    mapped_arguments: dict[str, Any] = field(repr=False)
    argument_mapping: dict[str, str]
    missing_parameters: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def executable(self) -> bool:
        return self.eligible and not self.missing_parameters and not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "function_id": self.function_id,
            "description": self.description,
            "semantic_group": self.semantic_group,
            "score": self.score,
            "semantic_similarity": self.semantic_similarity,
            "components": {
                "intent_match": self.intent_match,
                "argument_match": self.argument_match,
                "type_match": self.type_match,
                "installed_package_score": self.installed_package_score,
                "package_prior": self.package_prior,
            },
            "exact_alias": self.exact_alias,
            "matched_alias": self.matched_alias,
            "eligible": self.eligible,
            "executable": self.executable,
            "mapped_arguments": self.mapped_arguments,
            "argument_mapping": self.argument_mapping,
            "missing_parameters": list(self.missing_parameters),
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class ResolutionResult:
    requested_name: str
    status: Literal["resolved", "ambiguous", "not_found", "invalid_arguments"]
    selected_function: str | None
    score: float
    candidates: tuple[Candidate, ...]
    mapped_args: tuple[Any, ...] = field(default=(), repr=False)
    mapped_arguments: dict[str, Any] = field(default_factory=dict, repr=False)
    argument_mapping: dict[str, str] = field(default_factory=dict)
    missing_parameters: tuple[str, ...] = ()
    message: str = ""

    @property
    def ambiguous(self) -> bool:
        return self.status == "ambiguous"

    @property
    def executable(self) -> bool:
        return (
            self.status == "resolved" and self.selected_function is not None
            and not self.missing_parameters
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.requested_name,
            "status": self.status,
            "selected_function": self.selected_function,
            "ranking_score": self.score,
            "mapped_args": self.mapped_args,
            "mapped_arguments": self.mapped_arguments,
            "argument_mapping": self.argument_mapping,
            "missing_parameters": list(self.missing_parameters),
            "executable": self.executable,
            "message": self.message,
            "candidates": [c.to_dict() for c in self.candidates],
        }

    def __str__(self) -> str:
        lines = [f"Intent: {self.requested_name}", f"Status: {self.status}"]
        if self.selected_function:
            lines += [f"Selected: {self.selected_function}", f"Ranking score: {self.score:.4f}"]
            match = next(
                (c for c in self.candidates if c.function_id == self.selected_function), None
            )
            if match is not None and match.matched_alias:
                lines.append(f"Matched curated alias: {match.matched_alias}")
        if self.message:
            lines.append(self.message)
        if self.argument_mapping:
            lines.append("Mapped parameters:")
            lines += [f"  {src} -> {dst}" for src, dst in self.argument_mapping.items()]
        if self.missing_parameters:
            lines.append(f"Required before execution: {', '.join(self.missing_parameters)}")
        lines.append("Candidates (score; raw semantic similarity):")
        lines += [
            f"  {c.function_id:40s} {c.score:.4f}; {c.semantic_similarity:.4f}"
            + (f" [invalid: {'; '.join(c.errors)}]" if c.errors else
               " [excluded by intent/availability]" if not c.eligible else "")
            for c in self.candidates[:5]
        ]
        return "\n".join(lines)
