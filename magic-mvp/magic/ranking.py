from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class RankingWeights:
    semantic_similarity: float = 0.55
    argument_match: float = 0.20
    type_match: float = 0.15
    installed_package_score: float = 0.05
    package_prior: float = 0.05

    def __post_init__(self) -> None:
        values = tuple(vars(self).values())
        if any(not isfinite(v) or v < 0 for v in values) or abs(sum(values) - 1.0) > 1e-8:
            raise ValueError("Ranking weights must be nonnegative and sum to 1")

    def score(
        self, intent: float, arguments: float, types: float, installed: float, prior: float
    ) -> float:
        return (
            self.semantic_similarity * intent
            + self.argument_match * arguments
            + self.type_match * types
            + self.installed_package_score * installed
            + self.package_prior * prior
        )
