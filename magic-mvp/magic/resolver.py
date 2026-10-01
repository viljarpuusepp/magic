from __future__ import annotations

import inspect
from typing import Any

from .arguments import map_arguments
from .embeddings import Encoder, FunctionIndex, LocalSentenceEncoder
from .errors import AmbiguousIntentError, ResolutionError
from .executor import Executor
from .models import Candidate, ResolutionResult
from .ranking import RankingWeights
from .registry import Registry, normalize


class Resolver:
    def __init__(
        self,
        *,
        model_path=None,
        cache_dir=None,
        encoder: Encoder | None = None,
        weights: RankingWeights | None = None,
        top_k: int = 12,
        ambiguity_threshold: float = 0.035,
        minimum_similarity: float = 0.30,
    ) -> None:
        if not 0 <= ambiguity_threshold <= 1 or not 0 <= minimum_similarity <= 1:
            raise ValueError("Thresholds must be between 0 and 1")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 2:
            raise ValueError("Resolver top_k must be at least 2 for ambiguity detection")
        self.registry = Registry()
        self.index = FunctionIndex(
            self.registry, encoder or LocalSentenceEncoder(model_path), cache_dir
        )
        self.executor = Executor(self.registry)
        self.weights = weights or RankingWeights()
        self.top_k = top_k
        self.ambiguity_threshold = ambiguity_threshold
        self.minimum_similarity = minimum_similarity

    def resolve(
        self,
        intent: str,
        *args: Any,
        arguments: dict[str, Any] | None = None,
        execute: bool = False,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> ResolutionResult | Any:
        if not isinstance(intent, str) or not intent.strip():
            raise ValueError("intent must be a nonempty string")
        if arguments is not None:
            if not isinstance(arguments, dict) or not all(isinstance(k, str) for k in arguments):
                raise ValueError("arguments must be a dictionary with string keys")
            duplicates = arguments.keys() & kwargs.keys()
            if duplicates:
                raise ValueError(f"Arguments supplied twice: {sorted(duplicates)}")
            kwargs = {**arguments, **kwargs}
        requested_k = self.top_k if top_k is None else top_k
        if not isinstance(requested_k, int) or isinstance(requested_k, bool) or requested_k < 2:
            raise ValueError("top_k must be at least 2 for ambiguity detection")
        query = normalize(intent)
        generic_normal = query in {
            "normal distribution",
            "gaussian distribution",
            "normal",
            "gaussian",
        }
        has_sample_count = bool({normalize(k) for k in kwargs} & {"samples", "count", "size"})
        exact = self.registry.exact_matches(intent)
        aliases = self.registry.alias_matches(intent)
        generic_normal = generic_normal or any(
            match[0] in {"normal distribution", "gaussian distribution"}
            for match in aliases.values()
        )
        operation_words = {
            "draw",
            "generate",
            "sample",
            "samples",
            "sampling",
            "random",
            "density",
            "pdf",
            "cdf",
            "cumulative",
            "ppf",
            "inverse",
            "quantile",
            "object",
            "frozen",
        }
        underspecified_normal = (
            generic_normal
            and not has_sample_count
            and not operation_words.intersection(query.split())
        )
        strongest = max((match[1] for match in aliases.values()), default=0)
        anchored_ids = {fid for fid, match in aliases.items() if match[1] == strongest}
        anchored_groups = (
            {self.registry.get(fid).semantic_group for fid in anchored_ids}
            if not underspecified_normal
            else set()
        )
        # A qualified registered ID is an explicit implementation choice.
        fixed_ids = {d.id for d in exact if normalize(d.id) == query}
        retrieved = dict(self.index.search(query, requested_k))
        # Include exact aliases even if a transformer places them outside its top-k.
        for d in exact:
            retrieved.setdefault(d.id, self.index.similarity(query, d.id))
        for fid in anchored_ids:
            retrieved.setdefault(fid, self.index.similarity(query, fid))
        if underspecified_normal:
            for d in self.registry.functions:
                if d.semantic_group.startswith("normal."):
                    retrieved.setdefault(d.id, self.index.similarity(query, d.id))
        candidates = []
        for function_id, similarity in retrieved.items():
            d = self.registry.get(function_id)
            installed = self.registry.installed(d.package)
            runtime_signature = None
            if installed:
                try:
                    runtime_signature = inspect.signature(self.registry.load_callable(d.id))
                except (ValueError, TypeError):
                    pass
            mapped = map_arguments(d, args, kwargs, runtime_signature=runtime_signature)
            is_exact = any(d.id == e.id for e in exact)
            matched_alias = aliases[d.id][0] if d.id in anchored_ids else None
            intent_match = min(1.0, max(0.0, similarity) + (0.20 if matched_alias else 0.0))
            eligible = (
                installed
                and not mapped.errors
                and (not anchored_groups or d.semantic_group in anchored_groups)
                and (not fixed_ids or d.id in fixed_ids)
                and (is_exact or similarity >= self.minimum_similarity)
            )
            score = self.weights.score(
                intent_match, mapped.argument_match, mapped.type_match, float(installed), d.prior
            )
            candidates.append(
                Candidate(
                    d.id,
                    d.description,
                    d.semantic_group,
                    score,
                    similarity,
                    intent_match,
                    mapped.argument_match,
                    mapped.type_match,
                    float(installed),
                    d.prior,
                    is_exact,
                    matched_alias,
                    eligible,
                    mapped.args,
                    mapped.kwargs,
                    mapped.mapping,
                    mapped.missing,
                    mapped.errors,
                )
            )
        candidates.sort(key=lambda c: (not c.eligible, -round(c.score, 12), c.function_id))
        eligible = [c for c in candidates if c.eligible]
        if not eligible:
            status = (
                "invalid_arguments"
                if exact
                or any(c.semantic_similarity >= self.minimum_similarity for c in candidates)
                else "not_found"
            )
            result = ResolutionResult(
                intent,
                status,
                None,
                0.0,
                tuple(candidates),
                message="No registered function matches both the intent and supplied arguments.",
            )
        else:
            best = eligible[0]
            distinct = next(
                (c for c in eligible[1:] if c.semantic_group != best.semantic_group), None
            )
            ambiguous = distinct is not None and (
                underspecified_normal or best.score - distinct.score < self.ambiguity_threshold
            )
            if ambiguous:
                result = ResolutionResult(
                    intent,
                    "ambiguous",
                    None,
                    best.score,
                    tuple(candidates),
                    message=(
                        "Several different operations match. Specify sampling (samples=...), "
                        "density (normal_pdf), cumulative probability (normal_cdf), or an "
                        "inverse/quantile operation (normal_ppf)."
                        if underspecified_normal
                        else "Several different operations have similar scores. "
                        "Use a clearer intent "
                        "or a fully qualified registered function ID."
                    ),
                )
            else:
                result = ResolutionResult(
                    intent,
                    "resolved",
                    best.function_id,
                    best.score,
                    tuple(candidates),
                    best.mapped_args,
                    best.mapped_arguments,
                    best.argument_mapping,
                    best.missing_parameters,
                    "Ranking score is a weighted heuristic, not a probability.",
                )
        if execute:
            if result.ambiguous:
                raise AmbiguousIntentError(result)
            if result.status != "resolved":
                raise ResolutionError(result)
            return self.executor.execute(result)
        return result
