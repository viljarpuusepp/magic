"""Small, inspectable MVP evaluation; never executes the retrieved functions."""

from __future__ import annotations

import json
import platform
import statistics
import time
from importlib.metadata import version
from importlib.resources import files
from typing import Any

from .embeddings import DEFAULT_MODEL, DEFAULT_REVISION
from .registry import normalize
from .resolver import Resolver


def load_cases() -> list[dict[str, Any]]:
    return json.loads(files("magic").joinpath("data/benchmark.json").read_text(encoding="utf-8"))


def run_benchmark(resolver: Resolver, cases: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    cases = load_cases() if cases is None else cases
    if not cases:
        raise ValueError("The benchmark needs at least one case")
    start = time.perf_counter()
    resolver.index.search("warm the local semantic index", 2)
    for descriptor in resolver.registry.functions:
        if resolver.registry.installed(descriptor.package):
            resolver.registry.load_callable(descriptor.id)
    warmup_seconds = time.perf_counter() - start
    hits = {k: 0 for k in (1, 3, 5)}
    raw_hits = {k: 0 for k in (1, 3, 5)}
    mapping_total = mapping_correct = 0
    latencies = []
    details = []
    for case in cases:
        start = time.perf_counter()
        result = resolver.resolve(
            case["intent"],
            *case.get("args", []),
            arguments=case.get("arguments", {}),
            execute=False,
            top_k=max(resolver.top_k, 5),
        )
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)
        ranked = [c.function_id for c in result.candidates if c.eligible]
        raw = [fid for fid, _ in resolver.index.search(normalize(case["intent"]), 5)]
        expected = case["expected_function"]
        for k in hits:
            # Top-1 measures an actual unambiguous selection, not a silently chosen ambiguous best.
            hits[k] += (
                int(result.selected_function == expected) if k == 1 else int(expected in ranked[:k])
            )
            raw_hits[k] += int(expected in raw[:k])
        if "expected_mapping" in case:
            mapping_total += 1
            mapping_correct += int(
                result.selected_function == expected
                and result.argument_mapping == case["expected_mapping"]
            )
        details.append(
            {
                "intent": case["intent"],
                "expected_function": expected,
                "selected_function": result.selected_function,
                "status": result.status,
                "ranked_top5": ranked[:5],
                "semantic_top5": raw,
                "argument_mapping": result.argument_mapping,
                "latency_ms": round(elapsed, 3),
            }
        )
    ordered = sorted(latencies)
    return {
        "case_count": len(cases),
        **{f"top_{k}_accuracy": hits[k] / len(cases) for k in hits},
        **{f"semantic_top_{k}_accuracy": raw_hits[k] / len(cases) for k in raw_hits},
        "argument_mapping_cases": mapping_total,
        "argument_mapping_accuracy": mapping_correct / mapping_total if mapping_total else None,
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 3),
            "median": round(statistics.median(latencies), 3),
            "p95": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 3),
        },
        "warmup_seconds": round(warmup_seconds, 3),
        "latency_scope": (
            "warm CPU model, vector index and target imports; first resolution per intent"
        ),
        "registry_count": len(resolver.registry.functions),
        "registry_fingerprint": resolver.registry.fingerprint,
        "model": DEFAULT_MODEL,
        "model_revision": DEFAULT_REVISION,
        "encoder_identity": resolver.index.encoder.identity,
        "python": platform.python_version(),
        "dependencies": {
            name: version(name)
            for name in (
                "sentence-transformers",
                "transformers",
                "torch",
                "numpy",
                "scipy",
                "pandas",
                "scikit-learn",
            )
        },
        "cases": details,
    }
