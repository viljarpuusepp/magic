import inspect

import numpy as np
import pytest

from magic.arguments import map_arguments
from magic.ranking import RankingWeights


def test_registry_is_curated_and_complete(registry):
    assert len(registry.functions) >= 30
    assert {d.package for d in registry.functions} == {
        "numpy",
        "scipy",
        "pandas",
        "sklearn",
        "statistics",
    }
    for descriptor in registry.functions:
        assert isinstance(descriptor.signature, inspect.Signature)
        assert callable(registry.load_callable(descriptor.id))


def test_normal_alias_mapping(registry):
    result = map_arguments(
        registry.get("numpy.random.normal"), (), {"mean": 1, "std": 5, "samples": 100}
    )
    assert result.kwargs == {"loc": 1, "scale": 5, "size": 100}
    assert result.mapping == {"mean": "loc", "std": "scale", "samples": "size"}
    assert not result.errors
    assert not result.missing


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mean": 1, "loc": 2},
        {"std": 1, "scale": 2},
        {"unknown": 3},
        {"samples": 2.5},
        {"samples": -1},
        {"std": "five"},
    ],
)
def test_invalid_arguments_are_rejected(registry, kwargs):
    assert map_arguments(registry.get("numpy.random.normal"), (), kwargs).errors


def test_positional_alias_collision(registry):
    assert map_arguments(registry.get("numpy.random.normal"), (1,), {"mean": 2}).errors


def test_missing_arguments_are_reported(registry):
    result = map_arguments(registry.get("numpy.mean"), (), {})
    assert result.missing == ("a",)
    assert not result.errors


def test_runtime_signature_is_checked(registry):
    result = map_arguments(
        registry.get("numpy.mean"),
        ([1, 2],),
        {},
        runtime_signature=inspect.Signature(
            [inspect.Parameter("x", inspect.Parameter.KEYWORD_ONLY)]
        ),
    )
    assert any("Runtime signature" in e for e in result.errors)


def test_numeric_array_and_numpy_scalar_types(registry):
    assert not map_arguments(registry.get("numpy.mean"), (np.array([1, 2]),), {}).errors
    assert not map_arguments(registry.get("scipy.stats.norm.cdf"), (np.float64(1.96),), {}).errors


def test_weight_validation():
    with pytest.raises(ValueError):
        RankingWeights(semantic_similarity=-1)
    with pytest.raises(ValueError):
        RankingWeights(semantic_similarity=0.9)
