import socket

import numpy as np
import pandas as pd
import pytest

import magic
from magic import AmbiguousIntentError, ArgumentMappingError, ResolutionError, Resolver

pytestmark = pytest.mark.model


def test_required_dynamic_calls(model_path, tmp_path):
    magic.configure(model_path=model_path, cache_dir=tmp_path)
    assert magic.mean([1, 2, 3, 4]) == 2.5
    assert magic.standard_deviation([1, 2, 3, 4]) == pytest.approx(np.std([1, 2, 3, 4]))
    samples = magic.normal_distribution(mean=1, std=5, samples=100)
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (100,)
    assert np.isfinite(samples).all()
    assert magic.normal_cdf(1.96) == pytest.approx(0.9750021048517795)
    result = magic.resolve("reduce dimensionality using PCA", execute=False)
    assert result.selected_function == "sklearn.decomposition.PCA"


def test_explain_and_mapping_do_not_execute(resolver, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Explain mode executed a candidate")

    monkeypatch.setattr(resolver.executor, "execute", fail)
    result = resolver.resolve("normal distribution", mean=1, std=5, samples=100)
    assert result.selected_function == "numpy.random.normal"
    assert result.mapped_arguments == {"loc": 1, "scale": 5, "size": 100}
    assert result.argument_mapping == {"mean": "loc", "std": "scale", "samples": "size"}
    assert "probability" in result.message  # explicitly says the score is NOT a probability
    assert "Ranking score" in str(result)
    assert len(result.to_dict()["candidates"]) >= 3


def test_ambiguity_is_explained_and_never_executed(resolver):
    result = resolver.resolve("normal distribution", 1, 5)
    assert result.status == "ambiguous"
    assert result.selected_function is None
    assert not result.executable
    assert len({c.semantic_group for c in result.candidates if c.eligible}) >= 2
    with pytest.raises(AmbiguousIntentError):
        resolver.resolve("normal distribution", 1, 5, execute=True)


def test_unambiguous_density_and_quantile(resolver):
    assert resolver.resolve("normal pdf", 0, execute=True) == pytest.approx(0.3989422804)
    assert resolver.resolve("normal ppf", probability=0.975, execute=True) == pytest.approx(
        1.9599639845
    )


def test_numpy_and_series_type_preferences(resolver):
    assert resolver.resolve("mean", np.array([1, 2, 3])).selected_function == "numpy.mean"
    series = pd.Series([1.0, 3.0, np.nan])
    assert resolver.resolve("mean", series).selected_function == "pandas.Series.mean"
    assert resolver.resolve("mean", series, execute=True) == 2.0


def test_equivalent_implementations_are_not_ambiguous(resolver):
    assert resolver.resolve("mean", [1, 2, 3]).status == "resolved"


def test_generic_normal_phrase_is_ambiguous(resolver):
    assert resolver.resolve("please calculate a normal distribution", 1, 5).ambiguous


def test_specific_aliases_do_not_override_more_precise_operations(resolver):
    result = resolver.resolve("average squared prediction residuals")
    assert result.selected_function == "sklearn.metrics.mean_squared_error"
    frozen = resolver.resolve("create a frozen Gaussian distribution object")
    assert frozen.selected_function == "scipy.stats.norm"


def test_qualified_id_honors_the_implementation(resolver):
    assert resolver.resolve("statistics.mean", [1, 2, 3]).selected_function == "statistics.mean"
    assert resolver.resolve("statistics.stdev", [1, 2, 3], execute=True) == 1.0


def test_invalid_known_intent_never_switches_operation(resolver):
    result = resolver.resolve("normal cdf", 1.96, samples=100)
    assert result.status == "invalid_arguments"
    assert result.selected_function is None
    with pytest.raises(ResolutionError):
        resolver.resolve("normal cdf", 1.96, samples=100, execute=True)


def test_missing_required_values_are_searchable(resolver):
    result = resolver.resolve("average of these numbers")
    assert result.selected_function == "numpy.mean"
    assert result.missing_parameters == ("a",)
    assert not result.executable
    with pytest.raises(ArgumentMappingError):
        resolver.resolve("numpy.mean", execute=True)


def test_unknown_intent_and_invalid_input(resolver):
    result = resolver.resolve("compose an opera about medieval dragons")
    assert result.status == "not_found"
    with pytest.raises(ValueError):
        resolver.resolve("")
    with pytest.raises(ValueError):
        resolver.resolve("mean", arguments={"data": [1]}, data=[2])
    with pytest.raises(ValueError):
        resolver.resolve("mean", top_k=1)
    with pytest.raises(AttributeError):
        _ = magic.__does_not_exist__


def test_pca_estimator_is_runnable(resolver):
    pca = resolver.resolve("PCA", components=2, execute=True)
    data = np.random.default_rng(42).normal(size=(10, 4))
    assert pca.fit_transform(data).shape == (10, 2)


def test_cold_local_inference_never_connects(model_path, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Resolution attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    cold = Resolver(model_path=model_path, cache_dir=tmp_path)
    assert cold.resolve("normal cdf", 1.96, execute=True) > 0.97


def test_stable_ranking_across_resolver_instances(model_path, tmp_path):
    a = Resolver(model_path=model_path, cache_dir=tmp_path)
    b = Resolver(model_path=model_path, cache_dir=tmp_path)
    first = a.resolve("reduce dimensionality using PCA")
    second = b.resolve("reduce dimensionality using PCA")
    assert [c.function_id for c in first.candidates] == [c.function_id for c in second.candidates]
    assert [c.score for c in first.candidates] == [c.score for c in second.candidates]


def test_real_embeddings_are_normalized_and_retrieve_pca(resolver):
    vectors = resolver.index.vectors
    assert vectors.shape == (44, 384)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)
    assert "sklearn.decomposition.PCA" in [
        name for name, _ in resolver.index.search("reduce dimensionality using PCA", 3)
    ]
