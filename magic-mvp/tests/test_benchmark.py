import pytest

from magic.benchmark import load_cases, run_benchmark


@pytest.mark.model
def test_real_model_benchmark_quality_and_metrics(resolver):
    report = run_benchmark(resolver)
    assert report["case_count"] >= 30
    assert report["top_1_accuracy"] >= 0.8
    assert report["top_3_accuracy"] >= 0.95
    assert report["top_5_accuracy"] >= report["top_3_accuracy"]
    assert report["argument_mapping_cases"] >= 5
    assert report["argument_mapping_accuracy"] == 1.0
    assert report["latency_ms"]["mean"] > 0
    assert len(report["cases"]) == len(load_cases())


def test_empty_benchmark_is_rejected(resolver):
    with pytest.raises(ValueError):
        run_benchmark(resolver, [])
