import hashlib

import numpy as np
import pytest

from magic.embeddings import FunctionIndex, LocalSentenceEncoder
from magic.errors import ModelNotAvailableError


class CountingEncoder:
    identity = "cache-unit-test-v1"

    def __init__(self):
        self.batches = []

    def encode(self, texts):
        self.batches.append(tuple(texts))
        rows = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4])
            vector = np.random.default_rng(seed).normal(size=8)
            rows.append(vector / np.linalg.norm(vector))
        return np.array(rows, dtype=np.float32)


def test_vectors_are_cached_across_calls_and_instances(registry, tmp_path):
    first = CountingEncoder()
    index = FunctionIndex(registry, first, tmp_path)
    initial = index.search("calculate an average", 3)
    assert len(first.batches) == 2  # one registry batch, one query
    assert index.search("calculate an average", 3) == initial
    assert len(first.batches) == 2
    second = CountingEncoder()
    reopened = FunctionIndex(registry, second, tmp_path)
    assert reopened.search("calculate an average", 3) == initial
    assert len(second.batches) == 1  # saved registry vectors reused


def test_corrupt_cache_rebuilds(registry, tmp_path):
    index = FunctionIndex(registry, CountingEncoder(), tmp_path)
    index.cache_path.write_bytes(b"broken cache")
    assert len(index.search("average", 5)) == 5


def test_model_identity_invalidates_cache(registry, tmp_path):
    a, b = CountingEncoder(), CountingEncoder()
    b.identity = "different-model"
    assert (
        FunctionIndex(registry, a, tmp_path).cache_path
        != FunctionIndex(registry, b, tmp_path).cache_path
    )


@pytest.mark.parametrize("k", [0, -1, True, 2.5])
def test_invalid_top_k(registry, tmp_path, k):
    with pytest.raises(ValueError):
        FunctionIndex(registry, CountingEncoder(), tmp_path).search("average", k)


def test_missing_model_is_an_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(ModelNotAvailableError, match="download-model"):
        LocalSentenceEncoder(tmp_path / "nonexistent-model").encode(["average"])
