from __future__ import annotations

import os
from pathlib import Path

import pytest

from magic import Resolver
from magic.registry import Registry


@pytest.fixture(scope="session")
def registry():
    return Registry()


@pytest.fixture(scope="session")
def model_path():
    root_model = Path(__file__).resolve().parents[1] / "models" / "all-MiniLM-L6-v2"
    return os.environ.get("MAGIC_MODEL_PATH", str(root_model))


@pytest.fixture(scope="session")
def resolver(model_path, tmp_path_factory):
    return Resolver(model_path=model_path, cache_dir=tmp_path_factory.mktemp("real-index"))
