"""Local-only transformer inference and a small cached cosine-similarity index."""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Protocol

import numpy as np

from .errors import ModelNotAvailableError
from .registry import Registry

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class Encoder(Protocol):
    identity: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


class LocalSentenceEncoder:
    def __init__(self, model_path: str | Path | None = None) -> None:
        bundled_paths = (
            Path.cwd() / "models" / "all-MiniLM-L6-v2",
            Path(__file__).resolve().parents[1] / "models" / "all-MiniLM-L6-v2",
        )
        bundled = next((p for p in bundled_paths if (p / "modules.json").is_file()), None)
        self.model_path = str(
            model_path or os.environ.get("MAGIC_MODEL_PATH") or bundled or DEFAULT_MODEL
        )
        path = Path(self.model_path)
        identity = self.model_path + ":" + DEFAULT_REVISION
        if path.is_dir():
            for name in ("config.json", "modules.json", "model.safetensors", "pytorch_model.bin"):
                file = path / name
                if file.exists():
                    identity += f":{name}:{file.stat().st_size}:{file.stat().st_mtime_ns}"
        self.identity = hashlib.sha256(identity.encode()).hexdigest()
        self._model = None
        self._lock = threading.RLock()

    def encode(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(
                        self.model_path,
                        device="cpu",
                        local_files_only=True,
                        trust_remote_code=False,
                        revision=None if Path(self.model_path).is_dir() else DEFAULT_REVISION,
                    )
                    self._model.eval()
                except (ImportError, OSError, ValueError) as e:
                    raise ModelNotAvailableError(
                        f"Cannot load local sentence transformer {self.model_path!r}. "
                        "Run `python -m magic download-model --output models/all-MiniLM-L6-v2` "
                        "once, then configure(model_path='models/all-MiniLM-L6-v2') or set "
                        "MAGIC_MODEL_PATH. Resolution never downloads weights or calls an API."
                    ) from e
            return np.asarray(
                self._model.encode(
                    texts,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=32,
                ),
                dtype=np.float32,
            )


class FunctionIndex:
    """Exact vector search is sufficient for 44 entries; replace this layer to scale up."""

    def __init__(self, registry: Registry, encoder: Encoder, cache_dir: str | Path | None = None):
        self.registry = registry
        self.encoder = encoder
        self.descriptors = registry.functions
        self.ids = tuple(d.id for d in self.descriptors)
        cache_root = cache_dir or os.environ.get("MAGIC_CACHE_DIR")
        self.cache_dir = Path(cache_root) if cache_root else Path.home() / ".cache" / "magic"
        key = hashlib.sha256((registry.fingerprint + encoder.identity + ":v1").encode()).hexdigest()
        self.cache_path = self.cache_dir / f"functions-{key}.npz"
        self._vectors: np.ndarray | None = None
        self._lock = threading.RLock()
        self._queries: OrderedDict[str, np.ndarray] = OrderedDict()

    @property
    def vectors(self) -> np.ndarray:
        with self._lock:
            if self._vectors is None:
                self._vectors = self._load()
                if self._vectors is None:
                    self._vectors = self.encoder.encode(
                        [d.embedding_text for d in self.descriptors]
                    )
                    self._validate(self._vectors)
                    self.save()
            return self._vectors

    def _validate(self, vectors: np.ndarray) -> None:
        if (
            vectors.ndim != 2
            or vectors.shape[0] != len(self.ids)
            or vectors.shape[1] == 0
            or not np.isfinite(vectors).all()
        ):
            raise ValueError("Invalid embedding matrix")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4):
            raise ValueError("Embeddings must be normalized")

    def _load(self) -> np.ndarray | None:
        try:
            with np.load(self.cache_path, allow_pickle=False) as data:
                if tuple(data["ids"].tolist()) != self.ids:
                    return None
                vectors = data["vectors"].astype(np.float32)
                self._validate(vectors)
                return vectors
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            return None

    def save(self) -> None:
        if self._vectors is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=self.cache_dir, suffix=".npz", delete=False) as file:
            temp_path = Path(file.name)
            np.savez_compressed(file, ids=np.array(self.ids), vectors=self._vectors)
        try:
            os.replace(temp_path, self.cache_path)
        finally:
            temp_path.unlink(missing_ok=True)

    def query_vector(self, intent: str) -> np.ndarray:
        with self._lock:
            if intent not in self._queries:
                self._queries[intent] = self.encoder.encode([intent])[0]
            self._queries.move_to_end(intent)
            if len(self._queries) > 256:
                self._queries.popitem(last=False)
            return self._queries[intent]

    def search(self, intent: str, top_k: int = 12) -> list[tuple[str, float]]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        scores = self.vectors @ self.query_vector(intent)
        # Stable secondary key avoids depending on incidental registry/NumPy ordering.
        order = sorted(range(len(self.ids)), key=lambda i: (-float(scores[i]), self.ids[i]))
        return [(self.ids[i], float(scores[i])) for i in order[:top_k]]

    def similarity(self, intent: str, function_id: str) -> float:
        return float(self.vectors[self.ids.index(function_id)] @ self.query_vector(intent))
