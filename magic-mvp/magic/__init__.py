"""Call Python functions by intent, with local embeddings and deterministic execution."""

from __future__ import annotations

import threading
from typing import Any

from .errors import (
    AmbiguousIntentError,
    ArgumentMappingError,
    MagicError,
    ModelNotAvailableError,
    ResolutionError,
    UnsafeExecutionError,
)
from .models import ResolutionResult
from .ranking import RankingWeights
from .resolver import Resolver

__version__ = "0.1.0"
__all__ = [
    "AmbiguousIntentError",
    "ArgumentMappingError",
    "MagicError",
    "ModelNotAvailableError",
    "RankingWeights",
    "ResolutionError",
    "ResolutionResult",
    "Resolver",
    "UnsafeExecutionError",
    "configure",
    "get_resolver",
    "resolve",
]
_resolver: Resolver | None = None
_lock = threading.RLock()


def configure(**options: Any) -> Resolver:
    """Replace the default resolver. Loading weights stays lazy and local-only."""
    global _resolver
    with _lock:
        _resolver = Resolver(**options)
        return _resolver


def get_resolver() -> Resolver:
    global _resolver
    with _lock:
        if _resolver is None:
            _resolver = Resolver()
        return _resolver


def resolve(
    intent: str,
    *args: Any,
    arguments: dict[str, Any] | None = None,
    execute: bool = False,
    top_k: int | None = None,
    **kwargs: Any,
) -> ResolutionResult | Any:
    """Explain by default. Set execute=True to invoke the selected registered callable."""
    return get_resolver().resolve(
        intent, *args, arguments=arguments, execute=execute, top_k=top_k, **kwargs
    )


def __getattr__(name: str):
    if not name.isidentifier() or name.startswith("_"):
        raise AttributeError(name)

    def semantic_call(*args: Any, **kwargs: Any) -> Any:
        return get_resolver().resolve(name, *args, execute=True, **kwargs)

    semantic_call.__name__ = name
    semantic_call.__doc__ = f"Resolve and execute the semantic intent {name!r}."
    return semantic_call
