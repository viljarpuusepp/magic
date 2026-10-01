"""Actionable errors; no resolver failure silently falls back to another implementation."""


class MagicError(Exception):
    """Base exception for magic."""


class ModelNotAvailableError(MagicError):
    """The local sentence-transformer weights have not been installed."""


class ResolutionError(MagicError):
    def __init__(self, result):
        self.result = result
        super().__init__(str(result))


class AmbiguousIntentError(ResolutionError):
    """Several materially different operations match the intent."""


class ArgumentMappingError(MagicError, ValueError):
    """Arguments cannot be safely bound to the curated signature."""


class UnsafeExecutionError(MagicError):
    """The target is outside the allowlist or violates the execution policy."""
