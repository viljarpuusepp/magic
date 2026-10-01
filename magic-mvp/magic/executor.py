"""Execute one allowlisted callable after binding; never generate or evaluate Python code."""

from __future__ import annotations

import inspect
import re
from typing import Any

from .arguments import map_arguments
from .errors import ArgumentMappingError, ResolutionError, UnsafeExecutionError
from .models import ResolutionResult
from .registry import Registry


class Executor:
    def __init__(self, registry: Registry) -> None:
        self.registry = registry

    def execute(self, result: ResolutionResult) -> Any:
        if result.status != "resolved" or result.selected_function is None:
            raise ResolutionError(result)
        descriptor = self.registry.get(result.selected_function)
        target = self.registry.load_callable(descriptor.id)
        try:
            runtime_signature = inspect.signature(target)
        except (ValueError, TypeError):
            runtime_signature = None
        mapped = map_arguments(
            descriptor,
            result.mapped_args,
            result.mapped_arguments,
            runtime_signature=runtime_signature,
        )
        if mapped.errors or mapped.missing:
            detail = "; ".join(mapped.errors) or f"Missing required parameters: {mapped.missing}"
            raise ArgumentMappingError(f"{descriptor.id}: {detail}")
        if descriptor.effect == "local_read":
            values = descriptor.signature.bind(*mapped.args, **mapped.kwargs).arguments
            path = values.get("filepath_or_buffer")
            if isinstance(path, str) and re.match(r"^[a-zA-Z][\w+.-]*://", path):
                raise UnsafeExecutionError("read_csv accepts local paths or file objects only")
        return target(*mapped.args, **mapped.kwargs)
