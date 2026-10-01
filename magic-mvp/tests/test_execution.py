from dataclasses import replace
from pathlib import Path

import pytest

from magic.errors import ArgumentMappingError, ResolutionError, UnsafeExecutionError
from magic.executor import Executor
from magic.models import ResolutionResult


def resolved(function, *args, **kwargs):
    return ResolutionResult("test", "resolved", function, 1.0, (), args, kwargs)


def test_executor_rejects_target_outside_allowlist(registry):
    with pytest.raises(UnsafeExecutionError):
        Executor(registry).execute(resolved("os.system", "echo unsafe"))


def test_executor_revalidates_arguments(registry):
    with pytest.raises(ArgumentMappingError):
        Executor(registry).execute(resolved("numpy.mean"))
    with pytest.raises(ArgumentMappingError):
        Executor(registry).execute(resolved("numpy.mean", [1, 2], unknown=1))


def test_executor_rejects_ambiguity(registry):
    result = replace(resolved("numpy.mean", [1, 2]), status="ambiguous")
    with pytest.raises(ResolutionError):
        Executor(registry).execute(result)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.org/data.csv",
        "http://host/file",
        "s3://bucket/key",
        "ftp://host/file",
        "file:///etc/passwd",
    ],
)
def test_read_csv_never_fetches_a_url(registry, url):
    with pytest.raises(UnsafeExecutionError):
        Executor(registry).execute(resolved("pandas.read_csv", url))


def test_read_csv_accepts_a_local_file(registry, tmp_path):
    file = tmp_path / "data.csv"
    file.write_text("a,b\n1,2\n3,4\n")
    result = Executor(registry).execute(resolved("pandas.read_csv", file))
    assert result["a"].tolist() == [1, 3]


def test_no_eval_or_exec_in_package():
    import ast

    import magic

    for file in Path(magic.__file__).parent.rglob("*.py"):
        tree = ast.parse(file.read_text())
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"eval", "exec"}
            for node in ast.walk(tree)
        ), file
