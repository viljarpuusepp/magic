"""Exercise the curated bindings against installed libraries, including constructors."""

from io import StringIO

import pandas as pd
import pytest

from magic.executor import Executor
from magic.models import ResolutionResult

CASES = {
    **{
        f"numpy.{name}": (([1, 2, 3],), {})
        for name in (
            "mean",
            "median",
            "std",
            "var",
            "sum",
            "min",
            "max",
            "prod",
            "unique",
            "sort",
        )
    },
    "numpy.quantile": (([1, 2, 3], 0.5), {}),
    "numpy.percentile": (([1, 2, 3], 50), {}),
    "numpy.dot": (([1, 2], [3, 4]), {}),
    "numpy.linalg.norm": (([3, 4],), {}),
    "numpy.random.normal": ((), {"loc": 0, "scale": 1, "size": 3}),
    "numpy.random.uniform": ((), {"low": 0, "high": 1, "size": 3}),
    "numpy.random.exponential": ((), {"scale": 1, "size": 3}),
    "numpy.random.randint": ((), {"low": 0, "high": 5, "size": 3}),
    "scipy.stats.norm.pdf": ((0,), {}),
    "scipy.stats.norm.cdf": ((0,), {}),
    "scipy.stats.norm.ppf": ((0.5,), {}),
    "scipy.stats.norm.rvs": ((), {"size": 3, "random_state": 42}),
    "scipy.stats.norm": ((), {"loc": 0, "scale": 1}),
    "scipy.stats.ttest_ind": (([1, 2, 3], [3, 4, 5]), {}),
    "scipy.stats.pearsonr": (([1, 2, 3], [3, 5, 4]), {}),
    "scipy.stats.spearmanr": (([1, 2, 3], [3, 5, 4]), {}),
    "scipy.stats.zscore": (([1, 2, 3],), {}),
    **{
        f"statistics.{name}": (([1, 2, 3],), {})
        for name in (
            "mean",
            "median",
            "stdev",
            "variance",
        )
    },
    **{
        name: ((), {})
        for name in (
            "sklearn.decomposition.PCA",
            "sklearn.cluster.KMeans",
            "sklearn.linear_model.LinearRegression",
            "sklearn.linear_model.LogisticRegression",
            "sklearn.preprocessing.StandardScaler",
            "sklearn.preprocessing.MinMaxScaler",
        )
    },
    "sklearn.metrics.mean_squared_error": (([1, 2, 3], [2, 2, 4]), {}),
    "sklearn.metrics.accuracy_score": (([0, 1, 1], [0, 1, 0]), {}),
    "pandas.Series.mean": ((pd.Series([1, 2, 3]),), {}),
    "pandas.Series.median": ((pd.Series([1, 2, 3]),), {}),
    "pandas.read_csv": ((), {}),  # Fresh file object supplied per test.
    "pandas.concat": (([pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [2]})],), {}),
    "pandas.to_datetime": ((["2026-01-01", "2026-01-02"],), {}),
}


@pytest.mark.parametrize("function_id", sorted(CASES))
def test_every_curated_binding_executes(registry, function_id):
    assert set(CASES) == {d.id for d in registry.functions}
    args, kwargs = CASES[function_id]
    if function_id == "pandas.read_csv":
        args = (StringIO("a,b\n1,2\n3,4\n"),)
    result = ResolutionResult(
        "registry binding smoke test", "resolved", function_id, 1.0, (), args, kwargs
    )
    assert Executor(registry).execute(result) is not None
