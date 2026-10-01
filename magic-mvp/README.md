# magic — Semantic Function Resolver

`magic` resolves an intent to a curated Python callable using a **local sentence
transformer**, then maps arguments, validates signatures and types, and executes
the selected function. It is not an LLM: it generates neither text nor Python
programs and uses no external LLM/API. It works offline after installation.

## Install and run

Python 3.11+ is required. The distribution name is `semantic-function-magic`;
the Python import is `magic`. Avoid installing another package that uses that import.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1

# Optional CPU-only torch installation (avoids CUDA downloads):
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"

# Explicit one-time weights installation. This command requires internet.
python -m magic download-model --output models/all-MiniLM-L6-v2
```

The full `magic-mvp.zip` bundle already includes these pinned weights; skip the
download command when using it. A source-only archive or wheel requires the
one-time model installation. `constraints-tested.txt` records the versions used
for verification; optionally add `-c constraints-tested.txt` to the editable install.

The model at `models/all-MiniLM-L6-v2` is detected automatically from the current
project directory. For another location, configure it once in your program or
set `MAGIC_MODEL_PATH` to its absolute path. All resolution uses `local_files_only=True` and
`trust_remote_code=False`. A missing model produces a setup error; there is no
remote or non-transformer fallback.

```python
import magic

magic.configure(model_path="models/all-MiniLM-L6-v2")

print(magic.mean([1, 2, 3, 4]))  # 2.5
print(magic.standard_deviation([1, 2, 3, 4]))  # 1.118033988749895 (ddof=0)

x = magic.normal_distribution(mean=1, std=5, samples=100)
print(x.shape)  # (100,)
print(magic.normal_cdf(1.96))  # approximately 0.9750021

result = magic.resolve("reduce dimensionality using PCA", execute=False)
print(result.selected_function)  # sklearn.decomposition.PCA
print(result)
```

`python examples/demo.py` runs these examples, a PCA transform, and an ambiguity example.

## Explain, resolve, and execute

`magic.resolve()` explains **without execution by default**. Dynamic calls such
as `magic.normal_cdf(...)` execute. Explicit execution uses `execute=True`.

```python
resolution = magic.resolve("calculate cumulative normal probability", value=1.96, execute=False)
print(resolution.selected_function)  # scipy.stats.norm.cdf
print(resolution.argument_mapping)  # {'value': 'x'}
print(resolution.to_dict())  # scores, components, alternatives, errors

samples = magic.resolve(
    "generate Gaussian random values",
    arguments={"mean": 0, "std": 1, "samples": 1000},
    execute=True,
)
```

Aliases are explicit metadata: `mean -> loc`, `std -> scale`, `samples -> size`.
Unknown parameters, duplicate aliases, incompatible types, and missing required
arguments prevent execution. Explain mode accepts missing arguments so you can
search for a function before supplying data. It reports `missing_parameters` and
`executable`. For a fixed target use its registered ID, for example
`magic.resolve("statistics.stdev", [1, 2, 3, 4], execute=True)`.

Estimator intents return **unfitted estimators**, not an automatically trained pipeline:

```python
pca = magic.pca(components=2)
projected = pca.fit_transform([[1, 2, 3], [2, 1, 4], [3, 4, 2]])
```

Runtime types help select implementations. `magic.mean(numpy_array)` prefers
`numpy.mean`; `magic.mean(pandas_series)` prefers `pandas.Series.mean` and its
missing-value behavior. Generic standard deviation and variance default to
NumPy population statistics; use `sample_standard_deviation` / `sample_variance`
or `ddof=1` for sample estimates.

## Ambiguity and unknown intents

```python
result = magic.resolve("normal distribution", 1, 5)
assert result.ambiguous

try:
    magic.normal_distribution(1, 5)
except magic.AmbiguousIntentError as error:
    print(error.result.candidates)
```

A generic normal-distribution request without a named sample count is deliberately
ambiguous. Choose `normal_samples`, `normal_pdf`, `normal_cdf`, `normal_ppf`, or
`normal_distribution_object`; alternatively specify `samples=...`.
Other materially different operations trigger ambiguity when their ranking scores
are within the configured margin. Equivalent implementations of the same
operation (for example two arithmetic means) do not trigger ambiguity.
Unknown intents return `not_found`; invalid arguments return `invalid_arguments`.
Executing either raises `ResolutionError`. An ambiguity result never selects or
executes a function.

## Retrieval and ranking

The registry contains **44 functions across NumPy, SciPy, pandas, statistics, and
scikit-learn**. List them with `python -m magic registry` or inspect
`magic/data/functions.json`.

`all-MiniLM-L6-v2` embeds the name, description, aliases, and parameter meanings.
The normalized vectors are stored in a versioned local `.npz` cache; the registry
is not re-encoded on each call. Queries use exact cosine top-k search (default 12),
with curated alias matches added if necessary. The most specific matching alias
anchors the operation, ignoring grammar words and word order. Ranking uses:

| Component | Default weight |
|---|---:|
| Semantic similarity with a 0.20 curated-alias boost (capped at 1) | 0.55 |
| Argument compatibility | 0.20 |
| Runtime type compatibility | 0.15 |
| Installed package | 0.05 |
| Curated implementation preference | 0.05 |

Exact intents restrict selection to their semantic operation, so an invalid CDF
call cannot silently become a random sampler. Candidates with unknown arguments
or incompatible types cannot execute. Score ties use the function ID for stable
ordering. Scores are heuristics, **not statistical probabilities**. Sampling remains
random; deterministic ranking does not mean deterministic sampled values.

```python
magic.configure(
    model_path="models/all-MiniLM-L6-v2",
    cache_dir=".magic-cache",
    top_k=12,
    ambiguity_threshold=0.035,
    minimum_similarity=0.30,
    weights=magic.RankingWeights(),
)
```

`MAGIC_MODEL_PATH` and `MAGIC_CACHE_DIR` are environment alternatives. The model
loads lazily on CPU; model and registry vector construction are cold-start costs.

## Safety boundaries

Only packaged registry entries execute. Import paths come from that allowlist,
never from user text. The executor uses `importlib`, `getattr`, and
`inspect.signature`; it uses no `eval` or `exec`. Ranking does not execute
candidates. `read_csv` accepts local files and readable file objects and rejects
URL paths. No registry functions write files or run processes.

This is an allowlisted dispatcher, **not an operating-system sandbox**. Callers
control the supplied data and local paths; approved library code can allocate
memory and raise normal library exceptions. Large arrays and sample counts need
application-level resource limits. The resolver does not understand arbitrary
Python APIs, invent argument values from text, or discover the whole PyPI ecosystem.

## Tests and benchmark

Install the model first, then set its location for the CLI and tests:

```bash
# Linux/macOS
export MAGIC_MODEL_PATH="$PWD/models/all-MiniLM-L6-v2"
# Windows PowerShell instead:
# $env:MAGIC_MODEL_PATH = (Resolve-Path models/all-MiniLM-L6-v2).Path

pytest
ruff check .
python -m magic benchmark --output benchmark-results.json
python -m magic resolve "normal cumulative probability" --kwargs '{"value": 1.96}'
```

The test suite includes unit tests and real-model integration tests; a missing
model fails integration setup rather than silently substituting fake embeddings.
The benchmark reports Top-1, Top-3, Top-5 accuracy, argument mapping accuracy,
and latency. It measures unique queries with a warm model/index and does not
execute candidate functions. Detailed per-query results are saved for inspection.
The small curated evaluation set is an MVP smoke benchmark, not a claim about
unseen arbitrary Python functions. Extend it before expanding the registry.

Verified on Python 3.12.14 with the real CPU model: **93 tests passed** and Ruff
passed. On the 46 curated queries, final Top-1/Top-3/Top-5 were **100%** and
argument mapping was **100% (7 cases)**. Before alias/ranking rules, raw semantic
Top-1 was **89.1%** and Top-3 **97.8%**. Warm resolution averaged approximately
**15 ms**, with `OMP_NUM_THREADS=2` and `MKL_NUM_THREADS=2`; timing depends on hardware.
See `benchmark-results.json` for exact scores, dependencies, and every query.

## Project layout

`magic/registry.py` loads the metadata; `embeddings.py` owns local inference and
indexing; `arguments.py` maps and checks inputs; `ranking.py` defines the weights;
`resolver.py` selects and handles ambiguity; `executor.py` invokes the allowlisted
function. `models/` contains typed dataclasses. The index has a small interface so
a larger vector-search backend can be substituted later.

Automatic discovery, fine-tuning, cross-encoder reranking, and vector databases
are intentionally deferred until this curated MVP is validated.

The code is MIT licensed. Bundled MiniLM weights are Apache-2.0 licensed; their
upstream model card and license are retained in `models/all-MiniLM-L6-v2`.
