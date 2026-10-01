# Project: Semantic Function Resolver for Python

Build a Python package that allows users to call functionality by **semantic intent** instead of knowing the exact Python library, module, class, or function name.

The package must work **locally without any external LLM API**.
The semantic resolver should use a small local transformer/embedding model and deterministic Python logic.

The working project name can be:

```text
magic
```

Example desired usage:

```python
import magic

x = magic.normal_distribution(
    mean=1,
    std=5,
    samples=100
)
```

The resolver should internally determine that an appropriate implementation is for example:

```python
numpy.random.normal(
    loc=1,
    scale=5,
    size=100
)
```

Another example:

```python
magic.normal_cdf(1.96)
```

could resolve to:

```python
scipy.stats.norm.cdf(1.96)
```

The important point is that the user does **not** need to know the actual library API.

---

# 1. Core concept

The system maps:

```text
user intent
    ↓
semantic representation
    ↓
candidate Python functions
    ↓
ranking
    ↓
argument mapping
    ↓
selected Python function
    ↓
execution
```

Do not use OpenAI, Anthropic, Gemini, or any other remote LLM/API.

The system should remain usable offline once the required packages and embedding model are installed.

---

# 2. Architecture

Implement the solution in separate layers.

Suggested architecture:

```text
magic/
    __init__.py

    resolver.py
    registry.py
    embeddings.py
    ranking.py
    arguments.py
    executor.py

    discovery/
        __init__.py
        introspection.py
        docstrings.py

    models/
        function_descriptor.py
        resolution_result.py

    data/
        functions.json

tests/
examples/
README.md
pyproject.toml
```

Keep the components loosely coupled.

---

# 3. Function registry

Create a registry containing descriptions of callable Python functions.

Each function should have metadata such as:

```python
FunctionDescriptor(
    id="numpy.random.normal",
    package="numpy",
    module="numpy.random",
    function="normal",

    description="""
    Draw random samples from a normal Gaussian distribution.
    Useful for generating normally distributed random values.
    """,

    signature="normal(loc=0.0, scale=1.0, size=None)",

    aliases=[
        "normal distribution",
        "gaussian distribution",
        "random normal values",
        "normal samples"
    ],

    parameter_aliases={
        "mean": "loc",
        "mu": "loc",
        "std": "scale",
        "standard_deviation": "scale",
        "sigma": "scale",
        "samples": "size",
        "count": "size"
    }
)
```

For the first prototype, manually register approximately 20–50 common functions.

Focus initially on:

```text
numpy
scipy
pandas
statistics
sklearn
```

Example functions:

```text
numpy.mean
numpy.median
numpy.std
numpy.var
numpy.sum
numpy.random.normal
numpy.random.uniform

scipy.stats.norm.pdf
scipy.stats.norm.cdf
scipy.stats.norm.ppf
scipy.stats.ttest_ind
scipy.stats.pearsonr

statistics.mean
statistics.median
statistics.stdev

sklearn.linear_model.LinearRegression
sklearn.cluster.KMeans
sklearn.preprocessing.StandardScaler
sklearn.decomposition.PCA

pandas.read_csv
pandas.concat
```

---

# 4. Local transformer embeddings

Use a small local transformer model.

Preferred initial implementation:

```python
sentence-transformers
```

with a compact model such as:

```text
all-MiniLM-L6-v2
```

The embedding model must run locally.

Create embeddings for every function using text combining:

```text
function name
description
aliases
parameter meanings
package name
```

For example:

```text
numpy.random.normal

Generate random samples from a Gaussian or normal distribution.

Aliases:
normal distribution
Gaussian samples
random normal values

Parameters:
loc = mean
scale = standard deviation
size = number of samples
```

Store the resulting embeddings locally.

Do not recompute all embeddings on every function call.

---

# 5. Semantic resolution

Implement:

```python
resolver.resolve(...)
```

Example:

```python
result = resolver.resolve(
    "normal distribution",
    arguments={
        "mean": 1,
        "std": 5,
        "samples": 100
    }
)
```

The resolver should:

1. Encode the requested semantic intent.
2. Compare it with registered function embeddings.
3. Find the top-K candidates.
4. Apply deterministic ranking rules.
5. Map argument aliases.
6. Validate the selected function signature.
7. Return a `ResolutionResult`.

Example:

```python
ResolutionResult(
    requested_name="normal_distribution",
    selected_function="numpy.random.normal",
    score=0.94,
    mapped_arguments={
        "loc": 1,
        "scale": 5,
        "size": 100
    }
)
```

---

# 6. Ranking

Do not rely only on embedding similarity.

Use a weighted score.

Initial idea:

```text
final_score =
    0.55 * semantic_similarity
  + 0.20 * argument_match
  + 0.15 * type_match
  + 0.05 * installed_package_score
  + 0.05 * package_prior
```

Make the scoring weights configurable.

The resolver should prefer a function when:

* its semantic meaning matches the request,
* supplied arguments can be mapped to the function signature,
* supplied Python types are compatible,
* the required package is installed.

---

# 7. Argument mapping

Implement semantic argument aliases.

Example user call:

```python
magic.normal_distribution(
    mean=1,
    std=5,
    samples=100
)
```

Selected function:

```python
numpy.random.normal(
    loc=1,
    scale=5,
    size=100
)
```

Mapping:

```text
mean -> loc
std -> scale
samples -> size
```

The mapping should initially be deterministic and registry-driven.

Do not use a generative LLM for parameter mapping.

Also use Python `inspect.signature()` when possible.

---

# 8. Runtime type information

Use runtime Python types to improve selection.

Example:

```python
magic.mean(np_array)
```

should prefer:

```text
numpy.mean
```

while:

```python
magic.mean(pd_series)
```

may prefer:

```text
pandas.Series.mean
```

Type matching should therefore be part of the ranking system.

Do not execute candidate functions while ranking them.

---

# 9. Dynamic API

Support this syntax:

```python
magic.normal_distribution(
    mean=1,
    std=5,
    samples=100
)
```

The method `normal_distribution` does not need to be explicitly declared.

Implement the dynamic interface using:

```python
__getattr__
```

or another clean Python mechanism.

Conceptually:

```python
magic.normal_distribution(...)
```

should internally become:

```python
resolver.resolve(
    intent="normal distribution",
    arguments=...
)
```

and then execute the selected callable.

Also provide an explicit API:

```python
magic.resolve(
    "generate 100 normal random values",
    mean=1,
    std=5,
    samples=100
)
```

---

# 10. Safe execution

Do not dynamically execute arbitrary generated Python code.

Resolve only to functions that exist in the registry.

Execution should use:

```python
importlib
getattr
inspect
```

or equivalent safe Python mechanisms.

Avoid `eval()`.

Avoid `exec()`.

The registry acts as the allowlist of executable functions.

---

# 11. Explainability

The user must be able to inspect why a function was selected.

Support:

```python
result = magic.resolve(
    "normal distribution",
    mean=1,
    std=5,
    samples=100,
    execute=False
)

print(result)
```

Output should contain something similar to:

```text
Intent:
normal distribution

Selected:
numpy.random.normal

Confidence:
0.94

Mapped parameters:
mean -> loc
std -> scale
samples -> size

Alternative candidates:
scipy.stats.norm.rvs       0.91
random.gauss               0.82
torch.normal               0.79
```

Do not call this score a statistical probability unless it actually is one.

Use terms such as:

```text
similarity score
resolution score
ranking score
```

---

# 12. Ambiguity handling

The resolver must recognize ambiguous requests.

For example:

```python
magic.normal_distribution(1, 5)
```

may mean:

* generate normal samples,
* calculate PDF,
* calculate CDF,
* define a distribution object.

If the top two candidate scores are very close and their semantics are materially different, do not silently choose.

Return an ambiguity result such as:

```text
Ambiguous request.

Candidates:

1. numpy.random.normal
   Generate samples from a normal distribution

2. scipy.stats.norm.pdf
   Calculate normal probability density

3. scipy.stats.norm.cdf
   Calculate cumulative probability
```

Create a configurable ambiguity threshold.

---

# 13. First milestone

The first milestone should NOT attempt to index the whole PyPI ecosystem.

Build a reliable proof-of-concept first.

Milestone 1:

```text
5 packages
20–50 functions
local transformer
semantic retrieval
argument mapping
ranking
execution
tests
```

The goal is to verify that the architecture works.

---

# 14. Evaluation dataset

Create a small test dataset.

Example:

```python
TEST_CASES = [
    (
        "average of these numbers",
        "numpy.mean"
    ),
    (
        "standard deviation",
        "numpy.std"
    ),
    (
        "generate Gaussian random values",
        "numpy.random.normal"
    ),
    (
        "normal cumulative probability",
        "scipy.stats.norm.cdf"
    ),
    (
        "reduce dimensions using PCA",
        "sklearn.decomposition.PCA"
    ),
    (
        "group observations into clusters",
        "sklearn.cluster.KMeans"
    )
]
```

Measure at least:

```text
Top-1 accuracy
Top-3 accuracy
Top-5 accuracy
argument mapping accuracy
resolution latency
```

Add automated tests for these metrics.

---

# 15. Second milestone: automatic function discovery

After the manually curated registry works, implement automatic discovery.

Use Python introspection to inspect installed packages.

Potential information sources:

```python
module.__dict__
inspect.getmembers()
inspect.signature()
inspect.getdoc()
```

Extract:

```text
package
module
callable
signature
docstring
type hints
```

Generate a `FunctionDescriptor`.

Do not initially crawl the internet.

Only inspect locally installed Python packages.

---

# 16. Future architecture

Design the system so that later it could support:

```text
10,000+
100,000+
1,000,000+
```

function descriptors.

Future search could use:

```text
FAISS
HNSW
vector indexes
approximate nearest neighbour search
```

Do not add this complexity to the first implementation unless necessary.

Keep the embedding index behind an abstraction.

For example:

```python
class FunctionIndex:
    def add(...)
    def search(...)
    def save(...)
    def load(...)
```

---

# 17. Optional future ML work

Do not implement these before the basic resolver works, but keep the architecture compatible with them.

Possible later improvements:

### Reranker

Use a small cross-encoder to rerank the top semantic candidates.

### Fine-tuning

Fine-tune the embedding model using pairs:

```text
user intent
correct function
```

### Contrastive training

Create positive and negative function pairs.

Example:

```text
"calculate normal cumulative probability"
positive -> scipy.stats.norm.cdf
negative -> scipy.stats.norm.pdf
negative -> numpy.random.normal
```

### Context-aware resolution

Use:

```text
argument values
argument types
imports
variable types
current package context
```

as additional features.

---

# 18. Development requirements

Use:

```text
Python 3.11+
pytest
type hints
dataclasses or pydantic
ruff
```

Prefer simple, readable code.

Avoid unnecessary frameworks.

Do not use LangChain.

Do not use LlamaIndex.

Do not use remote APIs.

Do not introduce a vector database for the first prototype unless the number of functions makes it necessary.

---

# 19. README

Create a useful README demonstrating:

```python
import magic

x = magic.normal_distribution(
    mean=0,
    std=1,
    samples=1000
)

print(x)
```

and:

```python
resolution = magic.resolve(
    "calculate cumulative normal probability",
    value=1.96,
    execute=False
)

print(resolution.selected_function)
```

Expected:

```text
scipy.stats.norm.cdf
```

Also explain clearly:

```text
magic is not an LLM.

It uses a local transformer encoder to semantically retrieve Python
functions and deterministic logic to validate and execute them.
```

---

# 20. Important design principle

The transformer should answer only:

```text
Which registered Python function is semantically closest to this intent?
```

It should NOT generate arbitrary Python programs.

Use conventional Python mechanisms for:

```text
type checking
signature validation
argument mapping
imports
execution
error handling
```

This hybrid architecture is intentional:

```text
Transformer:
semantic understanding

Python:
deterministic execution
```

---

# 21. Start implementation

Start with a minimal but working end-to-end prototype.

Implement in this order:

```text
1. Project structure
2. FunctionDescriptor
3. Hard-coded registry
4. Local sentence-transformer embeddings
5. Semantic candidate search
6. Ranking
7. Argument aliases
8. Dynamic magic.xxx API
9. Safe function execution
10. ResolutionResult / explain mode
11. Unit tests
12. Evaluation script
13. README
```

After each major component, add tests.

Do not over-engineer the first version.

The first success criterion is that these calls work correctly:

```python
magic.mean([1, 2, 3, 4])

magic.standard_deviation([1, 2, 3, 4])

magic.normal_distribution(
    mean=1,
    std=5,
    samples=100
)

magic.normal_cdf(1.96)
```

and that:

```python
magic.resolve(
    "reduce the dimensionality of this dataset with PCA",
    execute=False
)
```

ranks:

```text
sklearn.decomposition.PCA
```

among the top candidates.
