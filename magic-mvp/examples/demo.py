"""Run from the project root: python examples/demo.py."""

from pathlib import Path

import numpy as np

import magic

model = Path(__file__).resolve().parents[1] / "models" / "all-MiniLM-L6-v2"
if model.is_dir():
    magic.configure(model_path=model)

print("Mean:", magic.mean([1, 2, 3, 4]))
print("Standard deviation:", magic.standard_deviation([1, 2, 3, 4]))
samples = magic.normal_distribution(mean=1, std=5, samples=100)
print("Normal samples:", samples.shape, samples[:5])
print("Normal CDF at 1.96:", magic.normal_cdf(1.96))
print(magic.resolve("reduce dimensionality using PCA", execute=False))
print(magic.resolve("normal distribution", mean=1, std=5, samples=100))

pca = magic.pca(components=2)
data = np.random.default_rng(42).normal(size=(10, 4))
print("PCA output:", pca.fit_transform(data).shape)
try:
    magic.normal_distribution(1, 5)
except magic.AmbiguousIntentError as e:
    print("\nExpected ambiguity:\n", e)
