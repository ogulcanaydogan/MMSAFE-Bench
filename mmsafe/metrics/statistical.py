"""Statistical utilities for safety metrics."""

from __future__ import annotations

import numpy as np


def bootstrap_confidence_interval(
    values: list[bool] | list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric.

    Returns (point_estimate, ci_lower, ci_upper).
    """
    if not values:
        return (0.0, 0.0, 0.0)

    arr = np.array(values, dtype=float)
    point = float(arr.mean())

    rng = np.random.default_rng(seed)
    n = len(arr)

    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        boot_means[i] = arr[indices].mean()

    alpha = (1 - ci) / 2
    lower = float(np.quantile(boot_means, alpha))
    upper = float(np.quantile(boot_means, 1 - alpha))
    return (point, lower, upper)


def wilson_score_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion.

    More accurate than normal approximation for small samples.
    """
    if total == 0:
        return (0.0, 0.0)

    from scipy import stats  # type: ignore[import-untyped]

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / total
    n = total

    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * (p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5 / denominator

    return (max(0.0, center - margin), min(1.0, center + margin))
