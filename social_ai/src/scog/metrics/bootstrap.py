"""Nonparametric bootstrap for a mean."""

from __future__ import annotations

import numpy as np


def bootstrap_mean_ci(
    values,
    *,
    n_boot: int = 1000,
    seed: int = 20261029,
    alpha: float = 0.05,
) -> dict:
    sample = np.asarray(values, dtype=float)
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        raise RuntimeError("bootstrap: zero finite values")
    rng = np.random.default_rng(seed)
    draws = rng.choice(sample, size=(n_boot, sample.size), replace=True)
    means = draws.mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {
        "n": int(sample.size),
        "mean": float(sample.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
    }
