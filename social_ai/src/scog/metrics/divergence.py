"""Jensen-Shannon divergence in bits (log base 2), not the square-root distance."""

from __future__ import annotations

import numpy as np


def js_divergence(p, q, *, base: float = 2.0) -> float:
    left = np.asarray(p, dtype=float)
    right = np.asarray(q, dtype=float)
    if left.shape != right.shape:
        raise ValueError("JS divergence inputs must have the same shape")
    if left.sum() <= 0 or right.sum() <= 0:
        raise ValueError("JS divergence inputs must have positive mass")
    left = left / left.sum()
    right = right / right.sum()
    mix = 0.5 * (left + right)

    def _kl(a, b) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])) / np.log(base))

    return 0.5 * _kl(left, mix) + 0.5 * _kl(right, mix)


def mean_item_js(counts_a: np.ndarray, counts_b: np.ndarray) -> dict:
    """Mean and 95th percentile of per-item JS. `counts_*` are (n_items, n_choices)."""
    a = np.asarray(counts_a, dtype=float)
    b = np.asarray(counts_b, dtype=float)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("item count matrices must share shape (items, choices)")
    values = np.array([js_divergence(a[i], b[i]) for i in range(a.shape[0])])
    return {
        "n_items": int(a.shape[0]),
        "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)),
        "per_item": values,
    }
