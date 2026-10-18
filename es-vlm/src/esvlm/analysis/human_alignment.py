"""RQ4: RSA and Jensen-Shannon vs human EQ / SQ-R tertiles.

The human workbook stays on the Mac. Run this locally from pulled
aggregate model outputs. Write aggregates only, never item text.
"""

from __future__ import annotations

import numpy as np


def jensen_shannon(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))


def rsa(rdm_a: np.ndarray, rdm_b: np.ndarray) -> float:
    """Spearman correlation of off-diagonal RDM entries."""

    from scipy.stats import spearmanr

    mask = np.triu(np.ones_like(rdm_a, dtype=bool), k=1)
    return float(spearmanr(rdm_a[mask], rdm_b[mask]).correlation)


def error_rdm(errors: np.ndarray) -> np.ndarray:
    """Pairwise absolute difference of item error rates."""

    e = np.asarray(errors, dtype=np.float64).reshape(-1)
    return np.abs(e[:, None] - e[None, :])
