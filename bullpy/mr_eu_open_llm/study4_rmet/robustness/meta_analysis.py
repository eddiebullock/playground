"""
Random-effects meta-analysis of per-model Pearson correlations (Fisher-z).

DerSimonian–Laird τ²; reports pooled r, 95% CI, Q, I².
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
from scipy import stats


def fisher_z(r: float) -> float:
    return float(np.arctanh(np.clip(r, -0.999999, 0.999999)))


def inv_fisher_z(z: float) -> float:
    return float(np.tanh(z))


def random_effects_meta_fisher_z(
    rs: Sequence[float],
    ns: Sequence[int],
) -> Dict[str, float]:
    """
    Pool correlations with DerSimonian–Laird random effects on Fisher-z scale.

    Within-study variance for z_i is 1/(n_i - 3).
    """
    rs = [float(r) for r in rs]
    ns = [int(n) for n in ns]
    if len(rs) != len(ns) or len(rs) == 0:
        return {"k": 0.0, "r_pooled": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "I2": float("nan")}
    z = np.array([fisher_z(r) for r in rs], dtype=float)
    v = np.array([1.0 / (n - 3) for n in ns], dtype=float)
    w = 1.0 / v
    z_fixed = float(np.sum(w * z) / np.sum(w))
    q = float(np.sum(w * (z - z_fixed) ** 2))
    k = len(rs)
    df = k - 1
    c = float(np.sum(w) - np.sum(w**2) / np.sum(w))
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    w_star = 1.0 / (v + tau2)
    z_re = float(np.sum(w_star * z) / np.sum(w_star))
    se_re = float(np.sqrt(1.0 / np.sum(w_star)))
    z_crit = stats.norm.ppf(0.975)
    i2 = max(0.0, (q - df) / q) * 100.0 if q > 0 else 0.0
    return {
        "k": float(k),
        "r_pooled": inv_fisher_z(z_re),
        "z_pooled": z_re,
        "se_z": se_re,
        "ci_low": inv_fisher_z(z_re - z_crit * se_re),
        "ci_high": inv_fisher_z(z_re + z_crit * se_re),
        "tau2": float(tau2),
        "Q": q,
        "df": float(df),
        "I2": float(i2),
        "p_Q": float(stats.chi2.sf(q, df)) if df > 0 else float("nan"),
    }
