"""
TOST (Fisher-z), JZS Bayes factor via pingouin, and bootstrap CI for Pearson r.

TOST eps default 0.30 ≈ Cohen's 'medium' correlation threshold (|r|≈.3).
LIMITATION: this bound is an analyst choice, not a universal standard —
report sensitivity at eps=0.20 and 0.40 alongside the default.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.stats import pearsonr

# Jeffreys / Lee & Wagenmakers-style verbal labels for BF10
_BF_LABELS: List[Tuple[float, str]] = [
    (100.0, "extreme evidence for H1"),
    (30.0, "very strong evidence for H1"),
    (10.0, "strong evidence for H1"),
    (3.0, "moderate evidence for H1"),
    (1.0, "anecdotal evidence for H1"),
]


def jeffreys_label(bf10: float) -> str:
    if not np.isfinite(bf10) or bf10 <= 0:
        return "undefined"
    if bf10 == 1.0:
        return "no preference"
    if bf10 > 1.0:
        for thr, lab in _BF_LABELS:
            if bf10 >= thr:
                return lab
        return "anecdotal evidence for H1"
    # prefer H0: label on BF01 = 1/BF10
    bf01 = 1.0 / bf10
    for thr, lab in _BF_LABELS:
        if bf01 >= thr:
            return lab.replace("H1", "H0")
    return "anecdotal evidence for H0"


def fisher_z(r: float) -> float:
    return float(np.arctanh(np.clip(r, -0.999999, 0.999999)))


def tost_correlation(
    x: np.ndarray,
    y: np.ndarray,
    eps: float = 0.30,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    TOST for Pearson r on Fisher-z scale with bounds [-eps, +eps].

    SE = 1/sqrt(n-3). Equivalence established if both one-sided tests
    reject at level alpha (equivalently TOST p = max(p1,p2) < alpha).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n <= 3:
        return {"n": float(n), "r": float("nan"), "eps": float(eps), "p_tost": float("nan"), "equivalent": 0.0}
    r, _ = pearsonr(x, y)
    z_r = fisher_z(float(r))
    se = 1.0 / np.sqrt(n - 3)
    z_lo = fisher_z(-float(eps))
    z_hi = fisher_z(float(eps))
    # H0: rho <= -eps  vs  H1: rho > -eps
    p_lower = float(stats.norm.sf((z_r - z_lo) / se))
    # H0: rho >= +eps  vs  H1: rho < +eps
    p_upper = float(stats.norm.cdf((z_r - z_hi) / se))
    p_tost = float(max(p_lower, p_upper))
    return {
        "n": float(n),
        "r": float(r),
        "eps": float(eps),
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_tost": p_tost,
        "equivalent": float(p_tost < alpha),
        "alpha": float(alpha),
    }


def pearson_bf(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """JZS Bayes factor for Pearson correlation via pingouin (Wetzels & Wagenmakers style)."""
    import pandas as pd
    import pingouin as pg

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    df = pd.DataFrame({"x": x[mask], "y": y[mask]})
    if len(df) < 5:
        return {"n": float(len(df)), "r": float("nan"), "BF10": float("nan"), "BF01": float("nan"), "label": "undefined"}
    res = pg.corr(df["x"], df["y"], method="pearson")
    # pingouin returns BF10 in column 'BF10' when available
    r = float(res["r"].iloc[0])
    bf10 = res["BF10"].iloc[0] if "BF10" in res.columns else np.nan
    try:
        bf10 = float(bf10)
    except (TypeError, ValueError):
        bf10 = float("nan")
    bf01 = (1.0 / bf10) if np.isfinite(bf10) and bf10 > 0 else float("nan")
    pcol = "p_val" if "p_val" in res.columns else ("p-val" if "p-val" in res.columns else None)
    p_pearson = float(res[pcol].iloc[0]) if pcol else float("nan")
    return {
        "n": float(len(df)),
        "r": r,
        "p_pearson": p_pearson,
        "BF10": bf10,
        "BF01": bf01,
        "label": jeffreys_label(bf10) if np.isfinite(bf10) else "undefined",
    }


def bootstrap_corr_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Percentile bootstrap 95% CI for Pearson r (resample item pairs)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 5:
        return {"r": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": float(n)}
    r_obs, _ = pearsonr(x, y)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ri, _ = pearsonr(x[idx], y[idx])
        boots[i] = ri
    lo = float(np.nanpercentile(boots, 100 * alpha / 2))
    hi = float(np.nanpercentile(boots, 100 * (1 - alpha / 2)))
    return {"r": float(r_obs), "ci_low": lo, "ci_high": hi, "n": float(n), "n_boot": float(n_boot)}


def equivalence_battery(x: np.ndarray, y: np.ndarray, seed: int = 42) -> Dict[str, object]:
    """TOST at eps 0.20/0.30/0.40 + BF + bootstrap CI."""
    out: Dict[str, object] = {
        "bootstrap": bootstrap_corr_ci(x, y, seed=seed),
        "bayes": pearson_bf(x, y),
        "tost": {},
    }
    for eps in (0.20, 0.30, 0.40):
        out["tost"][f"eps_{eps:.2f}"] = tost_correlation(x, y, eps=eps)
    return out
