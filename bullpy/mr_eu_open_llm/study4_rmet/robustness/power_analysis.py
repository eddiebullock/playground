"""
Fisher-z power analysis for Pearson correlations (item-level H1, n≈36).

Standard large-sample approximation: z = artanh(r), SE = 1/sqrt(n-3).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


def fisher_z(r: float) -> float:
    r = float(np.clip(r, -0.999999, 0.999999))
    return float(np.arctanh(r))


def achieved_power(
    n: int,
    r: float,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    """Power to detect Pearson correlation r against H0: rho=0 at sample size n."""
    if n <= 3:
        return float("nan")
    se = 1.0 / np.sqrt(n - 3)
    ncp = fisher_z(r) / se  # under true r (signed); use |r| for magnitude
    ncp = abs(ncp)
    if two_sided:
        z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
        # two-sided power
        power = float(
            stats.norm.sf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp)
        )
    else:
        z_crit = stats.norm.ppf(1.0 - alpha)
        power = float(stats.norm.sf(z_crit - ncp))
    return float(np.clip(power, 0.0, 1.0))


def min_detectable_r(
    n: int,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> float:
    """Smallest |r| detectable at given n, alpha, power (Fisher-z approximation)."""
    if n <= 3:
        return float("nan")
    lo, hi = 1e-6, 0.999
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        p = achieved_power(n, mid, alpha=alpha, two_sided=two_sided)
        if p < power:
            lo = mid
        else:
            hi = mid
    return float(hi)


def power_summary_table(
    n: int,
    rs: Optional[Iterable[float]] = None,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> pd.DataFrame:
    """Table of achieved power for hypothetical true r values at fixed n."""
    if rs is None:
        rs = (0.1, 0.2, 0.3, 0.4, 0.5)
    rows: List[Dict[str, float]] = []
    mdr = min_detectable_r(n, alpha=alpha, power=target_power, two_sided=True)
    for r in rs:
        rows.append(
            {
                "n": float(n),
                "true_r": float(r),
                "alpha": float(alpha),
                "achieved_power": achieved_power(n, float(r), alpha=alpha),
                "min_detectable_r_at_power": mdr,
                "target_power": float(target_power),
            }
        )
    return pd.DataFrame(rows)


def format_power_sentence(n: int, alpha: float = 0.05, power: float = 0.80, r_ref: float = 0.3) -> str:
    mdr = min_detectable_r(n, alpha=alpha, power=power)
    ap = achieved_power(n, r_ref, alpha=alpha)
    return (
        f"at n={n}, alpha={alpha}, power={power:.2f}, the minimum detectable |r| is "
        f"{mdr:.3f}; our achieved power to detect r={r_ref} is {100 * ap:.1f}%."
    )
