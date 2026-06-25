import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest
from statsmodels.stats.proportion import proportion_confint, proportions_ztest
from scipy.stats import fisher_exact

from config import CHANCE_LEVEL


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Wilson score confidence interval for a binomial proportion.
    """
    if n <= 0:
        return (np.nan, np.nan)
    low, high = proportion_confint(count=successes, nobs=n, alpha=alpha, method="wilson")
    return (float(low), float(high))


def binomial_vs_chance(successes: int, n: int, p0: float = CHANCE_LEVEL) -> float:
    """
    Exact binomial test vs a chance level p0.
    """
    if n <= 0:
        return np.nan
    return float(binomtest(k=successes, n=n, p=p0, alternative="greater").pvalue)


def two_proportion_ztest_vs_human(
    successes_model: int,
    n_model: int,
    successes_human: int,
    n_human: int,
) -> float:
    """
    Two-proportion z-test comparing model accuracy to human benchmark.
    """
    if n_model <= 0 or n_human <= 0:
        return np.nan
    count = np.array([successes_model, successes_human])
    nobs = np.array([n_model, n_human])
    stat, p = proportions_ztest(count=count, nobs=nobs, alternative="two-sided")
    _ = stat
    return float(p)


def fisher_exact_test(
    table: np.ndarray,
    alternative: str = "two-sided",
) -> float:
    """
    Fisher's exact test with Bonferroni correction handled by caller.
    """
    if table.shape != (2, 2):
        raise ValueError("fisher_exact_test expects a 2x2 contingency table")
    _, p = fisher_exact(table, alternative=alternative)
    return float(p)


def bonferroni_correction(
    p_values: Sequence[float],
    n_tests: Optional[int] = None,
) -> List[float]:
    """Bonferroni-adjusted p-values (capped at 1.0)."""
    n = n_tests if n_tests is not None else len(p_values)
    if n <= 0:
        return []
    return [min(1.0, float(p) * n) for p in p_values]


def cohens_h(p1: float, p2: float) -> float:
    """
    Cohen's h effect size for proportions.
    """
    p1 = float(p1)
    p2 = float(p2)
    return float(2.0 * np.arcsin(np.sqrt(p1)) - 2.0 * np.arcsin(np.sqrt(p2)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistical utilities for mental state recognition analyses.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a small demo of the statistical tests.",
    )

    args = parser.parse_args()
    if args.demo:
        successes = 35
        n = 118
        ci = wilson_ci(successes, n)
        p = binomial_vs_chance(successes, n, p0=CHANCE_LEVEL)
        print({"successes": successes, "n": n, "wilson_ci": ci, "p_binom_gt_chance": p})


if __name__ == "__main__":
    main()

