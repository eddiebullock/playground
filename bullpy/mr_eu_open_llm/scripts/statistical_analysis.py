from __future__ import annotations

"""
Nested model x dataset x condition statistical analysis.

Human benchmarks: O'Reilly (EU video/multimodal), Lassalle (EU audio-only).
Mindreading has no direct human benchmark comparisons.
"""

import logging
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from statsmodels.stats.power import NormalIndPower

from config import CHANCE_LEVEL, CONFIRMATORY_N_MODELS, HUMAN_BENCHMARKS

logger = logging.getLogger(__name__)

CHANCE = CHANCE_LEVEL
FISHER_ALPHA = 0.005
FISHER_N_PAIRS = 6  # C(4,2) for four open VLMs


def wilson_ci(n_correct: int, n_total: int, confidence: float = 0.95) -> Tuple[float, float]:
    if n_total <= 0:
        raise ValueError("n_total must be > 0")
    p_hat = n_correct / n_total
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + (z**2) / n_total
    center = (p_hat + (z**2) / (2 * n_total)) / denom
    half = (z / denom) * math.sqrt((p_hat * (1 - p_hat) / n_total) + ((z**2) / (4 * (n_total**2))))
    return max(0.0, center - half), min(1.0, center + half)


def binomial_test_vs_chance(n_correct: int, n_total: int, chance: float = CHANCE) -> float:
    return float(stats.binomtest(k=n_correct, n=n_total, p=chance, alternative="greater").pvalue)


def two_proportion_z_test(p1: float, n1: int, p2: float, n2: int) -> Tuple[float, float]:
    x1, x2 = p1 * n1, p2 * n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    return float(z), float(2 * stats.norm.sf(abs(z)))


def cohen_h(p1: float, p2: float) -> float:
    return float(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2)))


def fisher_exact_2x2(correct1: int, total1: int, correct2: int, total2: int) -> Tuple[float, float]:
    table = np.array([[correct1, total1 - correct1], [correct2, total2 - correct2]], dtype=int)
    odds_ratio, p_value = stats.fisher_exact(table, alternative="two-sided")
    return float(odds_ratio), float(p_value)


def bonferroni_correct(p_values: Sequence[float], alpha: float = 0.05) -> Tuple[List[float], float]:
    pvals = [float(p) for p in p_values]
    m = max(1, len(pvals))
    return [min(1.0, p * m) for p in pvals], alpha / m


def power_analysis(effect_size_h: float, n: int, alpha: float = 0.01) -> float:
    return float(NormalIndPower().power(effect_size=abs(effect_size_h), nobs1=n, alpha=alpha, ratio=1.0))


def _extract_counts(obj: Any) -> Tuple[int, int]:
    if isinstance(obj, (tuple, list)) and len(obj) == 2:
        return int(obj[0]), int(obj[1])
    if isinstance(obj, Mapping):
        if "n_correct" in obj and "n_total" in obj:
            return int(obj["n_correct"]), int(obj["n_total"])
        if "correct" in obj and "total" in obj:
            return int(obj["correct"]), int(obj["total"])
        if "accuracy" in obj and "n_total" in obj:
            n_total = int(obj["n_total"])
            return int(round(float(obj["accuracy"]) * n_total)), n_total
    raise ValueError(f"Cannot extract counts from {type(obj)}")


def run_all_analyses(results_dict: Mapping[str, Any]) -> Dict[str, Any]:
    """
    results_dict: {model: {dataset: {condition: counts}}}
    Legacy {model: {dataset: counts}} is treated as video_only.
    """
    models = sorted(results_dict.keys())
    counts: Dict[str, Dict[str, Dict[str, Tuple[int, int]]]] = {m: {} for m in models}

    for m in models:
        for dataset_name, payload in (results_dict.get(m) or {}).items():
            if isinstance(payload, Mapping) and any(
                k in payload for k in ("n_correct", "correct", "total", "n_total", "accuracy")
            ):
                n_correct, n_total = _extract_counts(payload)
                counts[m].setdefault(dataset_name, {})["video_only"] = (n_correct, n_total)
            elif isinstance(payload, Mapping):
                for cond, obj in payload.items():
                    n_correct, n_total = _extract_counts(obj)
                    counts[m].setdefault(dataset_name, {})[str(cond)] = (n_correct, n_total)
            else:
                n_correct, n_total = _extract_counts(payload)
                counts[m].setdefault(dataset_name, {})["video_only"] = (n_correct, n_total)

    out: Dict[str, Any] = {
        "benchmarks": {"chance": CHANCE, "human_benchmarks": HUMAN_BENCHMARKS},
        "per_model_dataset": {},
        "comparisons": {"vs_chance": {}, "vs_human_benchmark": {}, "pairwise_models_fisher": {}},
    }

    for m in models:
        out["per_model_dataset"].setdefault(m, {})
        for dataset_name, per_cond in counts[m].items():
            ds_key = "eu_emotion" if dataset_name == "eu_emotions" else dataset_name
            out["per_model_dataset"][m].setdefault(dataset_name, {})
            for cond, (n_correct, n_total) in per_cond.items():
                acc = n_correct / n_total if n_total else 0.0
                ci_lo, ci_hi = wilson_ci(n_correct, n_total)
                p_chance = binomial_test_vs_chance(n_correct, n_total)
                out["per_model_dataset"][m][dataset_name][cond] = {
                    "n_correct": n_correct,
                    "n_total": n_total,
                    "accuracy": acc,
                    "wilson_ci_95": (ci_lo, ci_hi),
                }
                out["comparisons"]["vs_chance"].setdefault(m, {}).setdefault(dataset_name, {})[cond] = {
                    "p_value": p_chance
                }

                bench = (HUMAN_BENCHMARKS.get(ds_key) or {}).get(cond)
                if bench and bench.get("accuracy") is not None and bench.get("n") is not None:
                    p_b, n_b = float(bench["accuracy"]), int(bench["n"])
                    z, p = two_proportion_z_test(acc, n_total, p_b, n_b)
                    h = cohen_h(acc, p_b)
                    out["comparisons"]["vs_human_benchmark"].setdefault(m, {}).setdefault(dataset_name, {})[cond] = {
                        "benchmark": bench,
                        "z": z,
                        "p_value_raw": p,
                        "cohen_h": h,
                        "power_alpha_0_01": power_analysis(h, n=min(n_total, n_b), alpha=0.01),
                    }
                else:
                    out["comparisons"]["vs_human_benchmark"].setdefault(m, {}).setdefault(dataset_name, {})[cond] = None

    for dataset_name in {ds for m in models for ds in counts[m]}:
        for cond in {c for m in models for c in counts[m].get(dataset_name, {})}:
            available = [m for m in models if cond in counts[m].get(dataset_name, {})]
            if len(available) < 2:
                continue
            raw_pvals: List[float] = []
            pairs: List[Dict[str, Any]] = []
            for i in range(len(available)):
                for j in range(i + 1, len(available)):
                    m1, m2 = available[i], available[j]
                    c1, t1 = counts[m1][dataset_name][cond]
                    c2, t2 = counts[m2][dataset_name][cond]
                    odds, p = fisher_exact_2x2(c1, t1, c2, t2)
                    raw_pvals.append(p)
                    pairs.append(
                        {
                            "model_a": m1,
                            "model_b": m2,
                            "odds_ratio": odds,
                            "p_value_raw": p,
                            "cohen_h": cohen_h(c1 / t1 if t1 else 0, c2 / t2 if t2 else 0),
                        }
                    )
            corrected, thresh = bonferroni_correct(raw_pvals, alpha=FISHER_ALPHA)
            for r, p_c in zip(pairs, corrected):
                r["p_value_bonferroni"] = p_c
                r["alpha_threshold"] = thresh
            out["comparisons"]["pairwise_models_fisher"][f"{dataset_name}:{cond}"] = pairs

    return out
