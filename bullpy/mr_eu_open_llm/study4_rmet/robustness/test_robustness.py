"""pytest sanity checks for study4_rmet/robustness (simulated data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr

from power_analysis import achieved_power, min_detectable_r
from equivalence_bayes import jeffreys_label, pearson_bf, tost_correlation
from meta_analysis import random_effects_meta_fisher_z
from trial_level_model import _fit_mixedlm


def _corr_data(n: int, r: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = r * x + np.sqrt(max(1e-9, 1 - r**2)) * rng.normal(size=n)
    return x, y


def test_min_detectable_r_decreases_with_n():
    r20 = min_detectable_r(20)
    r80 = min_detectable_r(80)
    assert r80 < r20


def test_achieved_power_increases_with_r():
    p1 = achieved_power(36, 0.1)
    p3 = achieved_power(36, 0.3)
    p5 = achieved_power(36, 0.5)
    assert p1 < p3 < p5


def test_tost_equivalence_near_zero_and_not_when_large():
    x0, y0 = _corr_data(80, 0.0, seed=1)
    t0 = tost_correlation(x0, y0, eps=0.30)
    # With n=80 and true r≈0, often establishes equivalence at eps=.3
    assert t0["p_tost"] < 0.5  # soft: should lean toward equivalence

    x1, y1 = _corr_data(80, 0.6, seed=2)
    t1 = tost_correlation(x1, y1, eps=0.30)
    assert t1["equivalent"] == 0.0
    assert t1["p_tost"] > t0["p_tost"]


def test_bf_direction():
    x0, y0 = _corr_data(36, 0.0, seed=3)
    x1, y1 = _corr_data(36, 0.6, seed=4)
    b0 = pearson_bf(x0, y0)
    b1 = pearson_bf(x1, y1)
    assert np.isfinite(b0["BF10"]) and np.isfinite(b1["BF10"])
    assert b1["BF10"] > 1.0
    assert b0["BF10"] < b1["BF10"]
    assert "H0" in jeffreys_label(0.2) or "anecdotal" in jeffreys_label(0.2)


def test_meta_pools_near_zero():
    rs = [0.02, -0.01, 0.03, 0.0, -0.02, 0.01]
    ns = [36] * 6
    m = random_effects_meta_fisher_z(rs, ns)
    assert abs(m["r_pooled"]) < 0.1
    assert m["ci_low"] < m["r_pooled"] < m["ci_high"]


def test_trial_level_recovers_interaction_and_smaller_se():
    """
    Generative: item slopes + agent interaction on eq_sensitivity.
    Assert LMM interaction SE is smaller than SE implied by item-level r test.
    """
    rng = np.random.default_rng(0)
    n_items, n_subj_h, n_rep_m = 36, 40, 10
    eq = rng.normal(size=n_items)
    eq = (eq - eq.mean()) / eq.std()
    item_slope = 0.15 + 0.05 * rng.normal(size=n_items)

    rows = []
    # humans: positive slope of correct on eq
    for s in range(n_subj_h):
        for i in range(n_items):
            eta = -0.2 + item_slope[i] * eq[i] + 0.05 * rng.normal()
            p = 1 / (1 + np.exp(-eta))
            rows.append(
                {
                    "item": i,
                    "correct": float(rng.random() < p),
                    "eq_sensitivity_z": float(eq[i]),
                    "agent_type": "human",
                    "subject_or_rep": f"h{s}",
                }
            )
    # model: flatter / near-zero slope → negative interaction vs human
    for r in range(n_rep_m):
        for i in range(n_items):
            eta = -0.2 + 0.02 * eq[i] + 0.05 * rng.normal()
            p = 1 / (1 + np.exp(-eta))
            rows.append(
                {
                    "item": i,
                    "correct": float(rng.random() < p),
                    "eq_sensitivity_z": float(eq[i]),
                    "agent_type": "modelX",
                    "subject_or_rep": f"m{r}",
                }
            )
    df = pd.DataFrame(rows)
    df["agent_type"] = pd.Categorical(df["agent_type"], categories=["human", "modelX"])
    fit, re_used = _fit_mixedlm(
        df, "correct ~ eq_sensitivity_z * C(agent_type)", re_formula="1"
    )
    # interaction term
    inter_name = [n for n in fit.params.index if "agent_type" in n and "eq_sensitivity" in n][0]
    inter_coef = float(fit.params[inter_name])
    inter_se = float(fit.bse[inter_name])
    assert inter_coef < 0  # model flatter than human
    assert inter_se < 0.2

    # Item-level correlation SE (Fisher-z) for n=36
    # Build item accuracies
    acc_h = df[df.agent_type == "human"].groupby("item")["correct"].mean().to_numpy()
    r_item, _ = pearsonr(eq, acc_h)
    se_item_z = 1 / np.sqrt(36 - 3)
    # Compare SE on roughly comparable scale: interaction SE should be << item Fisher SE
    # (power gain claim — soft check)
    assert inter_se < se_item_z


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
