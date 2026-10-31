"""Unit tests for Phase 2 metrics, gate logic, and option-order helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from paper_v2.metrics import (  # noqa: E402
    bootstrap_spearman_ci,
    holm_bonferroni,
    nested_ridge_preds,
    one_hot,
    permutation_p,
    residualise,
    spearman_from_preds,
)
from paper_v2.option_score import option_logprob_scores, shuffle_options  # noqa: E402
from paper_v2.probe import decide_branch, layer_passes_gate  # noqa: E402


def test_holm_known_values():
    adj = holm_bonferroni([0.01, 0.04, 0.03])
    assert abs(adj[0] - 0.03) < 1e-12
    assert abs(adj[1] - 0.06) < 1e-12
    assert abs(adj[2] - 0.06) < 1e-12


def test_residualise_orthogonal_to_predictor():
    rng = np.random.default_rng(0)
    z = rng.normal(size=80)
    y = 3.0 * z + rng.normal(size=80) * 0.01
    r = residualise(y, z)
    corr = np.corrcoef(r, z)[0, 1]
    assert abs(corr) < 1e-10


def test_grouped_cv_blocks_actor_leakage():
    rng = np.random.default_rng(0)
    n_actors, n_per = 10, 8
    actors = np.repeat(np.arange(n_actors), n_per)
    n = len(actors)
    X = one_hot([str(a) for a in actors])
    actor_mean = rng.normal(size=n_actors)
    y = actor_mean[actors] + 0.01 * rng.normal(size=n)
    loo = nested_ridge_preds(X, y, groups=None)
    grp = nested_ridge_preds(X, y, groups=actors)
    assert spearman_from_preds(y, loo) > 0.8
    rho_g = spearman_from_preds(y, grp)
    # LOGO + intercept can anti-correlate slightly; it must not positively decode.
    assert (not np.isfinite(rho_g)) or rho_g < 0.3


def test_nested_ridge_recovers_linear_target():
    rng = np.random.default_rng(0)
    n, d = 40, 6
    w = rng.normal(size=d)
    X = rng.normal(size=(n, d))
    y = X @ w
    preds = nested_ridge_preds(X, y, groups=None)
    assert spearman_from_preds(y, preds) > 0.9


def test_bootstrap_ci_excludes_zero_when_correlated():
    rng = np.random.default_rng(1)
    x = rng.normal(size=60)
    y = x + 0.05 * rng.normal(size=60)
    ci = bootstrap_spearman_ci(y, x, n_boot=400, seed=1)
    assert ci["excludes_zero"]
    assert ci["ci_lo"] > 0.7


def test_permutation_p_null_not_small():
    rng = np.random.default_rng(0)
    null = rng.normal(size=200)
    p = permutation_p(0.01, null, alternative="two-sided")
    assert p > 0.2


def test_gate_requires_all_four_checks():
    def row(**kwargs):
        base = {
            "t1_grouped_perm": {"p_perm_holm": 0.001},
            "delta_t1_minus_stimulus_grouped": {
                "delta_rho": 0.20,
                "excludes_zero": True,
            },
            "t1_category_residual_grouped": {"excludes_zero": True},
            "t1_confidence_residual_grouped": {"excludes_zero": True},
        }
        for k, v in kwargs.items():
            if isinstance(v, dict) and k in base:
                base[k].update(v)
            else:
                base[k] = v
        return base

    assert layer_passes_gate(row())["all"] is True
    assert layer_passes_gate(row(t1_grouped_perm={"p_perm_holm": 0.02}))["all"] is False
    assert layer_passes_gate(row(delta_t1_minus_stimulus_grouped={"delta_rho": 0.05}))["all"] is False


def test_decide_branch_ignores_molmo_at_chance():
    molmo = {"model": "molmo2", "above_chance": False, "any_layer_passes": True, "passing_layers": [4]}
    qwen = {"model": "qwen3vl", "above_chance": True, "any_layer_passes": False, "passing_layers": []}
    out = decide_branch([molmo, qwen])
    assert out["branch"] == "B"
    qwen["any_layer_passes"] = True
    qwen["passing_layers"] = [13]
    out = decide_branch([molmo, qwen])
    assert out["branch"] == "A"
    assert out["eligible_models"][0]["model"] == "qwen3vl"


def test_option_shuffle_preserves_set_and_is_deterministic():
    opts = ["A", "B", "C", "D", "E", "F"]
    a = shuffle_options(opts, seed=42, trial_id="x")
    b = shuffle_options(opts, seed=42, trial_id="x")
    c = shuffle_options(opts, seed=43, trial_id="x")
    assert a == b
    assert set(a) == set(opts)
    assert c != a


def test_option_logprob_softmax_and_argmax():
    scores = option_logprob_scores(lambda s: {"yes": 0.0, "no": -5.0}[s], ["yes", "no"])
    assert scores["argmax"] == "yes"
    assert scores["probs"]["yes"] > 0.99
    assert abs(sum(scores["probs"].values()) - 1.0) < 1e-12
