"""Tests for EU causal confusion axis builder."""

from __future__ import annotations

import numpy as np

from scripts.causal_eu_confusion_axes import (
    analyze_model_layer,
    axis_entanglement,
    mean_diff_axis,
    pair_trial_indices,
    select_difficulty_matched_pairs,
    tercile_indices,
)


def test_tercile_split() -> None:
    vals = list(range(9))
    high, low = tercile_indices(vals, high_tail=True)
    assert len(high) == 3
    assert len(low) == 3
    assert max(high) > min(low)


def test_pair_trial_indices() -> None:
    rows = [
        {"human_target_label": "Worried", "top_foil_label": "Disappointed"},
        {"human_target_label": "Happy", "top_foil_label": "Sad"},
        {"human_target_label": "Disappointed", "top_foil_label": "Worried"},
    ]
    idx = pair_trial_indices(rows, "Worried", "Disappointed")
    assert idx == [0, 2]


def test_analyze_model_layer_synthetic() -> None:
    rng = np.random.default_rng(42)
    n, d = 30, 16
    X = rng.standard_normal((n, d)).astype(np.float32)
    rows = []
    for i in range(n):
        rows.append(
            {
                "human_entropy": float(i) / n,
                "confusability_1_minus_p_target": float(i) / n,
                "human_target_label": "Worried" if i < 5 else "Happy",
                "top_foil_label": "Disappointed" if i < 5 else "Sad",
            }
        )
    out = analyze_model_layer("test", 4, X, rows, [("Worried", "Disappointed")], seed=42)
    assert out["n_trials"] == 30
    assert "own_effect_confusability" in out
    assert "difficulty_matched_control_pairs" in out
    assert out["pair_axes"][0]["entanglement_vs_entropy"]["cos_abs_pair_vs_generic"] >= 0.0
    high = [0, 1, 2, 3, 4]
    low = [25, 26, 27, 28, 29]
    ax = mean_diff_axis(X, high, low)
    assert ax.shape == (d,)
