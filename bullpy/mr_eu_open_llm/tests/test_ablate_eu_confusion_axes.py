"""Tests for EU ablation axis helpers."""

from __future__ import annotations

import numpy as np

from scripts.causal_eu_confusion_axes import axis_entanglement, select_difficulty_matched_pairs


def test_axis_entanglement_identical() -> None:
    v = np.array([1.0, 0.0, 0.0])
    out = axis_entanglement(v, v)
    assert out["cos_abs_pair_vs_generic"] == 1.0
    assert out["specificity_ratio_vs_generic"] == 0.0


def test_axis_entanglement_orthogonal() -> None:
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    out = axis_entanglement(a, b)
    assert out["cos_abs_pair_vs_generic"] == 0.0
    assert out["specificity_ratio_vs_generic"] == 1.0


def test_difficulty_matched_pairs() -> None:
    rows = []
    for i in range(20):
        rows.append(
            {
                "human_entropy": 1.0 if i < 5 else 0.2,
                "human_target_label": "Worried" if i < 5 else "Happy",
                "top_foil_label": "Disappointed" if i < 5 else "Sad",
            }
        )
    confused = [("Worried", "Disappointed")]
    out = select_difficulty_matched_pairs(rows, "Worried", "Disappointed", confused)
    assert out["n_confused_items"] == 5
    assert isinstance(out["matched_pairs"], list)
