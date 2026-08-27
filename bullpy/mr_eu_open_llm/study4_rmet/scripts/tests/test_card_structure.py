"""Tests for CARD structure + behavioural helpers (study4)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# tests/ -> scripts/ -> study4_rmet
STUDY4 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STUDY4 / "scripts"))

from build_card_rmet_structure import (  # noqa: E402
    _js_divergence,
    _shannon,
    build_structure,
    choice_dist,
    load_key,
)


def test_shannon_uniform():
    p = np.ones(4) / 4
    assert abs(_shannon(p) - np.log(4)) < 1e-9


def test_js_identical_zero():
    p = np.array([0.5, 0.25, 0.25, 0.0])
    assert _js_divergence(p, p) < 1e-9


def test_choice_dist_ignores_timeout():
    s = pd.Series([1, 1, 2, -1, 4, None])
    d = choice_dist(s, n_opt=4)
    assert abs(d.sum() - 1.0) < 1e-9
    assert d[0] == 0.5


def test_build_structure_smoke():
    card = STUDY4 / "data" / "processed" / "card_rmet_item_level.csv"
    key = STUDY4 / "data" / "rmet" / "answer_key" / "rmet_adult_answer_key.json"
    trait = STUDY4 / "results" / "human" / "item_trait_sensitivity.csv"
    if not card.exists() or not trait.exists():
        return  # skip if data absent
    df = pd.read_csv(card)
    item_df, choice, meta = build_structure(df, load_key(key), trait)
    assert len(item_df) == 36
    assert "human_entropy" in item_df.columns
    assert "trait_diagnosticity_eq_slope" in item_df.columns
    assert set(item_df["class_trait_diagnosticity"].unique()) <= {
        "high_diagnosticity",
        "low_diagnosticity",
    }
    assert "1" in choice["items"]
    assert len(choice["items"]["1"]["p_all"]) == 4


def test_steer_protocol_only(tmp_path):
    from steer_rmet_axes import write_protocol

    path = tmp_path / "proto.json"
    proto = write_protocol(
        path,
        model="qwen3vl",
        layer=4,
        alphas=[-1.0, 1.0],
        patch_modes=["last_token", "all_tokens"],
        n_samples=5,
        seed=42,
    )
    assert path.is_file()
    assert proto["C1_primary"] == "reuse_steer_dissociation_diag_vs_entropy"
    assert "diagnosticity" in proto["axes"]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
