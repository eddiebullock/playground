from __future__ import annotations

import json

import numpy as np
import pytest

from esvlm.analysis.separability import residualize
from esvlm.cli.extract_task import (
    is_modality_confound,
    leftover_dirs,
    paired_unit_dirs,
    pick_candidate_layer,
    safe_projection_auroc,
    train_hold_idx,
)
from esvlm.cli.steer import load_item_id_filter
from esvlm.vectors.extract import difference_of_means, directions_for_layers
from esvlm.vectors.validate import held_out_auroc


def test_task_stacks_form_a_direction() -> None:
    rng = np.random.default_rng(0)
    social = rng.normal(size=(20, 3, 8)) + np.array([2.0] + [0.0] * 7)
    mech = rng.normal(size=(24, 3, 8)) - np.array([2.0] + [0.0] * 7)
    vecs = directions_for_layers(social, mech)
    assert vecs.shape == (3, 8)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)
    held = held_out_auroc(social[:, 1, :], mech[:, 1, :], seed=0)
    assert held > 0.8
    flipped = difference_of_means(mech[:, 1, :], social[:, 1, :])
    assert float(vecs[1] @ flipped) < 0


def test_train_hold_and_paired_dirs() -> None:
    train, hold = train_hold_idx(36, hold_frac=0.33, seed=0)
    assert len(train) + len(hold) == 36
    assert len(hold) >= 4
    assert set(train).isdisjoint(hold)
    rng = np.random.default_rng(1)
    pos = rng.normal(size=(12, 4, 6)) + 1.5
    neg = pos - 3.0
    vecs = paired_unit_dirs(pos, neg)
    assert vecs.shape == (4, 6)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)
    auroc = safe_projection_auroc(pos[:, 2, :], neg[:, 2, :], vecs[2])
    assert auroc > 0.9


def test_visual_leftover_and_confound_gate() -> None:
    rng = np.random.default_rng(2)
    text = np.zeros((5, 8), dtype=np.float64)
    text[:, 0] = 1.0
    visual = np.zeros((5, 8), dtype=np.float64)
    visual[:, 1] = 1.0
    mixed = text + visual
    leftover, collinear = leftover_dirs(mixed, text)
    assert collinear == [False] * 5
    assert abs(float(leftover[0] @ np.array([1.0] + [0.0] * 7))) < 1e-6
    assert leftover[3, 1] > 0.9
    parallel, flags = leftover_dirs(text, text)
    assert flags == [True] * 5
    assert np.allclose(parallel, 0.0)
    assert np.isnan(safe_projection_auroc(rng.normal(size=(6, 8)), rng.normal(size=(6, 8)), parallel[0]))
    assert is_modality_confound(1.0)
    assert not is_modality_confound(0.51)
    scores = np.array([0.99, 0.99, 0.99, 0.99, 0.71, 0.88, 0.80])
    assert pick_candidate_layer(scores) == 5
    assert pick_candidate_layer(np.array([0.4, 0.9, np.nan]), min_layer=2) == 1


def test_residualize_matches_leftover_helper() -> None:
    raw = np.array([1.0, 1.0, 0.0])
    drop = np.array([1.0, 0.0, 0.0])
    got, flags = leftover_dirs(raw[None, :], drop[None, :])
    assert flags == [False]
    assert np.allclose(got[0], residualize(raw, drop))


def test_load_item_id_filter(tmp_path) -> None:
    path = tmp_path / "ids.json"
    path.write_text(json.dumps(["a", "b"]), encoding="utf-8")
    assert load_item_id_filter(path) == {"a", "b"}
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"hold_social_ids": ["x"], "hold_other_ids": ["y"]}), encoding="utf-8")
    assert load_item_id_filter(summary) == {"x"}
    assert load_item_id_filter(summary, key="hold_other_ids") == {"y"}
    with pytest.raises(SystemExit):
        load_item_id_filter(summary, key="missing")
