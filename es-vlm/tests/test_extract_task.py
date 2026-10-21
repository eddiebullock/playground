from __future__ import annotations

import json

import numpy as np
import pytest

from esvlm.analysis.separability import residualize
from esvlm.cli.extract_task import (
    TEXT_RESIDUALIZE_DESIGNS,
    decide_gates,
    is_modality_confound,
    leftover_dirs,
    load_design_items,
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


def test_wording_only_blocks_steer_on_collinear_leftover() -> None:
    visual = np.array([0.0] * 35 + [1.0])
    unique = [1e-7] * 36
    gates = decide_gates(raw_layer0=1.0, pick_scores=visual, unique_vs_text=unique)
    assert gates["wording_only"] is True
    assert gates["degenerate"] is True
    assert gates["steer_ok"] is False


def test_video_layer0_confound_blocks_steer() -> None:
    held = np.ones(36)
    gates = decide_gates(raw_layer0=1.0, pick_scores=held, unique_vs_text=None)
    assert gates["modality_confound"] is True
    assert gates["wording_only"] is False
    assert gates["steer_ok"] is False


def test_real_visual_leftover_can_steer() -> None:
    visual = np.array([0.52] * 4 + [0.81] + [0.70] * 31)
    unique = [0.4] * 36
    gates = decide_gates(raw_layer0=0.55, pick_scores=visual, unique_vs_text=unique)
    assert gates["wording_only"] is False
    assert gates["modality_confound"] is False
    assert gates["steer_ok"] is True
    assert gates["candidate_layer"] == 4


def test_residualize_matches_leftover_helper() -> None:
    raw = np.array([1.0, 1.0, 0.0])
    drop = np.array([1.0, 0.0, 0.0])
    got, flags = leftover_dirs(raw[None, :], drop[None, :])
    assert flags == [False]
    assert np.allclose(got[0], residualize(raw, drop))


def test_public_designs_load_and_residualize_flag(tmp_path, monkeypatch) -> None:
    from argparse import Namespace

    from esvlm.tasks.mmtom import mmtom_to_trial
    from esvlm.tasks.nextqa import pair_same_video, parse_nextqa_row, write_pair_manifest
    from esvlm.tasks.sdomain import piqa_to_trial, write_manifest

    assert "same_scene_nextqa" in TEXT_RESIDUALIZE_DESIGNS
    assert "mmtom_vs_piqa" not in TEXT_RESIDUALIZE_DESIGNS

    cw = parse_nextqa_row(
        {
            "video": "5",
            "qid": "1",
            "question": "why did they sit",
            "answer": "0",
            "type": "CW",
            "a0": "tired",
            "a1": "late",
            "a2": "lost",
            "a3": "cold",
            "a4": "hungry",
        }
    )
    do = parse_nextqa_row(
        {
            "video": "5",
            "qid": "2",
            "question": "what is on the table",
            "answer": "1",
            "type": "DO",
            "a0": "cup",
            "a1": "book",
            "a2": "bag",
            "a3": "phone",
            "a4": "key",
        }
    )
    assert cw is not None and do is not None
    nextqa_man = tmp_path / "pairs_val.json"
    write_pair_manifest(nextqa_man, pair_same_video([cw, do]))
    monkeypatch.setenv("ESVLM_NEXTQA_MANIFEST", str(nextqa_man))
    monkeypatch.setenv("ESVLM_NEXTQA_DIR", str(tmp_path))

    mm_trials = [
        mmtom_to_trial(
            {
                "question": (
                    f"Question: If agent {i} wants a plate, which is more likely? "
                    "(a) in the fridge. (b) not in the fridge."
                ),
                "answer": "a",
                "episode": i,
                "answer_list": ["a", "b"],
            },
            idx=i,
        )
        for i in range(8)
    ]
    mm_man = tmp_path / "mmtom.json"
    write_manifest(mm_man, "mmtom", mm_trials)
    monkeypatch.setenv("ESVLM_MMTOM_MANIFEST", str(mm_man))

    piqa_trials = []
    for i in range(8):
        trial = piqa_to_trial(
            {"goal": f"g{i}", "sol1": "a", "sol2": "b", "label": 0},
            split="valid",
            idx=i,
        )
        assert trial is not None
        piqa_trials.append(trial)
    piqa_man = tmp_path / "piqa.json"
    write_manifest(piqa_man, "piqa", piqa_trials)
    monkeypatch.setenv("ESVLM_PIQA_MANIFEST", str(piqa_man))
    monkeypatch.setenv("ESVLM_PIQA_DIR", str(tmp_path))

    args = Namespace(trial_limit_social=0, trial_limit_mechanical=0)
    social, other, sname, oname, paired = load_design_items("same_scene_nextqa", args)
    assert paired is True
    assert sname == "nextqa_social"
    assert oname == "nextqa_physical"
    assert len(social) == 1
    assert social[0].media_type == "none"

    tom, phys, tname, pname, tom_paired = load_design_items("mmtom_vs_piqa", args)
    assert tom_paired is False
    assert tname == "mmtom"
    assert pname == "piqa"
    assert len(tom) == 8
    assert len(phys) == 8


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
