"""Amendment A3 logic on made-up rows. No model, no clip, no rotation result."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from paper_v2.option_score import ABSTAIN_LABEL  # noqa: E402
from scripts import face_input_validity as fiv  # noqa: E402


def items():
    human = json.loads((ROOT / "data/eu_emotions_human_entropy.json").read_text())["trials"]
    return [
        {
            "trial_id": k,
            "label": v["human_target_label"],
            "actor": k.split("/")[-1][0].upper(),
            "options": v["human_options"],
        }
        for k, v in human.items()
    ]


def rows_for(n_actors, effect, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for actor in range(n_actors):
        for item in range(6):
            trial = f"{actor}_{item}"
            for rotation in fiv.ROTATIONS:
                rows.append({"trial_id": trial, "actor": str(actor), "rotation": rotation, "condition": "real", "status": "ok", "correct": rng.random() < 0.2 + effect})
                rows.append({"trial_id": trial, "actor": str(actor), "rotation": rotation, "condition": "mismatch", "status": "ok", "correct": rng.random() < 0.2})
    return rows


def test_statistics_are_the_a1_functions():
    assert fiv.holm.__module__ == "analyse_rotation"
    assert fiv.actor_sign_flip_p.__module__ == "analyse_rotation"
    assert fiv.actor_bootstrap_ci.__module__ == "analyse_rotation"


def test_partners_match_a1_seed_and_order_free():
    data = items()
    first = fiv.partners_for(data)
    assert first == fiv.partners_for(list(reversed(data)))
    assert len(first) == 243
    by = {i["trial_id"]: i for i in data}
    assert all(by[k]["label"] != by[v]["label"] and by[k]["actor"] != by[v]["actor"] for k, v in first.items())


def test_each_item_has_ten_cells_with_its_own_options():
    item, partner = items()[:2]
    cells = fiv.trial_cells(item, partner)
    assert len(cells) == 10
    assert {c["rotation"] for c in cells} == set(range(5))
    assert all(sorted(c["options"]) == sorted(item["options"]) for c in cells)
    assert all(c["options"][-1] == ABSTAIN_LABEL for c in cells)
    assert {c["face_item"] for c in cells if c["condition"] == "mismatch"} == {partner["trial_id"]}
    positions = {c["options"].index(item["label"]) for c in cells}
    assert positions == set(range(5))


def test_exact_scoring():
    options = ["A", "B", "C"]
    out = fiv.correct_from_scores(options, np.log([0.5, 0.3, 0.2]).tolist(), "A")
    assert out["correct"] and out["p_target"] == pytest.approx(0.5)
    assert not fiv.correct_from_scores(options, np.log([0.5, 0.3, 0.2]).tolist(), "B")["correct"]
    assert not fiv.correct_from_scores(options, [0.0, 0.0, -1.0], "A")["correct"]


def test_incomplete_items_are_left_out_and_failures_reported():
    rows = rows_for(4, 0.0)
    for row in rows:
        if row["trial_id"] == "0_0" and row["rotation"] == 2 and row["condition"] == "real":
            row.update(status="failed", reason="no_apex_frame")
    summary = fiv.summarise_model(rows)
    assert summary["n_items"] == 23 and summary["n_items_expected"] == 24
    assert summary["failure_reasons"] == {"no_apex_frame": 1}


def test_decision_rule_finds_a_planted_delta_only():
    summaries = {
        "a": fiv.summarise_model(rows_for(17, 0.5, 1)),
        "b": fiv.summarise_model(rows_for(17, 0.0, 2)),
        "c": fiv.summarise_model(rows_for(17, 0.0, 3)),
    }
    out = fiv.decide(summaries)
    assert out["a"]["vision_dependent"]
    assert not out["b"]["vision_dependent"] and not out["c"]["vision_dependent"]


def test_reading_table_is_the_plans():
    assert "grid limited" in fiv.reading(False, True)
    assert "No demonstrated" in fiv.reading(False, False)


def test_refuses_until_frozen(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["face_input_validity", "--model", "gemma4"])
    with pytest.raises(SystemExit, match="not frozen"):
        fiv.main()
