"""Golden-fixture regression tests for artifact pipeline."""

import json
from pathlib import Path

from scripts.augment_eval_artifact import augment_eval_json
from scripts.free_response_judge import judge_free_response
from scripts.selective_prediction import selective_prediction_report

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "eval_v2_mini.json"


def _load_fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_augment_eval_artifact_fixture():
    data = _load_fixture()
    out = augment_eval_json(data)
    am = out["artifact_metrics"]
    assert am["primary"] == "free_response_judge"
    assert am["free_response_judge_n"] == 3
    assert 0.0 < am["free_response_judge_accuracy"] < 1.0
    assert "selective_prediction" in am
    assert am["selective_prediction"]["n_scored"] == 3
    assert "tolerant_rescore" in am
    assert am["tolerant_rescore"]["strict_4afc_accuracy"] == 2 / 3


def test_free_response_judge_on_fixture_trials():
    data = _load_fixture()
    for t in data["trials"]:
        s1 = t["stage1"]
        r = judge_free_response(s1["free_response_text"], t["label"], use_llm=False)
        if t["trial_id"] == "fixture_afraid_03":
            assert r["correct"] is False
        else:
            assert r["correct"] is True


def test_selective_prediction_on_fixture():
    data = _load_fixture()
    rep = selective_prediction_report(data["trials"])
    assert rep["n_scored"] == 3
    assert rep["overall_accuracy"] == 2 / 3
    assert rep["expected_calibration_error"] is not None
