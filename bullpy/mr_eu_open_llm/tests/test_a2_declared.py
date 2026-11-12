"""Tests for the declared A2 extras and pre-run check (DECISIONS 2026-11-12). Synthetic cells only."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from paper_v2.schema import actor_id_from_eu_filename  # noqa: E402
from scripts import analyse_stimulus_interventions as an  # noqa: E402
from scripts import recovery_stimulus_interventions as rec  # noqa: E402
from scripts import run_stimulus_interventions as run  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name + "_under_test", ROOT / "research" / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ext = _load("a2_declared_extras")
pre = _load("a2_prerun_check")
PASSED = {k: True for k in an.GATE_KEYS}
P1 = next(c for c in an.CONFIRMATORY if c.id == "P1")


def items():
    human = json.loads((ROOT / "data/eu_emotions_human_entropy.json").read_text())["trials"]
    out = []
    for key, value in human.items():
        named = {k: v for k, v in value["human_response_distribution"].items() if k != run.ABSTAIN}
        out.append({"trial_id": key, "actor": actor_id_from_eu_filename(key), "label": value["human_target_label"],
                    "human_modal_label": max(sorted(named), key=lambda k: named[k]),
                    "weak_label": value["human_target_label"] in run.WEAK_LABELS})
    return out


def row(trial, actor, condition, happy, route="single_face"):
    scores = {"Happy": happy, "Sad": 0.0, "Neutral": 0.0}
    return {"trial_id": trial, "actor": actor, "label": "Happy", "human_modal_label": "Happy", "input_name": route,
            "condition": condition, "status": "ok", "blind": {"family_logprob": scores, "lexicon_mass": 0.5}}


def opposing_cells(other_fill_change):
    """P1 (mouth mask lowers Happy) with the neutral fill raising Happy instead, over 3 actors."""
    rows = []
    for n, actor in enumerate("BCD"):
        trial = f"t{n}"
        rows += [row(trial, actor, "original", 1.0), row(trial, actor, "mask_mouth_neutral", 2.0),
                 row(trial, actor, "mask_mouth_first", 1.0 + other_fill_change),
                 row(trial, actor, "mask_mouth_blur", 1.0 + other_fill_change)]
    return an.Cells(rows), ["B", "C", "D"]


def test_fill_rule_on_an_opposing_neutral_fill():
    cells, actors = opposing_cells(+0.5)  # other fills also raise Happy: they share the neutral sign
    agree = ext.fill_check(cells, P1, actors)
    assert agree["neutral_fill_mean"] > 0 and agree["fills_share_neutral_sign"] is True
    cells, actors = opposing_cells(-0.5)  # other fills lower Happy: they do not
    disagree = ext.fill_check(cells, P1, actors)
    assert disagree["fills_share_neutral_sign"] is False
    # The frozen field reads the predicted direction (a fall), so it says the opposite of the text's reading.
    frozen = an.test_family(cells, [P1], actors, flip_args={"max_exact_actors": 0, "draws": 255})["P1"]
    assert frozen["fills_agree"] is True
    assert ext.verdict_after_fill_rule("opposes_prediction", disagree) == "inconclusive"
    assert ext.verdict_after_fill_rule("opposes_prediction", agree) == "opposes_prediction"
    assert ext.verdict_after_fill_rule("matches_prediction", disagree) == "matches_prediction"


def test_intervals_reproduce_the_frozen_equivalence_and_shuffles_are_written():
    data = items()
    actors = sorted({i["actor"] for i in data})
    cells = an.Cells([dict(r, mcq=[]) for r in rec.synthetic_rows("human_like", data, 2)])
    frozen = {route: {"margin": 0.1, "results": an.analyse_route(cells, route, actors, PASSED, 0.1,
                                                                  flip_args={"max_exact_actors": 0, "draws": 255})}
              for route in ext.ROUTES}
    out = ext.extras(cells, frozen)
    for route in ext.ROUTES:
        for cid, r in out[route]["contrasts"].items():
            interval = r["equivalence_interval_90"]
            assert interval is None or interval[0] <= interval[1]
    assert out["single_face"]["shuffles_overall"] is None
    assert set(out["video_face"]["shuffles_overall"]) >= {"basic", "complex", "Happy"}


def test_prerun_check_catches_a_torn_line_and_a_foreign_model(tmp_path):
    root = tmp_path
    (root / "scripts").mkdir()
    (root / pre.ROOT_MARKER).write_text("")
    repro = root / pre.REPRO
    repro.mkdir(parents=True)
    for name in ("rotation_summary.json", "vision_gates_gemma4.json", pre.A3_NAME):
        (repro / name).write_text("{}")
    (repro / "input_delivery_gemma4.json").write_text(json.dumps(
        {"items": [{"single_face": {"input": {"pixel_shapes": {"pixel_values": [1, 2520, 768]}}}}]}))
    good = {"status": "ok", "input": {"pixel_shapes": {"pixel_values": [1, 2520, 768]}}}
    qwen = {"status": "ok", "input": {"pixel_shapes": {"pixel_values": [2304, 1536]}}}
    cells = root / "cells_gemma4_shard0.jsonl"
    cells.write_text(json.dumps(good) + "\n" + json.dumps(qwen) + "\n" + '{"status": "ok", "inp')
    report = root / "run_gemma4_shard0.json"
    report.write_text("{}")
    result = pre.check("gemma4", [cells], [report], root)
    assert result["unparseable_lines"]["cells_gemma4_shard0.jsonl"][0]["line"] == 3
    assert result["n_rows_not_matching_model"] == 1
    assert any("geometry" in f for f in result["failures"])
    wrong = root / "cells_qwen3vl_shard0.jsonl"
    wrong.write_text(json.dumps(good) + "\n")
    assert any("not named" in f for f in pre.check("gemma4", [wrong], [report], root)["failures"])
