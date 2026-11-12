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


a5pre = _load("a5_prerun_check")


def test_a5_prerun_check_catches_a_missing_model_and_an_empty_route(tmp_path):
    cells_dir = tmp_path / "results" / "stimulus_interventions"
    cells_dir.mkdir(parents=True)
    rows = [{"condition": "original", "status": "ok", "input_name": "single_face"},
            {"condition": "original", "status": "failed", "input_name": "video_face"}]
    (cells_dir / "cells_gemma4_shard0.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (cells_dir / "run_gemma4_shard0.json").write_text(json.dumps({"n_shards": 1}))
    repro = tmp_path / a5pre.REPRO
    repro.mkdir(parents=True)
    (repro / "rotation_summary.json").write_text(json.dumps({m: {"vision_dependent": True} for m in a5pre.MODELS}))
    result = a5pre.check(cells_dir, tmp_path)
    fails = " | ".join(result["failures"])
    assert result["models"]["gemma4"]["ok_original_cells"] == {"single_face": 1, "video_face": 0}
    assert "gemma4 video_face: no ok original cell" in fails
    assert "qwen3vl: cells and reports are not one complete shard set" in fails
    assert "A3 summary missing or not the pinned bytes" in fails
    assert "human norms file missing" in fails


a4pre = _load("a4_prerun_check")
a4gates = _load("a4_gates")


def _a4_manifest():
    rows = [{"stim_id": f"bwc_{i:03d}", "set": "main", "dataset": "basic_WC"} for i in range(30)]
    rows += [{"stim_id": f"cwc_{i:03d}", "set": "main", "dataset": "conversational_WC"} for i in range(30)]
    rows += [{"stim_id": "female_a_neutral", "set": "neutral", "dataset": "basic_WC"}]
    return rows


def _write_shards(folder, model, manifest, design, *, skip=None, dup=False, nan_first=False):
    from scripts import a4_run as run
    for k in range(a4pre.N_SHARDS):
        if k == skip:
            continue
        rows = []
        for cell in run.planned_cells(manifest, k, a4pre.N_SHARDS):
            row = {**cell, "status": "ok", "design_sha256": design}
            if cell["readout"] in ("blind", "blind_image_free"):
                row["family_logprob"] = {"Happy": -1.0}
            else:
                row["joint_logprob"] = [-1.0, -2.0]
            rows.append(row)
        if nan_first and rows:
            rows[0]["joint_logprob" if "joint_logprob" in rows[0] else "family_logprob"] = [float("nan")]
        if dup and rows:
            rows.append(dict(rows[0]))
        (folder / f"cells_{model}_{k}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_a4_prerun_check_catches_missing_shard_duplicates_and_non_finite(tmp_path):
    from scripts import a4_run as run
    manifest = _a4_manifest()
    design = "d" * 64
    _write_shards(tmp_path, "gemma4", manifest, design)
    ok = a4pre.check_model("gemma4", tmp_path, manifest, design, run)
    assert ok["failures"] == [] and ok["shards"] == list(range(12)) and ok["non_finite_rows"] == []
    _write_shards(tmp_path, "qwen3vl", manifest, design, skip=7, dup=True, nan_first=True)
    bad = a4pre.check_model("qwen3vl", tmp_path, manifest, design, run)
    text = " | ".join(bad["failures"])
    assert "not 0 to 11" in text and "duplicate cells" in text
    assert len(bad["non_finite_rows"]) >= 1
    clean = tmp_path / "clean"
    a4pre.write_clean(tmp_path, clean, ["qwen3vl"], bad["non_finite_rows"])
    assert (clean / "removed_non_finite_rows.json").exists()
    first = bad["non_finite_rows"][0]
    n_bad = sum(1 for r in bad["non_finite_rows"] if r["file"] == first["file"])  # the duplicate copies the bad row
    assert n_bad == 2
    assert sum(1 for _ in (clean / first["file"]).read_text().splitlines()) == \
        sum(1 for line in (tmp_path / first["file"]).read_text().splitlines() if line.strip()) - n_bad


def test_a4_gates_file_matches_its_builder():
    built = a4gates.build()
    committed = json.loads((ROOT / "research/reproduction/a4_gates.json").read_text())
    assert built == committed
    assert built["gemma4"] == {"a1_passed": True, "a3_passed": True, "g2_g4": True}
    assert built["molmo2"]["g2_g4"] is False


def test_a4_pins_parse_from_the_frozen_text():
    pins = a4pre.pinned_inputs(ROOT / "research/PREREG_A4_DRAFT.md")
    assert pins["research/reproduction/a4_v1_pilot.csv"].startswith("c735615a")
    assert len(pins) == 4
