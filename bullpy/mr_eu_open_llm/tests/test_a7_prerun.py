"""Tests for A7's pre-run check and its place in the Phase 3 runbook (2026-11-12). Synthetic files only."""
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts import a7_design  # noqa: E402
from scripts import run_stimulus_interventions as run  # noqa: E402
from scripts.a7_analyse import ROUTE  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name + "_under_test", ROOT / "research" / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pre = _load("a7_prerun_check")
book = _load("post_phase3")
MODELS = ("gemma4", "qwen3vl")


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _tree(root):
    (root / "scripts").mkdir()
    (root / pre.ROOT_MARKER).write_text("")
    research = root / "research"
    (research / "reproduction").mkdir(parents=True)
    (research / "DECISIONS.md").write_text(pre.DECISIONS_HEADING + ", recorded after Stage 0\n")
    geometry = root / pre.GEOMETRY
    geometry.parent.mkdir(parents=True)
    geometry.write_text(json.dumps({"items": {
        "T1": {"face_box": [0, 0, 100, 100], "face_width_px": 100},  # 16 tokens across: kept
        "T2": {"face_box": [0, 0, 100, 100], "face_width_px": 40},   # 6.4: under the floor
        "T3": {}}}))                                                  # no way to tell: under the floor
    steps, outputs = [], {}
    for m in MODELS:
        shutil.copy(ROOT / "research" / "reproduction" / f"a7_stage0_{m}.json", research / "reproduction")
        cells = f"results/stimulus_interventions/cells_{m}_shard0.jsonl"
        report = f"results/stimulus_interventions/run_{m}_shard0.json"
        out = f"research/reproduction/stimulus_interventions_{m}.json"
        _jsonl(root / cells, [{"trial_id": t} for t in ("T1", "T2", "T3")])
        (root / report).write_text(json.dumps({"delivery": [{ROUTE: {"tokens_across_view": 16.0}}]}))
        (root / out).write_text("{}")
        (research / "reproduction" / f"a2_prerun_{m}.json").write_text(json.dumps(
            {"sha256": {cells: _sha(root / cells), report: _sha(root / report)}}))
        steps.append({"command": ["python3", "-m", "scripts.analyse_stimulus_interventions", "--model", m,
                                  "--cells", cells, "--run_report", report, "--output", out, "--a2_frozen"]})
        outputs[out] = _sha(root / out)
        for k in range(2):
            _jsonl(root / "results" / "a7" / f"a7_cells_{m}_shard{k}.jsonl", [
                {"trial_id": f"T{k}", "input_name": ROUTE, "condition": c, "status": "ok",
                 "a7_design_sha256": a7_design.DESIGN_SHA256, "decision_date": pre.DECISION_DATE}
                for c in ("original", "edit")])
    (research / "reproduction" / "a2_run_note.json").write_text(json.dumps(
        {"status": "complete", "steps": steps, "outputs": outputs}))


def test_a7_prerun_passes_on_a_complete_tree_and_builds_the_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "check_geometry", lambda path: None)
    _tree(tmp_path)
    record, inputs = pre.check(tmp_path)
    assert record["failures"] == []
    assert inputs["below_token_floor"] == {m: ["T2", "T3"] for m in MODELS}
    for m in MODELS:
        assert inputs[m]["decisions"]["route"]["run"] is True and inputs[m]["decisions"]["token"]["run"] is False
        assert inputs[m]["a2_cells"] == [f"results/stimulus_interventions/cells_{m}_shard0.jsonl"]
        assert len(inputs[m]["stage1_cells"]) == 2
    assert "research/reproduction/a7_stage0_gemma4.json" in record["sha256"]


def test_a7_prerun_refuses_bad_rows_a_changed_a2_input_and_no_a2_run(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "check_geometry", lambda path: None)
    _tree(tmp_path)
    shard = tmp_path / "results" / "a7" / "a7_cells_qwen3vl_shard1.jsonl"
    rows = [json.loads(l) for l in shard.read_text().splitlines()]
    shard.write_text("".join(json.dumps(r) + "\n" for r in rows + [dict(rows[0], decision_date="2026-11-13")]) + '{"tri')
    a2_cells = tmp_path / "results/stimulus_interventions/cells_gemma4_shard0.jsonl"
    a2_cells.write_text(a2_cells.read_text() + json.dumps({"trial_id": "T4"}) + "\n")
    record, inputs = pre.check(tmp_path)
    assert inputs is None
    text = " ".join(record["failures"])
    assert "unparseable" in text and "decision_date" in text and "duplicated" in text
    assert "differs from A2's pre-run record" in text
    (tmp_path / "research/reproduction/a2_run_note.json").unlink()
    record, _ = pre.check(tmp_path)
    assert any("has not run" in f for f in record["failures"])


def test_runbook_knows_a7():
    assert book.expected_files("a7") == [f"results/a7/a7_cells_{m}_shard{k}.jsonl" for m in MODELS for k in range(2)]
    check, analysis = book.commands("a7")
    assert "research/scripts/a7_prerun_check.py" in check
    assert analysis[1:3] == ["-m", "scripts.a7_analyse"] and "--a7_frozen" in analysis and "--a2_frozen" in analysis
    assert book.outputs("a7") == ["research/reproduction/a7_analysis.json"]
