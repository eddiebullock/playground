"""Table 1 from locked outputs only (2026-11-13)."""
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t1 = _load("make_table1_under_test", ROOT / "research/scripts/make_table1.py")


def test_table1_on_the_locked_outputs_reads_nothing_unlocked():
    # A2, A4 and A9 are locked (run notes 2026-11-14), so every cell is read from a locked output.
    text = t1.build(ROOT)
    assert ("| Qwen3-VL | .402 (.00009) | .244 (.00009) | Face use on both inputs | .549 (.0002), confirmatory | "
            "Inconclusive; Inconclusive | Not aligned; replication not testable | single_face: Gate B pass; 4, 0, 0, 6<br>"
            "video_face: Gate B pass; 0, 0, 0, 6 |") in text
    assert ("| Gemma 4 | .228 (.00009) | .216 (.00014) | Face use on both inputs | .418 (.0006), confirmatory | "
            "Aligned; Mixed | Aligned; replicated on a second human sample (same lab and generator) | single_face: Gate B pass; "
            "5, 0, 0, 5<br>video_face: Gate B pass; 0, 1, 0, 5 |") in text
    assert "| Molmo2 |" in text and "not eligible (vision gates)" in text
    assert "results/a4/summary.json" in text and "results/a9/summary.json" in text
    assert "stimulus_interventions_gemma4.json" in text and "a2_extras_qwen3vl.json" in text


def test_table1_fills_a2_a4_a9_once_their_run_notes_exist(tmp_path):
    mf = _load("make_figures_for_table1", ROOT / "research/scripts/make_figures.py")
    repro = tmp_path / "research" / "reproduction"
    repro.mkdir(parents=True)
    for name in ("rotation_summary.json", "face_input_validity_summary.json",
                 *[f"same_clip_family_model_{m}.json" for m in t1.MODELS]):
        shutil.copy(ROOT / "research/reproduction" / name, repro / name)
    (tmp_path / "results/a4").mkdir(parents=True)
    (tmp_path / "results/a9").mkdir(parents=True)
    a4 = mf.synthetic_fig5()
    a4["models"] = {m: {"headline": {"basic": "Aligned", "conversational": "Mixed"}} for m in t1.ELIGIBLE}
    (tmp_path / "results/a4/summary.json").write_text(json.dumps(a4))
    a9 = {"models": {m: {"decision": "Aligned", "replication": {"reading": "replicated on a second human sample (same lab and generator)"}}
                     for m in t1.ELIGIBLE}}
    (tmp_path / "results/a9/summary.json").write_text(json.dumps(a9))
    for m, out in zip(t1.ELIGIBLE, list(mf.synthetic_fig6().values())[:2]):
        path = repro / f"stimulus_interventions_{m}.json"
        path.write_text(json.dumps(out))
        ext = {"a2_output_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for route in t1.ROUTES:
            rows = out[route]["results"]["primary_blind_acted"]
            ext[route] = {"contrasts": {cid: {"verdict_after_fill_rule": "inconclusive" if i == 0 else r["verdict"]}
                                        for i, (cid, r) in enumerate(rows.items())}}
        (repro / f"a2_extras_{m}.json").write_text(json.dumps(ext))
    text = t1.build(tmp_path)
    assert text.count("| pending") == 6  # A4, A9 and A2 for both eligible models, before any run note
    for a in ("a2", "a4", "a9"):
        (repro / f"{a}_run_note.json").write_text(json.dumps({"status": "complete"}))
    text = t1.build(tmp_path)
    assert "Aligned; Mixed" in text and "replicated on a second human sample" in text
    assert "single_face: Gate B pass;" in text and "pending |" not in text
