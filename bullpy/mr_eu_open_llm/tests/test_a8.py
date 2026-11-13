"""Amendment A8: design block, preflight, launcher, A1 sharding, families, wrappers, and the
three equivalence tests.

No model and no clip are used locally. The A1 sharding tests run the real frozen
`scripts.evaluate.run_evaluation` with model loading, media and generation stubbed, so they
test the shard mechanism on the frozen code path. Tests that need a GPU and the real model,
or the frozen A3 cell files, skip when those are absent.
"""

import hashlib
import json
import os
import re
import shutil
import sys
import types
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "scripts"))

import a8_design  # noqa: E402
import a8_family  # noqa: E402
import a8_launch  # noqa: E402
import a8_preflight as pre  # noqa: E402
import a8_rotation_shard as shard  # noqa: E402

DRAFT_TEXT = a8_design.DRAFT.read_text(encoding="utf-8")
LOOKUP = ROOT / "data/eu_emotions_human_entropy.json"


# Design block ---------------------------------------------------------------


def test_block_hash_is_pinned_and_stated():
    assert a8_design.block_hash() == a8_design.DESIGN_SHA256 == a8_design.stated_hash()
    assert a8_design.load_design()["amendment"] == "A8"


def test_frozen_files_match_their_pins_and_the_earlier_amendments():
    design = a8_design.load_design()
    a8_design.check_frozen_code(design)
    pins = design["frozen_code_sha256"]
    assert pins["research/scripts/analyse_rotation.py"].startswith("98890c5b")
    assert pins["scripts/face_input_validity.py"].startswith("8a681a65")
    assert pins["scripts/vision_gates.py"].startswith("6327b6f8")
    assert pins["scripts/analyse_stimulus_interventions.py"].startswith("aba4f533")
    assert pins["scripts/a4_analyse.py"].startswith("30f8f529")
    from scripts import a4_design, run_stimulus_interventions as run, stimulus_design

    assert design["frozen_a3_summary_sha256"] == stimulus_design.A3_SUMMARY_SHA256
    assert design["frozen_design_sha256"]["A2"] == run.design_hash()
    assert design["frozen_design_sha256"]["A4"] == a4_design.design_hash()


def test_every_module_reached_by_import_is_pinned():
    """Finding 7: the transitive import closure of the frozen entry points is pinned."""
    closure = a8_design.import_closure()
    pins = a8_design.load_design()["frozen_code_sha256"]
    assert "scripts/activation_forward.py" in closure and "research/scripts/same_clip_cue_analysis.py" in closure
    assert [f for f in closure if f not in pins] == []


def test_data_inputs_are_pinned_and_the_lookups_are_reproducible(tmp_path):
    design = a8_design.load_design()
    a8_design.check_data(design, ["data/eu_emotions_full_manifest.json", "data/eu_emotions_human_entropy.json",
                                  "research/reproduction/eu_au_measures.csv", "research/reproduction/a4_v1_pilot.csv"])
    from paper_v2.option_score import write_rotated_human_lookup

    for r in range(5):
        dest = tmp_path / f"eu_emotions_human_entropy_rot{r}.json"
        write_rotated_human_lookup(LOOKUP, dest, rotation=r)
        assert hashlib.sha256(dest.read_bytes()).hexdigest() == design["data_sha256"][f"data/eu_emotions_human_entropy_rot{r}.json"]
    with pytest.raises(SystemExit, match="missing"):
        a8_design.check_data(design, ["data/eu_emotions_human_entropy_rot0.json"], root=tmp_path / "none")


def test_an_edited_block_is_refused(tmp_path):
    edited = tmp_path / "draft.md"
    edited.write_text(DRAFT_TEXT.replace('"amendment": "A8"', '"amendment": "A9"'), encoding="utf-8")
    with pytest.raises(SystemExit, match="not the pinned"):
        a8_design.load_design(edited)


def test_both_models_are_active_by_the_block_alone():
    """Finding 6: membership is inside the hash; no note outside it can change a family."""
    design = a8_design.load_design()
    assert a8_design.active_models(design) == ["qwen3vl_32b", "gemma4_31b"]
    assert design["models"]["gemma4_31b"]["status"].startswith("admitted_2026-11-09")
    assert not hasattr(a8_design, "admitted")
    entry = dict(design["models"]["gemma4_31b"], status="conditional")
    assert not a8_design.is_active(entry)


def test_code_pins_are_pending_and_scored_runs_wait_on_them():
    assert a8_design.code_pins() == {}
    with pytest.raises(SystemExit, match="not pinned"):
        a8_design.check_own_code(require=True)


# Preflight ------------------------------------------------------------------


def fake_model(tmp_path):
    weights = tmp_path / "store" / "Qwen3-VL-32B-Instruct"
    weights.mkdir(parents=True)
    files = {}
    for name, payload in (("model-00001-of-00002.safetensors", b"abc"), ("model-00002-of-00002.safetensors", b"defg")):
        (weights / name).write_bytes(payload)
        files[name] = {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    (weights / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3VLForConditionalGeneration"], "model_type": "qwen3_vl",
        "text_config": {"num_hidden_layers": 64, "hidden_size": 5120},
    }))
    models_dir = tmp_path / "a8_models"
    models_dir.mkdir()
    (models_dir / "qwen3vl").symlink_to(weights)
    entry = dict(a8_design.load_design()["models"]["qwen3vl_32b"], weights_dir=str(weights), weight_files=files,
                 models_dir=str(models_dir))
    return models_dir, weights, entry


def test_preflight_accepts_the_right_weights(tmp_path):
    models_dir, weights, entry = fake_model(tmp_path)
    assert pre.check_weights_dir(models_dir, entry) == weights.resolve()
    assert pre.check_config(weights, entry["expected_config"])["text_hidden_size"] == 5120
    assert set(pre.check_files(weights, entry["weight_files"], full_hash=True)) == set(entry["weight_files"])


def test_preflight_requires_the_environment_variable_to_match(tmp_path, monkeypatch):
    """Finding 2: the variable the frozen loader reads must be the design's directory."""
    models_dir, weights, entry = fake_model(tmp_path)
    monkeypatch.delenv("MR_EU_HPC_MODELS_DIR", raising=False)
    with pytest.raises(SystemExit, match="not set"):
        pre.env_models_dir(entry)
    monkeypatch.setenv("MR_EU_HPC_MODELS_DIR", str(tmp_path / "study2_models"))
    with pytest.raises(SystemExit, match="not the design"):
        pre.env_models_dir(entry)
    monkeypatch.setenv("MR_EU_HPC_MODELS_DIR", str(models_dir))
    assert pre.env_models_dir(entry) == models_dir.resolve()


def test_preflight_refuses_the_small_sibling_and_a_missing_link(tmp_path):
    models_dir, weights, entry = fake_model(tmp_path)
    cfg = json.loads((weights / "config.json").read_text())
    cfg["text_config"] = {"num_hidden_layers": 36, "hidden_size": 4096}
    (weights / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(SystemExit, match="does not match"):
        pre.check_config(weights, entry["expected_config"])
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(SystemExit, match="does not exist"):
        pre.check_weights_dir(other, entry)
    (other / "qwen3vl").symlink_to(tmp_path)
    with pytest.raises(SystemExit, match="not the design"):
        pre.check_weights_dir(other, entry)


def test_preflight_refuses_changed_missing_or_extra_weight_files(tmp_path):
    models_dir, weights, entry = fake_model(tmp_path)
    (weights / "model-00002-of-00002.safetensors").write_bytes(b"defh")
    pre.check_files(weights, entry["weight_files"])
    with pytest.raises(SystemExit, match="sha256"):
        pre.check_files(weights, entry["weight_files"], full_hash=True)
    (weights / "model-00002-of-00002.safetensors").write_bytes(b"de")
    with pytest.raises(SystemExit, match="bytes"):
        pre.check_files(weights, entry["weight_files"])
    (weights / "extra.safetensors").write_bytes(b"x")
    with pytest.raises(SystemExit, match="extra"):
        pre.check_files(weights, entry["weight_files"])


def test_preflight_keeps_outputs_inside_a8_and_never_overwrites(tmp_path):
    roots = ["results/a8", "research/reproduction/a8"]
    assert pre.check_outputs([Path("results/a8/qwen3vl_32b/x.json")], roots, root=tmp_path)
    for bad in ("results/baseline/eu_emotions/qwen3vl/x.json", "research/reproduction/vision_gates_qwen3vl.json", "results/a8x/y"):
        with pytest.raises(SystemExit, match="outside"):
            pre.check_outputs([Path(bad)], roots, root=tmp_path)
    (tmp_path / "results/a8").mkdir(parents=True)
    (tmp_path / "results/a8/done.json").write_text("{}")
    with pytest.raises(SystemExit, match="exists"):
        pre.check_outputs([Path("results/a8/done.json")], roots, root=tmp_path)


QUOTA = """Filesystem/Project   GB       Quota    Limit    Grace  Files    Quota     Limit     Grace  User/Grp/Proj
/home                49.8     52.4     57.7     -      207498   N/A       N/A       N/A    U:eb2007
/rds-d7              1112.8   0.0      0.0      -      1038416  0         0         -      G:eb2007
/rds-d7/user/eb2007  1041.9   1099.5   1209.5   -      996131   1048576   1048576   -      P:eb2007
"""


def test_storage_rule_reads_the_rds_user_row():
    usage = pre.parse_quota(QUOTA)
    assert usage == {"gb": 1041.9, "files": 996131}
    stop = a8_design.load_design()["storage_stop"]
    pre.check_storage(usage, stop)
    with pytest.raises(SystemExit, match="over the stop rule"):
        pre.check_storage({"gb": 1080.1, "files": 1}, stop)
    with pytest.raises(SystemExit, match="over the stop rule"):
        pre.check_storage({"gb": 1.0, "files": 1040001}, stop)


# Launcher -------------------------------------------------------------------


def test_every_step_writes_only_under_the_a8_roots_with_explicit_paths():
    """Finding 3: every command names its outputs, and they sit under the A8 roots."""
    design = a8_design.load_design()
    ctx = {"a8_id": "qwen3vl_32b", "key": "qwen3vl", "rotation": 0, "condition": "real", "shard": 0, "n_shards": 4}
    roots = [ROOT / r for r in design["output_roots"]]
    for step in design["commands"]:
        spec = a8_launch.command(design, step, ctx)
        for out in spec["outputs"]:
            assert any(r in (ROOT / out).parents for r in roots), (step, out)
        if ("module" in spec or "path" in spec) and "declared_output_outside_roots" not in spec:
            argv = " ".join(spec["argv"])
            assert any(f in argv for f in ("--output", "--output_dir", "--reproduction")), step
            assert "results/stimulus_interventions/cells" not in argv and "results/a4/summary" not in argv
    a2 = a8_launch.command(design, "a2_run", ctx)["argv"]
    assert a2[a2.index("--output_dir") + 1] == "results/a8/qwen3vl_32b/stimulus_interventions"
    assert a2[a2.index("--reproduction") + 1] == "research/reproduction/a8/qwen3vl_32b"
    a4 = a8_launch.command(design, "a4_analyse", ctx)["argv"]
    assert a4[a4.index("--output") + 1] == "results/a8/a4/summary.json" and "--detector" in a4
    rq1 = a8_launch.command(design, "natural_rq1", ctx)
    assert rq1["argv"][rq1["argv"].index("--model") + 1] == "qwen3vl_32b"
    assert rq1["declared_output_outside_roots"] == "research/reproduction/same_clip_family_model_qwen3vl_32b.json"


def test_frozen_output_guard_detects_any_change(tmp_path):
    (tmp_path / "results/baseline/eu_emotions/qwen3vl").mkdir(parents=True)
    f = tmp_path / "results/baseline/eu_emotions/qwen3vl/eval.json"
    f.write_text("{}")
    patterns = a8_design.load_design()["frozen_outputs_guard"]
    files = a8_launch.frozen_output_files(patterns, ["qwen3vl"], root=tmp_path)
    before = a8_launch.snapshot(files)
    assert a8_launch.changed(before, a8_launch.frozen_output_files(patterns, ["qwen3vl"], root=tmp_path)) == []
    f.write_text('{"x": 1}')
    assert a8_launch.changed(before, a8_launch.frozen_output_files(patterns, ["qwen3vl"], root=tmp_path)) == [str(f)]
    new = tmp_path / "results/baseline/eu_emotions/qwen3vl/other.json"
    new.write_text("{}")
    assert str(new) in a8_launch.changed(before, a8_launch.frozen_output_files(patterns, ["qwen3vl"], root=tmp_path))


def test_out_of_memory_in_a_cell_error_field_fails_the_job(tmp_path):
    (tmp_path / "cells.jsonl").write_text(json.dumps({"status": "failed", "reason": "CUDA out of memory. Tried"}) + "\n")
    (tmp_path / "ok.jsonl").write_text(json.dumps({"status": "ok", "raw_model_output": "I am out of memory today"}) + "\n")
    assert a8_launch.oom_in_outputs([tmp_path], "a3") == [str(tmp_path / "cells.jsonl")]
    assert a8_launch.oom_in_outputs([tmp_path], "ladder") == []


def test_the_frozen_8b_gate_file_is_not_flagged():
    """Re-check R1: the expected float32 out-of-memory error in float32_failed is not a job failure."""
    gate = ROOT / "research/reproduction/vision_gates_qwen3vl.json"
    assert "out of memory" in gate.read_text().lower()
    assert a8_launch.oom_in_outputs([gate], "gates") == []
    for step in a8_launch.SCANNED_STEPS:
        assert not any(shard.is_oom(x) for x in a8_launch.error_texts(gate, step)), step


def test_gates_command_mirrors_the_frozen_job_for_each_key():
    """Re-check R1: --float32_max_frames 6 for qwen3vl only, after --output, as the frozen job."""
    design = a8_design.load_design()
    q = a8_launch.command(design, "gates", {"a8_id": "qwen3vl_32b", "key": "qwen3vl"})["argv"]
    g = a8_launch.command(design, "gates", {"a8_id": "gemma4_31b", "key": "gemma4"})["argv"]
    assert q[-2:] == ["--float32_max_frames", "6"] and q.index("--output") < q.index("--float32_max_frames")
    assert "--float32_max_frames" not in g
    frozen = (ROOT / "slurm_jobs/study3_vision_gates.sh").read_text()
    assert 'if [ "${MODEL}" = "qwen3vl" ]; then\n  FLOAT32_MAX_FRAMES="${FLOAT32_MAX_FRAMES:-6}"' in frozen
    for flag, value in (("--gates", "g2,g3,g4"), ("--dtypes", "bfloat16,float32"), ("--n_g3", "12")):
        assert q[q.index(flag) + 1] == value


def test_every_model_step_checks_its_data_inputs():
    """Re-check R6: gates and a4_run included."""
    design = a8_design.load_design()
    for step, spec in design["commands"].items():
        if spec["uses_model"]:
            assert design["data_by_step"].get(step), step
    assert "results/a4/manifest.csv" in design["data_by_step"]["a4_run"]


def test_models_dir_variable_is_set_from_the_design_and_a_different_value_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("MR_EU_HPC_MODELS_DIR", str(tmp_path / "study2_models"))
    with pytest.raises(SystemExit, match="not the design"):
        shard.set_models_dir(tmp_path / "a8_models")


def test_launcher_refuses_before_phase2_and_before_code_pins(monkeypatch):
    with pytest.raises(SystemExit, match="Phase 2"):
        a8_launch.main(["--a8_id", "qwen3vl_32b", "--step", "gates"])
    with pytest.raises(SystemExit, match="not pinned"):
        a8_launch.main(["--a8_id", "qwen3vl_32b", "--step", "gates", "--confirm_phase2_completed"])


def test_a_frozen_script_that_ends_with_sys_exit_still_finishes_its_step(tmp_path):
    """same_clip_family_analysis.py ends with sys.exit(main()). Run in the launcher's process, its clean
    exit must not skip the move of the declared output (natural_rq1); a refusal still fails the job."""
    assert (ROOT / "research/scripts/same_clip_family_analysis.py").read_text().rstrip().endswith("sys.exit(main())")
    script = tmp_path / "frozen.py"
    script.write_text("import sys\nfrom pathlib import Path\n\n\ndef main():\n    Path(sys.argv[1]).write_text('{}')\n\n\n"
                      "if __name__ == '__main__':\n    sys.exit(main())\n")
    declared, target = tmp_path / "outside.json", tmp_path / "a8" / "inside.json"
    a8_launch.run_frozen_step({"path": str(script), "argv": [str(declared)], "outputs": [str(target)]}, str(declared))
    assert target.read_text() == "{}" and not declared.exists()
    for body, code in (("raise SystemExit('refused: not frozen')", "refused: not frozen"), ("sys.exit(3)", 3)):
        bad = tmp_path / "bad.py"
        bad.write_text(f"import sys\n{body}\n")
        with pytest.raises(SystemExit) as info:
            a8_launch.run_frozen({"path": str(bad), "argv": []})
        assert info.value.code == code


def test_job_status_records_an_out_of_memory_crash():
    """A crash whose message is an out-of-memory error is recorded as out_of_memory (dated note 2026-11-09)."""
    oom = "OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"
    assert a8_launch.final_status(True, [], [], None) == "ok"
    assert a8_launch.final_status(False, [], [], None) == "failed"
    assert a8_launch.final_status(False, [], [], "SystemExit: refused") == "failed"
    assert a8_launch.final_status(False, [], [], oom) == "out_of_memory"
    assert a8_launch.final_status(True, [], ["cells.jsonl"], None) == "out_of_memory"
    assert a8_launch.final_status(False, ["frozen.json"], ["cells.jsonl"], oom) == "frozen_output_changed"


def test_a4_analyse_runs_single_threaded(monkeypatch):
    """A4 pre-data ruling 7 (DECISIONS 2026-11-12) holds for A8's run of the same frozen analysis."""
    for v in a8_launch.THREAD_VARS:
        monkeypatch.delenv(v, raising=False)
    with pytest.raises(SystemExit, match="single-threaded"):
        a8_launch.check_threads("a4_analyse")
    assert a8_launch.check_threads("gates") == {v: None for v in a8_launch.THREAD_VARS}
    for v in a8_launch.THREAD_VARS:
        monkeypatch.setenv(v, "1")
    assert a8_launch.check_threads("a4_analyse") == {v: "1" for v in a8_launch.THREAD_VARS}
    text = (ROOT / "slurm_jobs/a8_job.sh").read_text()
    block = text[text.index('if [ "${STEP}" = "a4_analyse" ]'):].split("\nfi")[0]
    assert all(f"{v}=1" in block for v in a8_launch.THREAD_VARS)


def test_slurm_file_never_sets_the_models_dir_and_mirrors_the_frozen_environments():
    text = (ROOT / "slurm_jobs/a8_job.sh").read_text()
    assert "unset MR_EU_HPC_MODELS_DIR" in text and "export MR_EU_HPC_MODELS_DIR" not in text
    assert "venvs/molmo2_cv411" in text and "venvs/au_tools/env/bin/activate" in text
    assert "research/scripts/a8_launch.py" in text
    au = text.index('source "${PROJECT_ROOT}/venvs/au_tools/env/bin/activate"')
    assert text.rfind("else", 0, au) > text.rfind("conda activate", 0, au)  # au_tools alone, as frozen


# A1 sharding ----------------------------------------------------------------


def test_shard_enumerate_restricts_only_the_trial_list():
    trials = [{"trial_id": f"t{i}"} for i in range(5)]
    filt = shard.ShardEnumerate([t["trial_id"] for t in trials], shard.shard_indices(5, 1, 2))
    assert [i for i, _ in filt(trials)] == [1, 3]
    assert list(filt([10, 20])) == [(0, 10), (1, 20)]
    assert list(filt(trials[:4])) == list(enumerate(trials[:4]))
    assert filt.matches == 1
    with pytest.raises(SystemExit):
        shard.shard_indices(5, 2, 2)


def test_argv_matches_the_frozen_rotation_script():
    a1 = a8_design.load_design()["a1"]
    argv = shard.evaluate_argv("qwen3vl", "sequential", 3, "mismatch", Path("o.json"), a1)
    text = (ROOT / "slurm_jobs/study3_rotation_eval.sh").read_text()
    for flag in ("--dataset eu_emotions", "--condition video_only", "--stage stage2", "--fps 1",
                 "--data_root data/eu_emotions", "--manifest data/eu_emotions_full_manifest.json"):
        assert flag in text and flag in " ".join(argv)
    for name, value in (("MAX_FRAMES", "16"), ("SEED", "42"), ("N_OPTIONS", "6"), ("FC_SAMPLES", "20"), ("MISMATCH_SEED", "42")):
        assert f'{name}="${{{name}:-{value}}}"' in text
    assert argv[argv.index("--human_options") + 1] == "data/eu_emotions_human_entropy_rot3.json"
    assert "--fc_sequential" in argv and argv[argv.index("--vision_mismatch_seed") + 1] == "42"
    assert '[ "${MODEL}" = "qwen3vl" ]' in text
    assert "python research/scripts/decoder_preflight.py" in text and "molmo2_cv411" in text
    assert "--write_rotation_lookups" in text


def test_the_runner_is_bound_to_the_design_per_model():
    """Finding 9: key, sampling, models and weights directories and output come from the block."""
    d = a8_design.load_design()
    ns = Namespace(cmd="run", a8_id="gemma4_31b", rotation=1, condition="real", shard=0, n_shards=4)
    run = shard.resolve_run(ns, d)
    assert (run["key"], run["sampling"]) == ("gemma4", "batched")
    assert str(run["weights_dir"]).endswith("gemma-4-31B-it")
    assert str(run["expected_output"]).endswith("results/a8/gemma4_31b/rotation/shards/rot1_real/shard0of4/eval.json")
    with pytest.raises(SystemExit, match="Output must be"):
        shard.check_output_place(ROOT / "results/a8/gemma4_31b/elsewhere/eval.json", run, test_mode=False)
    with pytest.raises(SystemExit, match="not an A8 model"):
        shard.resolve_run(Namespace(cmd="run", a8_id="qwen3vl", rotation=0, condition="real", shard=0, n_shards=1), d)


LABEL_LINE = re.compile(r"^\d+\) (.+)$", re.M)


class _Model:
    def eval(self):
        return self

    def to(self, _device):
        return self


FAIL_ON = {}


def _stub_generate(**kw):
    """Depends on the prompt, the frames (so the partner's video matters) and torch's RNG."""
    import torch

    clip = "|".join(map(str, kw["images"]))
    for marker, message in FAIL_ON.items():
        if marker in clip:
            raise RuntimeError(message)
    options = LABEL_LINE.findall(kw["prompt"])
    base = int(hashlib.sha256((kw["prompt"] + clip).encode()).hexdigest()[:8], 16)
    pick = (base + int(torch.randint(0, 7, (1,)).item())) % (len(options) + 1)
    return "EMOTION: unsure" if pick == len(options) else f"EMOTION: {options[pick]}\nREASONING: stub"


@pytest.fixture()
def stub_evaluate(monkeypatch, tmp_path):
    ev = shard.import_evaluate()
    weights = tmp_path / "equiv_models" / "qwen3vl"
    monkeypatch.setattr(ev, "AutoProcessor", types.SimpleNamespace(from_pretrained=lambda *a, **k: None))
    monkeypatch.setattr(ev, "load_hf_model_for_key", lambda **k: _Model())
    monkeypatch.setattr(ev, "resolve_model_path", lambda key: weights)
    monkeypatch.setattr(ev, "resolve_trial_media", lambda trial, **k: (Path(trial["stimulus_path"]), None, None))
    monkeypatch.setattr(ev, "load_stimulus_as_images", lambda path, fps, max_frames: ([Path(path).name], [0]))
    monkeypatch.setattr(ev, "prepare_images_for_model", lambda key, images, enforce_multi_frame: (images, {"n_frames_used": len(images), "multi_frame_strategy": "stub"}))
    monkeypatch.setattr(ev, "generate_model_response", _stub_generate)
    monkeypatch.setattr(ev, "generate_model_response_batch", lambda num_return_sequences, **kw: [_stub_generate(**kw) for _ in range(num_return_sequences)])
    monkeypatch.setattr(ev, "collect_run_metadata", lambda: {})
    monkeypatch.chdir(ROOT)
    FAIL_ON.clear()
    yield ev
    FAIL_ON.clear()


def _args(cmd, out, condition, tmp_path, max_trials=6, **extra):
    base = dict(cmd=cmd, a8_id=None, equivalence_label="test", equivalence_models_dir=tmp_path / "equiv_models",
                rotation=0, condition=condition, max_trials=max_trials, human_options=str(LOOKUP),
                output=out, test_mode=True)
    base.update(extra)
    return Namespace(**base)


def _sharded(ev, tmp_path, condition, n_shards, max_trials, tag=""):
    files = []
    for k in range(n_shards):
        out = tmp_path / f"{condition}{tag}" / f"shard{k}of{n_shards}" / "eval.json"
        shard.cmd_run(_args("run", out, condition, tmp_path, max_trials=max_trials, shard=k, n_shards=n_shards), ev)
        files.append(out)
    merged = tmp_path / f"merged_{condition}{tag}_{n_shards}.json"
    shard.merge(files, merged)
    return files, merged


def _frozen(ev, tmp_path, condition, max_trials, tag=""):
    out = tmp_path / f"frozen_{condition}{tag}" / "eval.json"
    shard.cmd_frozen(_args("frozen", out, condition, tmp_path, max_trials=max_trials), ev)
    return out


@pytest.mark.parametrize("condition,n_shards,max_trials", [("mismatch", 2, 6), ("real", 2, 6), ("mismatch", 3, 7)])
def test_a1_equivalence_on_the_frozen_code_path(tmp_path, stub_evaluate, condition, n_shards, max_trials):
    """Equivalence test 1, small: sharded and merged records equal the unsharded frozen run."""
    frozen = _frozen(stub_evaluate, tmp_path, condition, max_trials)
    _, merged = _sharded(stub_evaluate, tmp_path, condition, n_shards, max_trials)
    report = shard.compare(frozen, merged)
    assert report["identical"], report
    trials = json.loads(merged.read_text())["trials"]
    assert len(trials) == max_trials and any(t["stage2"]["fc_predictions"] for t in trials)


def test_a1_equivalence_on_all_243_items_with_the_full_partner_map(tmp_path, stub_evaluate):
    """Finding 8: the full-243 partner map with a restricted loop equals the unsharded run."""
    frozen = _frozen(stub_evaluate, tmp_path, "mismatch", None)
    _, merged = _sharded(stub_evaluate, tmp_path, "mismatch", 4, None)
    report = shard.compare(frozen, merged)
    assert report["identical"] and report["n_merged"] == 243, report
    loaded = shard.loaded_trials(stub_evaluate, a8_design.load_design()["a1"], None)
    partners = shard.partner_map(loaded, LOOKUP, 42)
    trials = json.loads(merged.read_text())["trials"]
    assert all(t["vision_source_trial_id"] == partners[t["trial_id"]] for t in trials)


@pytest.mark.parametrize("condition", ["real", "mismatch"])
def test_an_errored_trial_is_kept_scored_wrong_and_merged(tmp_path, stub_evaluate, condition):
    """Finding 1: an errored record stays, as the frozen pipeline keeps it, and the shard merges."""
    FAIL_ON["NF25_cut"] = "ValueError: decode failed"
    frozen = _frozen(stub_evaluate, tmp_path, condition, 6)
    files, merged = _sharded(stub_evaluate, tmp_path, condition, 2, 6)
    assert shard.compare(frozen, merged)["identical"]
    trials = json.loads(merged.read_text())["trials"]
    errored = [t for t in trials if t.get("error")]
    assert errored and all(t.get("vision_source_trial_id") is None for t in errored)
    side = json.loads(shard.sidecar_path(files[0]).read_text()) | json.loads(shard.sidecar_path(files[1]).read_text())
    assert json.loads(merged.read_text())["a8_merged"]["n_errored"] == len(errored)


def test_an_out_of_memory_trial_fails_the_shard_and_blocks_the_merge(tmp_path, stub_evaluate):
    """Finding 12: an OOM error is not scored as wrong; the shard fails and is rerun."""
    FAIL_ON["RF25_cut"] = "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate"
    out1 = tmp_path / "oom" / "shard0of2" / "eval.json"
    shard.cmd_run(_args("run", out1, "real", tmp_path, shard=0, n_shards=2), stub_evaluate)
    out2 = tmp_path / "oom" / "shard1of2" / "eval.json"
    with pytest.raises(SystemExit, match="out of memory"):
        shard.cmd_run(_args("run", out2, "real", tmp_path, shard=1, n_shards=2), stub_evaluate)
    assert json.loads(shard.sidecar_path(out2).read_text())["n_oom"] == 1
    with pytest.raises(SystemExit, match="out of memory"):
        shard.merge([out1, out2], tmp_path / "oom_merged.json")


def test_a_naive_manifest_split_would_change_partners(stub_evaluate):
    loaded = shard.loaded_trials(stub_evaluate, a8_design.load_design()["a1"], None)
    full = shard.partner_map(loaded, LOOKUP, 42)
    half = shard.partner_map(loaded[::2], LOOKUP, 42)
    assert any(full[k] != v for k, v in half.items())


def test_a_missing_lookup_is_refused_not_replaced(tmp_path, stub_evaluate):
    """Finding 4: no fallback to sampled foils or manifest labels."""
    with pytest.raises(SystemExit, match="missing"):
        shard.partner_map([], tmp_path / "eu_emotions_human_entropy_rot0.json", 42)
    out = tmp_path / "nolookup" / "eval.json"
    with pytest.raises(SystemExit, match="missing"):
        shard.cmd_run(_args("run", out, "real", tmp_path, shard=0, n_shards=1, human_options=str(tmp_path / "none.json")), stub_evaluate)


def test_merge_refuses_gaps_duplicates_mixed_runs_edits_and_test_mode_into_production(tmp_path, stub_evaluate):
    files, merged = _sharded(stub_evaluate, tmp_path, "real", 2, 4)
    with pytest.raises(SystemExit, match="exists"):
        shard.merge(files, merged)
    with pytest.raises(SystemExit, match="complete set"):
        shard.merge(files[:1], tmp_path / "gap.json")
    with pytest.raises(SystemExit, match="complete set|two shards"):
        shard.merge([files[0], files[0]], tmp_path / "twice.json")
    with pytest.raises(SystemExit, match="Test-mode"):
        shard.merge(files, ROOT / "results/a8/qwen3vl_32b/rotation/never_written.json")
    assert not (ROOT / "results/a8/qwen3vl_32b/rotation/never_written.json").exists()
    side_path = shard.sidecar_path(files[1])
    side = json.loads(side_path.read_text())
    for key, value in (("sampling", "batched"), ("visible_gpus", 9), ("resolved_model_path", "/elsewhere"),
                       ("evaluate_argv_without_output", ["--other"])):
        side_path.write_text(json.dumps({**side, key: value}))
        with pytest.raises(SystemExit, match="different runs"):
            shard.merge(files, tmp_path / f"mixed_{key}.json")
    side_path.write_text(json.dumps(side))
    obj = json.loads(files[1].read_text())
    obj["trials"][0]["stage2"]["prediction"] = "edited"
    files[1].write_text(json.dumps(obj))
    with pytest.raises(SystemExit, match="changed"):
        shard.merge(files, tmp_path / "edited.json")


def test_each_shard_needs_a_fresh_directory(tmp_path, stub_evaluate):
    out = tmp_path / "s" / "eval.json"
    out.parent.mkdir()
    (out.parent / "other.csv").write_text("x")
    with pytest.raises(SystemExit, match="not empty"):
        shard.cmd_run(_args("run", out, "real", tmp_path, max_trials=4, shard=0, n_shards=2), stub_evaluate)


def test_evaluate_is_hash_checked(tmp_path):
    copy = tmp_path / "evaluate.py"
    copy.write_bytes((ROOT / "scripts/evaluate.py").read_bytes() + b"\n")
    with pytest.raises(SystemExit, match="not the frozen"):
        shard.check_evaluate(copy)


def test_sidecar_records_mode_paths_versions_and_gpus(tmp_path, stub_evaluate):
    out = tmp_path / "rec" / "eval.json"
    report = shard.cmd_run(_args("run", out, "mismatch", tmp_path, shard=0, n_shards=1), stub_evaluate)
    for key in ("test_mode", "versions", "visible_gpus", "resolved_model_path", "evaluate_argv_without_output", "partner_map_sha256"):
        assert key in report
    assert report["test_mode"] is True and report["resolved_model_path"].endswith("equiv_models/qwen3vl")


@pytest.mark.skipif(
    os.environ.get("A8_A1_EQUIV_GPU") != "1",
    reason="GPU equivalence on Qwen3-VL-8B: set A8_A1_EQUIV_GPU=1 and A8_EQUIV_MODELS_DIR on a GPU node",
)
def test_a1_equivalence_on_qwen3vl_8b_gpu():
    """Equivalence test 1 on the cluster: the real 8B, 6 items, both conditions; frozen run twice."""
    import subprocess

    import torch

    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    models_dir = os.environ["A8_EQUIV_MODELS_DIR"]
    out_root = ROOT / "results/a8/equivalence"
    script = str(ROOT / "research/scripts/a8_rotation_shard.py")
    for condition in ("real", "mismatch"):
        base = ["--equivalence_label", "equivalence_qwen3vl_8b", "--equivalence_models_dir", models_dir,
                "--rotation", "0", "--condition", condition, "--max_trials", "6"]
        frozen = []
        for run in ("a", "b"):
            out = out_root / condition / f"frozen_{run}" / "eval.json"
            subprocess.run([sys.executable, script, "frozen", *base, "--output", str(out)], check=True, cwd=ROOT)
            frozen.append(out)
        shards = []
        for k in range(2):
            out = out_root / condition / f"shard{k}of2" / "eval.json"
            subprocess.run([sys.executable, script, "run", *base, "--shard", str(k), "--n_shards", "2", "--output", str(out)], check=True, cwd=ROOT)
            shards.append(str(out))
        merged = out_root / condition / "merged.json"
        subprocess.run([sys.executable, script, "merge", "--shards", *shards, "--output", str(merged)], check=True, cwd=ROOT)
        determinism = shard.compare(frozen[0], frozen[1])
        report = {"frozen_vs_frozen": determinism, "frozen_vs_merged": shard.compare(frozen[0], merged)}
        (out_root / condition / "equivalence_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        assert determinism["identical"], "frozen runs differ: A1 sharding not shown; a dated ruling is needed (design a1_gpu_rule)"
        assert report["frozen_vs_merged"]["identical"], report


# Families -------------------------------------------------------------------


def _delivery(base, a8_id, key, passed, launched=True):
    """A delivery file and, by default, the ok launcher record of the job that wrote it (inputs_from_launched_jobs)."""
    folder = a8_family.repro(base, a8_id)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"input_delivery_{key}.json").write_text(json.dumps({"verdicts": {"single_face": {"pass": passed}}}))
    if launched:
        _launch(base, a8_id, "delivery", stamp="d")


def _gates(base, a8_id, key, passed, launched=True):
    """A gate file and, by default, the ok launcher record of the job that wrote it (inputs_from_launched_jobs)."""
    folder = a8_family.repro(base, a8_id)
    folder.mkdir(parents=True, exist_ok=True)
    verdict = {g: {"pass": passed} for g in ("G2", "G3", "G4")}
    (folder / f"vision_gates_{key}.json").write_text(json.dumps({"smoke": False, "verdict": verdict}))
    if launched:
        _launch(base, a8_id, "gates", stamp="g")


def _launch(base, a8_id, step, status="ok", context=None, weights=None, stamp="0"):
    folder = base / "research/reproduction/a8" / a8_id / "launch"
    folder.mkdir(parents=True, exist_ok=True)
    weights = weights or a8_design.load_design()["models"][a8_id]["weights_dir"]
    tag = "_".join(str(v) for v in (context or {}).values())
    record = {"step": step, "status": status, "context": context or {}, "started_at_utc": stamp,
              "loader": {"resolve_model_path": str(Path(weights).resolve())}}
    (folder / f"{step}{'_' + tag if tag else ''}_{stamp}.json").write_text(json.dumps(record))


def test_a_step_that_fails_twice_counts_as_not_tested_and_others_go_on(tmp_path, a3_cells):
    """Re-check R4: a delivery job that never writes its file excludes only that model."""
    fiv, folder = a3_cells
    design = a8_design.load_design()
    _delivery(tmp_path, "qwen3vl_32b", "qwen3vl", True)
    path = a8_family.a3_cells_path(tmp_path, "qwen3vl_32b", "qwen3vl")
    path.parent.mkdir(parents=True)
    shutil.copy(folder / "cells_qwen3vl.jsonl", path)
    _launch(tmp_path, "qwen3vl_32b", "a3")
    _launch(tmp_path, "gemma4_31b", "delivery", status="started", stamp="1")
    with pytest.raises(SystemExit, match="no delivery file"):
        a8_family.a3_family(design, tmp_path)
    _launch(tmp_path, "gemma4_31b", "delivery", status="failed", stamp="2")
    family, excluded = a8_family.a3_family(design, tmp_path)
    assert [n for n, _ in family] == ["qwen3vl_32b"] and "retry" in excluded["gemma4_31b"]


def test_a3_inputs_need_a_launcher_record_with_the_design_weights(tmp_path, a3_cells):
    """Re-check R3: which weights produced the cells is checked, not only completeness."""
    fiv, folder = a3_cells
    design = a8_design.load_design()
    _delivery(tmp_path, "qwen3vl_32b", "qwen3vl", True)
    _delivery(tmp_path, "gemma4_31b", "gemma4", False)
    path = a8_family.a3_cells_path(tmp_path, "qwen3vl_32b", "qwen3vl")
    path.parent.mkdir(parents=True)
    shutil.copy(folder / "cells_qwen3vl.jsonl", path)
    with pytest.raises(SystemExit, match="no launcher record"):
        a8_family.a3_family(design, tmp_path)
    _launch(tmp_path, "qwen3vl_32b", "a3", weights="/home/eb2007/rds/hpc-work/study2/models/qwen3vl")
    with pytest.raises(SystemExit, match="loaded"):
        a8_family.a3_family(design, tmp_path)


def test_the_launcher_eligibility_follows_the_frozen_rules(tmp_path):
    """Re-check R2: A1 needs gates and A3 Pass; A2 and A4 need rq0_status at single_face and gates."""
    design = a8_design.load_design()
    a3 = tmp_path / a8_family.A3_SUMMARY
    a3.parent.mkdir(parents=True)
    a3.write_text(json.dumps({"qwen3vl_32b": {"vision_dependent": True}, "gemma4_31b": {"vision_dependent": False}}))
    a8_family.write_provenance(a3, [], {})
    _gates(tmp_path, "qwen3vl_32b", "qwen3vl", True)
    _gates(tmp_path, "gemma4_31b", "gemma4", True)
    assert a8_family.eligibility(design, tmp_path, "qwen3vl_32b", "a1_shard")["eligible"]
    with pytest.raises(SystemExit, match="not eligible"):
        a8_family.eligibility(design, tmp_path, "gemma4_31b", "a1_shard")
    a1 = tmp_path / a8_family.A1_SUMMARY
    a1.write_text(json.dumps({"qwen3vl_32b": {"vision_dependent": False}}))
    a8_family.write_provenance(a1, [], {})
    assert a8_family.eligibility(design, tmp_path, "qwen3vl_32b", "a4_run")["rq0_single_face"] is True
    with pytest.raises(SystemExit, match="not eligible"):
        a8_family.eligibility(design, tmp_path, "qwen3vl_32b", "a2_run")  # no delivery file
    _delivery(tmp_path, "qwen3vl_32b", "qwen3vl", True)
    assert a8_family.eligibility(design, tmp_path, "qwen3vl_32b", "a2_run")["eligible"]
    with pytest.raises(SystemExit, match="not eligible"):
        a8_family.eligibility(design, tmp_path, "gemma4_31b", "a4_run")


def test_a3_family_is_computed_from_delivery_files(tmp_path, a3_cells):
    """Finding 5: no hand-picked list; a failed delivery means not tested; a missing file stops."""
    fiv, folder = a3_cells
    design = a8_design.load_design()
    with pytest.raises(SystemExit, match="no delivery file"):
        a8_family.a3_family(design, tmp_path)
    _delivery(tmp_path, "qwen3vl_32b", "qwen3vl", True)
    _delivery(tmp_path, "gemma4_31b", "gemma4", False)
    with pytest.raises(SystemExit, match="does not exist"):
        a8_family.a3_family(design, tmp_path)
    path = a8_family.a3_cells_path(tmp_path, "qwen3vl_32b", "qwen3vl")
    path.parent.mkdir(parents=True)
    shutil.copy(folder / "cells_qwen3vl.jsonl", path)
    _launch(tmp_path, "qwen3vl_32b", "a3")
    family, excluded = a8_family.a3_family(design, tmp_path)
    assert [n for n, _ in family] == ["qwen3vl_32b"] and "gemma4_31b" in excluded


def test_an_incomplete_a3_cell_file_is_refused(tmp_path, a3_cells):
    """Finding 11: a partial run cannot be summarised as 0% lost."""
    fiv, folder = a3_cells
    lines = (folder / "cells_qwen3vl.jsonl").read_text().splitlines(keepends=True)
    short = tmp_path / "short.jsonl"
    short.write_text("".join(lines[:-10]))
    with pytest.raises(SystemExit, match="not 243"):
        a8_family.check_a3_cells_complete(short)
    partial = tmp_path / "partial.jsonl"
    partial.write_text("".join(lines) + '{"trial_id": "x"')
    with pytest.raises(SystemExit, match="partial"):
        a8_family.check_a3_cells_complete(partial)
    twice = tmp_path / "twice.jsonl"
    twice.write_text("".join(lines) + lines[0])
    with pytest.raises(SystemExit, match="duplicate"):
        a8_family.check_a3_cells_complete(twice)
    a8_family.check_a3_cells_complete(folder / "cells_qwen3vl.jsonl")


def test_an_a3_file_left_incomplete_by_two_oom_attempts_at_the_top_gpu_count_is_no_result(tmp_path, a3_cells):
    """Dated note 2026-11-09: a partial A3 file after two out-of-memory attempts at the largest GPU count
    used counts as no A3 result for that model, and the other model goes on. Until then the family waits."""
    fiv, folder = a3_cells
    design = a8_design.load_design()
    for a8_id, key in (("qwen3vl_32b", "qwen3vl"), ("gemma4_31b", "gemma4")):
        _delivery(tmp_path, a8_id, key, True)
        path = a8_family.a3_cells_path(tmp_path, a8_id, key)
        path.parent.mkdir(parents=True)
        shutil.copy(folder / f"cells_{key}.jsonl", path)
    _launch(tmp_path, "gemma4_31b", "a3")
    partial = a8_family.a3_cells_path(tmp_path, "qwen3vl_32b", "qwen3vl")
    partial.write_text("".join(partial.read_text().splitlines(keepends=True)[:100]))

    def attempt(status, gpus, stamp):
        _launch(tmp_path, "qwen3vl_32b", "a3", status=status, stamp=stamp)
        p = tmp_path / "research/reproduction/a8/qwen3vl_32b/launch" / f"a3_{stamp}.json"
        p.write_text(json.dumps(dict(json.loads(p.read_text()), visible_gpus=gpus)))

    attempt("out_of_memory", 2, "1")
    attempt("out_of_memory", 2, "2")
    attempt("out_of_memory", 4, "3")  # one out-of-memory attempt so far at the largest count: wait
    with pytest.raises(SystemExit, match="not 243"):
        a8_family.a3_family(design, tmp_path)
    attempt("failed", 4, "4")  # a crash with no out-of-memory evidence does not count
    with pytest.raises(SystemExit, match="not 243"):
        a8_family.a3_family(design, tmp_path)
    attempt("out_of_memory", 4, "5")
    family, excluded = a8_family.a3_family(design, tmp_path)
    assert [n for n, _ in family] == ["gemma4_31b"] and "dated note 2026-11-09" in excluded["qwen3vl_32b"]


def _a1_files(base, a8_id, gpus=1, test_mode=False):
    for r in a8_family.ROTATIONS:
        for c in a8_family.CONDITIONS:
            path = a8_family.merged_path(base, a8_id, r, c)
            path.parent.mkdir(parents=True, exist_ok=True)
            weights = str(Path(a8_design.load_design()["models"][a8_id]["weights_dir"]).resolve())
            side = {"visible_gpus": gpus, "test_mode": test_mode, "resolved_model_path": weights, "mode": "production"}
            path.write_text(json.dumps({"a8_merged": {"shard_sidecars": [side]}, "trials": []}))


def test_a1_family_follows_the_stop_rule_and_requires_every_file(tmp_path):
    design = a8_design.load_design()
    summary = tmp_path / a8_family.A3_SUMMARY
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({"qwen3vl_32b": {"vision_dependent": True}, "gemma4_31b": {"vision_dependent": True}}))
    _gates(tmp_path, "qwen3vl_32b", "qwen3vl", True)
    _gates(tmp_path, "gemma4_31b", "gemma4", False)
    with pytest.raises(SystemExit, match="0 of 10"):
        a8_family.a1_family(design, tmp_path)
    _a1_files(tmp_path, "qwen3vl_32b")
    family, excluded = a8_family.a1_family(design, tmp_path)
    assert family == ["qwen3vl_32b"] and "stop rule" in excluded["gemma4_31b"]
    _a1_files(tmp_path, "gemma4_31b")
    with pytest.raises(SystemExit, match="outside the A1 family"):
        a8_family.a1_family(design, tmp_path)
    assert a8_family.check_a1_run_settings(tmp_path, ["qwen3vl_32b"], design) == {"qwen3vl_32b": 1}
    path = a8_family.merged_path(tmp_path, "qwen3vl_32b", 0, "real")
    path.write_text(json.dumps({"a8_merged": {"shard_sidecars": [{"visible_gpus": 1, "test_mode": False, "resolved_model_path": "/8b", "mode": "production"}]}}))
    with pytest.raises(SystemExit, match="not a production run"):
        a8_family.check_a1_run_settings(tmp_path, ["qwen3vl_32b"], design)
    path = a8_family.merged_path(tmp_path, "qwen3vl_32b", 3, "mismatch")
    path.write_text(json.dumps({"a8_merged": {"shard_sidecars": [{"visible_gpus": 2, "test_mode": False}]}}))
    with pytest.raises(SystemExit, match="different GPU counts"):
        a8_family.check_a1_run_settings(tmp_path, ["qwen3vl_32b"])


def test_provenance_refuses_a_changed_output_or_input(tmp_path):
    out, inp = tmp_path / "out.json", tmp_path / "in.json"
    inp.write_text("1")
    out.write_text("2")
    a8_family.write_provenance(out, [inp], {})
    a8_family.check_provenance(out)
    inp.write_text("3")
    with pytest.raises(SystemExit, match="changed"):
        a8_family.check_provenance(out)
    inp.write_text("1")
    out.write_text("4")
    with pytest.raises(SystemExit, match="differs"):
        a8_family.check_provenance(out)


def test_provenance_written_on_one_machine_is_checked_on_another(tmp_path, monkeypatch):
    """Paths under the project root are stored relative to it, so a summary written on the cluster can be
    checked after a pull, and no record names one machine's folders."""
    first, second = tmp_path / "cluster", tmp_path / "mac"
    for root in (first, second):
        (root / "results/a8").mkdir(parents=True)
    monkeypatch.setattr(a8_design, "ROOT", first)
    out, inp = first / "research/reproduction/a8/summary.json", first / "results/a8/cells.jsonl"
    out.parent.mkdir(parents=True)
    inp.write_text("1")
    out.write_text("2")
    a8_family.write_provenance(out, [inp], {})
    record = json.loads(a8_family.provenance_path(out).read_text())
    assert record["output"] == "research/reproduction/a8/summary.json"
    assert list(record["inputs_sha256"]) == ["results/a8/cells.jsonl"]
    shutil.copytree(first / "research", second / "research")
    shutil.copy(inp, second / "results/a8/cells.jsonl")
    monkeypatch.setattr(a8_design, "ROOT", second)
    a8_family.check_provenance(second / "research/reproduction/a8/summary.json")
    (second / "results/a8/cells.jsonl").write_text("3")
    with pytest.raises(SystemExit, match="changed"):
        a8_family.check_provenance(second / "research/reproduction/a8/summary.json")


# A1 analysis ----------------------------------------------------------------


import a8_analyse_rotation as a1w  # noqa: E402


def test_a1_analysis_runs_the_frozen_main_over_the_a8_family(tmp_path):
    """The frozen statistics over synthetic merged files; Holm over one model is the identity."""
    human = json.loads(LOOKUP.read_text())["trials"]
    (tmp_path / "data").mkdir()
    shutil.copy(LOOKUP, tmp_path / "data" / LOOKUP.name)
    rng = np.random.default_rng(0)
    for r in a8_family.ROTATIONS:
        for c in a8_family.CONDITIONS:
            trials = []
            for tid, h in human.items():
                options = h["human_options"]
                target = h["human_target_label"]
                ok = rng.random() < (0.5 if c == "real" else 0.15)
                pred = target if ok else options[(options.index(target) + 1) % len(options)]
                trials.append({"trial_id": tid, "stage2": {"correct": ok, "prediction": pred, "options": options,
                                                           "scoring_label": target, "response_distribution": {pred: 1.0}}})
            path = a8_family.merged_path(tmp_path, "qwen3vl_32b", r, c)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"trials": trials}))
    out, err = tmp_path / "summary.json", tmp_path / "errored.json"
    a1w.analyse(["qwen3vl_32b"], out, err, base=tmp_path)
    result = json.loads(out.read_text())
    assert list(result) == ["qwen3vl_32b"]
    assert result["qwen3vl_32b"]["p_delta_holm"] == result["qwen3vl_32b"]["p_delta_actor_sign_flip"]
    assert result["qwen3vl_32b"]["vision_dependent"] is True
    a8_family.check_provenance(out)
    with pytest.raises(SystemExit, match="runs once"):
        a1w.analyse(["qwen3vl_32b"], out, tmp_path / "e2.json", base=tmp_path)


# A3 summary -----------------------------------------------------------------


import a8_face_input_validity_summary as a3w  # noqa: E402


def synthetic_a3_cells(fiv, path, seed, effect):
    human = json.loads(LOOKUP.read_text())["trials"]
    items = [{"trial_id": k, "label": v["human_target_label"], "actor": k.split("/")[-1][0].upper(), "options": v["human_options"]}
             for k, v in human.items()]
    partners = fiv.partners_for(items)
    by_id = {i["trial_id"]: i for i in items}
    rng = np.random.default_rng(seed)
    lines = []
    for item in items:
        for cell in fiv.trial_cells(item, by_id[partners[item["trial_id"]]]):
            scores = rng.normal(size=len(cell["options"]))
            if cell["condition"] == "real":
                scores[cell["options"].index(cell["label"])] += effect
            cell.update(status="ok", joint_logprob=[round(float(s), 4) for s in scores])
            cell.update(fiv.correct_from_scores(cell["options"], list(scores), cell["label"]))
            lines.append(json.dumps(cell) + "\n")
    path.write_text("".join(lines))


@pytest.fixture(scope="module")
def a3_cells(tmp_path_factory):
    fiv = a3w.load_frozen()
    folder = tmp_path_factory.mktemp("a3")
    for seed, (model, effect) in enumerate(zip(fiv.MODELS, (1.0, 0.6, 0.0))):
        synthetic_a3_cells(fiv, folder / f"cells_{model}.jsonl", seed, effect)
    return fiv, folder


def test_a3_wrapper_equals_the_frozen_summarise_on_synthetic_cells(a3_cells):
    """Equivalence test 2, on synthetic cells: identical bytes to the frozen --summarise."""
    fiv, folder = a3_cells
    fiv.summarise(Namespace(lookup=LOOKUP, output_dir=folder))
    frozen = (folder / "face_input_validity_summary.json").read_bytes()
    assert a3w.summary_bytes([(m, folder / f"cells_{m}.jsonl") for m in fiv.MODELS]) == frozen


def test_a3_production_path_uses_the_computed_family_and_writes_provenance(tmp_path, a3_cells):
    fiv, folder = a3_cells
    _delivery(tmp_path, "qwen3vl_32b", "qwen3vl", True)
    _delivery(tmp_path, "gemma4_31b", "gemma4", True)
    for a8_id, key in (("qwen3vl_32b", "qwen3vl"), ("gemma4_31b", "gemma4")):
        path = a8_family.a3_cells_path(tmp_path, a8_id, key)
        path.parent.mkdir(parents=True)
        shutil.copy(folder / f"cells_{key}.jsonl", path)
        _launch(tmp_path, a8_id, "a3")
    out = a3w.produce(tmp_path)
    data = json.loads(out.read_text())
    assert list(data) == ["gemma4_31b", "qwen3vl_32b"]
    record = a8_family.check_provenance(out)
    assert record["family"] == ["qwen3vl_32b", "gemma4_31b"]
    expected = a3w.summary_bytes([("qwen3vl_32b", folder / "cells_qwen3vl.jsonl"), ("gemma4_31b", folder / "cells_gemma4.jsonl")])
    assert out.read_bytes() == expected
    with pytest.raises(SystemExit, match="written once"):
        a3w.produce(tmp_path)


FROZEN_A3 = [ROOT / "results/face_input_validity" / f"cells_{m}.jsonl" for m in ("qwen3vl", "gemma4", "molmo2")]


def _close(a, b, path=""):
    """Equal, except floats may differ in the last digits (library version)."""
    if isinstance(a, dict) and isinstance(b, dict):
        assert set(a) == set(b), path
        return [d for k in a for d in _close(a[k], b[k], f"{path}/{k}")]
    if isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), path
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _close(x, y, f"{path}[{i}]")]
    if isinstance(a, float) and isinstance(b, float) and a != b:
        assert abs(a - b) <= 1e-12 * max(abs(a), abs(b)), (path, a, b)
        return [path]
    assert a == b, (path, a, b)
    return []


@pytest.mark.skipif(not all(p.exists() for p in FROZEN_A3), reason="frozen A3 cell files are not present")
def test_a3_wrapper_on_the_frozen_cell_files(tmp_path):
    """Equivalence test 2, on the frozen files.

    (a) The wrapper equals the frozen --summarise run here, byte for byte.
    (b) Against the pinned summary (made on the cluster): every field equal; floats may differ
        in the last digits, which comes from the library build, not the wrapper. The exact
        hash is required where the frozen environment runs (A8_FROZEN_ENV=1, cluster).
    """
    for path, digest in zip(FROZEN_A3, ("a3f302b3", "05eec2d4", "cd80b5c5")):
        assert hashlib.sha256(path.read_bytes()).hexdigest().startswith(digest)
    fiv = a3w.load_frozen()
    for path in FROZEN_A3:
        shutil.copy(path, tmp_path / path.name)
    fiv.summarise(Namespace(lookup=LOOKUP, output_dir=tmp_path))
    data = a3w.summary_bytes(list(zip(("qwen3vl", "gemma4", "molmo2"), FROZEN_A3)))
    assert data == (tmp_path / "face_input_validity_summary.json").read_bytes()
    pinned = ROOT / "research/reproduction/face_input_validity_summary.json"
    differing = _close(json.loads(data), json.loads(pinned.read_text()))
    assert all(d.endswith("accuracy_real_vs_chance_pooled_p") for d in differing), differing
    if os.environ.get("A8_FROZEN_ENV") == "1":
        assert hashlib.sha256(data).hexdigest() == a8_design.load_design()["frozen_a3_summary_sha256"]


# A2 loader ------------------------------------------------------------------


import a8_analyse_a2 as a2w  # noqa: E402

DELIVERED = {
    "gate_2_every_frame_changes_output": True,
    "gate_3_logged_indices_are_the_frames_sent": True,
    "gate_4_ids_unchanged_within_clip": True,
    "pixels_change_under_shuffle": True,
}


def a2_inputs(tmp_path):
    """Synthetic A2 shards in the frozen format, as tests/test_readout_and_analysis.py builds them."""
    from paper_v2.schema import actor_id_from_eu_filename
    from scripts import recovery_stimulus_interventions as rec
    from scripts import run_stimulus_interventions as run

    human = json.loads(LOOKUP.read_text())["trials"]
    data = []
    for key, value in human.items():
        named = {k: v for k, v in value["human_response_distribution"].items() if k != run.ABSTAIN}
        data.append({"trial_id": key, "actor": actor_id_from_eu_filename(key), "label": value["human_target_label"],
                     "human_modal_label": max(sorted(named), key=lambda k: named[k]),
                     "weak_label": value["human_target_label"] in run.WEAK_LABELS})
    rows = [r for r in rec.synthetic_rows("human_like", data, 7) if r["input_name"] in ("single_face", "video_face")]
    words = run.lexicon_words()
    design = run.design_hash()
    trials = sorted({r["trial_id"] for r in rows})
    for row in rows:
        row.update({"design_sha256": design, "mcq": [], "media_sha256": ["same"], "edit": {"mask_fraction": 0.05}, "position": 0})
        row["blind"]["word_logprob"] = [row["blind"]["family_logprob"][f] for f, _ in words]
    delivery = {r: {**DELIVERED, "repeat_forward_max_abs": 0.0, "tokens_across_view": 48.0} for r in ("single_face", "video_face")}
    free = dict(zip(rec.FAMILIES, np.zeros(21).tolist()))
    baselines = [{"input_name": r, "condition": "image_free", "family_logprob": free} for r in delivery]
    cells, reports = [], []
    out_dir = tmp_path / "results/a8/gemma4_31b/stimulus_interventions"
    out_dir.mkdir(parents=True)
    for k in range(3):
        cell = out_dir / f"cells_gemma4_shard{k}.jsonl"
        cell.write_text("".join(json.dumps(r) + "\n" for r in rows if trials.index(r["trial_id"]) % 3 == k))
        report = out_dir / f"run_gemma4_shard{k}.json"
        report.write_text(json.dumps({"model": "gemma4", "smoke": False, "design_sha256": design, "shard": k, "n_shards": 3, "delivery": [delivery], "baselines": baselines}))
        cells.append(str(cell))
        reports.append(str(report))
    geometry = tmp_path / "geometry.json"
    geometry.write_text(json.dumps({"items": {t: {"face_box": [0, 0, 400, 400], "face_width_px": 200.0, "width": 800, "height": 400} for t in trials}}))
    return cells, reports, geometry, run


def test_a2_production_path_gives_the_frozen_output_byte_for_byte(tmp_path, monkeypatch):
    """Equivalence test 3, through run(): frozen A3 summary with gemma4 renamed to gemma4_31b,
    A8 A1 summary, provenance checked, A1 view written; output equals the frozen analysis."""
    an = a2w.load_frozen()
    cells, reports, geometry, run = a2_inputs(tmp_path)
    monkeypatch.setattr(run, "check_geometry", lambda path: None)
    folder = tmp_path / "research/reproduction/a8/gemma4_31b"
    folder.mkdir(parents=True)
    for name in ("vision_gates_gemma4.json", "input_delivery_gemma4.json"):
        shutil.copy(ROOT / "research/reproduction" / name, folder / name)
    frozen_a3 = ROOT / "research/reproduction/face_input_validity_summary.json"
    a8_a3 = json.loads(frozen_a3.read_text())
    a8_a3["gemma4_31b"] = a8_a3.pop("gemma4")
    a3_path = tmp_path / a8_family.A3_SUMMARY
    a3_path.write_text(json.dumps(a8_a3, indent=2, sort_keys=True) + "\n")
    a1_path = tmp_path / a8_family.A1_SUMMARY
    a1_path.write_text(json.dumps({"gemma4_31b": {"vision_dependent": True}}))
    a8_family.write_provenance(a3_path, [], {})
    a8_family.write_provenance(a1_path, [], {})
    common = ["--geometry", str(geometry), "--a2_frozen"]
    wrapped = tmp_path / "results/a8/gemma4_31b/a2_summary.json"
    with pytest.raises(SystemExit, match="no a2_run job finished ok"):
        a2w.run("gemma4_31b", [*common, "--output", str(wrapped)], base=tmp_path)
    for k in range(3):
        _launch(tmp_path, "gemma4_31b", "a2_run", context={"shard": k, "n_shards": 3})
    _launch(tmp_path, "gemma4_31b", "gates")
    _launch(tmp_path, "gemma4_31b", "delivery")
    with pytest.raises(SystemExit, match="pre-run check first"):
        a2w.run("gemma4_31b", [*common, "--output", str(wrapped)], base=tmp_path)
    used_cells, used_reports, _ = a2w.shard_inputs(a8_design.load_design(), tmp_path, "gemma4_31b")
    names = [Path(p) for p in (*used_cells, *used_reports)]
    rel = lambda p: p.resolve().relative_to(tmp_path.resolve()).as_posix()
    (tmp_path / a8_family.A2_PRERUN.format(a8_id="gemma4_31b")).write_text(json.dumps(
        {"failures": [], "analysed_files": [rel(p) for p in names],
         "sha256": {rel(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in names}}))
    a2w.run("gemma4_31b", [*common, "--output", str(wrapped)], base=tmp_path)
    assert json.loads((folder / "rotation_summary.json").read_text()) == {"gemma4": {"vision_dependent": True}}
    record = a8_family.check_provenance(wrapped)  # the prereg's provenance file, checked as the next consumer would
    assert record["a8_id"] == "gemma4_31b" and len(record["inputs_sha256"]) == 3 + 3 + 5

    # The frozen analysis, frozen loader, on the same folder with the pinned A3 file in it.
    shutil.copy(frozen_a3, folder / frozen_a3.name)
    frozen = tmp_path / "frozen.json"
    a2w.run_frozen_main(an, ["--model", "gemma4", "--reproduction", str(folder), "--cells", *cells,
                             "--run_report", *reports, *common, "--output", str(frozen)], None)
    assert wrapped.read_bytes() == frozen.read_bytes()
    assert json.loads(frozen.read_text())["single_face"]["gates"]["rq0"] is True
    assert an.a3_summary.__module__ == an.__name__


def test_a2_wrapper_refuses_hand_set_paths_abbreviations_and_unrecorded_inputs(tmp_path):
    """Re-check R3: no pass-through of the wrapper's own options, in full or abbreviated."""
    for bad in (["--model", "gemma4"], ["--repro", "research/reproduction"], ["--reproduction=x"],
                ["--cells", "results/stimulus_interventions/cells_qwen3vl_shard0.jsonl"], ["--run", "r.json"],
                ["--c", "x"]):
        with pytest.raises(SystemExit, match="set by the wrapper"):
            a2w.run("gemma4_31b", [*bad, "--output", "results/a8/x.json"], base=tmp_path)
    with pytest.raises(SystemExit, match="abbreviation"):
        a2w.run("gemma4_31b", ["--out", "results/a8/x.json"], base=tmp_path)
    with pytest.raises(SystemExit, match="exactly once"):
        a2w.run("gemma4_31b", ["--output", "results/a8/x.json", "--output", "results/a8/y.json"], base=tmp_path)
    with pytest.raises(SystemExit, match="outside the A8 output roots"):
        a2w.run("gemma4_31b", ["--output", "results/stimulus_interventions/x.json"], base=tmp_path)
    with pytest.raises(SystemExit, match="provenance is missing"):
        a2w.run("gemma4_31b", ["--output", "results/a8/x.json"], base=tmp_path)


def test_the_frozen_loader_cannot_see_an_a8_summary_and_the_a8_loader_can(tmp_path):
    an = a2w.load_frozen()
    path = tmp_path / "face_input_validity_summary.json"
    path.write_text(json.dumps({"qwen3vl_32b": {"vision_dependent": True}}))
    assert an.a3_summary(tmp_path, tmp_path) == {}
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    loader = a2w.a3_loader(path, sha, lambda s: {"qwen3vl": s["qwen3vl_32b"]})
    assert loader(tmp_path, tmp_path) == {"qwen3vl": {"vision_dependent": True}}
    with pytest.raises(SystemExit, match="not the given"):
        a2w.a3_loader(path, "0" * 64, lambda s: s)


def test_a1_view_is_empty_for_a_model_outside_the_a1_family(tmp_path):
    a1 = tmp_path / "rotation_summary_a8.json"
    a1.write_text(json.dumps({"qwen3vl_32b": {"vision_dependent": False}}))
    path = a2w.write_a1_view(a1, "gemma4_31b", "gemma4", tmp_path / "g")
    assert json.loads(path.read_text()) == {}
    path = a2w.write_a1_view(a1, "qwen3vl_32b", "qwen3vl", tmp_path / "q")
    assert json.loads(path.read_text()) == {"qwen3vl": {"vision_dependent": False}}
    a1.write_text(json.dumps({"qwen3vl_32b": {"vision_dependent": True}}))
    with pytest.raises(SystemExit, match="other content"):
        a2w.write_a1_view(a1, "qwen3vl_32b", "qwen3vl", tmp_path / "q")


# A4 gates -------------------------------------------------------------------


def test_a4_gates_file_is_built_from_a8_outputs(tmp_path):
    import a8_a4_gates

    (tmp_path / "research/reproduction/a8").mkdir(parents=True)
    (tmp_path / a8_family.A3_SUMMARY).write_text(json.dumps({"qwen3vl_32b": {"vision_dependent": True}, "gemma4_31b": {"vision_dependent": True}}))
    (tmp_path / a8_family.A1_SUMMARY).write_text(json.dumps({"qwen3vl_32b": {"vision_dependent": False}}))
    _gates(tmp_path, "qwen3vl_32b", "qwen3vl", True)
    _gates(tmp_path, "gemma4_31b", "gemma4", False)
    out = a8_a4_gates.gates(tmp_path)
    assert out["qwen3vl"] == {"a8_id": "qwen3vl_32b", "a1_passed": False, "a3_passed": True, "g2_g4": True}
    assert out["gemma4"]["a1_passed"] is None and out["gemma4"]["g2_g4"] is False


# Pre-run checks for A8's A2 and A4 analyses (dated note 2026-11-13) ----------


import a8_prerun_check as a8pre  # noqa: E402


def _a4_manifest():
    rows = [{"stim_id": f"bwc_{i:03d}", "set": "main", "dataset": "basic_WC"} for i in range(30)]
    rows += [{"stim_id": f"cwc_{i:03d}", "set": "main", "dataset": "conversational_WC"} for i in range(30)]
    rows += [{"stim_id": "female_a_neutral", "set": "neutral", "dataset": "basic_WC"}]
    return rows


def _a8_a4_shards(base, key, manifest, design_hash, n, nan_in=None):
    from scripts import a4_run as run

    folder = base / a8pre.CELLS_A4
    folder.mkdir(parents=True, exist_ok=True)
    for k in range(n):
        rows = []
        for cell in run.planned_cells(manifest, k, n):
            row = {**cell, "status": "ok", "design_sha256": design_hash}
            if cell["readout"] in ("blind", "blind_image_free"):
                row["family_logprob"] = {"Happy": -1.0}
            else:
                row["joint_logprob"] = [-1.0, -2.0]
            rows.append(row)
        if k == nan_in:
            row = next(r for r in rows if "joint_logprob" in r)
            row["joint_logprob"] = [float("nan"), -2.0]
        (folder / f"cells_{key}_{k}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def _attempt(base, a8_id, step, status, gpus, stamp, context=None):
    _launch(base, a8_id, step, status=status, context=context, stamp=stamp)
    tag = "_".join(str(v) for v in (context or {}).values())
    p = base / "research/reproduction/a8" / a8_id / "launch" / f"{step}{'_' + tag if tag else ''}_{stamp}.json"
    p.write_text(json.dumps(dict(json.loads(p.read_text()), visible_gpus=gpus)))


def test_a8_a4_prerun_uses_the_launched_shard_count_and_filters_in_place(tmp_path):
    """A4's own row check at A8's run-setting shard count. Non-finite rows: the original is kept unchanged in
    unfiltered/, the filtered copy takes its name, which is all the frozen analysis reads."""
    from scripts import a4_analyse, a4_run as run

    design_hash, manifest = "d" * 64, _a4_manifest()
    weights = a8_design.load_design()["models"]["qwen3vl_32b"]["weights_dir"]
    for k in range(3):
        _launch(tmp_path, "qwen3vl_32b", "a4_run", context={"shard": k, "n_shards": 3}, stamp=str(k))
    _a8_a4_shards(tmp_path, "qwen3vl", manifest, design_hash, 3, nan_in=1)
    check = lambda: a8pre.check_a4_model(tmp_path, "qwen3vl_32b", "qwen3vl", weights, manifest, design_hash, run)
    out = check()
    assert out["failures"] == [] and out["n_shards"] == 3 and out["shards"] == [0, 1, 2]
    assert [r["file"] for r in out["non_finite_rows"]] == ["cells_qwen3vl_1.jsonl"]
    assert out["pending_filter"] == ["cells_qwen3vl_1.jsonl"]
    folder = tmp_path / a8pre.CELLS_A4
    before = (folder / "cells_qwen3vl_1.jsonl").read_bytes()
    done = a8pre.filter_in_place(folder, out["non_finite_rows"], out["pending_filter"])
    assert set(done) == {"cells_qwen3vl_1.jsonl"}
    assert (folder / "unfiltered/cells_qwen3vl_1.jsonl").read_bytes() == before
    assert len((folder / "cells_qwen3vl_1.jsonl").read_bytes().splitlines()) == len(before.splitlines()) - 1
    assert [p.name for p in a4_analyse.shard_files(folder, "qwen3vl")] == [f"cells_qwen3vl_{k}.jsonl" for k in range(3)]
    again = check()
    assert again["failures"] == [] and again["pending_filter"] == [] and again["filtered_earlier"] == ["cells_qwen3vl_1.jsonl"]
    (folder / "cells_qwen3vl_1.jsonl").write_bytes(before + b'{"extra": 1}\n')  # neither the original nor its filter
    assert any("differs from its original" in f for f in check()["failures"])
    _launch(tmp_path, "qwen3vl_32b", "a4_run", context={"shard": 0, "n_shards": 4}, stamp="9")
    assert "shard counts" in check()["failures"][0]


def test_an_interrupted_filter_is_completed_by_a_rerun(tmp_path, monkeypatch):
    """Review finding 5: a failed write never leaves a shard name empty, and the next run completes it."""
    from scripts import a4_run as run

    design_hash, manifest = "d" * 64, _a4_manifest()
    weights = a8_design.load_design()["models"]["qwen3vl_32b"]["weights_dir"]
    for k in range(3):
        _launch(tmp_path, "qwen3vl_32b", "a4_run", context={"shard": k, "n_shards": 3}, stamp=str(k))
    _a8_a4_shards(tmp_path, "qwen3vl", manifest, design_hash, 3, nan_in=1)
    folder = tmp_path / a8pre.CELLS_A4
    check = lambda: a8pre.check_a4_model(tmp_path, "qwen3vl_32b", "qwen3vl", weights, manifest, design_hash, run)
    out = check()
    real = Path.write_bytes

    def quota_full(self, data):
        if self.name.endswith(".filtered"):
            raise OSError("Disk quota exceeded")
        return real(self, data)

    monkeypatch.setattr(Path, "write_bytes", quota_full)
    with pytest.raises(OSError):
        a8pre.filter_in_place(folder, out["non_finite_rows"], out["pending_filter"])
    monkeypatch.setattr(Path, "write_bytes", real)
    assert (folder / "cells_qwen3vl_1.jsonl").exists() and (folder / "unfiltered/cells_qwen3vl_1.jsonl").exists()
    again = check()
    assert again["failures"] == [] and again["pending_filter"] == ["cells_qwen3vl_1.jsonl"]
    a8pre.filter_in_place(folder, again["non_finite_rows"], again["pending_filter"])
    final = check()
    assert final["failures"] == [] and final["pending_filter"] == [] and final["shards"] == [0, 1, 2]


def test_a8_a4_shards_need_a_launcher_record_each(tmp_path):
    from scripts import a4_run as run

    design_hash, manifest = "d" * 64, _a4_manifest()
    weights = a8_design.load_design()["models"]["gemma4_31b"]["weights_dir"]
    for k in (0, 1):
        _launch(tmp_path, "gemma4_31b", "a4_run", context={"shard": k, "n_shards": 3}, stamp=str(k))
    _a8_a4_shards(tmp_path, "gemma4", manifest, design_hash, 3)
    out = a8pre.check_a4_model(tmp_path, "gemma4_31b", "gemma4", weights, manifest, design_hash, run)
    assert "no launcher record with status ok" in out["failures"][0]


def test_a8_a4_set_aside_needs_spent_retries_and_never_overwrites(tmp_path):
    """Dated note 2026-11-09, read per shard: two out-of-memory attempts at the largest GPU count."""
    design = a8_design.load_design()
    folder = tmp_path / a8pre.CELLS_A4
    folder.mkdir(parents=True)
    (folder / "cells_gemma4_0.jsonl").write_text("{}\n")
    _attempt(tmp_path, "gemma4_31b", "a4_run", "out_of_memory", 4, "1", {"shard": 0, "n_shards": 2})
    with pytest.raises(SystemExit, match="no shard"):
        a8pre.set_aside_plan(tmp_path, design, "gemma4_31b")
    _attempt(tmp_path, "gemma4_31b", "a4_run", "out_of_memory", 4, "2", {"shard": 0, "n_shards": 2})
    plan = a8pre.set_aside_plan(tmp_path, design, "gemma4_31b")
    assert plan["shards_out_of_retries"] == [0] and plan["files"] == ["cells_gemma4_0.jsonl"]
    assert (folder / "cells_gemma4_0.jsonl").exists()  # planning moves nothing
    done = a8pre.apply_set_aside(tmp_path, plan)
    assert (folder / "no_result/cells_gemma4_0.jsonl").exists() and set(done["moved_sha256"]) == {"cells_gemma4_0.jsonl"}
    (folder / "cells_gemma4_0.jsonl").write_text('{"later": 1}\n')
    with pytest.raises(SystemExit, match="nothing is overwritten"):
        a8pre.set_aside_plan(tmp_path, design, "gemma4_31b")
    with pytest.raises(SystemExit, match="not an active A8 model|Unknown|not"):
        a8pre.set_aside_plan(tmp_path, design, "gemma4_31")


def _a4_tree(base, monkeypatch):
    """Both A8 models eligible for A4 (A1 and A3 Pass, gates passed), the gates file built with provenance,
    the synthetic manifest, and the input pins switched off (they are tested with the real files elsewhere)."""
    import a8_a4_gates

    for v in a8_launch.THREAD_VARS:
        monkeypatch.setenv(v, "1")
    monkeypatch.setattr(a8_design, "check_data", lambda design, names, root=None: {})
    monkeypatch.setattr(a8pre.a4pre, "pinned_inputs", lambda path: {})
    for a8_id, key in (("qwen3vl_32b", "qwen3vl"), ("gemma4_31b", "gemma4")):
        _gates(base, a8_id, key, True)
    for summary in (a8_family.A3_SUMMARY, a8_family.A1_SUMMARY):
        (base / summary).write_text(json.dumps({m: {"vision_dependent": True} for m in ("qwen3vl_32b", "gemma4_31b")}))
        a8_family.write_provenance(base / summary, [], {})
    a8_a4_gates.produce(base)
    (base / a8pre.MANIFEST_A4).parent.mkdir(parents=True, exist_ok=True)
    import pandas

    pandas.DataFrame(_a4_manifest()).to_csv(base / a8pre.MANIFEST_A4, index=False)


def test_a8_a4_prerun_accounts_for_every_eligible_or_launched_model(tmp_path, monkeypatch):
    """Review finding 1: a model launched for A4 (or eligible for it) without shard files fails the check;
    it is not silently outside the family."""
    from scripts import a4_design

    _a4_tree(tmp_path, monkeypatch)
    for k in range(2):
        _launch(tmp_path, "qwen3vl_32b", "a4_run", context={"shard": k, "n_shards": 2}, stamp=str(k))
    _a8_a4_shards(tmp_path, "qwen3vl", _a4_manifest(), a4_design.design_hash(), 2)
    result = a8pre.check_a4(tmp_path)
    assert result["members"]["gemma4_31b"] == ["eligible"]
    assert any("gemma4_31b: eligible for A4 but no a4_run job has started" in f for f in result["failures"])
    _launch(tmp_path, "gemma4_31b", "a4_run", status="started", context={"shard": 0, "n_shards": 2}, stamp="0")
    result = a8pre.check_a4(tmp_path)
    assert any(f.startswith("gemma4_31b: launcher records") for f in result["failures"])
    deferred = a8pre.check_a4(tmp_path, deferred=["gemma4_31b"], deferral_note="PREREG_A8 dated note")
    assert any("--deferred, but A4 jobs or cells exist" in f for f in deferred["failures"])
    assert any("gemma4_31" in f for f in a8pre.check_a4(tmp_path, no_result=["gemma4_31"])["failures"])  # no traceback


def test_a8_a4_prerun_moves_files_only_when_everything_else_passes(tmp_path, monkeypatch):
    """Review findings 2 and 4: the set-aside and the filter happen only on a passing check, a rerun is safe,
    every record hashes what was moved, and the analysis needs the passing record and those bytes."""
    from scripts import a4_design

    _a4_tree(tmp_path, monkeypatch)
    design_hash = a4_design.design_hash()
    for k in range(2):
        _launch(tmp_path, "qwen3vl_32b", "a4_run", context={"shard": k, "n_shards": 2}, stamp=str(k))
    _a8_a4_shards(tmp_path, "qwen3vl", _a4_manifest(), design_hash, 2, nan_in=0)
    folder = tmp_path / a8pre.CELLS_A4
    (folder / "cells_gemma4_0.jsonl").write_text('{"partial": true}\n')
    for stamp in ("1", "2"):
        _attempt(tmp_path, "gemma4_31b", "a4_run", "out_of_memory", 4, stamp, {"shard": 0, "n_shards": 2})
    held = a8pre.check_a4(tmp_path, apply_filter=False, no_result=["gemma4_31b"])
    assert any("--apply_filter" in f for f in held["failures"]) and any("no file was moved" in f for f in held["failures"])
    assert (folder / "cells_gemma4_0.jsonl").exists() and not (folder / "unfiltered").exists()
    done = a8pre.check_a4(tmp_path, apply_filter=True, no_result=["gemma4_31b"])
    assert done["failures"] == [], done["failures"]
    assert (folder / "no_result/cells_gemma4_0.jsonl").exists() and (folder / "unfiltered/cells_qwen3vl_0.jsonl").exists()
    assert done["analysed_files"] == [f"results/a8/a4/cells_qwen3vl_{k}.jsonl" for k in range(2)]
    assert "results/a8/a4/no_result/cells_gemma4_0.jsonl" in done["sha256"]
    rerun = a8pre.check_a4(tmp_path, apply_filter=True, no_result=["gemma4_31b"])
    assert rerun["failures"] == [] and "results/a8/a4/no_result/cells_gemma4_0.jsonl" in rerun["sha256"]
    record = tmp_path / a8_family.A4_PRERUN
    record.write_text(json.dumps(rerun))
    a8_family.check_prerun_record(record, a8_family.a4_cells_files(tmp_path), tmp_path)
    (folder / "cells_qwen3vl_1.jsonl").write_text((folder / "cells_qwen3vl_1.jsonl").read_text() + "\n")
    with pytest.raises(SystemExit, match="not the file"):
        a8_family.check_prerun_record(record, a8_family.a4_cells_files(tmp_path), tmp_path)
    record.write_text(json.dumps({**rerun, "failures": ["x"]}))
    with pytest.raises(SystemExit, match="did not pass"):
        a8_family.check_prerun_record(record, a8_family.a4_cells_files(tmp_path), tmp_path)
    record.unlink()
    with pytest.raises(SystemExit, match="pre-run check first"):
        a8_family.check_prerun_record(record, a8_family.a4_cells_files(tmp_path), tmp_path)


def test_a8_a2_prerun_reads_the_a8_models_own_inputs(tmp_path):
    """A2 rulings 2, 3 and 6 on A8's own files; the small sibling's gate files are never needed."""
    base, a8_id, key = tmp_path, "gemma4_31b", "gemma4"
    (base / "scripts").mkdir()
    (base / a8pre.a2pre.ROOT_MARKER).write_text("")
    for k in range(2):
        _launch(base, a8_id, "a2_run", context={"shard": k, "n_shards": 2}, stamp=str(k))
    _launch(base, a8_id, "gates", stamp="g")
    _launch(base, a8_id, "delivery", stamp="d")
    folder = base / "results/a8" / a8_id / "stimulus_interventions"
    folder.mkdir(parents=True)
    good = {"status": "ok", "input": {"pixel_shapes": {"pixel_values": [1, 2520, 768]}}}
    other = {"status": "ok", "input": {"pixel_shapes": {"pixel_values": [2304, 1536]}}}
    (folder / "cells_gemma4_shard0.jsonl").write_text(json.dumps(good) + "\n" + json.dumps(other) + "\n")
    (folder / "cells_gemma4_shard1.jsonl").write_text(json.dumps(good) + "\n" + '{"status": "ok", "inp')
    for k in range(2):
        (folder / f"run_gemma4_shard{k}.json").write_text("{}")
    repro = a8_family.repro(base, a8_id)
    (repro / f"input_delivery_{key}.json").write_text(json.dumps(
        {"items": [{"single_face": {"input": {"pixel_shapes": {"pixel_values": [1, 2520, 768]}}}}]}))
    (repro / f"vision_gates_{key}.json").write_text("{}")
    for summary in (a8_family.A3_SUMMARY, a8_family.A1_SUMMARY):
        (base / summary).write_text("{}")
        a8_family.write_provenance(base / summary, [], {})
    result = a8pre.check_a2(a8_id, base)
    fails = " | ".join(result["failures"])
    assert result["n_rows_not_matching_model"] == 1 and "cells_gemma4_shard1.jsonl" in result["unparseable_lines"]
    assert "geometry.json is missing" in fails
    assert "launcher records" not in fails and "summary" not in fails and "gate input" not in fails
    assert len(result["analysed_files"]) == 4
    assert not (base / "research/reproduction/vision_gates_gemma4.json").exists()
    (base / a8_family.A1_SUMMARY).write_text('{"changed": true}')
    assert "A8 A1 summary" in " | ".join(a8pre.check_a2(a8_id, base)["failures"])
    assert "gemma4_31" in a8pre.check_a2("gemma4_31", base)["failures"][0]  # an unknown id is a failure, not a traceback


def test_a8_prerun_check_waits_for_the_code_pins_like_every_a8_entry_point():
    with pytest.raises(SystemExit, match="not pinned"):
        a8pre.main(["a4"])
    with pytest.raises(SystemExit, match="not pinned"):
        a8pre.main(["a2", "--a8_id", "qwen3vl_32b"])


def test_a_crashed_gates_job_leaves_no_gate_result_and_does_not_hold_up_the_other_model(tmp_path):
    """Review finding 3: vision_gates.py saves its file before it loads the model, so a crashed job leaves a
    file without a verdict. Two such attempts are no gate result (not passed), and the other model goes on."""
    import a8_a4_gates

    design = a8_design.load_design()
    _gates(tmp_path, "qwen3vl_32b", "qwen3vl", True)
    repro = a8_family.repro(tmp_path, "gemma4_31b")
    repro.mkdir(parents=True, exist_ok=True)
    (repro / "vision_gates_gemma4.json").write_text(json.dumps({"model": "gemma4", "smoke": False, "stages": {}}))
    _launch(tmp_path, "gemma4_31b", "gates", status="failed", stamp="1")
    with pytest.raises(SystemExit, match="not read yet"):  # one failed attempt: the retry decides
        a8_family.gate_result(design, tmp_path, "gemma4_31b")
    _launch(tmp_path, "gemma4_31b", "gates", status="failed", stamp="2")
    assert a8_family.gate_result(design, tmp_path, "gemma4_31b") is False
    for summary in (a8_family.A3_SUMMARY, a8_family.A1_SUMMARY):
        (tmp_path / summary).write_text(json.dumps({"qwen3vl_32b": {"vision_dependent": True}}))
    assert a8_a4_gates.gates(tmp_path)["gemma4"]["g2_g4"] is False
    for r in a8_family.ROTATIONS:
        for c in a8_family.CONDITIONS:
            p = a8_family.merged_path(tmp_path, "qwen3vl_32b", r, c)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}")
    family, excluded = a8_family.a1_family(design, tmp_path)
    assert family == ["qwen3vl_32b"] and "gemma4_31b" in excluded


def test_a_retry_still_in_the_queue_is_not_an_ended_attempt(tmp_path, monkeypatch):
    """Review finding 6: a started record whose Slurm job is still queued or running has not ended, so the
    family code waits instead of concluding not tested."""
    design = a8_design.load_design()
    _launch(tmp_path, "gemma4_31b", "delivery", status="failed", stamp="1")
    _launch(tmp_path, "gemma4_31b", "delivery", status="started", stamp="2")
    p = tmp_path / "research/reproduction/a8/gemma4_31b/launch/delivery_2.json"
    p.write_text(json.dumps(dict(json.loads(p.read_text()), slurm_job_id="41000001")))
    monkeypatch.setattr(a8_family, "job_in_queue", lambda job: job == "41000001")
    with pytest.raises(SystemExit, match="no delivery file from a finished delivery job"):
        a8_family.delivery_status(design, tmp_path, "gemma4_31b")
    monkeypatch.setattr(a8_family, "job_in_queue", lambda job: False)  # the job has left the queue: it ended
    assert a8_family.delivery_status(design, tmp_path, "gemma4_31b") == "not_tested"


def test_a4_gates_file_is_written_once_with_provenance(tmp_path):
    """PREREG_A8_DRAFT.md: the gates file A4 reads carries a provenance file that the next consumer checks."""
    import a8_a4_gates

    (tmp_path / "research/reproduction/a8").mkdir(parents=True)
    for summary in (a8_family.A3_SUMMARY, a8_family.A1_SUMMARY):
        (tmp_path / summary).write_text(json.dumps({"qwen3vl_32b": {"vision_dependent": True}}))
        a8_family.write_provenance(tmp_path / summary, [], {})
    _gates(tmp_path, "qwen3vl_32b", "qwen3vl", True)
    with pytest.raises(SystemExit, match="no gates job has ended ok"):  # gemma4_31b's gates job has not ended
        a8_a4_gates.produce(tmp_path)
    assert not (tmp_path / a8_a4_gates.OUTPUT).exists()  # nothing written when the build refuses
    _gates(tmp_path, "gemma4_31b", "gemma4", False)
    path = a8_a4_gates.produce(tmp_path)
    record = a8_family.check_provenance(path)
    assert len(record["inputs_sha256"]) == 4  # two summaries and the two gate files
    with pytest.raises(SystemExit, match="exists"):
        a8_a4_gates.produce(tmp_path)
    (tmp_path / a8_family.A1_SUMMARY).write_text("{}")
    with pytest.raises(SystemExit, match="changed"):
        a8_family.check_provenance(path)


def test_gate_and_delivery_files_are_read_only_from_jobs_that_ended_ok(tmp_path):
    """Block rule inputs_from_launched_jobs for gate and delivery files: a file with no ok job behind it is
    not read; two failed attempts with no file (not_tested_after_retry), or a file left by two out-of-memory
    attempts at the largest GPU count (dated note 2026-11-09), mean no result; otherwise the caller waits."""
    design = a8_design.load_design()
    _gates(tmp_path, "qwen3vl_32b", "qwen3vl", True, launched=False)
    _delivery(tmp_path, "qwen3vl_32b", "qwen3vl", True, launched=False)
    with pytest.raises(SystemExit, match="no gates job has ended ok"):
        a8_family.gate_result(design, tmp_path, "qwen3vl_32b")
    with pytest.raises(SystemExit, match="no delivery file from a finished delivery job"):
        a8_family.delivery_status(design, tmp_path, "qwen3vl_32b")

    def attempt(step, status, gpus, stamp):
        _launch(tmp_path, "qwen3vl_32b", step, status=status, stamp=stamp)
        p = tmp_path / "research/reproduction/a8/qwen3vl_32b/launch" / f"{step}_{stamp}.json"
        p.write_text(json.dumps(dict(json.loads(p.read_text()), visible_gpus=gpus)))

    for stamp in ("1", "2"):
        attempt("gates", "out_of_memory", 2, stamp)
        attempt("delivery", "out_of_memory", 2, stamp)
    assert a8_family.gate_result(design, tmp_path, "qwen3vl_32b") is False  # a file left by spent retries: not passed
    assert a8_family.delivery_status(design, tmp_path, "qwen3vl_32b") == "not_tested"
    _launch(tmp_path, "qwen3vl_32b", "gates", stamp="3")
    _launch(tmp_path, "qwen3vl_32b", "delivery", stamp="3")
    assert a8_family.gate_result(design, tmp_path, "qwen3vl_32b") is True
    assert a8_family.delivery_status(design, tmp_path, "qwen3vl_32b") == "passed"
    for name in ("gates", "delivery"):  # two plain failures and no file: not tested, as the frozen rule says
        _launch(tmp_path, "gemma4_31b", name, status="failed", stamp="1")
        _launch(tmp_path, "gemma4_31b", name, status="started", stamp="2")
    assert a8_family.gate_result(design, tmp_path, "gemma4_31b") is False
    assert a8_family.delivery_status(design, tmp_path, "gemma4_31b") == "not_tested"
    _launch(tmp_path, "gemma4_31b", "gates", stamp="3", weights="/home/eb2007/rds/hpc-work/study2/models/gemma4")
    with pytest.raises(SystemExit, match="loaded"):
        a8_family.gate_result(design, tmp_path, "gemma4_31b")


# The A8 pull (post_phase3.py pull a8) ------------------------------------------


def _runbook():
    import importlib.util

    spec = importlib.util.spec_from_file_location("post_phase3_for_a8", ROOT / "research/scripts/post_phase3.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a8_pull_copies_outputs_and_only_hashes_cells_and_shards():
    book = _runbook()
    remote = {"research/reproduction/a8/face_input_validity_summary_a8.json": "a",
              "research/reproduction/a8/qwen3vl_32b/launch/gates_1.json": "b",
              "results/a8/a4/summary.json": "c",
              "results/a8/a4/cells_qwen3vl_0.jsonl": "d",
              "results/a8/a4/unfiltered/cells_qwen3vl_0.jsonl": "e",
              "results/a8/qwen3vl_32b/rotation/eval_v2_eu_emotions_qwen3vl_32b_video_only_seed42_rot0_real.json": "f"}
    copy, hashed = book.a8_plan(remote)
    assert copy == ["research/reproduction/a8/face_input_validity_summary_a8.json",
                    "research/reproduction/a8/qwen3vl_32b/launch/gates_1.json", "results/a8/a4/summary.json"]
    assert hashed == sorted(set(remote) - set(copy))


def test_a8_pull_checks_provenance_with_the_script_it_sends(tmp_path, monkeypatch, capsys):
    """The script the pull runs on the cluster, run here on a synthetic tree: ok, then a changed input."""
    book = _runbook()
    monkeypatch.setattr(a8_design, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    out = tmp_path / a8_family.A3_SUMMARY
    inp = tmp_path / "results/a8/qwen3vl_32b/face_input_validity/cells_qwen3vl.jsonl"
    out.parent.mkdir(parents=True)
    inp.parent.mkdir(parents=True)
    inp.write_text("1")
    out.write_text("2")
    a8_family.write_provenance(out, [inp], {})
    monkeypatch.chdir(tmp_path)
    exec(compile(book.A8_PROVENANCE_CHECK, "a8_provenance_check", "exec"), {})
    assert json.loads(capsys.readouterr().out) == {a8_family.A3_SUMMARY: "ok"}
    inp.write_text("3")
    exec(compile(book.A8_PROVENANCE_CHECK, "a8_provenance_check", "exec"), {})
    assert "changed" in json.loads(capsys.readouterr().out)[a8_family.A3_SUMMARY]


def test_a8_pull_runs_once(tmp_path, monkeypatch):
    book = _runbook()
    monkeypatch.setattr(book, "ROOT", tmp_path)
    (tmp_path / book.REPRO).mkdir(parents=True)
    (tmp_path / book.REPRO / "a8_pull_manifest.json").write_text("{}")
    with pytest.raises(SystemExit, match="runs once"):
        book.main(["pull", "a8", "--jobs", "40399722"])
