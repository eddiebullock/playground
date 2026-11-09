"""Amendment A8: design block, preflight, A1 sharding, and the three equivalence tests.

No model and no clip are used locally. The A1 sharding test runs the real frozen
`scripts.evaluate.run_evaluation` with model, media and generation stubbed, so it tests the
shard mechanism on the frozen code path. Tests that need a GPU and the real model, or the
frozen A3 cell files, skip when those are absent.
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
import a8_preflight as pre  # noqa: E402
import a8_rotation_shard as shard  # noqa: E402

DRAFT_TEXT = a8_design.DRAFT.read_text(encoding="utf-8")


# Design block ---------------------------------------------------------------


def test_block_hash_is_pinned_and_stated():
    assert a8_design.block_hash() == a8_design.DESIGN_SHA256 == a8_design.stated_hash()
    design = a8_design.load_design()
    assert design["amendment"] == "A8"


def test_every_frozen_file_matches_its_pin_and_the_earlier_amendments_pins():
    design = a8_design.load_design()
    a8_design.check_frozen_code(design)
    pins = design["frozen_code_sha256"]
    # Values pinned by A2, A3 and A4 themselves.
    assert pins["research/scripts/analyse_rotation.py"].startswith("98890c5b")
    assert pins["scripts/face_input_validity.py"].startswith("8a681a65")
    assert pins["scripts/vision_gates.py"].startswith("6327b6f8")
    assert pins["scripts/analyse_stimulus_interventions.py"].startswith("aba4f533")
    assert pins["scripts/a4_analyse.py"].startswith("30f8f529")
    from scripts import stimulus_design

    assert design["frozen_a3_summary_sha256"] == stimulus_design.A3_SUMMARY_SHA256
    from scripts import run_stimulus_interventions as run

    assert design["frozen_design_sha256"]["A2"] == run.design_hash()


def test_an_edited_block_is_refused(tmp_path):
    edited = tmp_path / "draft.md"
    edited.write_text(DRAFT_TEXT.replace('"amendment": "A8"', '"amendment": "A9"'), encoding="utf-8")
    with pytest.raises(SystemExit, match="not the pinned"):
        a8_design.load_design(edited)


def test_gemma_is_conditional_until_a_dated_note_admits_it(tmp_path):
    design = a8_design.load_design()
    # The live draft admits Gemma 4 31B by the dated note of 2026-11-09.
    assert a8_design.active_models(design) == ["qwen3vl_32b", "gemma4_31b"]
    bare = tmp_path / "bare.md"
    bare.write_text(re.sub(r"(?m)^- \d{4}-\d{2}-\d{2}: admit `gemma4_31b`.*$", "", DRAFT_TEXT), encoding="utf-8")
    assert a8_design.active_models(design, bare) == ["qwen3vl_32b"]
    with pytest.raises(SystemExit, match="dated note"):
        a8_design.model(design, "gemma4_31b", bare)
    noted = tmp_path / "draft.md"
    noted.write_text(bare.read_text(encoding="utf-8").rstrip("\n") + "\n\n- 2026-11-10: admit `gemma4_31b`, storage approved; no output exists.\n", encoding="utf-8")
    assert a8_design.active_models(design, noted) == ["qwen3vl_32b", "gemma4_31b"]
    assert a8_design.model(design, "gemma4_31b", noted)["code_key"] == "gemma4"


def test_code_pins_are_pending_and_scored_runs_wait_on_them():
    assert a8_design.code_pins() == {}
    with pytest.raises(SystemExit, match="not pinned"):
        a8_design.check_own_code(require=True)


def test_run_settings_are_not_in_the_block():
    """Shard and GPU counts are run settings, so no model entry fixes one."""
    for entry in a8_design.load_design()["models"].values():
        assert not any("shard" in k or "gpu" in k for k in entry)


# Preflight ------------------------------------------------------------------


def fake_model(tmp_path, entry_overrides=None):
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
    entry = dict(a8_design.load_design()["models"]["qwen3vl_32b"], weights_dir=str(weights), weight_files=files)
    entry.update(entry_overrides or {})
    return models_dir, weights, entry


def test_preflight_accepts_the_right_weights(tmp_path):
    models_dir, weights, entry = fake_model(tmp_path)
    assert pre.check_weights_dir(models_dir, entry) == weights.resolve()
    assert pre.check_config(weights, entry["expected_config"])["text_hidden_size"] == 5120
    rows = pre.check_files(weights, entry["weight_files"], full_hash=True)
    assert set(rows) == set(entry["weight_files"])


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
    pre.check_files(weights, entry["weight_files"])  # same size passes the quick check
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
    a1 = json.loads(a8_design.block_bytes())["a1"]
    argv = shard.evaluate_argv("qwen3vl", "sequential", 3, "mismatch", Path("o.json"), a1)
    text = (ROOT / "slurm_jobs/study3_rotation_eval.sh").read_text()
    for flag in ("--dataset eu_emotions", "--condition video_only", "--stage stage2", "--fps 1",
                 "--data_root data/eu_emotions", "--manifest data/eu_emotions_full_manifest.json"):
        assert flag in text and flag in " ".join(argv)
    for name, value in (("MAX_FRAMES", "16"), ("SEED", "42"), ("N_OPTIONS", "6"), ("FC_SAMPLES", "20"), ("MISMATCH_SEED", "42")):
        assert f'{name}="${{{name}:-{value}}}"' in text
    assert argv[argv.index("--human_options") + 1] == "data/eu_emotions_human_entropy_rot3.json"
    assert "--fc_sequential" in argv and argv[argv.index("--vision_mismatch_seed") + 1] == "42"
    assert '[ "${MODEL}" = "qwen3vl" ]' in text  # sequential for qwen3vl, batched otherwise
    real = shard.evaluate_argv("gemma4", "batched", 0, "real", Path("o.json"), a1)
    assert "--fc_sequential" not in real and "--vision_mismatch_seed" not in real


LABEL_LINE = re.compile(r"^\d+\) (.+)$", re.M)


class _Model:
    def eval(self):
        return self

    def to(self, _device):
        return self


def _stub_generate(**kw):
    """Depends on the prompt, the frames (so the partner's video matters) and torch's RNG."""
    import torch

    options = LABEL_LINE.findall(kw["prompt"])
    base = int(hashlib.sha256((kw["prompt"] + "|".join(map(str, kw["images"]))).encode()).hexdigest()[:8], 16)
    pick = (base + int(torch.randint(0, 7, (1,)).item())) % (len(options) + 1)
    return "EMOTION: unsure" if pick == len(options) else f"EMOTION: {options[pick]}\nREASONING: stub"


@pytest.fixture()
def stub_evaluate(monkeypatch):
    ev = shard.import_evaluate(None)
    monkeypatch.setattr(ev, "AutoProcessor", types.SimpleNamespace(from_pretrained=lambda *a, **k: None))
    monkeypatch.setattr(ev, "load_hf_model_for_key", lambda **k: _Model())
    monkeypatch.setattr(ev, "resolve_model_path", lambda key: Path("/nonexistent/model"))
    monkeypatch.setattr(ev, "resolve_trial_media", lambda trial, **k: (Path(trial["stimulus_path"]), None, None))
    monkeypatch.setattr(ev, "load_stimulus_as_images", lambda path, fps, max_frames: ([Path(path).name], [0]))
    monkeypatch.setattr(ev, "prepare_images_for_model", lambda key, images, enforce_multi_frame: (images, {"n_frames_used": len(images), "multi_frame_strategy": "stub"}))
    monkeypatch.setattr(ev, "generate_model_response", _stub_generate)
    monkeypatch.setattr(ev, "generate_model_response_batch", lambda num_return_sequences, **kw: [_stub_generate(**kw) for _ in range(num_return_sequences)])
    monkeypatch.setattr(ev, "collect_run_metadata", lambda: {})
    monkeypatch.chdir(ROOT)
    return ev


def _args(cmd, out, condition, sampling="sequential", **extra):
    base = dict(cmd=cmd, key="qwen3vl", sampling=sampling, rotation=0, condition=condition, models_dir=None,
                max_trials=extra.pop("max_trials", 6), human_options="data/eu_emotions_human_entropy.json",
                output=out, no_design_gate=False)
    base.update(extra)
    return Namespace(**base)


def _sharded(ev, tmp_path, condition, n_shards, max_trials, sampling="sequential"):
    files = []
    for k in range(n_shards):
        out = tmp_path / f"{condition}_{sampling}" / f"shard{k}of{n_shards}" / "eval.json"
        shard.cmd_run(_args("run", out, condition, sampling, max_trials=max_trials, shard=k, n_shards=n_shards, label="test"), ev)
        files.append(out)
    merged = tmp_path / f"merged_{condition}_{sampling}_{n_shards}.json"
    shard.merge(files, merged)
    return files, merged


@pytest.mark.parametrize("condition,sampling,n_shards,max_trials", [
    ("mismatch", "sequential", 2, 6),
    ("real", "sequential", 2, 6),
    ("mismatch", "batched", 3, 7),
])
def test_a1_equivalence_on_the_frozen_code_path(tmp_path, stub_evaluate, condition, sampling, n_shards, max_trials):
    """Equivalence test 1, locally: sharded and merged records equal the unsharded frozen run."""
    ev = stub_evaluate
    frozen = tmp_path / "frozen" / "eval.json"
    frozen.parent.mkdir()
    shard.cmd_frozen(_args("frozen", frozen, condition, sampling, max_trials=max_trials), ev)
    files, merged = _sharded(ev, tmp_path, condition, n_shards, max_trials, sampling)
    report = shard.compare(frozen, merged)
    assert report["identical"], report
    trials = json.loads(merged.read_text())["trials"]
    assert len(trials) == max_trials
    if condition == "mismatch":
        loaded = shard.loaded_trials(ev, json.loads(a8_design.block_bytes())["a1"], max_trials)
        partners = shard.partner_map(loaded, ROOT / "data/eu_emotions_human_entropy.json", 42)
        assert all(t["vision_source_trial_id"] == partners[t["trial_id"]] for t in trials)
        assert any(t["vision_source_trial_id"] != t["trial_id"] for t in trials)
    # The test can fail: a different seed changes the records.
    assert any(t["stage2"]["fc_predictions"] for t in trials)


def test_naive_manifest_split_would_change_partners(stub_evaluate):
    a1 = json.loads(a8_design.block_bytes())["a1"]
    loaded = shard.loaded_trials(stub_evaluate, a1, None)
    lookup = ROOT / "data/eu_emotions_human_entropy.json"
    full = shard.partner_map(loaded, lookup, 42)
    half = shard.partner_map(loaded[::2], lookup, 42)
    assert any(full[k] != v for k, v in half.items())


def test_merge_refuses_gaps_duplicates_mixed_runs_and_edits(tmp_path, stub_evaluate):
    files, merged = _sharded(stub_evaluate, tmp_path, "real", 2, 4)
    with pytest.raises(SystemExit, match="exists"):
        shard.merge(files, merged)
    with pytest.raises(SystemExit, match="complete set"):
        shard.merge(files[:1], tmp_path / "gap.json")
    with pytest.raises(SystemExit, match="complete set|two shards"):
        shard.merge([files[0], files[0]], tmp_path / "twice.json")
    side_path = files[1].parent / shard.SIDECAR_NAME
    side = json.loads(side_path.read_text())
    side_path.write_text(json.dumps({**side, "sampling": "batched"}))
    with pytest.raises(SystemExit, match="different runs"):
        shard.merge(files, tmp_path / "mixed.json")
    side_path.write_text(json.dumps(side))
    obj = json.loads(files[1].read_text())
    obj["trials"][0]["stage2"]["prediction"] = "edited"
    files[1].write_text(json.dumps(obj))
    with pytest.raises(SystemExit, match="changed"):
        shard.merge(files, tmp_path / "edited.json")


def test_a_shard_is_never_overwritten(tmp_path, stub_evaluate):
    out = tmp_path / "s" / "eval.json"
    out.parent.mkdir()
    out.write_text("{}")
    with pytest.raises(SystemExit, match="never overwritten"):
        shard.cmd_run(_args("run", out, "real", max_trials=4, shard=0, n_shards=2, label="t"), stub_evaluate)


def test_evaluate_is_hash_checked(tmp_path, monkeypatch):
    copy = tmp_path / "evaluate.py"
    copy.write_bytes((ROOT / "scripts/evaluate.py").read_bytes() + b"\n")
    with pytest.raises(SystemExit, match="not the frozen"):
        shard.check_evaluate(copy)


@pytest.mark.skipif(
    os.environ.get("A8_A1_EQUIV_GPU") != "1",
    reason="GPU equivalence on Qwen3-VL-8B: set A8_A1_EQUIV_GPU=1 and A8_EQUIV_MODELS_DIR on a GPU node",
)
def test_a1_equivalence_on_qwen3vl_8b_gpu(tmp_path):
    """Equivalence test 1 on the cluster: the real 8B model, 6 items, both conditions."""
    import subprocess

    import torch

    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    models_dir = os.environ["A8_EQUIV_MODELS_DIR"]
    out_root = Path(os.environ.get("A8_EQUIV_OUT", ROOT / "results/a8/equivalence"))
    script = str(ROOT / "research/scripts/a8_rotation_shard.py")
    gate = ["--no_design_gate"] if os.environ.get("A8_EQUIV_NO_GATE") == "1" else []
    for condition in ("real", "mismatch"):
        base = ["--key", "qwen3vl", "--sampling", "sequential", "--rotation", "0", "--condition", condition,
                "--models_dir", models_dir, "--max_trials", "6", *gate]
        frozen = out_root / condition / "frozen" / "eval.json"
        frozen.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, script, "frozen", *base, "--output", str(frozen)], check=True, cwd=ROOT)
        shards = []
        for k in range(2):
            out = out_root / condition / f"shard{k}of2" / "eval.json"
            subprocess.run([sys.executable, script, "run", *base, "--shard", str(k), "--n_shards", "2",
                            "--label", "equivalence_qwen3vl_8b", "--output", str(out)], check=True, cwd=ROOT)
            shards.append(str(out))
        merged = out_root / condition / "merged.json"
        subprocess.run([sys.executable, script, "merge", "--shards", *shards, "--output", str(merged)], check=True, cwd=ROOT)
        report = shard.compare(frozen, merged)
        (out_root / condition / "equivalence_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        assert report["identical"], report


# A3 summary -----------------------------------------------------------------


import a8_face_input_validity_summary as a3w  # noqa: E402


def synthetic_a3_cells(fiv, path, seed, effect):
    """Cells in the frozen schema for all 243 items, scored by the frozen correct_from_scores."""
    human = json.loads((ROOT / "data/eu_emotions_human_entropy.json").read_text())["trials"]
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
    fiv.summarise(Namespace(lookup=ROOT / "data/eu_emotions_human_entropy.json", output_dir=folder))
    frozen = (folder / "face_input_validity_summary.json").read_bytes()
    wrapped = a3w.summary_bytes([(m, folder / f"cells_{m}.jsonl") for m in fiv.MODELS])
    assert wrapped == frozen


def test_a3_wrapper_holm_is_over_the_a8_family_only(a3_cells):
    fiv, folder = a3_cells
    one = json.loads(a3w.summary_bytes([("qwen3vl_32b", folder / "cells_qwen3vl.jsonl")]))
    entry = one["qwen3vl_32b"]
    assert entry["p_delta_holm"] == entry["p_delta_actor_sign_flip"]
    three = json.loads(a3w.summary_bytes([(m, folder / f"cells_{m}.jsonl") for m in fiv.MODELS]))
    assert three["qwen3vl"]["p_delta_holm"] >= entry["p_delta_holm"]


def test_a3_wrapper_refuses_names_outside_the_family_and_missing_files(tmp_path, a3_cells):
    fiv, folder = a3_cells
    with pytest.raises(SystemExit, match="A8 family"):
        a3w.main(["--cells", f"qwen3vl={folder / 'cells_qwen3vl.jsonl'}", "--output", str(tmp_path / "o.json")])
    with pytest.raises(SystemExit, match="does not exist"):
        a3w.main(["--allow_any_names", "--cells", f"x={tmp_path / 'none.jsonl'}", "--output", str(tmp_path / "o.json")])


FROZEN_A3 = [ROOT / "results/face_input_validity" / f"cells_{m}.jsonl" for m in ("qwen3vl", "gemma4", "molmo2")]


@pytest.mark.skipif(not all(p.exists() for p in FROZEN_A3), reason="frozen A3 cell files are on the cluster only")
def test_a3_wrapper_reproduces_the_frozen_summary_hash():
    """Equivalence test 2, on the frozen files: the pinned summary, byte for byte."""
    data = a3w.summary_bytes(list(zip(("qwen3vl", "gemma4", "molmo2"), FROZEN_A3)))
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

    human = json.loads((ROOT / "data/eu_emotions_human_entropy.json").read_text())["trials"]
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
    for k in range(3):
        cell = tmp_path / f"cells_{k}.jsonl"
        cell.write_text("".join(json.dumps(r) + "\n" for r in rows if trials.index(r["trial_id"]) % 3 == k))
        report = tmp_path / f"run_{k}.json"
        report.write_text(json.dumps({"model": "gemma4", "smoke": False, "design_sha256": design, "shard": k, "n_shards": 3, "delivery": [delivery], "baselines": baselines}))
        cells.append(str(cell))
        reports.append(str(report))
    geometry = tmp_path / "geometry.json"
    geometry.write_text(json.dumps({"items": {t: {"face_box": [0, 0, 400, 400], "face_width_px": 200.0, "width": 800, "height": 400} for t in trials}}))
    repro = tmp_path / "repro"
    repro.mkdir()
    for name in ("vision_gates_gemma4.json", "input_delivery_gemma4.json"):
        shutil.copy(ROOT / "research/reproduction" / name, repro / name)
    (repro / "rotation_summary.json").write_text(json.dumps({"gemma4": {"vision_dependent": True}}))
    return cells, reports, geometry, repro, run


def test_a2_loader_swap_gives_the_frozen_output_byte_for_byte(tmp_path, monkeypatch):
    """Equivalence test 3: the frozen A3 file through the A8 loader, against the frozen loader."""
    an = a2w.load_frozen()
    cells, reports, geometry, repro, run = a2_inputs(tmp_path)
    monkeypatch.setattr(run, "check_geometry", lambda path: None)
    frozen_a3 = ROOT / "research/reproduction/face_input_validity_summary.json"
    shutil.copy(frozen_a3, repro / frozen_a3.name)

    def argv(out):
        return ["--model", "gemma4", "--cells", *cells, "--run_report", *reports, "--geometry", str(geometry),
                "--reproduction", str(repro), "--output", str(out), "--a2_frozen"]

    a2w.run_frozen_main(an, argv(tmp_path / "frozen.json"), None)
    sha = a8_design.load_design()["frozen_a3_summary_sha256"]
    loader = a2w.a3_loader(frozen_a3, sha, lambda s: s)
    a2w.run_frozen_main(an, argv(tmp_path / "wrapped.json"), loader)
    frozen, wrapped = (tmp_path / "frozen.json").read_bytes(), (tmp_path / "wrapped.json").read_bytes()
    assert frozen == wrapped
    result = json.loads(frozen)
    assert result["single_face"]["gates"]["rq0"] is True  # the gates path is exercised, not short-circuited
    assert an.a3_summary.__module__ == an.__name__  # restored after the call


def test_the_frozen_loader_cannot_see_an_a8_summary_and_the_a8_loader_can(tmp_path):
    an = a2w.load_frozen()
    summary = {"qwen3vl_32b": {"vision_dependent": True}}
    path = tmp_path / "face_input_validity_summary.json"
    path.write_text(json.dumps(summary))
    assert an.a3_summary(tmp_path, tmp_path) == {}
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    loader = a2w.a3_loader(path, sha, lambda s: {"qwen3vl": s["qwen3vl_32b"]})
    assert loader(tmp_path, tmp_path) == {"qwen3vl": {"vision_dependent": True}}
    with pytest.raises(SystemExit, match="not the given"):
        a2w.a3_loader(path, "0" * 64, lambda s: s)


def test_a1_view_is_written_once_under_the_code_key(tmp_path):
    a1 = tmp_path / "rotation_summary_a8.json"
    a1.write_text(json.dumps({"qwen3vl_32b": {"vision_dependent": False}}))
    folder = tmp_path / "a8" / "qwen3vl_32b"
    path = a2w.write_a1_view(a1, "qwen3vl_32b", "qwen3vl", folder)
    assert json.loads(path.read_text()) == {"qwen3vl": {"vision_dependent": False}}
    a2w.write_a1_view(a1, "qwen3vl_32b", "qwen3vl", folder)  # same content: allowed
    a1.write_text(json.dumps({"qwen3vl_32b": {"vision_dependent": True}}))
    with pytest.raises(SystemExit, match="other content"):
        a2w.write_a1_view(a1, "qwen3vl_32b", "qwen3vl", folder)


# A1 analysis ----------------------------------------------------------------


import a8_analyse_rotation as a1w  # noqa: E402


def test_a1_analysis_points_the_frozen_module_at_a8_files_and_family():
    runner, mod, _ = a1w.configure(["qwen3vl_32b"])
    assert mod.MODELS == ("qwen3vl_32b",)
    assert mod.EVAL.format(m="qwen3vl_32b", r=2, c="real") == (
        "results/a8/qwen3vl_32b/rotation/eval_v2_eu_emotions_qwen3vl_32b_video_only_seed42_rot2_real.json")
    assert runner.sha256(runner.TARGET) == a8_design.load_design()["frozen_code_sha256"]["research/scripts/analyse_rotation.py"]
