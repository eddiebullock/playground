"""Dry run of research/scripts/post_a1.sh on synthetic A1 rotation files.

The synthetic files carry the schema analyse_rotation.py and the model route read
(trials[].trial_id, vision_source_trial_id, stage2.options in rotation order,
stage2.correct, stage2.response_distribution, stage2.fc_n_samples). Every file is marked
synthetic and lives under pytest's tmp_path, never in results/. No real eval is read.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("PYTHON", "/opt/anaconda3/envs/mr_eu_open_llm/bin/python")
SCRIPT = REPO / "research/scripts/post_a1.sh"
HUMAN = REPO / "data/eu_emotions_human_entropy.json"
AU_CSV = REPO / "research/reproduction/eu_au_measures.csv"

sys.path.insert(0, str(REPO / "src"))
from paper_v2.option_score import rotate_options  # noqa: E402

_spec = importlib.util.spec_from_file_location("post_a1_helpers", REPO / "research/scripts/post_a1_helpers.py")
helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)

pytestmark = pytest.mark.skipif(not (HUMAN.is_file() and AU_CSV.is_file()),
                                reason="needs the human lookup and eu_au_measures.csv")


def build_synthetic(root: Path, skip=()):
    """30 obviously synthetic eval files at root/<model>/<eval name>."""
    trials = json.loads(HUMAN.read_text())["trials"]
    ids = sorted(trials)
    rng = np.random.default_rng(20261108)
    written = []
    for rel in helpers.expected():
        if rel in skip:
            continue
        model = rel.split("/")[3]
        rotation = int(rel.split("_rot")[1][0])
        condition = "real" if rel.endswith("_real.json") else "mismatch"
        out = []
        for k, trial_id in enumerate(ids):
            rec = trials[trial_id]
            options = rotate_options(rec["human_options"], rotation)
            weights = rng.dirichlet(np.ones(len(options)))
            if condition == "real":
                weights[options.index(rec["human_target_label"])] += 1.0
            weights /= weights.sum()
            dist = {o: round(float(w), 4) for o, w in zip(options, weights)}
            dist["__unparsed__"] = 0.0
            pred = max(options, key=dist.get)
            source = trial_id if condition == "real" else ids[(k + 1) % len(ids)]
            out.append({
                "trial_id": trial_id, "label": rec["human_target_label"],
                "vision_source_trial_id": source, "error": None,
                "stage2": {"options": options, "prediction": pred,
                           "correct": pred == rec["human_target_label"],
                           "response_distribution": dist, "fc_n_samples": 20,
                           "raw_model_output": "SYNTHETIC TEST DATA, NOT A MODEL OUTPUT"},
            })
        payload = {"SYNTHETIC": "test fixture for post_a1.sh; not a model result",
                   "model": f"SYNTHETIC_{model}", "condition": condition,
                   "rotation": rotation, "trials": out}
        path = root / model / Path(rel).name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
        written.append(path)
    return written


def run_dry(src: Path, work: Path, extra=()):
    env = dict(os.environ)
    return subprocess.run(["bash", str(SCRIPT), "--dry-run", str(src), "--workdir", str(work),
                           "--n-perm", "19", *extra], capture_output=True, text=True, env=env,
                          cwd=str(REPO), timeout=1800)


def test_dry_run_end_to_end(tmp_path):
    src, work = tmp_path / "synthetic_a1", tmp_path / "work"
    build_synthetic(src)
    repo_manifest = REPO / "research/reproduction/a1_pull_manifest.json"
    existed = repo_manifest.exists()
    proc = run_dry(src, work)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    manifest = json.loads((work / "a1_pull_manifest.json").read_text())
    assert manifest["mode"] == "dry-run" and manifest["synthetic"] is True
    assert manifest["status"] == "complete" and manifest["exit_code"] == 0
    assert len(manifest["git_commit"]) == 40
    files = manifest["files"]
    assert len(files) == 30 and all(f["match"] for f in files)
    assert sorted(f["path"] for f in files) == sorted(helpers.expected())
    for f in files:
        assert f["sha256_remote"] == helpers.sha256(src / f["path"].split("/", 3)[3])
    names = [c["name"] for c in manifest["commands"]]
    assert names == ["analyse_rotation", "model_route_qwen3vl", "model_route_gemma4", "model_route_molmo2"]
    assert all(c["exit_code"] == 0 for c in manifest["commands"])
    route = manifest["commands"][1]["cmd"]
    assert route[route.index("--au-csv") + 1] == "research/reproduction/eu_au_measures.csv"
    assert route[route.index("--model-evals") + 1] == (
        "results/baseline/eu_emotions/qwen3vl/eval_v2_eu_emotions_qwen3vl_video_only_seed42_rot*_real.json")
    assert manifest["script_sha256"]["research/scripts/analyse_rotation.py"] == \
        "98890c5bff65f13abeb0549b61fe414dc9fa30082b84bcb564671683ec12e24f"
    assert manifest["commands"][0]["cmd"][1] == "research/scripts/analyse_rotation_run.py"
    assert manifest["script_sha256"]["research/scripts/analyse_rotation_run.py"] == \
        helpers.sha256(REPO / "research/scripts/analyse_rotation_run.py")
    assert manifest["slurm_check_skipped_reason"] is None
    assert len(manifest["outputs_sha256"]) == 5
    for out, digest in manifest["outputs_sha256"].items():
        assert digest is not None and out.startswith(str(work.resolve()))

    summary = json.loads((work / "tree/research/reproduction/rotation_summary.json").read_text())
    assert set(summary) == {"qwen3vl", "gemma4", "molmo2"}
    assert summary["gemma4"]["n_items"] == 243
    assert summary["gemma4"]["accuracy_real"] > summary["gemma4"]["accuracy_mismatch"]
    route_out = json.loads((work / "tree/research/reproduction/same_clip_family_model_molmo2.json").read_text())
    assert route_out["coverage"]["n_dropped"] == 0
    sidecar = json.loads((work / "tree/research/reproduction/rotation_errored_trials.json").read_text())
    assert sidecar["total_errored"] == 0
    # With no errored trials the wrapper output is byte-identical to the frozen script.
    tree = work / "tree"
    direct = subprocess.run([PYTHON, "research/scripts/analyse_rotation.py", "--output", str(work / "direct.json")],
                            cwd=str(tree), capture_output=True)
    assert direct.returncode == 0, direct.stderr
    assert (work / "direct.json").read_bytes() == (tree / "research/reproduction/rotation_summary.json").read_bytes()
    wrapped_stdout = Path(manifest["commands"][0]["stdout_log"]).read_bytes()
    assert direct.stdout == wrapped_stdout
    # Nothing went to the repo.
    assert repo_manifest.exists() == existed


def test_dry_run_stops_on_missing_file(tmp_path):
    src, work = tmp_path / "synthetic_a1", tmp_path / "work"
    missing = helpers.expected()[7]
    build_synthetic(src, skip={missing})
    proc = run_dry(src, work)
    assert proc.returncode != 0
    assert "STOP" in proc.stderr
    manifest = json.loads((work / "a1_pull_manifest.json").read_text())
    assert manifest["status"] == "stopped before analysis"
    assert manifest["commands"] == []


def test_dry_run_refuses_dir_inside_repo(tmp_path):
    proc = subprocess.run(["bash", str(SCRIPT), "--dry-run", str(REPO / "tests"),
                           "--workdir", str(tmp_path / "w")], capture_output=True, text=True)
    assert proc.returncode != 0 and "outside the repo" in proc.stderr


def test_real_mode_rejects_dry_run_options():
    proc = subprocess.run(["bash", str(SCRIPT), "--n-perm", "5"], capture_output=True, text=True)
    assert proc.returncode != 0 and "dry runs only" in proc.stderr


def test_sacct_check():
    done = "\n".join(f"{a}_{t}|COMPLETED" for a in helpers.ARRAYS for t in range(5))
    assert helpers.check_sacct(done)[0]
    assert not helpers.check_sacct(done.replace("39982138_3|COMPLETED", "39982138_3|FAILED"))[0]
    assert not helpers.check_sacct(done.replace("39982138_4|COMPLETED", "39982138_[4]|PENDING"))[0]
    assert not helpers.check_sacct("39982137_[0-4]|PENDING\n39982138_[0-4]|PENDING")[0]
    assert not helpers.check_sacct("")[0]


def test_verify_detects_hash_mismatch(tmp_path):
    rels = helpers.expected()
    for rel in rels:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("SYNTHETIC")
    digest = helpers.sha256(tmp_path / rels[0])
    remote = tmp_path / "remote.txt"
    lines = [f"{digest}  rds/hpc-work/study3/{r}" for r in rels]
    remote.write_text("\n".join(lines) + "\n")
    out = tmp_path / "files.json"
    assert helpers.main(["verify", str(remote), str(tmp_path), "--json-out", str(out)]) == 0
    (tmp_path / rels[3]).write_text("SYNTHETIC, changed")
    assert helpers.main(["verify", str(remote), str(tmp_path), "--json-out", str(out)]) == 1


def test_errored_trial_scores_wrong_and_zero_p_target(tmp_path):
    from scipy.stats import spearmanr

    src, work = tmp_path / "synthetic_a1", tmp_path / "work"
    build_synthetic(src)
    trials = json.loads(HUMAN.read_text())["trials"]
    ids = sorted(trials)
    errored = ids[10]
    real = []
    for r in range(5):
        path = src / "gemma4" / f"eval_v2_eu_emotions_gemma4_video_only_seed42_rot{r}_real.json"
        payload = json.loads(path.read_text())
        for t in payload["trials"]:
            if t["trial_id"] == errored:
                t["error"] = "SYNTHETIC RuntimeError: test fixture"
                t["stage2"] = {"options": t["stage2"]["options"], "prediction": None, "correct": None}
        path.write_text(json.dumps(payload))
        real.append({t["trial_id"]: t["stage2"] for t in payload["trials"]})

    proc = run_dry(src, work, extra=["--skip-slurm-check", "SYNTHETIC: task rerun under new job id"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    manifest = json.loads((work / "a1_pull_manifest.json").read_text())
    assert manifest["slurm_check_skipped_reason"] == "SYNTHETIC: task rerun under new job id"
    assert manifest["slurm"]["skipped"] is True

    sidecar = json.loads((work / "tree/research/reproduction/rotation_errored_trials.json").read_text())
    assert sidecar["total_errored"] == 5
    for r in range(5):
        assert sidecar["counts"]["gemma4"]["real"][str(r)] == {"n_errored": 1, "trial_ids": [errored]}
        assert sidecar["counts"]["gemma4"]["mismatch"][str(r)]["n_errored"] == 0
        assert sidecar["counts"]["qwen3vl"]["real"][str(r)]["n_errored"] == 0

    summary = json.loads((work / "tree/research/reproduction/rotation_summary.json").read_text())["gemma4"]
    acc = np.array([np.mean([bool(run[k]["correct"]) for run in real]) for k in ids])
    assert acc[ids.index(errored)] == 0.0
    assert summary["accuracy_real"] == pytest.approx(float(acc.mean()), abs=1e-12)
    pt = np.array([np.mean([(run[k].get("response_distribution") or {}).get(trials[k]["human_target_label"], 0)
                            for run in real]) for k in ids])
    assert pt[ids.index(errored)] == 0.0
    ht = np.array([trials[k]["p_target"] for k in ids])
    assert summary["secondary_rho_ptarget_human"] == pytest.approx(float(spearmanr(pt, ht)[0]), abs=1e-12)

    # The frozen script alone crashes on this input; the wrapper is what makes it run.
    direct = subprocess.run([PYTHON, "research/scripts/analyse_rotation.py", "--output", str(work / "direct.json")],
                            cwd=str(work / "tree"), capture_output=True, text=True)
    assert direct.returncode != 0 and "response_distribution" in direct.stderr


def test_wrapper_refuses_changed_frozen_file(tmp_path):
    spec = importlib.util.spec_from_file_location("analyse_rotation_run", REPO / "research/scripts/analyse_rotation_run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    copy = tmp_path / "analyse_rotation.py"
    copy.write_bytes((REPO / "research/scripts/analyse_rotation.py").read_bytes() + b"# changed\n")
    with pytest.raises(SystemExit, match="not the frozen"):
        run.load_frozen(copy)


def test_skip_slurm_check_needs_reason():
    proc = subprocess.run(["bash", str(SCRIPT), "--skip-slurm-check"], capture_output=True, text=True)
    assert proc.returncode != 0 and "needs a REASON" in proc.stderr
