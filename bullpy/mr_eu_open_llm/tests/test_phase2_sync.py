"""Offline tests for research/scripts/phase2_sync_check.sh and its helper (ROADMAP 2.1).

Nothing here contacts the cluster. The end-to-end tests run the shell script against a
synthetic repo in tmp_path, with `ssh` and `squeue` replaced by stubs on PATH: the stub
ssh runs the remote command locally inside a fake home under tmp_path, so rsync copies
between two temp directories.
"""

import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "research/scripts/phase2_sync_check.sh"
HELPER = REPO / "research/scripts/phase2_sync_helpers.py"
PYTHON = os.environ.get("PYTHON", "/opt/anaconda3/envs/mr_eu_open_llm/bin/python")
if not Path(PYTHON).is_file():
    PYTHON = sys.executable

_spec = importlib.util.spec_from_file_location("phase2_sync_helpers", HELPER)
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)

H = {k: (c * 64) for k, c in zip("abcdef", "abcdef")}


# ------------------------------------------------------------------ the real preregs

@pytest.fixture(scope="module")
def real():
    return h.collect(REPO)


def test_real_preregs_parse_cleanly(real):
    assert real["errors"] == []
    assert real["conflicts"] == []
    am = real["amendments"]
    assert set(am) == {"A2", "A4", "A5", "A6", "A7"}
    assert [k for k, v in am.items() if not v["frozen"]] == ["A6"]
    uniq = {k: {p["path"] for p in v["pins"]} for k, v in am.items()}
    assert {k: len(v) for k, v in uniq.items()} == {"A2": 6, "A4": 22, "A5": 35, "A6": 51, "A7": 22}
    assert len(real["files"]) == 112


def test_real_known_pins(real):
    by = {f["path"]: f for f in real["files"]}
    assert by["scripts/stimulus_design.py"]["sha256"].startswith("89d256c1")
    assert set(by["scripts/analyse_stimulus_interventions.py"]["amendments"]) == {"A2", "A4", "A5", "A6", "A7"}
    assert by["slurm_jobs/study3_stimulus_interventions.sh"]["amendments"] == ["A2"]
    assert by["research/reproduction/a4_recovery.json"]["amendments"] == ["A4"]
    assert by["tests/test_a5.py"]["amendments"] == ["A5"]
    assert by["scripts/a7_toy.py"]["amendments"] == ["A7"]
    assert by["research/reproduction/crema_calibration_clips.csv"]["amendments"] == ["A7"]
    # A6-only pins are not frozen; shared pins are frozen through the other amendment
    assert by["scripts/frame_sampling.py"]["frozen"] is False
    assert by["scripts/a6_design.py"]["frozen"] is False
    assert by["src/paper_v2/schema.py"]["frozen"] is True
    # A2 lists human_response_predictor.py as code but pins no hash for it
    assert "scripts/human_response_predictor.py" not in by


def test_real_unparsed_items_are_reported(real):
    vals = {(u["amendment"], u["value"][:8]) for u in real["unparsed"]}
    assert ("A2", "0ae22118") in vals          # design_sha256, not a file
    assert ("A4", "2ef1161e") in vals          # manifest hash, no path written
    assert ("A7", "9e40cfcf") in vals
    pending = [u for u in real["unparsed"] if u["value"] == "PENDING"]
    assert pending and all(u["amendment"] == "A6" for u in pending)
    assert [c["path"] for c in real["cluster_only"]] == ["results/a4/manifest.csv"]


def test_real_working_tree_matches_frozen_pins(real):
    rows = h.local_check(real, REPO)
    bad = [r["path"] for r in rows if r["frozen"] and r["local_status"] != "match"]
    assert bad == []


def test_only_data_pins_are_outside_the_sync(real):
    excl = sorted(f["path"] for f in real["files"] if h.excluded(f["path"]))
    assert excl == ["data/eu_emotions_full_manifest.json", "data/eu_emotions_human_entropy.json"]


# ------------------------------------------------------------------ synthetic parsing

SYN = """# Amendment X

**Status: FROZEN.** Frozen `design_sha256`: `{a}`.

Frozen code, sha256: `scripts/one.py` `{b}`; `research/scripts/two.py` `{c}`.

## Design block

```json
{{
  "amendment": "X",
  "a2_design_sha256": "{a}",
  "code": {{"scripts/one.py": "{b}", "config.py": "{d}"}}
}}
```

## Method

`scripts/ignored.py` `{e}` is outside any freeze section.

## Freeze (2026-01-01)

- `slurm_jobs/job.sh`: `{d}`

```json
{{"x": 1}}
```

```
not/a/text/pin.py {e}
```

geometry: PENDING

## Code hashes

```text
scripts/three.py {f}
```

## Later
"""


def syn_text(**over):
    vals = dict(H)
    vals.update(over)
    return SYN.format(**vals)


def test_synthetic_forms():
    pins, unparsed = h.parse_amendment(syn_text(), "X", True)
    got = {(p["path"], p["sha256"][0], p["section"].split(" ")[0]) for p in pins}
    assert ("scripts/one.py", "b", "preamble") in got
    assert ("research/scripts/two.py", "c", "preamble") in got
    assert ("scripts/one.py", "b", "Design") in got
    assert ("config.py", "d", "Design") in got
    assert ("slurm_jobs/job.sh", "d", "Freeze") in got
    assert ("scripts/three.py", "f", "Code") in got
    # the bare ``` fence after a json fence in the Freeze section is a text fence
    assert ("not/a/text/pin.py", "e", "Freeze") in got
    assert not any(p["path"] == "scripts/ignored.py" for p in pins)
    kinds = {(u["value"][:1], u["section"].split(" ")[0]) for u in unparsed}
    assert ("a", "preamble") in kinds and ("a", "Design") in kinds
    assert ("P", "Freeze") in kinds


def write_repo(root: Path, files: dict, texts: dict):
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
    for name, text in texts.items():
        p = root / f"research/PREREG_{name}_DRAFT.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


def sha(b: bytes) -> str:
    return h.hashlib.sha256(b).hexdigest()


def make_fake_repo(root: Path):
    """Five amendments, each pinning real files in a synthetic repo."""
    files = {
        "scripts/frozen_a.py": b"print('a')\n",
        "scripts/frozen_b.py": b"print('b')\n",
        "scripts/a6_only.py": b"print('a6')\n",
        "slurm_jobs/job.sh": b"#!/bin/bash\n",
        "research/reproduction/out.json": b"{}\n",
        "data/pinned_manifest.json": b"[1, 2]\n",
        "data/big/clip.bin": b"\x00" * 10,
        "results/secret.json": b"{}\n",
        "stimuli/face.png": b"PNG",
        "scripts/__pycache__/x.pyc": b"\x00",
        "scripts/unpinned.py": b"x = 1\n",
    }
    s = {k: sha(v) for k, v in files.items()}
    texts = {
        "A2": f"# A2\n\nFrozen code, sha256: `scripts/frozen_a.py` `{s['scripts/frozen_a.py']}`; "
              f"`slurm_jobs/job.sh` `{s['slurm_jobs/job.sh']}`.\n\n## Deviations\n",
        "A4": f"# A4\n\nFrozen code, sha256: `scripts/frozen_b.py` `{s['scripts/frozen_b.py']}`.\n\n## Q\n",
        "A5": f"# A5\n\n## Freeze (x)\n\nTo be frozen inputs: `data/pinned_manifest.json` "
              f"`{s['data/pinned_manifest.json']}`; `scripts/frozen_a.py` `{s['scripts/frozen_a.py']}`.\n",
        "A6": f"# A6\n\n## Freeze (prepared, not tagged)\n\n`scripts/a6_only.py` `{s['scripts/a6_only.py']}`\n",
        "A7": f"# A7\n\n## Code hashes\n\n```text\nresearch/reproduction/out.json "
              f"{s['research/reproduction/out.json']}\n```\n",
    }
    write_repo(root, files, texts)
    (root / "research/scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(SCRIPT, root / "research/scripts/phase2_sync_check.sh")
    shutil.copy2(HELPER, root / "research/scripts/phase2_sync_helpers.py")
    return files


def run_helper(repo: Path, *args, stdin=""):
    env = dict(os.environ, PHASE2_SKIP_CLUSTER_ONLY_PINS="1")
    return subprocess.run([PYTHON, str(HELPER), "--repo", str(repo), *args], input=stdin,
                          capture_output=True, text=True, env=env)


def test_local_check_cli(tmp_path):
    make_fake_repo(tmp_path)
    r = run_helper(tmp_path, "local-check")
    assert r.returncode == 0, r.stderr
    assert "data/pinned_manifest.json" in r.stderr  # pinned but not synced
    # A6-only change: warning, not a stop, unless --strict-a6
    (tmp_path / "scripts/a6_only.py").write_text("changed\n")
    r = run_helper(tmp_path, "local-check")
    assert r.returncode == 0 and "A6-only" in r.stderr
    assert run_helper(tmp_path, "local-check", "--strict-a6").returncode == 1
    # frozen change: stop
    (tmp_path / "scripts/frozen_b.py").write_text("changed\n")
    r = run_helper(tmp_path, "local-check")
    assert r.returncode == 1 and "STOP" in r.stderr
    (tmp_path / "scripts/frozen_b.py").unlink()
    r = run_helper(tmp_path, "local-check")
    assert r.returncode == 1 and "MISSING" in r.stdout


def test_conflicting_frozen_pins_stop(tmp_path):
    make_fake_repo(tmp_path)
    p = tmp_path / "research/PREREG_A4_DRAFT.md"
    p.write_text(p.read_text() + f"\n## Freeze\n\n`scripts/frozen_a.py` `{H['f']}`\n")
    info = h.collect(tmp_path, cluster_only_pins=[])
    assert info["conflicts"] and info["conflicts"][0]["path"] == "scripts/frozen_a.py"
    r = run_helper(tmp_path, "local-check")
    assert r.returncode == 2


def test_missing_prereg_is_an_error(tmp_path):
    make_fake_repo(tmp_path)
    (tmp_path / "research/PREREG_A7_DRAFT.md").unlink()
    assert run_helper(tmp_path, "pins").returncode == 1
    assert run_helper(tmp_path, "local-check").returncode == 2


def test_cluster_only_pin_requires_hash_in_text(tmp_path):
    make_fake_repo(tmp_path)
    info = h.collect(tmp_path)
    assert any("cluster-only pin" in e for e in info["errors"])


# ------------------------------------------------------------------ queue gate

def test_jobs_gate():
    ok, _ = h.jobs_gate("")
    assert ok
    ok, msgs = h.jobs_gate("39982137_[0-4]|s3_rotation|PENDING\n")
    assert not ok and "s3_rotation" in msgs[0]
    ok, _ = h.jobs_gate("39982138_2|s3_rotation|RUNNING\n", other_jobs_ok="reviewed")
    assert not ok  # no override for the rotation jobs
    ok, _ = h.jobs_gate("4100|cas_chain_7|RUNNING\n")
    assert ok
    ok, msgs = h.jobs_gate("4101|s3_a6|PENDING\n")
    assert not ok and "--other-jobs-ok" in msgs[0]
    ok, _ = h.jobs_gate("4101|s3_a6|PENDING\n", other_jobs_ok="reads nothing synced")
    assert ok
    ok, _ = h.jobs_gate("4102|s3_rotation|COMPLETED\n")
    assert ok
    ok, _ = h.jobs_gate("garbage line\n")
    assert not ok


# ------------------------------------------------------------------ annotate, compare

def test_annotate_flags_pins(tmp_path):
    make_fake_repo(tmp_path)
    info = h.collect(tmp_path, cluster_only_pins=[])
    out, touched = h.annotate("<f.st...... scripts/frozen_a.py\n<f+++++++++ scripts/unpinned.py\n", info)
    assert touched == ["scripts/frozen_a.py"]
    assert "PINNED" in out[0] and "PINNED" not in out[1]


def test_compare_exit_codes(tmp_path):
    files = make_fake_repo(tmp_path)
    info = h.collect(tmp_path, cluster_only_pins=[])
    good = {f["path"]: f["sha256"] for f in info["files"]}
    doc, code = h.compare(info, good, "check-only", "~/x", False, repo=tmp_path)
    assert code == 0 and doc["all_frozen_match"] and doc["n_files"] == 6
    row = next(r for r in doc["files"] if r["path"] == "data/pinned_manifest.json")
    assert row["synced_by_rsync"] is False and row["local_sha256"] == sha(files["data/pinned_manifest.json"])
    bad = dict(good, **{"scripts/a6_only.py": H["a"]})
    assert h.compare(info, bad, "x", "~/x", False, repo=tmp_path)[1] == 3
    bad = dict(good, **{"scripts/frozen_b.py": None})
    doc, code = h.compare(info, bad, "x", "~/x", False, repo=tmp_path)
    assert code == 1 and doc["n_mismatch_frozen"] == 1
    remote = h.parse_remote(f"{H['a']}  scripts/x.py\nMISSING  scripts/y.py\n")
    assert remote == {"scripts/x.py": H["a"], "scripts/y.py": None}
    with pytest.raises(ValueError):
        h.parse_remote("sha256sum: oops\n")


def test_excludes():
    assert h.excluded("data/x.json") and h.excluded("results/a/b.json")
    assert h.excluded("study4_rmet/data/rmet/stimuli/a.png")
    assert h.excluded("research/figures/fig1.png") and h.excluded("x/stimuli/a.txt")
    assert h.excluded("scripts/__pycache__/a.cpython-311.pyc") and h.excluded(".git/HEAD")
    assert not h.excluded("scripts/a4_design.py")
    assert not h.excluded("research/reproduction/a4_recovery.json")
    assert not h.excluded("research/scripts/data_utils.py")


# ------------------------------------------------------------------ end to end, stubbed SSH

FAKE_SSH = r"""#!/bin/bash
# Test stub: drop ssh options and the host, run the command in the fake remote home.
while [ $# -gt 0 ]; do
  case "$1" in
    -o|-l|-p|-i|-F) shift 2 ;;
    -*) shift ;;
    *) break ;;
  esac
done
shift  # host
cd "$FAKE_REMOTE_HOME" || exit 255
exec /bin/bash -c "$*"
"""

FAKE_SQUEUE = """#!/bin/bash
cat "$FAKE_SQUEUE_FILE"
"""

# GNU coreutils on the cluster; macOS has shasum with the same output format.
FAKE_SHA256SUM = """#!/bin/bash
[ "$1" = "--" ] && shift
exec shasum -a 256 "$@"
"""


@pytest.fixture
def e2e(tmp_path):
    if any(shutil.which(t) is None for t in ("rsync", "perl", "shasum")):
        pytest.skip("needs rsync, perl and shasum")
    repo = tmp_path / "repo"
    files = make_fake_repo(repo)
    home = tmp_path / "home"
    remote = home / "rds/hpc-work/study3"
    (remote / "scripts").mkdir(parents=True)
    (remote / "scripts/remote_only.py").write_text("keep me\n")
    # the cluster already holds the data file (data/ is never synced)
    (remote / "data").mkdir()
    (remote / "data/pinned_manifest.json").write_bytes(files["data/pinned_manifest.json"])
    binp = tmp_path / "bin"
    binp.mkdir()
    for name, body in (("ssh", FAKE_SSH), ("squeue", FAKE_SQUEUE),
                       ("sha256sum", FAKE_SHA256SUM)):
        (binp / name).write_text(body)
        (binp / name).chmod((binp / name).stat().st_mode | stat.S_IEXEC)
    queue = tmp_path / "squeue.txt"
    queue.write_text("")
    env = dict(os.environ, PATH=f"{binp}:{os.environ['PATH']}", FAKE_REMOTE_HOME=str(home),
               FAKE_SQUEUE_FILE=str(queue), PYTHON=PYTHON, TMPDIR=str(tmp_path),
               PHASE2_SKIP_CLUSTER_ONLY_PINS="1")

    def run(*args):
        return subprocess.run(["/bin/bash", str(repo / "research/scripts/phase2_sync_check.sh"), *args],
                              capture_output=True, text=True, env=env, timeout=120)

    return {"repo": repo, "remote": remote, "queue": queue, "run": run}


def test_e2e_dry_run_changes_nothing(e2e):
    r = e2e["run"]()
    assert r.returncode == 1, r.stdout + r.stderr  # pinned code is not on the fake cluster yet
    assert "rsync would update" in r.stdout and "PINNED" in r.stdout
    assert "MISMATCH [FROZEN] scripts/frozen_a.py" in r.stderr
    assert "MISMATCH [FROZEN] data/pinned_manifest.json" not in r.stderr
    assert not (e2e["remote"] / "scripts/frozen_a.py").exists()
    assert not (e2e["repo"] / "research/reproduction/phase2_hash_check.json").exists()


def test_e2e_apply_refused_while_rotation_queued(e2e):
    e2e["queue"].write_text("39982137_[0-4]|s3_rotation|PENDING\n")
    r = e2e["run"]("--apply")
    assert r.returncode == 1 and "s3_rotation" in r.stderr
    assert not (e2e["remote"] / "scripts/frozen_a.py").exists()
    e2e["queue"].write_text("5|s3_a4|RUNNING\n")
    r = e2e["run"]("--apply")
    assert r.returncode == 1 and not (e2e["remote"] / "scripts/frozen_a.py").exists()


def test_e2e_apply_syncs_and_logs(e2e):
    e2e["queue"].write_text("77|cas_chain|RUNNING\n")
    r = e2e["run"]("--apply")
    assert r.returncode == 0, r.stdout + r.stderr
    remote = e2e["remote"]
    assert (remote / "scripts/frozen_a.py").read_text() == "print('a')\n"
    assert (remote / "scripts/unpinned.py").exists()
    assert (remote / "scripts/remote_only.py").exists()          # never --delete
    for gone in ("results/secret.json", "stimuli/face.png", "data/big/clip.bin",
                 "scripts/__pycache__/x.pyc", ".git"):
        assert not (remote / gone).exists(), gone
    out = json.loads((e2e["repo"] / "research/reproduction/phase2_hash_check.json").read_text())
    assert out["mode"] == "apply" and out["rsync_applied"] and out["all_frozen_match"]
    assert out["n_files"] == 6 and all(r_["match"] for r_ in out["files"])
    # a later check-only run sees a tampered data file on the cluster and fails
    (remote / "data/pinned_manifest.json").write_text("tampered\n")
    r = e2e["run"]("--check-only")
    assert r.returncode == 1 and "MISMATCH [FROZEN] data/pinned_manifest.json" in r.stderr


def test_e2e_local_mismatch_stops_before_ssh(e2e):
    (e2e["repo"] / "scripts/frozen_b.py").write_text("edited\n")
    e2e["queue"].write_text("")
    r = e2e["run"]("--apply")
    assert r.returncode == 1 and "local check failed" in r.stderr
    assert not (e2e["remote"] / "scripts/frozen_a.py").exists()


def test_script_rejects_unknown_args():
    r = subprocess.run(["/bin/bash", str(SCRIPT), "--delete"], capture_output=True, text=True)
    assert r.returncode == 2
    assert "--delete" not in SCRIPT.read_text().split("set -euo pipefail", 1)[1].replace("never --delete", "").replace("no --delete", "").replace("Never --delete", "")
