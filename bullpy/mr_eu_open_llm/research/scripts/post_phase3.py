#!/usr/bin/env python3
"""Phase 3 runbook: pull finished cells unread, then run each frozen analysis once (ROADMAP Phase 3).

Two subcommands, run by the coordinator from the project root on the Mac:

  pull AMENDMENT --jobs ID [ID ...]
      AMENDMENT is a2, a4, a7, a8 or a9 (a7: the Stage 1 route-arm shards; a8: see below). Over the SSH control socket: every task of every named Slurm array must
      be COMPLETED; every expected file must exist on the cluster. Records the remote sha256 of each file,
      copies them with scp into a staging folder, checks each copy's sha256, and only then moves them to
      the same relative path here. Refuses to overwrite a local file. Never prints or opens a file's
      contents. Writes `research/reproduction/<amendment>_pull_manifest.json`.

  analyse AMENDMENT
      AMENDMENT is a2, a4, a5, a7 or a9. A5 and A7 read A2's cells and output, so they run after A2.
      For A7 the pre-run check also writes the inputs file `a7_analyse` reads (added 2026-11-12). Runs the amendment's pre-run check (DECISIONS 2026-11-12), and only
      if it passes, the frozen analysis with the frozen command, single-threaded BLAS, once: it refuses if
      the output already exists. For A2 it then runs the declared extras. Writes a run note,
      `research/reproduction/<amendment>_run_note.json`, with the commands, environment, times, the
      pre-run record and every output's sha256. For A4 and A9 the check always gets a folder for a
      filtered copy; it writes one only if some rows have non-finite log probabilities (the ruling), and
      the analysis then reads that folder, which the run note names.

  pull a8 --jobs ID [ID ...]
      A8's analyses run on the cluster through its launcher (ROADMAP Part 3, A8 runbook), so A8 has its own
      pull (added 2026-11-13). Every listed job must be COMPLETED, and every A8 provenance file must check on
      the cluster, where its inputs are. Then `research/reproduction/a8/` and A8's A4 summary are copied with
      sha256 checks, and the sha256 of every other file under `results/a8/` is recorded without copying it.
      Writes `research/reproduction/a8_pull_manifest.json`.

Nothing here reads a result. The coordinator reads the outputs only after the run note exists, then
records them in RESULTS_REGISTRY.md.

Usage:
    PYTHONPATH=.:src python research/scripts/post_phase3.py pull a2 --jobs 40468625 40468626
    PYTHONPATH=.:src python research/scripts/post_phase3.py analyse a2
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
HOST = "eb2007@login.hpc.cam.ac.uk"
SSH = ["ssh", "-o", "ControlPath=/Users/eb2007/.ssh/cm-%r@%h:%p", "-o", "ControlMaster=no", "-o", "BatchMode=yes",
       "-o", "ConnectTimeout=10"]
SCP = ["scp", "-q", "-o", "ControlPath=/Users/eb2007/.ssh/cm-%r@%h:%p", "-o", "ControlMaster=no", "-o", "BatchMode=yes"]
REMOTE_ROOT = "rds/hpc-work/study3"  # relative to the remote home, for scp
MODELS = ("gemma4", "qwen3vl")
SINGLE_THREAD = {v: "1" for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
REPRO = "research/reproduction"
CLEAN = {"a4": "results/a4_clean", "a9": "results/a9_clean"}  # written only if rows are removed


def expected_files(amendment: str) -> List[str]:
    if amendment == "a2":
        files = ["results/stimulus_interventions/geometry.json"]
        for m in MODELS:
            files += [f"results/stimulus_interventions/cells_{m}_shard{k}.jsonl" for k in range(3)]
            files += [f"results/stimulus_interventions/run_{m}_shard{k}.json" for k in range(3)]
        return files
    if amendment == "a4":
        return ["results/a4/manifest.csv"] + [f"results/a4/cells_{m}_{k}.jsonl" for m in MODELS for k in range(12)]
    if amendment == "a9":
        return ["results/a9/manifest.csv"] + [f"results/a9/cells_{m}_{k}.jsonl" for m in MODELS for k in range(12)]
    if amendment == "a7":  # Stage 1 route arm, N_SHARDS=2 (DECISIONS 2026-11-10, 2026-11-12)
        return [f"results/a7/a7_cells_{m}_shard{k}.jsonl" for m in MODELS for k in range(2)]
    raise SystemExit(f"nothing to pull for {amendment}")


def outputs(amendment: str) -> Dict[str, List[str]]:
    """The files each analysis writes; the run refuses if any exists."""
    return {
        "a2": [f"{REPRO}/stimulus_interventions_{m}.json" for m in MODELS] + [f"{REPRO}/a2_extras_{m}.json" for m in MODELS],
        "a4": ["results/a4/summary.json"],
        "a5": ["results/a5/a5_analysis.json"],
        "a7": [f"{REPRO}/a7_analysis.json"],
        "a9": ["results/a9/summary.json"],
    }[amendment]


def commands(amendment: str, cells_dir: Optional[str] = None) -> List[List[str]]:
    """The pre-run check first, then the frozen analysis, exactly as the amendments name them."""
    if amendment == "a2":
        out: List[List[str]] = []
        for m in MODELS:
            cells = [f"results/stimulus_interventions/cells_{m}_shard{k}.jsonl" for k in range(3)]
            reports = [f"results/stimulus_interventions/run_{m}_shard{k}.json" for k in range(3)]
            out.append([PY, "research/scripts/a2_prerun_check.py", "--model", m, "--cells", *cells, "--run_report", *reports,
                        "--record", f"{REPRO}/a2_prerun_{m}.json"])
        for m in MODELS:
            cells = [f"results/stimulus_interventions/cells_{m}_shard{k}.jsonl" for k in range(3)]
            reports = [f"results/stimulus_interventions/run_{m}_shard{k}.json" for k in range(3)]
            out.append([PY, "-m", "scripts.analyse_stimulus_interventions", "--model", m, "--cells", *cells,
                        "--run_report", *reports, "--output", f"{REPRO}/stimulus_interventions_{m}.json", "--a2_frozen"])
            out.append([PY, "research/scripts/a2_declared_extras.py", "--model", m, "--cells", *cells, "--run_report", *reports,
                        "--a2_output", f"{REPRO}/stimulus_interventions_{m}.json", "--output", f"{REPRO}/a2_extras_{m}.json"])
        return out
    if amendment == "a4":
        check = [PY, "research/scripts/a4_prerun_check.py", "--record", f"{REPRO}/a4_prerun.json",
                 "--clean_dir", CLEAN["a4"]]
        return [check, [PY, "-m", "scripts.a4_analyse", "--cells_dir", cells_dir or "results/a4",
                        "--gates", f"{REPRO}/a4_gates.json", "--detector", f"{REPRO}/a4_v1_pilot.csv",
                        "--output", "results/a4/summary.json"]]
    if amendment == "a5":
        return [[PY, "research/scripts/a5_prerun_check.py", "--record", f"{REPRO}/a5_prerun.json"],
                [PY, "-m", "scripts.a5_analyse", "--output", "results/a5/a5_analysis.json", "--a5_frozen", "--a2_frozen"]]
    if amendment == "a7":
        return [[PY, "research/scripts/a7_prerun_check.py", "--record", f"{REPRO}/a7_prerun.json",
                 "--inputs", f"{REPRO}/a7_inputs.json"],
                [PY, "-m", "scripts.a7_analyse", "--inputs", f"{REPRO}/a7_inputs.json",
                 "--output", f"{REPRO}/a7_analysis.json", "--a7_frozen", "--a2_frozen"]]
    if amendment == "a9":
        check = [PY, "research/scripts/a9_prerun_check.py", "--record", f"{REPRO}/a9_prerun.json",
                 "--clean_dir", CLEAN["a9"]]
        return [check, [PY, "-m", "scripts.a9_analyse", "--cells_dir", cells_dir or "results/a9",
                        "--gates", f"{REPRO}/a4_gates.json", "--output", "results/a9/summary.json"]]
    raise SystemExit(f"unknown amendment {amendment}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ssh(command: str, timeout: int = 90) -> str:
    result = subprocess.run(["perl", "-e", f"alarm {timeout}; exec @ARGV", *SSH, HOST, command],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"SSH failed ({result.returncode}): {result.stderr.strip()[:300]}. Re-authenticate if needed.")
    return result.stdout


def array_states(jobs: Sequence[str]) -> Dict[str, List[str]]:
    text = ssh("sacct -X -n -P -o JobID,State -j " + ",".join(jobs))
    states: Dict[str, List[str]] = {}
    for line in text.splitlines():
        if "|" in line:
            job, state = line.split("|", 1)
            states.setdefault(job.split("_")[0], []).append(f"{job}:{state.strip()}")
    return states


def pull(amendment: str, jobs: Sequence[str], waive: Sequence[str] = (), reason: str = "") -> int:
    manifest_path = ROOT / REPRO / f"{amendment}_pull_manifest.json"
    if manifest_path.exists():
        raise SystemExit(f"{manifest_path} exists: this pull runs once.")
    states = array_states(jobs)
    if waive and not reason:
        raise SystemExit("--waive needs --reason, and the task's replacement job must be in --jobs.")
    bad = [s for job in jobs for s in states.get(job, [f"{job}:MISSING"])
           if not s.endswith(":COMPLETED") and s.split(":")[0] not in waive]
    if bad or any(job not in states for job in jobs):
        raise SystemExit(f"Not every task is COMPLETED: {bad[:12]}")
    files = expected_files(amendment)
    for rel in files:
        if (ROOT / rel).exists():
            raise SystemExit(f"{rel} already exists here; refusing to overwrite.")
    listing = ssh("cd ~/" + REMOTE_ROOT + " && sha256sum " + " ".join(shlex.quote(f) for f in files), timeout=300)
    remote = {line.split(None, 1)[1].strip(): line.split(None, 1)[0] for line in listing.splitlines() if line.strip()}
    missing = [f for f in files if f not in remote]
    if missing:
        raise SystemExit(f"Missing on the cluster: {missing[:12]}")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    staging = ROOT / "research" / "logs" / f"pull_{amendment}_{stamp}"
    staging.mkdir(parents=True)
    for rel in files:
        dest = staging / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(["perl", "-e", "alarm 900; exec @ARGV", *SCP, f"{HOST}:{REMOTE_ROOT}/{rel}", str(dest)])
        if result.returncode != 0:
            raise SystemExit(f"scp failed for {rel}")
        if sha256(dest) != remote[rel]:
            raise SystemExit(f"{rel}: local sha256 differs from the cluster's")
    for rel in files:
        target = ROOT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging / rel), str(target))
    record = {"amendment": amendment, "pulled_at": now(), "jobs": {j: states[j] for j in jobs},
              "waived_tasks": list(waive), "waiver_reason": reason,
              "files": {rel: remote[rel] for rel in files}, "contents_read": False}
    manifest_path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(f"pulled {len(files)} files; manifest {manifest_path.relative_to(ROOT)}")
    return 0


def analyse(amendment: str) -> int:
    note_path = ROOT / REPRO / f"{amendment}_run_note.json"
    if note_path.exists():
        raise SystemExit(f"{note_path} exists: the analysis has run.")
    for rel in outputs(amendment):
        if (ROOT / rel).exists():
            raise SystemExit(f"{rel} exists: the analysis has run, and runs once.")
    env = {**os.environ, **SINGLE_THREAD, "PYTHONPATH": f"{ROOT}:{ROOT / 'src'}"}
    steps: List[Dict[str, Any]] = []
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    cells_dir: Optional[str] = None
    plan = commands(amendment)
    index = 0
    while index < len(plan):
        command = plan[index]
        started = now()
        result = subprocess.run(command, cwd=ROOT, env=env)
        steps.append({"command": [Path(c).name if c == PY else c for c in command], "started": started,
                      "finished": now(), "exit": result.returncode})
        if result.returncode != 0:
            failed = {"amendment": amendment, "git_commit": git, "steps": steps, "status": "stopped",
                      "reason": "pre-run check failed" if "prerun" in " ".join(command) else "analysis step failed"}
            (ROOT / "research" / "logs").mkdir(parents=True, exist_ok=True)
            (ROOT / "research" / "logs" / f"{amendment}_run_stopped_{started.replace(':', '')}.json").write_text(
                json.dumps(failed, indent=1) + "\n", encoding="utf-8")
            raise SystemExit(f"Stopped at: {' '.join(command[1:4])}. Nothing after it ran. See research/logs/.")
        if amendment in CLEAN and index == 0:
            check_record = json.loads((ROOT / REPRO / f"{amendment}_prerun.json").read_text(encoding="utf-8"))
            cells_dir = check_record.get("clean_dir")  # set only when rows were removed
            plan = commands(amendment, cells_dir)
        index += 1
    note = {"amendment": amendment, "git_commit": git, "python": PY, "threads": SINGLE_THREAD, "cells_dir_used": cells_dir,
            "steps": steps, "outputs": {rel: sha256(ROOT / rel) for rel in outputs(amendment) if (ROOT / rel).exists()},
            "status": "complete"}
    note_path.write_text(json.dumps(note, indent=1) + "\n", encoding="utf-8")
    print(f"done; run note {note_path.relative_to(ROOT)}")
    return 0


A8_COPY_ROOT = "research/reproduction/a8"  # copied in full: summaries, gate files, launch records, provenance
A8_COPY_ALSO = ("results/a8/a4/summary.json",)  # A4's output; the cells and shards stay on the cluster
A8_HASH_ROOT = "results/a8"  # every other A8 file: sha256 recorded in the manifest, not copied
REMOTE_PY = "~/rds/hpc-work/study2/conda_envs/mr_eu_open_llm/bin/python"  # the project env; the login python is 3.6
A8_PROVENANCE_CHECK = """\
import glob, json, sys
sys.path.insert(0, "research/scripts")
import a8_family
out = {}
for p in sorted(glob.glob("research/reproduction/a8/**/*.provenance.json", recursive=True)
                + glob.glob("results/a8/**/*.provenance.json", recursive=True)):
    target = p[:-len(".provenance.json")]
    try:
        a8_family.check_provenance(a8_family.located(target))
        out[target] = "ok"
    except SystemExit as exc:
        out[target] = str(exc)
print(json.dumps(out))
"""


def a8_plan(remote: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """A8 files to copy (outputs, records, provenance) and files only hashed (cells, shards, merged A1 files)."""
    copy = sorted(p for p in remote if p.startswith(A8_COPY_ROOT + "/") or p in A8_COPY_ALSO)
    return copy, sorted(p for p in remote if p not in copy)


def pull_a8(jobs: Sequence[str], waive: Sequence[str] = (), reason: str = "") -> int:
    """A8's analyses run on the cluster, where their inputs are (ROADMAP Part 3, A8 runbook). Every listed
    job must be COMPLETED; every provenance file must check on the cluster, where its inputs are; then the
    A8 outputs are copied with their sha256 checked, and every other A8 file's sha256 is recorded."""
    manifest_path = ROOT / REPRO / "a8_pull_manifest.json"
    if manifest_path.exists():
        raise SystemExit(f"{manifest_path} exists: this pull runs once.")
    if waive and not reason:
        raise SystemExit("--waive needs --reason, and the task's replacement job must be in --jobs.")
    states = array_states(jobs)
    bad = [s for job in jobs for s in states.get(job, [f"{job}:MISSING"])
           if not s.endswith(":COMPLETED") and s.split(":")[0] not in waive]
    if bad or any(job not in states for job in jobs):
        raise SystemExit(f"Not every task is COMPLETED: {bad[:12]}")
    provenance = json.loads(ssh(f"cd ~/{REMOTE_ROOT} && {REMOTE_PY} -c {shlex.quote(A8_PROVENANCE_CHECK)}", timeout=600))
    failed = {k: v for k, v in provenance.items() if v != "ok"}
    if failed:
        raise SystemExit(f"Provenance does not check on the cluster: {list(failed.items())[:6]}")
    listing = ssh(f"cd ~/{REMOTE_ROOT} && find {A8_COPY_ROOT} {A8_HASH_ROOT} -type f -print0 | xargs -0 sha256sum",
                  timeout=1800)
    remote = {line.split(None, 1)[1].strip(): line.split(None, 1)[0] for line in listing.splitlines() if line.strip()}
    copy, hashed = a8_plan(remote)
    if not copy:
        raise SystemExit("No A8 output on the cluster.")
    for rel in copy:
        if (ROOT / rel).exists():
            raise SystemExit(f"{rel} already exists here; refusing to overwrite.")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    staging = ROOT / "research" / "logs" / f"pull_a8_{stamp}"
    staging.mkdir(parents=True)
    for rel in copy:
        dest = staging / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(["perl", "-e", "alarm 900; exec @ARGV", *SCP, f"{HOST}:{REMOTE_ROOT}/{rel}", str(dest)])
        if result.returncode != 0:
            raise SystemExit(f"scp failed for {rel}")
        if sha256(dest) != remote[rel]:
            raise SystemExit(f"{rel}: local sha256 differs from the cluster's")
    for rel in copy:
        target = ROOT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging / rel), str(target))
    record = {"amendment": "a8", "pulled_at": now(), "jobs": {j: states[j] for j in jobs},
              "waived_tasks": list(waive), "waiver_reason": reason, "provenance_checked_on_cluster": provenance,
              "files": {rel: remote[rel] for rel in copy}, "on_cluster_only": {rel: remote[rel] for rel in hashed},
              "contents_read": False}
    manifest_path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(f"pulled {len(copy)} A8 files, recorded {len(hashed)} more; manifest {manifest_path.relative_to(ROOT)}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pull")
    p.add_argument("amendment", choices=("a2", "a4", "a7", "a8", "a9"))
    p.add_argument("--jobs", nargs="+", required=True)
    p.add_argument("--waive", nargs="*", default=(), help="array tasks (e.g. 40589226_7) rerun under a job in --jobs")
    p.add_argument("--reason", default="")
    a = sub.add_parser("analyse")
    a.add_argument("amendment", choices=("a2", "a4", "a5", "a7", "a9"))
    args = ap.parse_args(argv)
    if args.cmd == "pull" and args.amendment == "a8":
        return pull_a8(args.jobs, args.waive, args.reason)
    if args.cmd == "pull":
        return pull(args.amendment, args.jobs, args.waive, args.reason)
    return analyse(args.amendment)


if __name__ == "__main__":
    raise SystemExit(main())
