#!/usr/bin/env python
"""Helpers for research/scripts/post_a1.sh, the one-off post-A1 runbook.

Nothing here opens an eval file for anything but hashing. Subcommands:

    expected                 print the 30 expected relative eval paths, one per line
    check-sacct FILE         check `sacct -X -n -P -o JobID,State` output for the A1 arrays
    hash-file FILE           print the sha256 of FILE
    verify REMOTE LOCALDIR   compare `sha256sum` output against files under LOCALDIR
    run LOG NAME -- CMD...   run CMD, append its record to LOG (JSON lines), return its code
    manifest ...             assemble the manifest JSON
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

MODELS = ("qwen3vl", "gemma4", "molmo2")
ROTATIONS = range(5)
CONDITIONS = ("real", "mismatch")
EVAL_DIR = "results/baseline/eu_emotions"
NAME = "eval_v2_eu_emotions_{m}_video_only_seed42_rot{r}_{c}.json"
ARRAYS = ("39982137", "39982138")
TASKS_PER_ARRAY = 5


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def expected():
    """Relative paths under the repo root, in analyse_rotation.py's EVAL pattern."""
    return [f"{EVAL_DIR}/{m}/{NAME.format(m=m, r=r, c=c)}"
            for m in MODELS for c in CONDITIONS for r in ROTATIONS]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_sacct(text, arrays=ARRAYS, tasks=TASKS_PER_ARRAY):
    """Return (ok, message, states). Every task of every array must be COMPLETED."""
    states = {a: {} for a in arrays}
    for line in text.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        job, state = line.split("|")[:2]
        base, _, task = job.partition("_")
        if base in states and task and "." not in task:
            states[base][task] = state.split()[0]
    problems = []
    for a in arrays:
        got = states[a]
        if any(t.startswith("[") for t in got):
            problems.append(f"array {a} still has pending tasks {sorted(t for t in got if t.startswith('['))}")
        done = {t: s for t, s in got.items() if not t.startswith("[")}
        if len(done) != tasks:
            problems.append(f"array {a} shows {len(done)} tasks, expected {tasks}")
        bad = {t: s for t, s in done.items() if s != "COMPLETED"}
        if bad:
            problems.append(f"array {a} tasks not COMPLETED: {bad}")
    return (not problems, "; ".join(problems) or "all tasks COMPLETED", states)


def parse_sha256sum(text):
    """`sha256sum` lines -> {relative path: hash}. Paths keep everything from results/."""
    out = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        digest, path = line.split(None, 1)
        path = path.lstrip("*").strip()
        rel = path[path.index(EVAL_DIR):] if EVAL_DIR in path else path
        out[rel] = digest
    return out


def cmd_expected(_):
    print("\n".join(expected()))


def cmd_check_sacct(args):
    ok, msg, states = check_sacct(Path(args.file).read_text())
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"ok": ok, "message": msg, "states": states},
                                                  indent=1, sort_keys=True) + "\n")
    print(msg)
    return 0 if ok else 1


def cmd_hash_file(args):
    print(sha256(args.file))


def cmd_verify(args):
    remote = parse_sha256sum(Path(args.remote).read_text())
    want = expected()
    problems = []
    if sorted(remote) != sorted(want):
        problems.append(f"remote hash list does not match the 30 expected names: "
                        f"missing {sorted(set(want) - set(remote))}, extra {sorted(set(remote) - set(want))}")
    rows = []
    for rel in want:
        local = Path(args.localdir) / rel
        got = sha256(local) if local.is_file() else None
        match = got is not None and got == remote.get(rel)
        if not match:
            problems.append(f"hash mismatch or missing: {rel}")
        rows.append({"path": rel, "sha256_remote": remote.get(rel), "sha256_local": got,
                     "match": match, "bytes": local.stat().st_size if local.is_file() else None})
    Path(args.json_out).write_text(json.dumps(rows, indent=1) + "\n")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(f"{len(rows)} files, all hashes match")
    return 0


def cmd_run(args):
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    log_dir = Path(args.log).parent
    stdout_path = log_dir / f"{args.name}.stdout.log"
    stderr_path = log_dir / f"{args.name}.stderr.log"
    started, t0 = now(), time.time()
    with open(stdout_path, "wb") as so, open(stderr_path, "wb") as se:
        code = subprocess.run(cmd, stdout=so, stderr=se, cwd=args.cwd).returncode
    record = {"name": args.name, "cmd": cmd, "cwd": args.cwd, "started": started,
              "finished": now(), "seconds": round(time.time() - t0, 1), "exit_code": code,
              "stdout_log": str(stdout_path), "stderr_log": str(stderr_path)}
    with open(args.log, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[{args.name}] exit {code} in {record['seconds']} s")
    return code


def cmd_manifest(args):
    commands = []
    if Path(args.commands).is_file():
        commands = [json.loads(l) for l in Path(args.commands).read_text().splitlines() if l.strip()]
    outputs = {}
    for p in args.outputs or []:
        path = Path(p)
        outputs[p] = sha256(path) if path.is_file() else None
    scripts = {}
    for p in args.scripts or []:
        scripts[p] = sha256(Path(args.repo) / p)
    manifest = {
        "what": "A1 rotation eval pull and analysis (research/scripts/post_a1.sh)",
        "mode": args.mode,
        "synthetic": args.mode == "dry-run",
        "started": args.started,
        "finished": now(),
        "git_commit": args.git_commit,
        "git_dirty": args.git_dirty == "1",
        "analysis_root": args.analysis_root,
        "source": args.source,
        "slurm": json.loads(Path(args.slurm).read_text()) if args.slurm and Path(args.slurm).is_file() else None,
        "slurm_check_skipped_reason": args.slurm_skip_reason or None,
        "remote_count": args.remote_count,
        "files": json.loads(Path(args.files).read_text()) if args.files and Path(args.files).is_file() else [],
        "script_sha256": scripts,
        "commands": commands,
        "outputs_sha256": outputs,
        "status": args.status,
        "exit_code": args.exit_code,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"manifest: {args.out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd_name", required=True)
    sub.add_parser("expected").set_defaults(fn=cmd_expected)
    p = sub.add_parser("check-sacct")
    p.add_argument("file")
    p.add_argument("--json-out")
    p.set_defaults(fn=cmd_check_sacct)
    p = sub.add_parser("hash-file")
    p.add_argument("file")
    p.set_defaults(fn=cmd_hash_file)
    p = sub.add_parser("verify")
    p.add_argument("remote")
    p.add_argument("localdir")
    p.add_argument("--json-out", required=True)
    p.set_defaults(fn=cmd_verify)
    p = sub.add_parser("run")
    p.add_argument("log")
    p.add_argument("name")
    p.add_argument("--cwd", required=True)
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("manifest")
    for flag in ("out", "mode", "started", "git-commit", "git-dirty", "analysis-root", "source",
                 "slurm", "remote-count", "files", "commands", "status", "repo", "slurm-skip-reason"):
        p.add_argument(f"--{flag}")
    p.add_argument("--exit-code", type=int)
    p.add_argument("--outputs", nargs="*")
    p.add_argument("--scripts", nargs="*")
    p.set_defaults(fn=cmd_manifest)
    args = ap.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
