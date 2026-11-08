#!/usr/bin/env python
"""Run the frozen analyse_rotation.py with one disclosed guard for errored trials.

Decision (DECISIONS, recorded before any A1 file was read): an errored trial already
scores as wrong on the primary estimand, because bool(None) is False. Only the secondary
p_target crashes on it, since scripts/evaluate.py writes errored trials without
`response_distribution`. Here such a stage2 gets an empty distribution, so the trial puts
0 mass on the target: the same "no vote" treatment.

The frozen file is not edited. It is imported by path after its sha256 is checked, its
`load` is wrapped, and its own `main` runs with the same CLI. The number of patched
trials per model, condition and rotation goes to a sidecar JSON. With no errored trials
the output is byte-identical to running analyse_rotation.py directly.

    python research/scripts/analyse_rotation_run.py [--output PATH] [--errored-out PATH]

A trial missing between files still raises, as in the frozen script: that is a job
problem, fixed by a rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = Path(__file__).resolve().with_name("analyse_rotation.py")
FROZEN_SHA256 = "98890c5bff65f13abeb0549b61fe414dc9fa30082b84bcb564671683ec12e24f"
DEFAULT_SIDECAR = ROOT / "research/reproduction/rotation_errored_trials.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen(path=TARGET):
    digest = sha256(path)
    if digest != FROZEN_SHA256:
        raise SystemExit(f"{path} sha256 {digest} is not the frozen {FROZEN_SHA256}")
    spec = importlib.util.spec_from_file_location("analyse_rotation", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wrap(mod):
    """Replace mod.load; return the dict it fills with patched-trial counts."""
    counts = {}
    original = mod.load

    def load(model, condition):
        runs = original(model, condition)
        for rotation, run in zip(mod.ROTATIONS, runs):
            patched = []
            for trial_id, stage2 in run.items():
                if isinstance(stage2, dict) and not isinstance(stage2.get("response_distribution"), dict):
                    stage2["response_distribution"] = {}
                    patched.append(trial_id)
            counts.setdefault(model, {}).setdefault(condition, {})[str(rotation)] = {
                "n_errored": len(patched), "trial_ids": sorted(patched)}
        return runs

    mod.load = load
    return counts


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--errored-out", type=Path, default=DEFAULT_SIDECAR)
    known, rest = pre.parse_known_args(argv)
    mod = load_frozen()
    counts = wrap(mod)
    saved = sys.argv
    sys.argv = [str(TARGET), *rest]
    try:
        mod.main()
    finally:
        sys.argv = saved
    total = sum(r["n_errored"] for m in counts.values() for c in m.values() for r in c.values())
    sidecar = {
        "what": "stage2 entries without response_distribution, given an empty one for p_target only",
        "analyse_rotation_sha256": FROZEN_SHA256,
        "wrapper_sha256": sha256(Path(__file__).resolve()),
        "total_errored": total,
        "counts": counts,
    }
    known.errored_out.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
