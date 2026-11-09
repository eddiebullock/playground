#!/usr/bin/env python
"""A8: the frozen A1 analysis over the A8 A1 family.

Runs research/scripts/analyse_rotation.py through the errored-trial wrapper
(research/scripts/analyse_rotation_run.py, DECISIONS 2026-11-08), both hash-checked. Two
module constants of the frozen analysis are set for this process only: `MODELS`, to the A8
A1 family computed by `a8_family.a1_family` (G2 to G4 passed and A3 Pass, every merged file
present, one GPU count per model, no test-mode shard), and `EVAL`, to the merged A8 files.
Holm then runs over the A8 family alone, and the frozen statistics are untouched.

    python research/scripts/a8_analyse_rotation.py

Writes research/reproduction/a8/rotation_summary_a8.json, the errored-trial sidecar, and a
provenance file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a8_design  # noqa: E402
import a8_family  # noqa: E402

A8_EVAL = "results/a8/{m}/rotation/eval_v2_eu_emotions_{m}_video_only_seed42_rot{r}_{c}.json"
ERRORED = "research/reproduction/a8/rotation_errored_trials_a8.json"


def configure(models: Sequence[str], base: Optional[Path] = None):
    """The frozen module, wrapped for errored trials, pointed at the A8 files and family.

    `base` replaces the frozen module's ROOT only in tests, where the files live elsewhere.
    """
    import analyse_rotation_run as runner

    mod = runner.load_frozen()
    counts = runner.wrap(mod)
    mod.MODELS = tuple(models)
    mod.EVAL = A8_EVAL
    if base is not None:
        mod.ROOT = Path(base)
    return runner, mod, counts


def analyse(models: Sequence[str], output: Path, errored_out: Path, excluded=None, gpus=None, base: Optional[Path] = None) -> None:
    """The frozen main over these models. Production callers pass the computed family."""
    for path in (output, errored_out):
        if Path(path).exists():
            raise SystemExit(f"{path} exists; the A8 A1 analysis runs once.")
    runner, mod, counts = configure(models, base)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    saved = sys.argv
    sys.argv = ["analyse_rotation.py", "--output", str(output)]
    try:
        mod.main()
    finally:
        sys.argv = saved
    total = sum(r["n_errored"] for m in counts.values() for c in m.values() for r in c.values())
    Path(errored_out).write_text(json.dumps({
        "what": "stage2 entries without response_distribution, given an empty one for p_target only",
        "analyse_rotation_sha256": runner.FROZEN_SHA256,
        "wrapper_sha256": a8_design.file_sha256(Path(__file__).resolve()),
        "a8_models": list(models), "total_errored": total, "counts": counts,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    root = Path(base) if base is not None else a8_design.ROOT
    inputs = [root / A8_EVAL.format(m=m, r=r, c=c) for m in models for r in a8_family.ROTATIONS for c in a8_family.CONDITIONS]
    a8_family.write_provenance(Path(output), inputs, {
        "family": list(models), "excluded": excluded or {}, "visible_gpus": gpus or {},
        "wrapper_sha256": a8_design.file_sha256(Path(__file__).resolve()), "design_sha256": a8_design.block_hash(),
    })


def main(argv: Optional[Sequence[str]] = None) -> None:
    root = a8_design.ROOT
    os.chdir(root)
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    design = a8_design.load_design()
    a8_design.check_frozen_code(design)
    a8_design.check_own_code(require=True)
    a8_family.check_provenance(root / a8_family.A3_SUMMARY)
    family, excluded = a8_family.a1_family(design, root)
    if not family:
        raise SystemExit(f"No A8 model is in the A1 family: {excluded}")
    gpus = a8_family.check_a1_run_settings(root, family, design)
    analyse(family, root / a8_family.A1_SUMMARY, root / ERRORED, excluded, gpus)


if __name__ == "__main__":
    main()
