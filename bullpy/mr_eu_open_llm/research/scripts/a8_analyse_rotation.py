#!/usr/bin/env python
"""A8: the frozen A1 analysis over the A8 family.

Runs research/scripts/analyse_rotation.py through the errored-trial wrapper
(research/scripts/analyse_rotation_run.py, DECISIONS 2026-11-08), both hash-checked. Two
module constants of the frozen analysis are set for this process only: `MODELS`, to the
active A8 ids (confirmed plus admitted, from the A8 design), and `EVAL`, to the merged A8
files. So Holm runs over the A8 family alone, and the frozen statistics are untouched.

    python research/scripts/a8_analyse_rotation.py \
        [--output research/reproduction/a8/rotation_summary_a8.json] \
        [--errored-out research/reproduction/a8/rotation_errored_trials_a8.json]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a8_design  # noqa: E402

A8_EVAL = "results/a8/{m}/rotation/eval_v2_eu_emotions_{m}_video_only_seed42_rot{r}_{c}.json"
DEFAULT_OUT = a8_design.ROOT / "research/reproduction/a8/rotation_summary_a8.json"
DEFAULT_ERRORED = a8_design.ROOT / "research/reproduction/a8/rotation_errored_trials_a8.json"


def configure(models: Sequence[str]):
    """The frozen module, wrapped for errored trials, pointed at the A8 files and family."""
    import analyse_rotation_run as runner

    mod = runner.load_frozen()
    counts = runner.wrap(mod)
    mod.MODELS = tuple(models)
    mod.EVAL = A8_EVAL
    return runner, mod, counts


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--errored-out", type=Path, default=DEFAULT_ERRORED)
    ap.add_argument("--models", nargs="+", help="Tests only; default is the active A8 models.")
    args = ap.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; the A8 A1 analysis runs once.")
    if args.models:
        models = args.models
    else:
        design = a8_design.load_design()
        a8_design.check_frozen_code(design)
        models = a8_design.active_models(design)
    runner, mod, counts = configure(models)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    saved = sys.argv
    sys.argv = ["analyse_rotation.py", "--output", str(args.output)]
    try:
        mod.main()
    finally:
        sys.argv = saved
    import json

    total = sum(r["n_errored"] for m in counts.values() for c in m.values() for r in c.values())
    args.errored_out.write_text(json.dumps({
        "what": "stage2 entries without response_distribution, given an empty one for p_target only",
        "analyse_rotation_sha256": runner.FROZEN_SHA256,
        "wrapper_sha256": a8_design.file_sha256(Path(__file__).resolve()),
        "a8_models": list(models),
        "total_errored": total,
        "counts": counts,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
