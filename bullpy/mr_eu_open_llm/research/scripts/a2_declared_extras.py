#!/usr/bin/env python3
"""Declared A2 extras, fixed before any A2 cell exists (DECISIONS 2026-11-12, A2 rulings 1, 4 and 5).

Runs once, after the frozen `scripts/analyse_stimulus_interventions.py --a2_frozen`, on the same cells.
It changes nothing in the frozen output; it adds three things the frozen output does not hold, each
computed with the frozen functions and their defaults:

1. Fill rule on Opposes rows (ruling 1). PREREG_A2 line 144 says a mask effect counts only if the
   first-frame and blur fills have the same sign as the neutral fill; the decision table and the code
   apply this to Matches only, and `fills_agree` means "other fills in the predicted direction". The
   ruling takes the stricter reading: an Opposes verdict on a contrast with other fills stands only if
   every other fill's mean actor effect has the neutral fill's sign. This script writes each fill's
   signed mean and `verdict_after_fill_rule`, which equals the frozen verdict except that Opposes
   becomes Inconclusive where the fills do not share the neutral fill's sign.
2. 90 percent equivalence intervals (ruling 4): `actor_interval(actor_effects(...), 0.90)` with the
   frozen draws and seed, the same call `equivalence_pass` makes. The script checks that each interval
   reproduces the frozen `null_equivalence` boolean and stops if any does not.
3. Shuffles overall (ruling 5): the change from `original` to the mean of `shuffle_a` and `shuffle_b`
   by label and scope, with its actor 95 percent interval, as `descriptive_changes` does for one
   condition. Descriptive only.

Only the primary analysis (`results.primary_blind_acted`) is covered; sensitivity rows keep the frozen
verdicts, and an Opposes verdict there is reported as not fill-checked.

Usage (project root, after the frozen analysis):
    PYTHONPATH=.:src python research/scripts/a2_declared_extras.py --model gemma4 \
        --cells results/stimulus_interventions/cells_gemma4_shard*.jsonl \
        --run_report results/stimulus_interventions/run_gemma4_shard*.json \
        --a2_output <frozen output>.json --output <extras>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import analyse_stimulus_interventions as an  # noqa: E402  (frozen, tag a2-freeze)
from scripts import run_stimulus_interventions as run  # noqa: E402
from scripts.stimulus_design import BASIC_LABELS, CONFIRMATORY, SHUFFLES, Contrast  # noqa: E402

ROUTES = ("single_face", "video_face")


def _mean(values: np.ndarray) -> float:
    present = values[~np.isnan(values)]
    return float(present.mean()) if present.size else float("nan")


def fill_check(cells: an.Cells, contrast: Contrast, actors: Sequence[str]) -> Optional[Dict[str, Any]]:
    """Each fill's signed mean actor effect, and whether the other fills share the neutral fill's sign."""
    if not contrast.other_fills:
        return None
    neutral = _mean(an.actor_effects(cells, contrast, actors))
    second = contrast.other_fills_b or (contrast.b,) * len(contrast.other_fills)
    others = [_mean(an.actor_effects(cells, contrast, actors, a=fill, b=side))
              for fill, side in zip(contrast.other_fills, second)]
    share = bool(np.isfinite(neutral) and neutral != 0 and all(
        np.isfinite(o) and np.sign(o) == np.sign(neutral) for o in others))
    return {"neutral_fill_mean": neutral,
            "other_fills": [{"a": list(f), "b": list(s), "mean": o} for f, s, o in zip(contrast.other_fills, second, others)],
            "fills_share_neutral_sign": share}


def verdict_after_fill_rule(frozen: str, check: Optional[Mapping[str, Any]]) -> str:
    if frozen == "opposes_prediction" and check is not None and not check["fills_share_neutral_sign"]:
        return "inconclusive"
    return frozen


def equivalence_interval(cells: an.Cells, contrast: Contrast, actors: Sequence[str]) -> Optional[List[float]]:
    interval = an.actor_interval(an.actor_effects(cells, contrast, actors), 0.90)
    return list(interval) if interval else None


def shuffles_overall(cells: an.Cells, route: str, actors: Sequence[str]) -> Optional[Dict[str, Any]]:
    if not set(SHUFFLES) <= set(an.INPUTS[route]["conditions"]):
        return None
    labels = sorted({i["label"] for i in cells.items.values()})
    groups = {"basic": tuple(BASIC_LABELS), "complex": tuple(x for x in labels if x not in BASIC_LABELS)}
    rows: Dict[str, Any] = {}
    for name, members in {**{x: (x,) for x in labels}, **groups}.items():
        values = an.actor_effects(cells, Contrast("d", route, members, SHUFFLES), actors)
        present = values[~np.isnan(values)]
        interval = an.actor_interval(values)
        rows[name] = {"n_actors": int(present.size), "mean_change": float(present.mean()) if present.size else None,
                      "ci95": list(interval) if interval else None}
    return rows


def extras(cells: an.Cells, frozen: Mapping[str, Any]) -> Dict[str, Any]:
    actors = sorted({r["actor"] for r in cells.items.values()})
    out: Dict[str, Any] = {}
    for route in ROUTES:
        block = frozen[route]
        primary = block["results"]["primary_blind_acted"]
        margin = block["margin"]
        rows: Dict[str, Any] = {}
        for contrast in [c for c in CONFIRMATORY if c.route == route]:
            row = primary[contrast.id]
            check = fill_check(cells, contrast, actors)
            interval = equivalence_interval(cells, contrast, actors)
            inside = bool(margin is not None and interval is not None and interval[0] > -margin and interval[1] < margin)
            if inside != bool(row["null_equivalence"]):
                raise SystemExit(f"{route} {contrast.id}: the 90% interval does not reproduce the frozen null_equivalence.")
            rows[contrast.id] = {"frozen_verdict": row["verdict"], "fill_rule": check,
                                 "verdict_after_fill_rule": verdict_after_fill_rule(row["verdict"], check),
                                 "equivalence_interval_90": interval, "margin": margin}
        out[route] = {"contrasts": rows, "shuffles_overall": shuffles_overall(cells, route, actors)}
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--cells", type=Path, nargs="+", required=True)
    ap.add_argument("--run_report", type=Path, nargs="+", required=True)
    ap.add_argument("--a2_output", type=Path, required=True, help="the frozen analysis output, already written")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    frozen = json.loads(args.a2_output.read_text(encoding="utf-8"))
    design = run.design_hash()
    if frozen.get("design_sha256") != design or frozen.get("model") != args.model:
        raise SystemExit("The frozen output is not this model's run of this design.")
    rows, _ = an.load_shards(args.cells, args.run_report, args.model, design)
    result = {"model": args.model, "design_sha256": design,
              "a2_output_sha256": hashlib.sha256(args.a2_output.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              **extras(an.Cells(rows), frozen)}
    args.output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
