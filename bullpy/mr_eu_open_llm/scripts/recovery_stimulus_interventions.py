"""
Rule 7: synthetic observers with known behaviour go through the real A2 analysis.

Each observer writes cells in the format of scripts/run_stimulus_interventions.py, over
the real 243 items, actors and labels, with made-up scores. No model, clip or result is
read. A contrast counts as recovered if it reaches step-down p < .05 in its direction.
Pass rule, fixed before this was run: expected contrasts recovered in at least 80 percent
of replicates, and any other contrast in a replicate at most 10 percent of the time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set

import numpy as np

from scripts import analyse_stimulus_interventions as an
from scripts import stimulus_interventions as si
from scripts.run_stimulus_interventions import INPUTS, LEXICON, OPTION_TO_FAMILY

FAMILIES = sorted(LEXICON)
EVIDENCE = 2.0
ACTOR_SD, ITEM_SD, CELL_SD = 0.3, 0.5, 0.2
FLIPS = {"max_exact_actors": 0, "draws": 8191}


def human_like_shift(condition: str, label: str, low: bool) -> Dict[str, float]:
    """Family shifts stating every confirmatory prediction, as a positive control."""
    shift: Dict[str, float] = {}
    if condition.startswith("mask_mouth") and label in ("Happy", "Disgusted"):
        shift[label] = -EVIDENCE
    if condition.startswith("mask_eyes_") and label in ("Afraid", "Sad"):
        shift[label] = -EVIDENCE
    if condition.startswith("mask_brows") and label == "Angry":
        shift[label] = -EVIDENCE
    if condition.startswith("mask_mouth") and label in ("Happy", "Sad", "Angry"):
        shift["Neutral"] = EVIDENCE
    if condition.startswith("mask_mouth") and label == "Disgusted":
        shift["Angry"] = EVIDENCE
    if condition == "sf_high" and label == "Happy":
        shift[label] = -EVIDENCE
    if condition == "sf_low" and label == "Sad":
        shift[label] = -EVIDENCE
    if condition in an.SHUFFLES and (label in ("Happy", "Surprised") or low):
        shift[label] = -EVIDENCE
    if condition == "trunc_first_third":
        partner = {"Afraid": "Surprised", "Surprised": "Afraid", "Angry": "Disgusted", "Disgusted": "Angry"}.get(label)
        if partner:
            shift[partner] = EVIDENCE
    return shift


def evidence_kept(observer: str, condition: str) -> float:
    if observer == "mouth_only" and ("mask_mouth" in condition or "mask_eyesmouth" in condition):
        return 0.0
    if observer == "eyes_only" and ("mask_eyes" in condition):
        return 0.0
    return 1.0


def synthetic_rows(observer: str, items: Sequence[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    prior = rng.normal(0, 1, len(FAMILIES))
    actor_effect = {a: rng.normal(0, ACTOR_SD, len(FAMILIES)) for a in sorted({i["actor"] for i in items})}
    rows = []
    for item in items:
        family = OPTION_TO_FAMILY[item["label"]]
        base = prior + actor_effect[item["actor"]] + rng.normal(0, ITEM_SD, len(FAMILIES))
        low = an.is_low_intensity(item["trial_id"])
        for route, spec in INPUTS.items():
            bag = rng.normal(0, CELL_SD, len(FAMILIES))
            for condition in spec["conditions"]:
                noise = rng.normal(0, CELL_SD, len(FAMILIES))
                if observer == "frame_bag" and spec["kind"] != "image" and condition in ("original", "reverse", *an.SHUFFLES):
                    noise = bag
                scores = base + noise
                if observer != "prior_only":
                    scores = scores + (FAMILIES.index(family) == np.arange(len(FAMILIES))) * EVIDENCE * evidence_kept(observer, condition)
                if observer == "human_like":
                    for name, value in human_like_shift(condition, item["label"], low).items():
                        scores[FAMILIES.index(OPTION_TO_FAMILY[name])] += value
                rows.append(
                    {
                        **item,
                        "input_name": route,
                        "condition": condition,
                        "status": "ok",
                        "blind": {"family_logprob": dict(zip(FAMILIES, scores.tolist())), "lexicon_mass": 0.5},
                    }
                )
    return rows


EXPECTED: Dict[str, Set[str]] = {
    "prior_only": set(),
    "frame_bag": set(),
    "mouth_only": {"P1", "P2"},
    "eyes_only": {"P3", "P4"},
    "human_like": {c.id for c in an.CONFIRMATORY},
}


def recover(observer: str, items: Sequence[Dict[str, Any]], replicates: int) -> Dict[str, Any]:
    actors = sorted({i["actor"] for i in items})
    ids = [c.id for c in an.CONFIRMATORY]
    wins = {k: 0 for k in ids}
    any_unexpected = 0
    for replicate in range(replicates):
        cells = an.Cells(synthetic_rows(observer, items, si.stable_seed(observer, replicate) % 2**32))
        hits: Set[str] = set()
        for route in ("single_face", "video_face"):
            family = [c for c in an.CONFIRMATORY if c.route == route]
            result = an.test_family(cells, family, actors, flip_args=FLIPS)
            hits |= {k for k, row in result.items() if row["p_maxt"] < an.ALPHA and row["mean_change"] * row["direction"] > 0}
        for key in hits:
            wins[key] += 1
        any_unexpected += bool(hits - EXPECTED[observer])
    expected = EXPECTED[observer]
    sensitivity = min((wins[k] / replicates for k in expected), default=None)
    unexpected = any_unexpected / replicates
    return {
        "expected": sorted(expected),
        "recovery_rate_by_contrast": {k: wins[k] / replicates for k in ids},
        "lowest_rate_on_expected": sensitivity,
        "share_replicates_with_an_unexpected_contrast": unexpected,
        "pass": bool((sensitivity is None or sensitivity >= 0.8) and unexpected <= 0.10),
    }


def main() -> None:
    from scripts.run_stimulus_interventions import load_items

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, default=Path("data/eu_emotions_full_manifest.json"))
    parser.add_argument("--lookup", type=Path, default=Path("data/eu_emotions_human_entropy.json"))
    parser.add_argument("--replicates", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("research/reproduction/recovery_stimulus_interventions.json"))
    args = parser.parse_args()
    items = load_items(args.manifest, args.lookup, Path("."))
    keep = ("trial_id", "actor", "label", "human_modal_label", "weak_label")
    items = [{k: i[k] for k in keep} for i in items]
    out = {
        "settings": {"evidence": EVIDENCE, "actor_sd": ACTOR_SD, "item_sd": ITEM_SD, "cell_sd": CELL_SD, "replicates": args.replicates, **FLIPS},
        "observers": {name: recover(name, items, args.replicates) for name in EXPECTED},
    }
    out["pass"] = all(v["pass"] for v in out["observers"].values())
    args.output.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: {"pass": v["pass"], "low": v["lowest_rate_on_expected"], "fp": v["share_replicates_with_an_unexpected_contrast"]} for k, v in out["observers"].items()}, indent=1))


if __name__ == "__main__":
    main()
