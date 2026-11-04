"""
Analysis for draft amendment A2: Gate B, confirmatory contrasts, null rule, human tables.

Reads the cells written by scripts/run_stimulus_interventions.py. Per model and route,
with no pooling across models. Scripts/recovery_stimulus_interventions.py runs this exact
code on synthetic observers. Statistics are imported from research/scripts/analyse_rotation.py
for Gate B, and from scripts/stimulus_interventions.py for step-down maxT.

No analysis may run on real cells until A2 is frozen, so main refuses without --a2_frozen.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "scripts"))
from analyse_rotation import actor_bootstrap_ci, actor_sign_flip_p  # noqa: E402

from scripts import stimulus_interventions as si  # noqa: E402
from scripts.run_stimulus_interventions import OPTION_TO_FAMILY, human_comparable  # noqa: E402

ALPHA = 0.05
B_RULES = {"coverage": 0.05, "image_rms": 0.25, "sign_flip_p": 0.01, "repeat": 1e-3}
MARGIN_FRACTION = 0.10
ORIGINAL = ("original",)
SHUFFLES = ("shuffle_a", "shuffle_b")


@dataclass(frozen=True)
class Contrast:
    id: str
    route: str
    labels: Tuple[str, ...]
    a: Tuple[str, ...]
    outcome: str = "acted"
    direction: int = -1
    b: Tuple[str, ...] = ORIGINAL
    versus_labels: Tuple[str, ...] = ()  # interaction: labels minus these, inside one actor set
    low_intensity: bool = False
    other_fills: Tuple[Tuple[str, ...], ...] = field(default=())


def _mask(region: str, fill: str = "neutral") -> Tuple[str, ...]:
    return (f"mask_{region}_{fill}",)


def _fills(region: str) -> Tuple[Tuple[str, ...], ...]:
    return (_mask(region, "first"), _mask(region, "blur"))


# P11 is out until the aperture dose is verified (see the plan).
CONFIRMATORY: Tuple[Contrast, ...] = (
    Contrast("P1", "single_face", ("Happy",), _mask("mouth"), other_fills=_fills("mouth")),
    Contrast("P2", "single_face", ("Disgusted",), _mask("mouth"), other_fills=_fills("mouth")),
    Contrast("P3", "single_face", ("Afraid",), _mask("eyes"), other_fills=_fills("eyes")),
    Contrast("P4", "single_face", ("Sad",), _mask("eyes"), other_fills=_fills("eyes")),
    Contrast("P5", "single_face", ("Angry",), _mask("brows"), other_fills=_fills("brows")),
    Contrast(
        "P6", "single_face", ("Happy", "Disgusted"), _mask("mouth"), b=_mask("eyes"), versus_labels=("Afraid",)
    ),
    Contrast("P7", "single_face", ("Happy", "Sad", "Angry"), _mask("mouth"), "Neutral", 1, other_fills=_fills("mouth")),
    Contrast("P8", "single_face", ("Disgusted",), _mask("mouth"), "Angry", 1, other_fills=_fills("mouth")),
    Contrast("P9", "single_face", ("Happy",), ("sf_low",), direction=1, b=("sf_high",)),
    Contrast("P10", "single_face", ("Sad",), ("sf_low",), direction=-1, b=("sf_high",)),
    Contrast("P12", "video_face", ("Happy", "Surprised"), SHUFFLES),
    Contrast("P13", "video_face", (), SHUFFLES, low_intensity=True),
    Contrast("P14a", "video_face", ("Afraid",), ("trunc_first_third",), "Surprised", 1, ("trunc_last_third",)),
    Contrast("P14b", "video_face", ("Surprised",), ("trunc_first_third",), "Afraid", 1, ("trunc_last_third",)),
    Contrast("P15a", "video_face", ("Angry",), ("trunc_first_third",), "Disgusted", 1, ("trunc_last_third",)),
    Contrast("P15b", "video_face", ("Disgusted",), ("trunc_first_third",), "Angry", 1, ("trunc_last_third",)),
)


def is_low_intensity(trial_id: str) -> bool:
    return "Low Intensity" in trial_id


class Cells:
    """Rows indexed by item, route and condition; failed cells are held out and counted."""

    def __init__(self, rows: Sequence[Dict[str, Any]]) -> None:
        self.ok: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self.failed = [r for r in rows if r.get("status") != "ok"]
        self.items: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            self.items.setdefault(row["trial_id"], row)
            if row.get("status") == "ok":
                self.ok[(row["trial_id"], row["input_name"], row["condition"])] = row


def clr_of(row: Dict[str, Any], family: str) -> float:
    scores = row["blind"]["family_logprob"]
    return float(si.clr([scores[k] for k in sorted(scores)])[sorted(scores).index(family)])


def mcq_score(row: Dict[str, Any], family: str) -> float:
    """Mean over rotations of an option's log-ratio against its row. NaN if not an option."""
    values = []
    for rotation in row["mcq"]:
        families = [OPTION_TO_FAMILY.get(o) for o in rotation["options"]]
        if family in families:
            values.append(float(si.clr(rotation["joint_logprob"])[families.index(family)]))
    return float(np.mean(values)) if values else float("nan")


def outcome_family(contrast: Contrast, item: Dict[str, Any], target: str) -> str:
    if contrast.outcome != "acted":
        return contrast.outcome
    label = item["human_modal_label"] if target == "human_mode" else item["label"]
    return OPTION_TO_FAMILY[label]


def item_delta(
    cells: Cells, trial: str, contrast: Contrast, conditions: Tuple[Tuple[str, ...], Tuple[str, ...]], readout: str, target: str
) -> float:
    item = cells.items[trial]
    family = outcome_family(contrast, item, target)
    score = clr_of if readout == "blind" else mcq_score
    sides = []
    for side in conditions:
        rows = [cells.ok.get((trial, contrast.route, c)) for c in side]
        if any(r is None for r in rows):
            return float("nan")
        sides.append(float(np.mean([score(r, family) for r in rows])))
    return sides[0] - sides[1]


def in_contrast(item: Dict[str, Any], labels: Sequence[str], low: bool) -> bool:
    if low:
        return is_low_intensity(item["trial_id"])
    return item["label"] in labels


def actor_effects(
    cells: Cells,
    contrast: Contrast,
    actors: Sequence[str],
    *,
    a: Optional[Tuple[str, ...]] = None,
    readout: str = "blind",
    target: str = "acted",
    exclude: Sequence[str] = (),
) -> np.ndarray:
    """Actor means of within-item change; NaN for an actor with no item in the contrast."""
    conditions = (a or contrast.a, contrast.b)

    def by_actor(labels: Sequence[str]) -> Dict[str, float]:
        values: Dict[str, List[float]] = defaultdict(list)
        for trial, item in cells.items.items():
            if trial in exclude or not in_contrast(item, labels, contrast.low_intensity):
                continue
            delta = item_delta(cells, trial, contrast, conditions, readout, target)
            if not np.isnan(delta):
                values[item["actor"]].append(delta)
        return {k: float(np.mean(v)) for k, v in values.items()}

    first = by_actor(contrast.labels)
    second = by_actor(contrast.versus_labels) if contrast.versus_labels else None
    out = np.full(len(actors), np.nan)
    for index, actor in enumerate(actors):
        if actor not in first:
            continue
        if second is None:
            out[index] = first[actor]
        elif actor in second:
            out[index] = first[actor] - second[actor]
    return out


def test_family(
    cells: Cells, contrasts: Sequence[Contrast], actors: Sequence[str], *, flip_args: Optional[Dict[str, Any]] = None, **kwargs: Any
) -> Dict[str, Dict[str, Any]]:
    """Step-down maxT over one route's family. Directions as preregistered."""
    matrix = np.column_stack([actor_effects(cells, c, actors, **kwargs) for c in contrasts])
    result = si.signflip_maxt(matrix, alternative=[c.direction for c in contrasts], **(flip_args or {}))
    out = {}
    for index, contrast in enumerate(contrasts):
        column = matrix[:, index]
        present = column[~np.isnan(column)]
        same_sign_fills = None
        if contrast.other_fills:
            same_sign_fills = all(
                float(np.nanmean(actor_effects(cells, contrast, actors, a=fill, **kwargs))) * contrast.direction > 0
                for fill in contrast.other_fills
            )
        out[contrast.id] = {
            "route": contrast.route,
            "n_actors": int(present.size),
            "mean_change": float(present.mean()) if present.size else float("nan"),
            "direction": contrast.direction,
            "p_raw": result["p_raw"][index],
            "p_maxt": result["p_maxt"][index],
            "fills_agree": same_sign_fills,
        }
    return out


def verdict(row: Dict[str, Any], gates_ok: bool, null_ok: bool) -> str:
    """The draft's decision table. Opposite direction needs a two-sided test; see two_sided."""
    if not gates_ok:
        return "inconclusive"
    agrees = row["mean_change"] * row["direction"] > 0
    if row["p_maxt"] < ALPHA and agrees and row["fills_agree"] is not False:
        return "matches_prediction"
    if row.get("p_maxt_two_sided", 1.0) < ALPHA and not agrees:
        return "opposes_prediction"
    return "null" if null_ok else "inconclusive"


def equivalence_pass(values: np.ndarray, margin: float, *, draws: int = 10000, seed: int = si.SEED) -> bool:
    """90 percent actor-bootstrap CI inside the margin. values are actor means."""
    present = values[~np.isnan(values)]
    if present.size < 2:
        return False
    rng = np.random.default_rng(seed)
    means = present[rng.integers(0, present.size, size=(draws, present.size))].mean(1)
    low, high = np.quantile(means, [0.05, 0.95])
    return bool(low > -margin and high < margin)


def gate_b(cells: Cells, route: str, partners: Dict[str, str], image_free: Optional[Dict[str, float]] = None, repeat_max_abs: float = 0.0) -> Dict[str, Any]:
    """B1 to B5 on one route. image_free is the family_logprob of the baseline row."""
    trials = sorted(t for t in cells.items if (t, route, "original") in cells.ok)
    originals = {t: cells.ok[(t, route, "original")] for t in trials}
    masses = [r["blind"]["lexicon_mass"] for r in originals.values()]
    fams = sorted(next(iter(originals.values()))["blind"]["family_logprob"])
    matrix = np.array([[r["blind"]["family_logprob"][f] for f in fams] for r in originals.values()])
    centred = matrix - matrix.mean(1, keepdims=True)
    prior = centred.mean(0)
    index = {t: i for i, t in enumerate(trials)}

    def signal(adjust: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        values, actors = [], []
        for trial in trials:
            partner = partners.get(trial)
            if partner not in index:
                continue
            f = fams.index(OPTION_TO_FAMILY[cells.items[trial]["label"]])
            values.append((centred - adjust)[index[trial], f] - (centred - adjust)[index[partner], f])
            actors.append(cells.items[trial]["actor"])
        return np.array(values), actors

    b3 = None
    if image_free is not None:
        free = np.array([image_free[f] for f in fams])
        free = free - free.mean()
        b3 = float(np.mean(np.sqrt(((centred - free) ** 2).mean(1))))
    values, actors = signal(np.zeros_like(prior))
    p4, ci4 = actor_sign_flip_p(values, actors), actor_bootstrap_ci(values, actors)
    adjusted, _ = signal(prior)
    p5, ci5 = actor_sign_flip_p(adjusted, actors), actor_bootstrap_ci(adjusted, actors)
    checks = {
        "B1_repeat_max_abs": repeat_max_abs,
        "B1": bool(repeat_max_abs <= B_RULES["repeat"]),
        "B2_median_lexicon_mass": float(np.median(masses)),
        "B2": bool(np.median(masses) >= B_RULES["coverage"]),
        "B3_rms_nats": b3,
        "B3": bool(b3 is not None and b3 >= B_RULES["image_rms"]),
        "B4_mean": float(values.mean()),
        "B4_p": p4,
        "B4_ci": ci4,
        "B4": bool(p4 < B_RULES["sign_flip_p"] and ci4[0] > 0),
        "B5_mean": float(adjusted.mean()),
        "B5_p": p5,
        "B5_ci": ci5,
        "B5": bool(p5 < B_RULES["sign_flip_p"] and ci5[0] > 0),
        "mismatch_margin_basis": float(np.mean(np.abs(values))),
    }
    checks["pass"] = all(checks[k] for k in ("B1", "B2", "B3", "B4", "B5"))
    return checks


def human_tables(cells: Cells, route: str, human: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """Change in divergence from the human distribution. Descriptive, weak labels apart."""
    grouped: Dict[Tuple[str, bool], List[float]] = defaultdict(list)
    for (trial, name, condition), row in cells.ok.items():
        original = cells.ok.get((trial, name, "original"))
        if name != route or condition == "original" or original is None or trial not in human:
            continue
        before = si.js_divergence(*human_comparable(original["blind"]["family_logprob"], human[trial]))
        after = si.js_divergence(*human_comparable(row["blind"]["family_logprob"], human[trial]))
        grouped[(condition, bool(row.get("weak_label")))].append(after - before)
    return {
        f"{condition}|{'weak_label' if weak else 'other'}": {"n": len(v), "mean_change_js": float(np.mean(v))}
        for (condition, weak), v in sorted(grouped.items())
    }


def analyse_route(
    cells: Cells,
    route: str,
    actors: Sequence[str],
    gates_ok: bool,
    margin: float,
    exclude: Sequence[str] = (),
    flip_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    family = [c for c in CONFIRMATORY if c.route == route]
    primary = test_family(cells, family, actors, exclude=exclude, flip_args=flip_args)
    two_sided = si.signflip_maxt(
        np.column_stack([actor_effects(cells, c, actors, exclude=exclude) for c in family]), **(flip_args or {})
    )
    for index, contrast in enumerate(family):
        row = primary[contrast.id]
        row["p_maxt_two_sided"] = two_sided["p_maxt"][index]
        values = actor_effects(cells, contrast, actors, exclude=exclude)
        row["null_equivalence"] = equivalence_pass(values, margin)
        row["verdict"] = verdict(row, gates_ok, row["null_equivalence"])
    return {
        "primary_blind_acted": primary,
        "against_human_modal_label": test_family(cells, family, actors, target="human_mode", exclude=exclude, flip_args=flip_args),
        "mcq_bridge": test_family(cells, family, actors, readout="mcq", exclude=exclude, flip_args=flip_args),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--run_report", type=Path, required=True)
    parser.add_argument("--lookup", type=Path, default=Path("data/eu_emotions_human_entropy.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke_items", default="")
    parser.add_argument("--a2_frozen", action="store_true")
    args = parser.parse_args()
    if not args.a2_frozen:
        raise SystemExit("A2 is not frozen. No analysis runs on real cells before that.")

    from paper_v2.option_score import vision_mismatch_partners

    rows = [json.loads(l) for l in args.cells.read_text(encoding="utf-8").splitlines()]
    cells = Cells(rows)
    actors = sorted({r["actor"] for r in cells.items.values()})
    human = {k: v["human_response_distribution"] for k, v in json.loads(args.lookup.read_text())["trials"].items()}
    ids = sorted(cells.items)
    partners = vision_mismatch_partners(
        ids, [cells.items[i]["label"] for i in ids], [cells.items[i]["actor"] for i in ids], seed=42
    )
    report = json.loads(args.run_report.read_text(encoding="utf-8"))
    baselines = {(b["input_name"], b["condition"]): b for b in report.get("baselines") or []}
    out: Dict[str, Any] = {"failed_cells": len(cells.failed)}
    for route in ("single_face", "video_face"):
        free = baselines.get((route, "image_free"), {}).get("family_logprob")
        repeat = max(
            [d[route]["repeat_forward_max_abs"] for d in report.get("delivery", []) if route in d and "failed" not in d[route]],
            default=float("nan"),
        )
        gate = gate_b(cells, route, partners, free, repeat)
        margin = MARGIN_FRACTION * gate["mismatch_margin_basis"]
        weak = [t for t, i in cells.items.items() if i.get("weak_label")]
        mode_wrong = [t for t, i in cells.items.items() if i["label"] != i["human_modal_label"]]
        smoke = [s for s in args.smoke_items.split(",") if s]
        out[route] = {
            "gate_b": gate,
            "margin": margin,
            "results": analyse_route(cells, route, actors, gate["pass"], margin),
            "human_distribution": human_tables(cells, route, human),
            "sensitivity": {
                "without_smoke_items": analyse_route(cells, route, actors, gate["pass"], margin, exclude=smoke),
                "without_weak_labels": analyse_route(cells, route, actors, gate["pass"], margin, exclude=weak),
                "without_mode_not_target": analyse_route(cells, route, actors, gate["pass"], margin, exclude=mode_wrong),
            },
        }
    args.output.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
