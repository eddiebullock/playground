"""
Analysis for draft amendment A2: Gate B, confirmatory contrasts, null rule, human tables.

Reads the cells written by scripts/run_stimulus_interventions.py. Per model and route,
with no pooling across models. Scripts/recovery_stimulus_interventions.py runs this exact
code on synthetic observers. Statistics are imported from research/scripts/analyse_rotation.py
for Gate B, and from scripts/stimulus_interventions.py for step-down maxT.

The contrasts, thresholds and null-rule controls are in scripts/stimulus_design.py. No
analysis may run on real cells until A2 is frozen, so main refuses without --a2_frozen.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from analyse_rotation import actor_bootstrap_ci, actor_sign_flip_p  # noqa: E402

from scripts import run_stimulus_interventions as run  # noqa: E402
from scripts import stimulus_interventions as si  # noqa: E402
from scripts.run_stimulus_interventions import INPUTS, LEXICON, OPTION_TO_FAMILY, human_comparable, lexicon_words  # noqa: E402
from scripts.stimulus_design import (  # noqa: E402
    ALPHA,
    B_RULES,
    BASIC_LABELS,
    BOOTSTRAP_DRAWS,
    CONFIRMATORY,
    GATE_KEYS,
    MARGIN_FRACTION,
    MIN_REGION_TOKENS,
    N_ITEMS,
    ORIGINAL,
    SHUFFLES,
    SMOKE_ITEMS,
    TOKEN_FLOOR,
    Contrast,
)


def is_low_intensity(trial_id: str) -> bool:
    return "Low Intensity" in trial_id


class Cells:
    """Rows indexed by item, route and condition; failed cells are held out and counted."""

    def __init__(self, rows: Sequence[Dict[str, Any]]) -> None:
        self.rows = list(rows)
        self.ok: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self.failed = [r for r in rows if r.get("status") != "ok"]
        self.items: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            self.items.setdefault(row["trial_id"], row)
            if row.get("status") == "ok":
                self.ok[(row["trial_id"], row["input_name"], row["condition"])] = row

    def without(self, drop: Callable[[Dict[str, Any]], bool]) -> "Cells":
        """The same items with some cells removed, for a sensitivity analysis."""
        return Cells([r for r in self.rows if not drop(r)])


def clr_of(row: Dict[str, Any], family: str, leave_out: Optional[str] = None) -> float:
    """A family's log score against the mean over families, without leave_out if given.

    A contrast that predicts a rise in another label leaves the acted label out of the
    reference. Otherwise a fall in the acted label lowers the mean and raises every other
    label by the same amount, which reads as a rise nobody predicted.
    """
    scores = row["blind"]["family_logprob"]
    names = [k for k in sorted(scores) if k != leave_out]
    return float(si.clr([scores[k] for k in names])[names.index(family)])


def mcq_score(row: Dict[str, Any], family: str, leave_out: Optional[str] = None) -> float:
    """Mean over rotations of an option's log-ratio against its row. NaN if not an option."""
    values = []
    for rotation in row["mcq"]:
        families = [OPTION_TO_FAMILY.get(o) for o in rotation["options"]]
        kept = [i for i, f in enumerate(families) if f != leave_out]
        names = [families[i] for i in kept]
        if family in names:
            scores = si.clr([rotation["joint_logprob"][i] for i in kept])
            values.append(float(scores[names.index(family)]))
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
    leave_out = OPTION_TO_FAMILY[item["label"]] if contrast.outcome != "acted" else None
    score = clr_of if readout == "blind" else mcq_score
    sides = []
    for side in conditions:
        rows = [cells.ok.get((trial, contrast.route, c)) for c in side]
        if any(r is None for r in rows):
            return float("nan")
        sides.append(float(np.mean([score(r, family, leave_out) for r in rows])))
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
    b: Optional[Tuple[str, ...]] = None,
    readout: str = "blind",
    target: str = "acted",
    exclude: Sequence[str] = (),
) -> np.ndarray:
    """Actor means of within-item change; NaN for an actor with no item in the contrast."""
    conditions = (a or contrast.a, b or contrast.b)

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
            second = contrast.other_fills_b or (contrast.b,) * len(contrast.other_fills)
            same_sign_fills = all(
                float(np.nanmean(actor_effects(cells, contrast, actors, a=fill, b=side, **kwargs))) * contrast.direction > 0
                for fill, side in zip(contrast.other_fills, second)
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


def gates_passed(gates: Mapping[str, Any]) -> bool:
    """Every named gate must be exactly True. A missing gate has not passed."""
    return all(gates.get(key) is True for key in GATE_KEYS)


def verdict(row: Dict[str, Any], gates: Mapping[str, Any], null_ok: bool) -> str:
    """The draft's decision table. Opposite direction needs a two-sided test; see two_sided."""
    if not gates_passed(gates):
        return "inconclusive"
    one_sided, two_sided = row.get("p_maxt", float("nan")), row.get("p_maxt_two_sided", float("nan"))
    agrees = row["mean_change"] * row["direction"] > 0
    if one_sided < ALPHA and agrees and row["fills_agree"] is not False:
        return "matches_prediction"
    if two_sided < ALPHA and not agrees:
        return "opposes_prediction"
    # A significant contrast is never a null, whatever its fills or its interval did.
    if null_ok and one_sided >= ALPHA and two_sided >= ALPHA:
        return "null"
    return "inconclusive"


def _actor_means(values: np.ndarray, draws: int, seed: int) -> Optional[np.ndarray]:
    present = values[~np.isnan(values)]
    if present.size < 2:
        return None
    rng = np.random.default_rng(seed)
    return present[rng.integers(0, present.size, size=(draws, present.size))].mean(1)


def actor_interval(
    values: np.ndarray, level: float = 0.95, *, draws: int = BOOTSTRAP_DRAWS, seed: int = si.SEED
) -> Optional[Tuple[float, float]]:
    """Actor-bootstrap percentile interval of a contrast. values are actor means."""
    means = _actor_means(values, draws, seed)
    if means is None:
        return None
    low, high = np.quantile(means, [(1 - level) / 2, (1 + level) / 2])
    return float(low), float(high)


def equivalence_pass(values: np.ndarray, margin: float, *, draws: int = BOOTSTRAP_DRAWS, seed: int = si.SEED) -> bool:
    """90 percent actor-bootstrap CI inside the margin. values are actor means."""
    interval = actor_interval(values, 0.90, draws=draws, seed=seed)
    return bool(interval is not None and interval[0] > -margin and interval[1] < margin)


def mean_absolute_change(cells: Cells, contrast: Contrast, condition: str, actors: Sequence[str], exclude: Sequence[str] = ()) -> float:
    """Item-level size of a condition against the original, on the acted label, by actor."""
    values: Dict[str, List[float]] = defaultdict(list)
    plain = Contrast("dose", contrast.route, (), (condition,))
    for trial, item in cells.items.items():
        wanted = in_contrast(item, contrast.labels + contrast.versus_labels, contrast.low_intensity)
        if trial in exclude or not wanted:
            continue
        delta = item_delta(cells, trial, plain, (plain.a, plain.b), "blind", "acted")
        if not np.isnan(delta):
            values[item["actor"]].append(abs(delta))
    means = [float(np.mean(values[a])) for a in actors if a in values]
    return float(np.mean(means)) if means else float("nan")


def region_tokens(cells: Cells, contrast: Contrast, frame_tokens: Optional[Mapping[str, float]]) -> Optional[float]:
    """Median edited area in tokens, over a mask contrast's items. None if it cannot be read.

    frame_tokens gives each clip's whole frame in square tokens on this route. The smallest
    region in the contrast decides, since the rule covers every region it edits.
    """
    conditions = [c for c in contrast.a + contrast.b if c.startswith("mask_")]
    if not conditions:
        return float("inf")
    if frame_tokens is None:
        return None
    sizes = []
    for condition in conditions:
        areas = []
        for trial, item in cells.items.items():
            row = cells.ok.get((trial, contrast.route, condition))
            wanted = in_contrast(item, contrast.labels + contrast.versus_labels, contrast.low_intensity)
            if row is None or not wanted or trial not in frame_tokens:
                continue
            areas.append(row["edit"]["mask_fraction"] * frame_tokens[trial])
        if not areas:
            return None
        sizes.append(float(np.median(areas)))
    return min(sizes)


def null_conditions(
    cells: Cells,
    contrast: Contrast,
    row: Dict[str, Any],
    actors: Sequence[str],
    gates: Mapping[str, Any],
    margin: Optional[float],
    frame_tokens: Optional[Mapping[str, float]],
    exclude: Sequence[str] = (),
) -> Dict[str, Any]:
    """The six conditions of the null rule. Anything missing leaves its condition False."""
    usable_margin = margin is not None and bool(np.isfinite(margin)) and margin > 0
    area = region_tokens(cells, contrast, frame_tokens)
    doses = {c: mean_absolute_change(cells, contrast, c, actors, exclude) for c in contrast.dose}
    joint = False
    if contrast.joint is not None and usable_margin:
        values = actor_effects(cells, contrast, actors, a=(contrast.joint,), b=ORIGINAL, exclude=exclude)
        joint = equivalence_pass(values, margin)
    interval = row.get("ci95")
    checks = {
        "c1_rq0_gates_gate_b": all(gates.get(k) is True for k in ("rq0", "g2_g4", "gate_b")),
        "c2_delivery_and_region": gates.get("delivery") is True and area is not None and area >= MIN_REGION_TOKENS,
        "c3_dose": usable_margin and bool(doses) and all(bool(v > margin) for v in doses.values()),
        "c4_joint_ablation": bool(joint),
        "c5_equivalence": usable_margin and bool(row.get("null_equivalence")),
        "c6_divergence_reported": interval is not None and bool(np.isfinite(row["mean_change"])),
    }
    return {
        **checks,
        "region_tokens": area,
        "dose_changes": doses,
        "joint_condition": contrast.joint,
        "can_reach_null": contrast.joint is not None,
        "null_ok": all(checks.values()),
    }


def gate_b(cells: Cells, route: str, partners: Dict[str, str], image_free: Optional[Dict[str, float]] = None, repeat_max_abs: float = 0.0) -> Dict[str, Any]:
    """B1 to B5 on one route. image_free is the family_logprob of the baseline row."""
    trials = sorted(t for t in cells.items if (t, route, "original") in cells.ok)
    originals = {t: cells.ok[(t, route, "original")] for t in trials}
    masses = [r["blind"]["lexicon_mass"] for r in originals.values()]
    fams = sorted(next(iter(originals.values()))["blind"]["family_logprob"])
    matrix = np.array([[r["blind"]["family_logprob"][f] for f in fams] for r in originals.values()])
    centred = matrix - matrix.mean(1, keepdims=True)
    index = {t: i for i, t in enumerate(trials)}

    def family_of(trial: str, target: str = "acted") -> Optional[str]:
        label = cells.items[trial]["human_modal_label" if target == "human_mode" else "label"]
        return OPTION_TO_FAMILY.get(label)

    acted = np.array([fams.index(family_of(t)) for t in trials])

    def face_specific(target: str = "acted", exclude: Sequence[str] = ()) -> Tuple[np.ndarray, List[str]]:
        """B4: s_i(x_i) minus s_i(x_d(i)), the item's own family on its own and donor faces."""
        values, actors = [], []
        for trial in trials:
            partner, family = partners.get(trial), family_of(trial, target)
            if partner not in index or family is None or trial in exclude:
                continue
            f = fams.index(family)
            values.append(centred[index[trial], f] - centred[index[partner], f])
            actors.append(cells.items[trial]["actor"])
        return np.array(values), actors

    def beyond_prior() -> Tuple[np.ndarray, List[str]]:
        """B5: the acted family on its own clip against that family on clips of other labels."""
        values = []
        for i, trial in enumerate(trials):
            others = acted != acted[i]
            values.append(centred[i, acted[i]] - centred[others, acted[i]].mean())
        return np.array(values), [cells.items[t]["actor"] for t in trials]

    def summary(values: np.ndarray, actors: Sequence[str]) -> Dict[str, Any]:
        return {"mean": float(values.mean()), "p": actor_sign_flip_p(values, actors),
                "ci": actor_bootstrap_ci(values, actors), "n_items": int(values.size)}

    b3 = None
    if image_free is not None:
        free = np.array([image_free[f] for f in fams])
        free = free - free.mean()
        b3 = float(np.mean(np.sqrt(((centred - free) ** 2).mean(1))))
    values, actors = face_specific()
    p4, ci4 = actor_sign_flip_p(values, actors), actor_bootstrap_ci(values, actors)
    adjusted, adjusted_actors = beyond_prior()
    p5, ci5 = actor_sign_flip_p(adjusted, adjusted_actors), actor_bootstrap_ci(adjusted, adjusted_actors)
    flagged = [t for t in trials if cells.items[t].get("weak_label")]
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
        "B4_against_human_mode": summary(*face_specific("human_mode")),
        "B4_without_flagged_items": summary(*face_specific(exclude=flagged)),
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


def top_family(row: Dict[str, Any]) -> str:
    scores = row["blind"]["family_logprob"]
    return max(sorted(scores), key=lambda k: scores[k])


def flip_rates(cells: Cells, route: str) -> Dict[str, Any]:
    """Share of items whose top family moves from the original. Secondary."""
    flips: Dict[str, List[bool]] = defaultdict(list)
    for (trial, name, condition), row in cells.ok.items():
        base = cells.ok.get((trial, name, "original"))
        if name == route and condition != "original" and base is not None:
            flips[condition].append(top_family(row) != top_family(base))
    return {c: {"n": len(v), "flip_rate": float(np.mean(v))} for c, v in sorted(flips.items())}


def shuffle_nuisance(cells: Cells) -> Optional[float]:
    """Flip rate between the two shuffles of one item, the measured nuisance rate."""
    pairs = []
    for trial in cells.items:
        a = cells.ok.get((trial, "video_face", SHUFFLES[0]))
        b = cells.ok.get((trial, "video_face", SHUFFLES[1]))
        if a is not None and b is not None:
            pairs.append(top_family(a) != top_family(b))
    return float(np.mean(pairs)) if pairs else None


def failure_table(cells: Cells) -> Dict[str, int]:
    counts = Counter((r["input_name"], r["condition"], r["label"], r.get("reason", "")) for r in cells.failed)
    return {"|".join(key): n for key, n in sorted(counts.items())}


def descriptive_changes(cells: Cells, route: str, actors: Sequence[str]) -> Dict[str, Any]:
    """Change from the original by condition, label and scope. No test and no direction.

    This holds the eye views, reverse, static_apex, shuffles overall and every complex
    label, which the prereg reports and does not test.
    """
    labels = sorted({i["label"] for i in cells.items.values()})
    groups = {"basic": tuple(BASIC_LABELS), "complex": tuple(x for x in labels if x not in BASIC_LABELS)}
    out: Dict[str, Any] = {}
    for condition in INPUTS[route]["conditions"]:
        if condition == "original":
            continue
        rows: Dict[str, Any] = {}
        for name, members in {**{x: (x,) for x in labels}, **groups}.items():
            values = actor_effects(cells, Contrast("d", route, members, (condition,)), actors)
            present = values[~np.isnan(values)]
            interval = actor_interval(values)
            rows[name] = {
                "n_actors": int(present.size),
                "mean_change": float(present.mean()) if present.size else None,
                "ci95": list(interval) if interval else None,
            }
        out[condition] = rows
    return out


def analyse_route(
    cells: Cells,
    route: str,
    actors: Sequence[str],
    gates: Mapping[str, Any],
    margin: Optional[float],
    exclude: Sequence[str] = (),
    flip_args: Optional[Dict[str, Any]] = None,
    frame_tokens: Optional[Mapping[str, float]] = None,
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
        interval = actor_interval(values)
        row["ci95"] = list(interval) if interval else None
        row["null_equivalence"] = bool(margin is not None and equivalence_pass(values, margin))
        row["null_rule"] = null_conditions(cells, contrast, row, actors, gates, margin, frame_tokens, exclude)
        row["verdict"] = verdict(row, gates, row["null_rule"]["null_ok"])
    return {
        "primary_blind_acted": primary,
        "against_human_modal_label": test_family(cells, family, actors, target="human_mode", exclude=exclude, flip_args=flip_args),
        "mcq_bridge": test_family(cells, family, actors, readout="mcq", exclude=exclude, flip_args=flip_args),
    }


def _edit(row: Dict[str, Any]) -> Dict[str, Any]:
    return row.get("edit") or {}


def align_flagged(row: Dict[str, Any]) -> bool:
    return bool(_edit(row).get("align_flagged"))


def donor_is_sent_frame(row: Dict[str, Any]) -> bool:
    return bool(_edit(row).get("donor_is_sent_frame"))


def own_neutral_donor(row: Dict[str, Any]) -> bool:
    return bool((_edit(row).get("neutral_donor") or {}).get("self"))


def canonical_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rows scored on each family's first word, in place of the sum over its forms."""
    words = lexicon_words()
    places = {family: words.index((family, LEXICON[family][0])) for family in LEXICON}
    out = []
    for row in rows:
        blind = row.get("blind")
        if blind is None:
            out.append(row)
            continue
        scores = {family: blind["word_logprob"][place] for family, place in places.items()}
        out.append({**row, "blind": {**blind, "family_logprob": scores}})
    return out


def frame_tokens_by_clip(items: Mapping[str, Dict[str, Any]], across: Optional[float], geometry: Mapping[str, Any]) -> Dict[str, float]:
    """Each clip's whole frame in square tokens, given tokens across the face crop."""
    out = {}
    for trial in items:
        record = geometry.get(trial) or {}
        if across is None or not record.get("face_box") or not record.get("height"):
            continue
        per_px = across / (record["face_box"][2] - record["face_box"][0])
        out[trial] = record["width"] * record["height"] * per_px**2
    return out


def below_token_floor(items: Mapping[str, Dict[str, Any]], across: Optional[float], geometry: Mapping[str, Any]) -> List[str]:
    """Clips with under 8 tokens across the face, or with no way to tell (DECISIONS 2026-11-04)."""
    out = []
    for trial in items:
        record = geometry.get(trial) or {}
        if across is None or not record.get("face_box") or not record.get("face_width_px"):
            out.append(trial)
            continue
        view = record["face_box"][2] - record["face_box"][0]
        if across * record["face_width_px"] / view < TOKEN_FLOOR:
            out.append(trial)
    return out


def rq0_status(a1_passed: Optional[bool], a3_passed: Optional[bool], route: str) -> bool:
    """The A1 and A3 table. A pass on A3 alone admits single_face and nothing else."""
    if a3_passed is not True:
        return False
    return a1_passed is True or (a1_passed is False and route == "single_face")


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def outside_gates(reproduction: Path, model: str, route: str) -> Dict[str, Any]:
    """RQ0, G2 to G4 and delivery from their own output files. A missing file has not passed."""
    a1 = (_read_json(reproduction / "rotation_summary.json") or {}).get(model, {}).get("vision_dependent")
    a3 = (_read_json(reproduction / "face_input_validity_summary.json") or {}).get(model, {}).get("vision_dependent")
    vision = _read_json(reproduction / f"vision_gates_{model}.json") or {}
    verdicts = vision.get("verdict") or {}
    g2_g4 = not vision.get("smoke", True) and all((verdicts.get(g) or {}).get("pass") is True for g in ("G2", "G3", "G4"))
    delivery = (_read_json(reproduction / f"input_delivery_{model}.json") or {}).get("verdicts", {}).get(route, {})
    return {"rq0": rq0_status(a1, a3, route), "a1": a1, "a3": a3, "g2_g4": g2_g4, "delivery_file": delivery.get("pass") is True}


def delivery_rows(reports: Sequence[Dict[str, Any]], route: str) -> List[Dict[str, Any]]:
    return [d[route] for report in reports for d in report.get("delivery", []) if route in d]


def delivery_passed(reports: Sequence[Dict[str, Any]], route: str) -> bool:
    """Delivery rows of this run, across every shard. No rows means not passed."""
    rows = delivery_rows(reports, route)
    if not rows or any("failed" in r for r in rows):
        return False
    return all(
        r.get(key) is True
        for r in rows
        for key in (
            "gate_2_every_frame_changes_output",
            "gate_3_logged_indices_are_the_frames_sent",
            "gate_4_ids_unchanged_within_clip",
        )
    ) and all(run.order_check(r) for r in rows) and (
        route != "video_face" or all("pixels_change_under_shuffle" in r for r in rows)
    )


def sham_passed(cells: Cells, route: str) -> bool:
    """The sham must send the same pixels as the original. Only single_face has a sham."""
    if route != "single_face":
        return True
    pairs = [
        (row["media_sha256"], cells.ok[(trial, name, "original")]["media_sha256"])
        for (trial, name, condition), row in cells.ok.items()
        if name == route and condition == "sham_self_fill" and (trial, name, "original") in cells.ok
    ]
    return bool(pairs) and all(a == b for a, b in pairs)


def load_shards(cell_paths: Sequence[Path], report_paths: Sequence[Path], model: str, expected: str, n_items: int = N_ITEMS) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """All shards of one full run, or nothing. Partners and B1 need every shard."""
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(report_paths)]
    rows = [json.loads(l) for p in sorted(cell_paths) for l in p.read_text(encoding="utf-8").splitlines() if l]
    if any(r.get("model") != model or r.get("smoke") for r in reports):
        raise SystemExit("Every report must be a full run of this model.")
    hashes = {r.get("design_sha256") for r in reports} | {r.get("design_sha256") for r in rows}
    if hashes != {expected}:
        raise SystemExit(f"Design hashes {sorted(map(str, hashes))} are not this design, {expected}.")
    shards = sorted(r.get("shard") for r in reports)
    if len({r.get("n_shards") for r in reports}) != 1 or shards != list(range(reports[0]["n_shards"])):
        raise SystemExit(f"Shards {shards} are not one complete set.")
    keys = [(r["trial_id"], r["input_name"], r["condition"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise SystemExit("Duplicate cells across shards.")
    trials = {r["trial_id"] for r in rows}
    if len(trials) != n_items:
        raise SystemExit(f"Expected {n_items} items, found {len(trials)}.")
    expected_cells = {(t, r, c) for t in trials for r in {k[1] for k in keys} for c in INPUTS[r]["conditions"]}
    if expected_cells - set(keys):
        raise SystemExit(f"{len(expected_cells - set(keys))} cells are missing. Failed cells must still be present.")
    return rows, reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--cells", type=Path, nargs="+", required=True, help="Every shard's cells file.")
    parser.add_argument("--run_report", type=Path, nargs="+", required=True, help="Every shard's report.")
    parser.add_argument("--lookup", type=Path, default=Path("data/eu_emotions_human_entropy.json"))
    parser.add_argument("--geometry", type=Path, default=Path("results/stimulus_interventions/geometry.json"))
    parser.add_argument("--reproduction", type=Path, default=Path("research/reproduction"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--a2_frozen", action="store_true")
    args = parser.parse_args()
    if not args.a2_frozen:
        raise SystemExit("A2 is not frozen. No analysis runs on real cells before that.")

    from paper_v2.option_score import vision_mismatch_partners

    run.check_geometry(args.geometry)
    geometry = json.loads(args.geometry.read_text(encoding="utf-8"))["items"]
    design_sha256 = run.design_hash()
    rows, reports = load_shards(args.cells, args.run_report, args.model, design_sha256)
    cells = Cells(rows)
    actors = sorted({r["actor"] for r in cells.items.values()})
    human = {k: v["human_response_distribution"] for k, v in json.loads(args.lookup.read_text())["trials"].items()}
    ids = sorted(cells.items)
    partners = vision_mismatch_partners(
        ids, [cells.items[i]["label"] for i in ids], [cells.items[i]["actor"] for i in ids], seed=42
    )
    baselines = {(b["input_name"], b["condition"]): b for b in reports[0].get("baselines") or []}
    out: Dict[str, Any] = {
        "model": args.model,
        "design_sha256": design_sha256,
        "n_items": len(ids),
        "failed_cells": len(cells.failed),
        "failures_by_route_condition_label_reason": failure_table(cells),
        "shuffle_nuisance_flip_rate": shuffle_nuisance(cells),
    }
    weak = [t for t, i in cells.items.items() if i.get("weak_label")]
    mode_wrong = [t for t, i in cells.items.items() if i["label"] != i["human_modal_label"]]
    variants = {
        "without_align_flagged": cells.without(align_flagged),
        "without_donor_is_sent_frame": cells.without(donor_is_sent_frame),
        "without_own_neutral_donor": cells.without(own_neutral_donor),
        "canonical_word_only": Cells(canonical_rows(rows)),
    }
    for route in ("single_face", "video_face"):
        free = baselines.get((route, "image_free"), {}).get("family_logprob")
        delivered = delivery_rows(reports, route)
        repeat = max([d["repeat_forward_max_abs"] for d in delivered if "failed" not in d], default=float("nan"))
        gate = gate_b(cells, route, partners, free, repeat)
        margin = MARGIN_FRACTION * gate["mismatch_margin_basis"]
        outside = outside_gates(args.reproduction, args.model, route)
        gates = {
            "rq0": outside["rq0"],
            "g2_g4": outside["g2_g4"],
            "gate_b": gate["pass"],
            "delivery": outside["delivery_file"] and delivery_passed(reports, route),
            "sham": sham_passed(cells, route),
        }
        across_values = [d["tokens_across_view"] for d in delivered if "failed" not in d]
        across = float(np.median(across_values)) if across_values else None
        frame_tokens = frame_tokens_by_clip(cells.items, across, geometry)
        small = below_token_floor(cells.items, across, geometry)
        common = {"frame_tokens": frame_tokens}
        out[route] = {
            "gates": {**gates, "a1": outside["a1"], "a3": outside["a3"]},
            "gate_b": gate,
            "margin": margin,
            "results": analyse_route(cells, route, actors, gates, margin, **common),
            "human_distribution": human_tables(cells, route, human),
            "flip_rates": flip_rates(cells, route),
            "descriptive_changes": descriptive_changes(cells, route, actors),
            "share_excluded_under_8_tokens": len(small) / len(ids),
            "sensitivity": {
                "without_smoke_items": analyse_route(cells, route, actors, gates, margin, exclude=SMOKE_ITEMS, **common),
                "without_weak_labels": analyse_route(cells, route, actors, gates, margin, exclude=weak, **common),
                "without_mode_not_target": analyse_route(cells, route, actors, gates, margin, exclude=mode_wrong, **common),
                "clips_at_8_tokens_or_more": analyse_route(cells, route, actors, gates, margin, exclude=small, **common),
                **{k: analyse_route(v, route, actors, gates, margin, **common) for k, v in variants.items()},
            },
        }
    args.output.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
