"""
Analysis of draft amendment A6 (RQ5, shared errors on CREMA-D). See research/PREREG_A6_DRAFT.md.

Reads the cells written by scripts/a6_run.py. Per model, never pooled. Each stage is run on
its own, and a stage reads only what the stage before it locked. scripts/a6_recovery.py runs
this exact code on synthetic observers. Statistics come from frozen A2 and A3 code, imported
unchanged: si.signflip_maxt, an.actor_interval, an.rq0_status, an.outside_gates, an.gate_b,
fiv.read_cells, fiv.decide. Nothing here reads a model output unless asked to by main, and
main refuses until every PENDING pin is filled and --a6_frozen is passed.
"""

from __future__ import annotations

import argparse
import json
from itertools import permutations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest, spearmanr

from scripts import a6_design as d
from scripts import a6_run as ar
from scripts import analyse_stimulus_interventions as an
from scripts import face_input_validity as fiv
from scripts import stimulus_interventions as si

D = d.DESIGN
OPTIONS: Tuple[str, ...] = tuple(D["options"])
FAMILIES: Tuple[str, ...] = tuple(D["family_map"][o] for o in OPTIONS)
INDEX = {label: i for i, label in enumerate(OPTIONS)}
PAIRS: Tuple[Tuple[str, str], ...] = tuple((p[0], p[1]) for p in D["shared_family"])
PAIR_NAMES = tuple("-".join(p) for p in PAIRS)
ALPHA = float(D["alpha"])
NO_EMOTION = "No Emotion"
CONTRASTS = d.contrasts(D)
CONDITIONS = tuple(D["conditions"])
EDIT_ROWS = tuple(D["edit_order_rows"])
ORIGINAL = CONDITIONS.index("original")
GATE_KEYS = ("rq0", "g2_g4", "delivery", "geometry")
DIRECTIONS = (f"{D['target_pair'][0]}>{D['target_pair'][1]}", f"{D['target_pair'][1]}>{D['target_pair'][0]}")
DIRECTION_OF = {"C1": DIRECTIONS[0], "C2": DIRECTIONS[1]}
FULL_SETTINGS = {"flips": int(D["signflip_draws"]), "boots": int(D["bootstrap_draws"]), "seed": int(D["seed"])}

NOT_TESTED = "Not tested"
NO_FACE_USE = "No face use on CREMA-D"
NOT_SHARED = "Not shared"
OPTION_DEPENDENT = "Option-dependent"
SAME_CUE = "Same cue"
ONE_DIRECTION = "Same cue, one direction"
OPPOSES = "Opposes the stated cue"
NOT_THIS_CUE = "Not this cue"
NO_TESTED_FEATURE = "No tested feature"
INCONCLUSIVE = "Inconclusive"
VERDICTS = (NOT_TESTED, NO_FACE_USE, NOT_SHARED, OPTION_DEPENDENT, SAME_CUE, ONE_DIRECTION, OPPOSES,
            NOT_THIS_CUE, NO_TESTED_FEATURE, INCONCLUSIVE)
SCOPE = "This model, these 91 actors, this clip set, single apex frames."
REASON_CODES = ("shared_unresolved", "option_unresolved", "cue_unresolved", "sham_failed",
                "region_too_small", "dose_failed")


class Cells:
    """Stage 1 and Stage 3 cells of one model as arrays. A failed cell stays NaN and is counted."""

    def __init__(self, items: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> None:
        self.items = sorted(items, key=lambda i: i["trial_id"])
        self.ids = [i["trial_id"] for i in self.items]
        self.index = {t: n for n, t in enumerate(self.ids)}
        self.actors = sorted({i["actor"] for i in self.items})
        actor_index = {a: n for n, a in enumerate(self.actors)}
        self.actor = np.array([actor_index[i["actor"]] for i in self.items])
        self.emo = np.array([INDEX[i["label"]] for i in self.items])
        n, k = len(self.items), len(OPTIONS)
        self.mcq1 = np.full((n, k, k), np.nan)
        self.mcq3 = np.full((n, len(CONDITIONS), len(EDIT_ROWS), k), np.nan)
        self.blind1 = np.full((n, k), np.nan)
        self.blind3 = np.full((n, len(D["blind_conditions"]), k), np.nan)
        self.bridge = np.full((n, len(EDIT_ROWS), k), np.nan)
        self.span = np.full((n, len(CONDITIONS)), np.nan)
        self.flagged = np.zeros((n, len(CONDITIONS), len(EDIT_ROWS)), dtype=bool)
        self.media: Dict[Tuple[str, int, str], str] = {}
        self.failed: List[Mapping[str, Any]] = []
        self.ingest(rows)

    def ingest(self, rows: Sequence[Mapping[str, Any]]) -> None:
        condition_index = {c: i for i, c in enumerate(CONDITIONS)}
        blind_index = {c: i for i, c in enumerate(D["blind_conditions"])}
        edit_index = {r: i for i, r in enumerate(EDIT_ROWS)}
        orders = [list(o) for o in D["orders"]]
        positions = [[INDEX[o] for o in order] for order in orders]
        for row in rows:
            if row["status"] != "ok":
                self.failed.append(row)
                continue
            i = self.index[row["trial_id"]]
            stage, route, readout, condition = row["stage"], row["route"], row["readout"], row["condition"]
            if readout == "blind":
                scores = row["blind"]["family_logprob"]
                values = [scores[f] for f in FAMILIES]
                if stage == 1:
                    self.blind1[i] = values
                else:
                    self.blind3[i, blind_index[condition]] = values
                continue
            order = row["order"]
            if row["options"] != orders[order]:
                raise SystemExit(f"{row['trial_id']}: options are not order {order}.")
            target = None
            if route == D["bridge_route"]:
                target = self.bridge[i, edit_index[order]]
            elif stage == 1:
                target = self.mcq1[i, order]
            else:
                c = condition_index[condition]
                target = self.mcq3[i, c, edit_index[order]]
                edit = row.get("edit") or {}
                if "span_view_fraction" in edit:
                    self.span[i, c] = edit["span_view_fraction"]
                self.flagged[i, c, edit_index[order]] = bool(edit.get("align_flagged"))
                if condition in ("original", "sham_self_fill"):
                    self.media[(row["trial_id"], order, condition)] = "|".join(row["media_sha256"])
            target[positions[order]] = row["joint_logprob"]

    def subset(self, keep: np.ndarray) -> "Cells":
        """The same clips with others blanked, so a sensitivity sees those clips as absent."""
        out = object.__new__(Cells)
        out.__dict__.update(self.__dict__)
        for name in ("mcq1", "mcq3", "blind1", "blind3", "bridge"):
            array = getattr(self, name).copy()
            array[~keep] = np.nan
            setattr(out, name, array)
        return out

    def actor_mean(self, values: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Mean over each actor's clips under mask. NaN clips leave; an empty actor is NaN."""
        ok = ~np.isnan(values)
        if mask is not None:
            ok = ok & mask
        total = np.bincount(self.actor[ok], weights=values[ok], minlength=len(self.actors))
        count = np.bincount(self.actor[ok], minlength=len(self.actors))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(count > 0, total / count, np.nan)


def top_score(logs: np.ndarray) -> np.ndarray:
    """The highest option score, NaN for an absent cell."""
    return np.where(np.isnan(logs), -np.inf, logs).max(-1) + np.where(np.isnan(logs).all(-1), np.nan, 0.0)


def softmax(logs: np.ndarray) -> np.ndarray:
    """Exact option probabilities: joint logprobs renormalised over the six options."""
    top = top_score(logs)[..., None]
    raw = np.exp(logs - top)
    return raw / raw.sum(-1, keepdims=True)


def error_share(p: np.ndarray, emo: np.ndarray, partner: int) -> np.ndarray:
    """The share of clip i's non-acted probability that goes to the partner. NaN if none."""
    rows = np.arange(len(p))
    errors = p.copy()
    errors[rows, emo] = 0.0
    total = errors.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(total > 0, p[:, partner] / total, np.nan)
    out[emo == partner] = np.nan
    return out


def share_without(p: np.ndarray, first: int, second: int, b: int) -> np.ndarray:
    """r_i(B without A, C): B's share of the probability outside A and C. NaN if none is left."""
    kept = [k for k in range(p.shape[-1]) if k not in (first, second)]
    total = p[..., kept].sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, p[..., b] / total, np.nan)


def directed_excess(cells: Any, p: np.ndarray, a: int, b: int) -> np.ndarray:
    """X_a(A to B) per actor, averaged over the comparison emotions C other than A and B.

    For each C, the same two options leave the denominator on clips of A and on clips of C,
    so the two means are shares of the same four options. A clip's own label is then out on
    both sides, and a softmax over additive option priors gives exactly zero at any accuracy.
    p has clips first and options last. With an order axis between, r is taken on each order
    and then averaged, so that a position effect stays additive inside each order. A clip
    missing an order is NaN.
    """
    gaps = []
    for c in range(p.shape[-1]):
        if c not in (a, b):
            r = share_without(p, a, c, b)
            r = r.mean(1) if r.ndim == 2 else r
            gaps.append(cells.actor_mean(r, cells.emo == a) - cells.actor_mean(r, cells.emo == c))
    stacked = np.array(gaps)
    present = ~np.isnan(stacked)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(present.any(0), np.where(present, stacked, 0.0).sum(0) / present.sum(0), np.nan)


def pair_excess(cells: Any, p: np.ndarray, a: int, b: int) -> np.ndarray:
    """X per actor for an unordered pair: the mean of the two directed excesses."""
    return (directed_excess(cells, p, a, b) + directed_excess(cells, p, b, a)) / 2.0


def pair_matrix(cells: Any, p: np.ndarray) -> np.ndarray:
    return np.column_stack([pair_excess(cells, p, INDEX[a], INDEX[b]) for a, b in PAIRS])


def directed_table(cells: Any, p: np.ndarray) -> Dict[str, float]:
    """X over actors for all 30 directed cells, in a fixed order."""
    return {f"{OPTIONS[a]}>{OPTIONS[b]}": float(np.nanmean(directed_excess(cells, p, a, b)))
            for a, b in permutations(range(len(OPTIONS)), 2)}


def order_probs(cells: Cells, orders: Sequence[int]) -> np.ndarray:
    """Option probabilities on the given orders: clips, orders, options."""
    return softmax(cells.mcq1[:, list(orders)])


def interval(values: np.ndarray, level: float, settings: Mapping[str, Any]) -> Optional[List[float]]:
    found = an.actor_interval(values, level, draws=settings["boots"], seed=settings["seed"])
    return None if found is None else [found[0], found[1]]


def flip_test(matrix: np.ndarray, directions: Sequence[int], settings: Mapping[str, Any]) -> Dict[str, Any]:
    return si.signflip_maxt(matrix, alternative=directions, draws=settings["flips"], seed=settings["seed"])


def inside(ci: Optional[Sequence[float]], margin: Optional[float]) -> bool:
    return bool(ci is not None and margin is not None and np.isfinite(margin) and margin > 0
                and ci[0] > -margin and ci[1] < margin)


# ---------------------------------------------------------------------------
# Stage 1: validity on CREMA-D
# ---------------------------------------------------------------------------


def correct_matrix(logs: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Target strictly highest, per clip and order. Ties are wrong, as in A3. NaN if absent."""
    is_top = logs == top_score(logs)[..., None]
    picked = np.take_along_axis(is_top, target[:, None, None], axis=2)[..., 0]
    out = (picked & (is_top.sum(-1) == 1)).astype(np.float64)
    out[np.isnan(logs).any(-1)] = np.nan
    return out


def partner_index(cells: Cells, partners: Mapping[str, str]) -> np.ndarray:
    return np.array([cells.index[partners[t]] for t in cells.ids])


def stage1(cells: Cells, partners: Mapping[str, str], settings: Mapping[str, Any], *, descriptive: bool = True) -> Dict[str, Any]:
    """RQ0's design on single_face. a_i and b_i over six orders, on the clip's and partner's face."""
    partner = partner_index(cells, partners)
    correct_real = correct_matrix(cells.mcq1, cells.emo)
    correct_partner = correct_matrix(cells.mcq1[partner], cells.emo)
    complete = ~np.isnan(correct_real).any(1) & ~np.isnan(correct_partner).any(1)
    a, b = correct_real.mean(1), correct_partner.mean(1)
    delta = (a - b)[complete]
    actors = cells.actor[complete]
    lost = 1.0 - complete.mean()
    out: Dict[str, Any] = {
        "n_items": int(complete.sum()),
        "n_items_expected": len(cells.ids),
        "share_items_lost": float(lost),
        "over_lost_item_limit": bool(lost > D["lost_item_limit"]),
        "accuracy_real": float(a[complete].mean()) if complete.any() else None,
        "accuracy_mismatch": float(b[complete].mean()) if complete.any() else None,
        "delta_real_minus_mismatch": float(delta.mean()) if delta.size else None,
        "delta_ci_actor_bootstrap": fiv.actor_bootstrap_ci(delta, actors, n_boot=settings["boots"], seed=settings["seed"]) if delta.size else [float("nan")] * 2,
        "p_delta_actor_sign_flip": fiv.actor_sign_flip_p(delta, actors, n_perm=settings["flips"], seed=settings["seed"]) if delta.size else 1.0,
    }
    if descriptive and delta.size:
        out["secondary"] = stage1_secondary(cells, correct_real, complete)
    return out


def stage1_secondary(cells: Cells, correct: np.ndarray, complete: np.ndarray) -> Dict[str, Any]:
    """Accuracy against 1/6, by target position, and against human p_target within emotion."""
    n_correct = int(np.nansum(correct[complete]))
    n_trials = int(complete.sum() * correct.shape[1])
    by_position: Dict[int, List[float]] = {}
    for row, order in enumerate(D["orders"]):
        for position, label in enumerate(order):
            mask = complete & (cells.emo == INDEX[label])
            by_position.setdefault(position, []).extend(correct[mask, row].tolist())
    model = softmax(cells.mcq1).mean(1)[np.arange(len(cells.ids)), cells.emo]
    people = np.array([i["human_p_target"] for i in cells.items], dtype=np.float64)
    within_model, within_people = model[complete].copy(), people[complete].copy()
    emo = cells.emo[complete]
    for label in np.unique(emo):
        within_model[emo == label] -= within_model[emo == label].mean()
        within_people[emo == label] -= within_people[emo == label].mean()
    return {
        "accuracy_real_vs_chance_pooled_p": float(binomtest(n_correct, n_trials, 1 / 6, alternative="greater").pvalue),
        "accuracy_by_target_position": {str(k): float(np.mean(v)) for k, v in sorted(by_position.items())},
        "mean_target_probability_real": float(model[complete].mean()),
        "within_emotion_r_ptarget_human": float(np.corrcoef(within_model, within_people)[0, 1]),
    }


def decide_stage1(summaries: Mapping[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """A3's rule, frozen in fiv.decide: Holm over the eligible models, and CI above 0."""
    return fiv.decide(dict(summaries)) if summaries else {}


# ---------------------------------------------------------------------------
# Stage 2: is the error shared, and is it the options?
# ---------------------------------------------------------------------------


def outcome_keep(cells: Any, contrast: Mapping[str, Any], reference: str, controls: Sequence[str]) -> np.ndarray:
    """Which options enter the reference of y, per clip.

    fixed       the primary: the acted label, No Emotion and the control emotions are all
                out, on every clip. That leaves the partner and Happy, so no clip's own
                label is in its reference, and the reference is the same on every clip.
    two_out     the acted label and No Emotion out, on every clip, as first drafted.
    own_out     each clip's own label and No Emotion out, so the reference varies by clip.
    with_none   fixed, with No Emotion kept in.
    """
    n, k = len(cells.emo), len(OPTIONS)
    keep = np.ones((n, k), dtype=bool)
    acted, none = INDEX[contrast["acted"]], INDEX[NO_EMOTION]
    if reference == "own_out":
        keep[np.arange(n), cells.emo] = False
        keep[:, none] = False
        keep[:, INDEX[contrast["partner"]]] = True
        return keep
    if reference not in ("fixed", "two_out", "with_none"):
        raise ValueError(reference)
    keep[:, acted] = False
    if reference != "with_none":
        keep[:, none] = False
    if reference in ("fixed", "with_none"):
        for label in D["control_emotions"]:
            keep[:, INDEX[label]] = False
    return keep


def outcome(logs: np.ndarray, partner: int, keep: np.ndarray) -> np.ndarray:
    """y: the CLR of the partner against the options kept. logs has clips first, options last."""
    extra = logs.ndim - 2
    weights = keep.reshape(keep.shape[0], *([1] * extra), keep.shape[1]).astype(np.float64)
    return logs[..., partner] - (logs * weights).sum(-1) / weights.sum(-1)


def contrast_clips(cells: Any, contrast: Mapping[str, Any], controls: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    acted = cells.emo == INDEX[contrast["acted"]]
    control = np.isin(cells.emo, [INDEX[c] for c in controls])
    return acted, control


def cue_margin(cells: Cells, partners: Mapping[str, str], contrast: Mapping[str, Any]) -> Optional[float]:
    """10% of the mean absolute change in y under full mismatch, on the contrast's clips.

    The face is the clip's partner, the prompt is unchanged, and the orders are the edit
    orders, so this is the same outcome on the same clips and orders as the contrast.
    """
    controls = D["control_emotions"]
    keep = outcome_keep(cells, contrast, D["outcome_reference"], controls)
    acted, control = contrast_clips(cells, contrast, controls)
    b = INDEX[contrast["partner"]]
    rows = list(EDIT_ROWS)
    own = outcome(cells.mcq1[:, rows], b, keep)
    other = outcome(cells.mcq1[partner_index(cells, partners)][:, rows], b, keep)
    change = np.abs(other - own).mean(1)
    means = cells.actor_mean(change, acted | control)
    if np.isnan(means).all():
        return None
    return float(D["cue_margin_fraction"]) * float(np.nanmean(means))


def stage2(cells: Cells, partners: Mapping[str, str], settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Model X on the family, O, X on non-adjacent orders, blind X, and the locked margins."""
    directions = [1] * len(PAIRS)
    all_orders = range(len(D["orders"]))
    x = pair_matrix(cells, order_probs(cells, all_orders))
    test = flip_test(x, directions, settings)
    blind = pair_matrix(cells, softmax(cells.blind1))
    blind_test = flip_test(blind, directions, settings)
    near = np.column_stack([pair_excess(cells, order_probs(cells, d.adjacent_rows(D, p)), INDEX[p[0]], INDEX[p[1]]) for p in PAIRS])
    far = np.column_stack([pair_excess(cells, order_probs(cells, d.far_rows(D, p)), INDEX[p[0]], INDEX[p[1]]) for p in PAIRS])
    option = near - far
    option_test = flip_test(option, directions, settings)
    far_test = flip_test(far, directions, settings)
    pairs: Dict[str, Dict[str, Any]] = {}
    for j, name in enumerate(PAIR_NAMES):
        pairs[name] = {
            "X": float(np.nanmean(x[:, j])),
            "ci95": interval(x[:, j], D["ci_decision"], settings),
            "ci90": interval(x[:, j], D["ci_equivalence"], settings),
            "p_raw": test["p_raw"][j],
            "p_maxt": test["p_maxt"][j],
            "X_adjacent": float(np.nanmean(near[:, j])),
            "X_far": float(np.nanmean(far[:, j])),
            "X_far_ci90": interval(far[:, j], D["ci_equivalence"], settings),
            "X_far_p_maxt": far_test["p_maxt"][j],
            "O": float(np.nanmean(option[:, j])),
            "O_p_maxt": option_test["p_maxt"][j],
            "blind_X": float(np.nanmean(blind[:, j])),
            "blind_p_maxt": blind_test["p_maxt"][j],
        }
    target = pairs[PAIR_NAMES[0]]
    margin = float(D["shared_margin"])
    x_passes = bool(target["p_maxt"] < ALPHA and target["X"] > 0)
    upper_below = bool(target["ci90"] is not None and target["ci90"][1] < margin)
    out: Dict[str, Any] = {
        "pairs": pairs,
        "x_passes": x_passes,
        "x_null": bool(not x_passes and upper_below),
        "o_significant": bool(target["O_p_maxt"] < ALPHA and target["O"] > 0),
        "far_inside_margin": bool(target["X_far_ci90"] is not None and target["X_far_ci90"][1] < margin),
        "far_passes": bool(target["X_far_p_maxt"] < ALPHA and target["X_far"] > 0),
        "blind_significant": bool(target["blind_p_maxt"] < ALPHA and target["blind_X"] > 0),
    }
    out["directions"] = direction_states(cells, settings)
    out["state"] = stage2_state(out)
    out["reaches_stage3"] = out["state"] == "stage3"
    out["cue_margin"] = {c["id"]: cue_margin(cells, partners, c) for c in CONTRASTS}
    return out


def direction_states(cells: Cells, settings: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Which directions of the target pair carry an excess, locked before Stage 3.

    carries     step-down p under alpha over the two directed cells, mean above zero.
    absent      not that, and the upper 90% limit is below shared_margin.
    unresolved  neither. It is kept, so doubt about a direction never lightens the null.
    """
    a, b = (INDEX[x] for x in D["target_pair"])
    p = order_probs(cells, range(len(D["orders"])))
    values = np.column_stack([directed_excess(cells, p, a, b), directed_excess(cells, p, b, a)])
    test = flip_test(values, [1, 1], settings)
    out: Dict[str, Dict[str, Any]] = {}
    for j, name in enumerate(DIRECTIONS):
        mean = float(np.nanmean(values[:, j]))
        ci90 = interval(values[:, j], D["ci_equivalence"], settings)
        carries = bool(test["p_maxt"][j] < ALPHA and mean > 0)
        absent = bool(not carries and ci90 is not None and ci90[1] < D["shared_margin"])
        out[name] = {"X": mean, "ci90": ci90, "p_maxt": test["p_maxt"][j],
                     "state": "carries" if carries else "absent" if absent else "unresolved"}
    return out


def stage2_state(s: Mapping[str, Any]) -> str:
    """Where Stage 2 leaves a model. Exhaustive, so every model gets one of these."""
    if not s["x_passes"]:
        return "not_shared" if s["x_null"] else "shared_unresolved"
    if not s["o_significant"] or s["far_passes"]:
        return "stage3"
    if s["far_inside_margin"] and not s["blind_significant"]:
        return "option_dependent"
    return "option_unresolved"


# ---------------------------------------------------------------------------
# Stage 3: cue-targeted test
# ---------------------------------------------------------------------------


def clip_delta(cells: Cells, contrast: Mapping[str, Any], condition: str, reference: str, controls: Sequence[str],
               rows: Optional[Sequence[int]] = None) -> np.ndarray:
    """delta(i, c): y under the condition minus y on the original, same order, over orders."""
    keep = outcome_keep(cells, contrast, reference, controls)
    b = INDEX[contrast["partner"]]
    c = CONDITIONS.index(condition)
    chosen = list(range(len(EDIT_ROWS))) if rows is None else list(rows)
    edited = outcome(cells.mcq3[:, c][:, chosen], b, keep)
    plain = outcome(cells.mcq3[:, ORIGINAL][:, chosen], b, keep)
    return (edited - plain).mean(1)


def double_difference(cells: Cells, contrast: Mapping[str, Any], condition: str, *, reference: Optional[str] = None,
                      controls: Optional[Sequence[str]] = None, rows: Optional[Sequence[int]] = None) -> np.ndarray:
    """e(a): the actor's mean delta on clips of A, minus on clips of the control emotions."""
    chosen = list(D["control_emotions"]) if controls is None else list(controls)
    delta = clip_delta(cells, contrast, condition, reference or D["outcome_reference"], chosen, rows)
    acted, control = contrast_clips(cells, contrast, chosen)
    return cells.actor_mean(delta, acted) - cells.actor_mean(delta, control)


def plain_change(cells: Cells, contrast: Mapping[str, Any], condition: str, *, reference: Optional[str] = None,
                 rows: Optional[Sequence[int]] = None) -> np.ndarray:
    """The simple change on clips of A, per actor, with no double difference."""
    controls = list(D["control_emotions"])
    delta = clip_delta(cells, contrast, condition, reference or D["outcome_reference"], controls, rows)
    return cells.actor_mean(delta, contrast_clips(cells, contrast, controls)[0])


def acted_change(cells: Cells, contrast: Mapping[str, Any], condition: str) -> float:
    """Dose: signed change in the acted label's CLR over six options, mean of actor means."""
    c = CONDITIONS.index(condition)
    acted = INDEX[contrast["acted"]]
    edited, plain = cells.mcq3[:, c], cells.mcq3[:, ORIGINAL]
    change = ((edited[..., acted] - edited.mean(-1)) - (plain[..., acted] - plain.mean(-1))).mean(1)
    means = cells.actor_mean(change, cells.emo == acted)
    return float(np.nanmean(means)) if not np.isnan(means).all() else float("nan")


def span_tokens(cells: Cells, contrast: Mapping[str, Any], tokens_across: Optional[float]) -> Optional[float]:
    """Median over the contrast's clips of the region's narrowest side, in tokens."""
    if tokens_across is None:
        return None
    acted, control = contrast_clips(cells, contrast, D["control_emotions"])
    fractions = cells.span[:, CONDITIONS.index(f"mask_{contrast['region']}_neutral")][acted | control]
    fractions = fractions[~np.isnan(fractions)]
    return float(np.median(fractions)) * float(tokens_across) if fractions.size else None


def sham_equal(cells: Cells) -> bool:
    """Every sham must send the same pixels as the original on the same clip and order."""
    pairs = [(v, cells.media.get((t, o, "original"))) for (t, o, c), v in cells.media.items() if c == "sham_self_fill"]
    return bool(pairs) and all(a == b for a, b in pairs)


def condition_family(region: str) -> Dict[str, str]:
    return {fill: f"mask_{region}_{fill}" for fill in si.FILL_TYPES}


def stage3(cells: Cells, margins: Mapping[str, Optional[float]], tokens_across: Optional[float],
           settings: Mapping[str, Any], *, reference: Optional[str] = None, controls: Optional[Sequence[str]] = None,
           rows: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """C1 to C4, neutral fill, step-down maxT, the fill rule, and each null condition."""
    kwargs = {"reference": reference, "controls": controls, "rows": rows}
    effects = np.column_stack([double_difference(cells, c, condition_family(c["region"])["neutral"], **kwargs) for c in CONTRASTS])
    directions = [c["direction"] for c in CONTRASTS]
    one_sided = flip_test(effects, directions, settings)
    two_sided = flip_test(effects, [0] * len(CONTRASTS), settings)
    wanted = np.isin(cells.emo, [INDEX[x] for x in D["stage3_emotions"]])
    lost = float(np.isnan(cells.mcq3[wanted]).any(axis=(1, 2, 3)).mean())
    out: Dict[str, Any] = {"contrasts": {}, "sham_equal": sham_equal(cells), "share_clips_lost": lost,
                           "over_lost_item_limit": bool(lost > D["lost_item_limit"])}
    for j, contrast in enumerate(CONTRASTS):
        values = effects[:, j]
        margin = margins.get(contrast["id"])
        direction = contrast["direction"]
        fills = condition_family(contrast["region"])
        mean = float(np.nanmean(values)) if not np.isnan(values).all() else float("nan")
        fill_values = {fill: double_difference(cells, contrast, name, **kwargs) for fill, name in fills.items() if fill != "neutral"}
        other_fills = {fill: {"mean": float(np.nanmean(v)), "ci95": interval(v, D["ci_decision"], settings)} for fill, v in fill_values.items()}
        plain = float(np.nanmean(plain_change(cells, contrast, fills["neutral"], reference=reference, rows=rows)))
        whole = double_difference(cells, contrast, "mask_all_neutral", **kwargs)
        whole_mean = float(np.nanmean(whole))
        whole_ci = interval(whole, D["ci_equivalence"], settings)
        keep_mid = double_difference(cells, contrast, "mask_uplow_neutral", **kwargs)
        dose = acted_change(cells, contrast, fills["blur"])
        span = span_tokens(cells, contrast, tokens_across)
        ci90, ci95 = interval(values, D["ci_equivalence"], settings), interval(values, D["ci_decision"], settings)
        significant_one = one_sided["p_maxt"][j] < ALPHA and mean * direction > 0
        significant_opposite = two_sided["p_maxt"][j] < ALPHA and mean * direction < 0
        fill_rule = all(beyond_zero(v["ci95"], direction) for v in other_fills.values()) and plain * direction > 0
        fill_rule_opposite = all(beyond_zero(v["ci95"], -direction) for v in other_fills.values()) and plain * direction < 0
        usable = margin is not None and np.isfinite(margin) and margin > 0
        conditions = {
            "c2_delivery": bool(span is not None and span >= D["min_span_tokens"]),
            "c3_dose": bool(usable and abs(dose) > margin),
            "c4_pair_can_move": bool(usable and abs(whole_mean) > margin),
            "c5_equivalence": bool(inside(ci90, margin) and two_sided["p_maxt"][j] >= ALPHA and one_sided["p_maxt"][j] >= ALPHA),
            "c6_reported": bool(ci95 is not None and np.isfinite(mean)),
        }
        row = {
            "mean": mean, "ci90": ci90, "ci95": ci95, "direction": direction, "margin": margin,
            "p_raw": one_sided["p_raw"][j], "p_maxt": one_sided["p_maxt"][j], "p_maxt_two_sided": two_sided["p_maxt"][j],
            "other_fills": other_fills, "plain_change": plain, "fill_rule": bool(fill_rule),
            "dose": dose, "span_tokens": span, "mask_all_mean": whole_mean, "mask_all_ci90": whole_ci,
            "mask_uplow_ci90": interval(keep_mid, D["ci_equivalence"], settings),
            "null_conditions": conditions,
            "match": bool(significant_one and fill_rule),
            "opposes": bool(significant_opposite and fill_rule_opposite),
            "significant": bool(two_sided["p_maxt"][j] < ALPHA or one_sided["p_maxt"][j] < ALPHA),
        }
        row["null"] = bool(all(conditions.values()) and out["sham_equal"] and not row["significant"])
        row["pair_inside_margin"] = inside(whole_ci, margin)
        row["status"] = "match" if row["match"] else "opposes" if row["opposes"] else "null" if row["null"] else "unresolved"
        out["contrasts"][contrast["id"]] = row
    return out


def beyond_zero(ci: Optional[Sequence[float]], direction: int) -> bool:
    """The interval lies wholly on the given side of zero."""
    return bool(ci is not None and (ci[0] > 0 if direction > 0 else ci[1] < 0))


def reading(stage: Mapping[str, Any]) -> str:
    """The table of C1 and C2 against C3 in the draft. A description, not a verdict."""
    rows = stage["contrasts"]
    pair = (rows["C1"], rows["C2"])
    if any(r["opposes"] for r in pair):
        return "The mid-face separates this pair in this model, against Yan"
    falls = all(r["match"] for r in pair)
    rises = rows["C3"]["match"]
    if falls and rises:
        return "Fits Yan and Carbon together: the nose carries the confusion, and the mouth separates"
    if falls:
        return "Fits Yan. Carbon's analogue not shown"
    if rises:
        return "Fits Carbon's analogue. Not Yan's cue"
    return "Neither"


# ---------------------------------------------------------------------------
# The decision table
# ---------------------------------------------------------------------------


class PrimeHolder:
    """One row per prime, in OPTIONS order, as if one actor had read each prime."""

    emo = np.arange(len(OPTIONS))
    actor = np.zeros(len(OPTIONS), dtype=int)
    actors = ["primes"]
    actor_mean = Cells.actor_mean


def text_ranks(primes: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
    """Rank of each directed cell of X_text, 1 being highest, from primes by orders by options."""
    if primes is None:
        return None
    p = softmax(primes)
    holder = PrimeHolder()
    values = {(a, b): float(directed_excess(holder, p, a, b)[0]) for a, b in permutations(range(len(OPTIONS)), 2)}
    ordered = sorted(values, key=lambda key: -values[key])
    a, b = (INDEX[x] for x in D["target_pair"])
    ranks = {key: ordered.index(key) + 1 for key in ((a, b), (b, a))}
    limit = int(D["text_neighbour_rank"])
    return {"values": {f"{OPTIONS[x]}>{OPTIONS[y]}": v for (x, y), v in values.items()},
            "target_ranks": {f"{OPTIONS[x]}>{OPTIONS[y]}": r for (x, y), r in ranks.items()},
            "label_neighbours_live": all(r <= limit for r in ranks.values())}


def gates_passed(gates: Mapping[str, Any]) -> bool:
    """Every named gate must be exactly True. A missing gate has not passed."""
    return all(gates.get(key) is True for key in GATE_KEYS)


def verdict(gates: Mapping[str, Any], first: Optional[Mapping[str, Any]], second: Optional[Mapping[str, Any]],
            third: Optional[Mapping[str, Any]], primes: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The decision table, read in order. Exactly one row applies, or the analysis stops."""

    def result(name: str, reason: Optional[str] = None, **riders: Any) -> Dict[str, Any]:
        return {"verdict": name, "reason_code": reason, "riders": riders, "scope": SCOPE}

    if not gates_passed(gates) or first is None or first.get("over_lost_item_limit") is not False:
        return result(NOT_TESTED)
    if first.get("vision_dependent") is not True:
        return result(NO_FACE_USE)
    if second is None:
        raise SystemExit("Stage 2 is missing for a model that passed Stage 1.")
    state = second["state"]
    if state == "not_shared":
        return result(NOT_SHARED)
    if state == "shared_unresolved":
        return result(INCONCLUSIVE, "shared_unresolved")
    if state == "option_dependent":
        return result(OPTION_DEPENDENT)
    if state == "option_unresolved":
        return result(INCONCLUSIVE, "option_unresolved")
    if state != "stage3":
        raise SystemExit(f"Stage 2 state {state} is not in the table.")
    if third is None:
        raise SystemExit("Stage 3 is missing for a model that reached it.")
    if third.get("over_lost_item_limit") is not False:
        return result(NOT_TESTED)
    if third["sham_equal"] is not True:
        return result(INCONCLUSIVE, "sham_failed")
    states = {cid: second["directions"][name]["state"] for cid, name in DIRECTION_OF.items()}
    active = [third["contrasts"][cid] for cid in ("C1", "C2") if states[cid] != "absent"]
    dropped = [third["contrasts"][cid] for cid in ("C1", "C2") if states[cid] == "absent"]
    used = {"directions_used": [DIRECTION_OF[cid] for cid in ("C1", "C2") if states[cid] != "absent"]}
    if not active:
        return result(INCONCLUSIVE, "cue_unresolved", **used)
    if len(active) == 2 and all(c["match"] for c in active):
        return result(SAME_CUE, **used)
    if any(c["opposes"] for c in active):
        return result(OPPOSES, **used)
    if any(c["match"] for c in active):
        return result(ONE_DIRECTION, **used)
    if not all(c["null_conditions"]["c2_delivery"] for c in active):
        return result(INCONCLUSIVE, "region_too_small", **used)
    if not any(c["significant"] for c in active) and not all(c["null_conditions"]["c3_dose"] for c in active):
        return result(INCONCLUSIVE, "dose_failed", **used)
    riders = {
        **used,
        "text_prime": primes,
        "label_neighbours_live": bool(primes and primes.get("label_neighbours_live")),
        "sufficient_not_necessary": all(inside(c["mask_uplow_ci90"], c["margin"]) for c in active),
    }
    # A significant contrast contradicts the Stage 2 reading that its direction has no excess.
    # It cannot count for a cue, and it stops a null too, so a null is never read past it.
    if any(c["significant"] for c in dropped):
        return result(INCONCLUSIVE, "cue_unresolved", **used)
    if all(c["null"] for c in active):
        return result(NOT_THIS_CUE, **riders)
    if not any(c["significant"] for c in active) and all(c["pair_inside_margin"] for c in active):
        return result(NO_TESTED_FEATURE, **riders)
    return result(INCONCLUSIVE, "cue_unresolved", **used)


# ---------------------------------------------------------------------------
# Descriptive rows and sensitivities: all reported, none changes a verdict
# ---------------------------------------------------------------------------


def pattern(cells: Cells, human: Mapping[str, Any]) -> Dict[str, Any]:
    """Rank correlation over the 30 directed cells, and error consistency (Geirhos et al.)."""
    model = directed_table(cells, order_probs(cells, range(len(D["orders"]))))
    people = human["directed"]
    keys = sorted(model)
    rho = float(spearmanr([model[k] for k in keys], [people[k]["X"] for k in keys])[0])
    p = softmax(cells.mcq1).mean(1)
    complete = ~np.isnan(p).any(1)
    model_right = p[complete].argmax(1) == cells.emo[complete]
    human_right = np.array([i["human_modal_correct"] for i in cells.items], dtype=bool)[complete]
    observed = float((model_right == human_right).mean())
    expected = float(model_right.mean() * human_right.mean() + (1 - model_right.mean()) * (1 - human_right.mean()))
    return {"directed": model, "rho_with_human": rho, "ceiling": human.get("split_half_rank_ceiling"),
            "error_consistency": (observed - expected) / (1 - expected) if expected < 1 else None,
            "agreement_observed": observed, "agreement_expected": expected}


def bridge(cells: Cells, settings: Mapping[str, Any]) -> Dict[str, Any]:
    """video_face and single_face X on the same orders. Descriptive only."""
    if np.isnan(cells.bridge).all():
        return {}
    video = softmax(cells.bridge)
    still = order_probs(cells, EDIT_ROWS)
    a, b = (INDEX[x] for x in D["target_pair"])
    x_video, x_still = pair_excess(cells, video, a, b), pair_excess(cells, still, a, b)
    return {"X_video_face": float(np.nanmean(x_video)), "ci95_video_face": interval(x_video, 0.95, settings),
            "X_single_face_same_orders": float(np.nanmean(x_still)), "ci95_single_face_same_orders": interval(x_still, 0.95, settings)}


def all_partners_b(cells: Cells) -> Dict[str, Any]:
    """Stage 1 with every other-emotion, other-actor face in place of the one partner."""
    correct = correct_matrix(cells.mcq1, cells.emo)
    a = correct.mean(1)
    out = np.full(len(cells.ids), np.nan)
    for i in range(len(cells.ids)):
        pool = (cells.emo != cells.emo[i]) & (cells.actor != cells.actor[i])
        hits = correct_matrix(cells.mcq1[pool], np.full(pool.sum(), cells.emo[i]))
        out[i] = np.nanmean(hits)
    delta = a - out
    keep = ~np.isnan(delta)
    return {"delta": float(delta[keep].mean()), "ci95": fiv.actor_bootstrap_ci(delta[keep], cells.actor[keep]),
            "p": fiv.actor_sign_flip_p(delta[keep], cells.actor[keep])}


def stage2_sensitivity(cells: Cells, settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Directed cells of the target pair, with one-sided tests and intervals."""
    a, b = (INDEX[x] for x in D["target_pair"])
    p = order_probs(cells, range(len(D["orders"])))
    out = {}
    for name, (x, y) in {"Anger>Disgust": (a, b), "Disgust>Anger": (b, a)}.items():
        values = directed_excess(cells, p, x, y)
        out[name] = {"X": float(np.nanmean(values)), "ci95": interval(values, 0.95, settings),
                     "n_actors_above_zero": int(np.nansum(values > 0))}
    return out


def stage3_sensitivities(cells: Cells, margins: Mapping[str, Optional[float]], tokens_across: Optional[float],
                         settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Outcome, control, order and clip variants of Stage 3, each with its own maxT."""

    def run(variant: Cells = cells, **kwargs: Any) -> Dict[str, Any]:
        result = stage3(variant, margins, tokens_across, settings, **kwargs)
        return {k: {f: v[f] for f in ("mean", "ci95", "p_maxt", "p_maxt_two_sided", "status")} for k, v in result["contrasts"].items()}

    out: Dict[str, Any] = {
        "reference_with_no_emotion": run(reference="with_none"),
        "reference_two_out_as_first_drafted": run(reference="two_out"),
        "reference_own_label_out": run(reference="own_out"),
        **{f"control_{c}_only": run(controls=[c]) for c in D["control_emotions"]},
        **{f"edit_order_{EDIT_ROWS[i]}_only": run(rows=[i]) for i in range(len(EDIT_ROWS))},
    }
    plain = {}
    equal_weight = {}
    for contrast in CONTRASTS:
        name = condition_family(contrast["region"])["neutral"]
        values = plain_change(cells, contrast, name)
        plain[contrast["id"]] = {"mean": float(np.nanmean(values)), "ci95": interval(values, 0.95, settings)}
        delta = clip_delta(cells, contrast, name, D["outcome_reference"], D["control_emotions"])
        acted, control = contrast_clips(cells, contrast, D["control_emotions"])
        equal_weight[contrast["id"]] = float(np.nanmean(delta[acted]) - np.nanmean(delta[control]))
    out["plain_change_on_acted_clips"] = plain
    out["equal_weight_clips"] = equal_weight
    flagged = cells.flagged
    if flagged.any():
        unflagged = object.__new__(Cells)
        unflagged.__dict__.update(cells.__dict__)
        unflagged.mcq3 = cells.mcq3.copy()
        unflagged.mcq3[flagged] = np.nan
        out["without_align_flagged"] = run(unflagged)
    return out


def clip_sensitivities(cells: Cells, partners: Mapping[str, str], margins: Optional[Mapping[str, Optional[float]]],
                       tokens_across: Optional[float], small: Sequence[str], settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Primary rows repeated on subsets of clips: human mode, the token floor, short decodes."""
    subsets = {
        "human_mode_is_intended": np.array([bool(i["human_modal_correct"]) for i in cells.items]),
        "at_or_above_token_floor": ~np.isin(cells.ids, list(small)),
        "without_short_decode_h264": ~np.isin(cells.ids, list(ar.SHORT_DECODE_CLIPS)),
    }
    sexes = sorted({i.get("sex") for i in cells.items})
    for sex in sexes:
        subsets[f"sex_{sex}_descriptive"] = np.array([i.get("sex") == sex for i in cells.items])
    out: Dict[str, Any] = {}
    for name, keep in subsets.items():
        variant = cells.subset(keep)
        p = order_probs(variant, range(len(D["orders"])))
        x = pair_matrix(variant, p)
        row: Dict[str, Any] = {"n_clips": int(keep.sum()), "pairs": {}}
        test = flip_test(x, [1] * len(PAIRS), settings)
        for j, pair in enumerate(PAIR_NAMES):
            row["pairs"][pair] = {"X": float(np.nanmean(x[:, j])), "ci95": interval(x[:, j], 0.95, settings), "p_maxt": test["p_maxt"][j]}
        if margins is not None and not np.isnan(variant.mcq3).all():
            result = stage3(variant, margins, tokens_across, settings)
            row["stage3"] = {k: {f: v[f] for f in ("mean", "ci95", "p_maxt", "status")} for k, v in result["contrasts"].items()}
        out[name] = row
    return out


# ---------------------------------------------------------------------------
# Loading cells: complete shards only
# ---------------------------------------------------------------------------


def expected_keys(items: Sequence[Mapping[str, Any]], stage: int) -> set:
    return {ar.cell_key(c) for item in items for c in ar.stage_plan(item, stage)}


def load_shards(cell_paths: Sequence[Path], report_paths: Sequence[Path], model: str, stage: int, expected: str,
                items: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """All shards of one full stage, or nothing. A bad final line is dropped by A3's rule."""
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(report_paths)]
    rows = [r for p in sorted(cell_paths) for r in fiv.read_cells(p)["rows"]]
    if any(r.get("model") != model or r.get("mechanics") for r in reports):
        raise SystemExit("Every report must be a full run of this model.")
    hashes = {r.get("design_sha256") for r in reports} | {r.get("design_sha256") for r in rows}
    if hashes != {expected}:
        raise SystemExit(f"Design hashes {sorted(map(str, hashes))} are not this design, {expected}.")
    shards = sorted(r.get("shard") for r in reports)
    counts = {r.get("n_shards") for r in reports}
    if len(counts) != 1 or shards != list(range(reports[0]["n_shards"])):
        raise SystemExit(f"Shards {shards} are not one complete set.")
    keys = [ar.cell_key(r) for r in rows]
    if len(keys) != len(set(keys)):
        raise SystemExit("Duplicate cells across shards.")
    gaps = expected_keys(items, stage) - set(keys)
    if gaps:
        raise SystemExit(f"{len(gaps)} cells are missing. Failed cells must still be present.")
    return rows, reports


def load_gates(reproduction: Path, model: str, baselines: Mapping[str, Any], geometry_ok: bool,
               a3_output_dir: Path = an.A3_OUTPUT_DIR) -> Dict[str, Any]:
    """A2's frozen gate loader, plus this run's delivery rows. Absent is not passed."""
    route = D["primary_route"]
    outside = an.outside_gates(reproduction, model, route, a3_output_dir)
    return {
        "rq0": outside["rq0"],
        "g2_g4": outside["g2_g4"],
        "delivery": bool(outside["delivery_file"] and an.delivery_passed([baselines], route)),
        "geometry": bool(geometry_ok),
        "a1": outside["a1"],
        "a3": outside["a3"],
    }


def geometry_passed(records: Mapping[str, Any], items: Sequence[Mapping[str, Any]]) -> bool:
    """Pinned already by check_geometry; here, too few clips lack a box or apex landmarks."""
    bad = 0
    for item in items:
        record = records.get(item["trial_id"]) or {}
        apex = record.get("apex")
        bad += bool(record.get("clip_failure") or record.get("box_failure") or apex is None
                    or (record.get("landmarks") or {}).get(str(apex)) is None)
    return bad / len(items) <= D["lost_item_limit"]


def tokens_across_view(baselines: Mapping[str, Any]) -> Optional[float]:
    rows = [r["tokens_across_view"] for r in an.delivery_rows([baselines], D["primary_route"]) if "failed" not in r]
    return float(np.median(rows)) if rows else None


def prime_table(baselines: Mapping[str, Any]) -> Optional[np.ndarray]:
    rows = [r for r in baselines.get("baselines", []) if r["condition"].startswith("prime_") and r["status"] == "ok"]
    if not rows:
        return None
    out = np.full((len(OPTIONS), len(D["orders"]), len(OPTIONS)), np.nan)
    for row in rows:
        out[INDEX[row["condition"][len("prime_"):]], row["order"], [INDEX[o] for o in row["options"]]] = row["joint_logprob"]
    return out


def gate_b_on_six(rows: Sequence[Mapping[str, Any]], partners: Mapping[str, str], baselines: Mapping[str, Any]) -> Dict[str, Any]:
    """Gate B's B1 to B5 by the frozen rules, on CREMA-D's six families. Reported, gates nothing."""
    route = D["primary_route"]
    family = dict(D["family_map"])
    six = list(FAMILIES)

    def narrow(scores: Mapping[str, float]) -> Dict[str, float]:
        return {f: scores[f] for f in six}

    pseudo = []
    for row in rows:
        if row["readout"] != "blind" or row["stage"] != 1 or row["status"] != "ok":
            continue
        scores = narrow(row["blind"]["family_logprob"])
        pseudo.append({
            "trial_id": row["trial_id"], "input_name": route, "condition": "original", "status": "ok",
            "actor": row["actor"], "label": family[row["label"]], "human_modal_label": family[row["human_modal_label"]],
            "blind": {"family_logprob": scores, "lexicon_mass": float(sum(np.exp(v) for v in scores.values()))},
        })
    free = next((r["family_logprob"] for r in baselines.get("baselines", []) if r.get("readout") == "blind" and r["condition"] == "image_free" and r["status"] == "ok"), None)
    delivered = [r for r in an.delivery_rows([baselines], route) if "failed" not in r]
    repeat = max([r["repeat_forward_max_abs"] for r in delivered], default=float("nan"))
    return an.gate_b(an.Cells(pseudo), route, dict(partners), narrow(free) if free else None, repeat)


def failure_table(cells: Cells) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in cells.failed:
        key = "|".join(str(row.get(k)) for k in ("route", "readout", "condition", "label", "reason"))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# Command line, for the real run after the freeze
# ---------------------------------------------------------------------------


def stem(args: argparse.Namespace, stage: int) -> Tuple[List[Path], List[Path]]:
    pattern_cells = f"cells_{{}}_stage{stage}*.jsonl"
    pattern_run = f"run_{{}}_stage{stage}*.json"
    return (sorted(args.results.glob(pattern_cells.format(args.model))), sorted(args.results.glob(pattern_run.format(args.model))))


def check_ready(args: argparse.Namespace) -> None:
    if not args.a6_frozen:
        raise SystemExit("A6 is not frozen. No analysis runs on real cells before that.")
    d.check_frozen(D)
    d.check_pins(D)
    d.check_human_target(D)
    d.check_geometry(D, args.geometry)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stage", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument("--models", nargs="+", default=list(D["models"]))
    parser.add_argument("--model", help="Stage 2 and 3: one model at a time.")
    parser.add_argument("--results", type=Path, default=d.OUTPUT_DIR)
    parser.add_argument("--geometry", type=Path, default=d.GEOMETRY)
    parser.add_argument("--reproduction", type=Path, default=Path("research/reproduction"))
    parser.add_argument("--a6_frozen", action="store_true")
    args = parser.parse_args()
    check_ready(args)
    items = ar.load_items(d.SELECTION, Path("."))
    partners = ar.partners_for(items)
    human = json.loads(d.HUMAN_TARGET.read_text(encoding="utf-8"))
    design = d.block_hash()
    geometry = json.loads(args.geometry.read_text(encoding="utf-8"))
    geometry_ok = geometry_passed(geometry["items"], items)

    def read(model: str, stage: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        args.model = model
        cell_paths, report_paths = stem(args, stage)
        rows, _reports = load_shards(cell_paths, report_paths, model, stage, design, items)
        return rows, json.loads((args.results / f"baselines_{model}.json").read_text(encoding="utf-8"))

    def write(name: str, out: Dict[str, Any]) -> None:
        out["design_sha256"] = design
        path = args.results / name
        if path.exists():
            raise SystemExit(f"{path} is already written and is not overwritten.")
        path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(path)

    if args.stage == 1:
        summaries: Dict[str, Dict[str, Any]] = {}
        all_gates: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}
        for model in args.models:
            rows, baselines = read(model, 1)
            cells = Cells(items, rows)
            gates = load_gates(args.reproduction, model, baselines, geometry_ok)
            all_gates[model] = gates
            result = stage1(cells, partners, FULL_SETTINGS)
            extra[model] = {"gate_b_descriptive": gate_b_on_six(rows, partners, baselines),
                            "all_partner_faces": all_partners_b(cells),
                            "failures": failure_table(cells),
                            "human_ceiling_p_target_within_emotion": json.loads(
                                (args.reproduction / "crema_d_reliability.json").read_text(encoding="utf-8")
                            )["selected_clips"]["p_target_within_emotion"]}
            if gates_passed(gates) and not result["over_lost_item_limit"]:
                summaries[model] = result
            else:
                extra[model]["summary_not_eligible"] = result
        decided = decide_stage1(summaries)
        write("a6_stage1.json", {"gates": all_gates, "decided": decided, "descriptive": extra,
                                 "eligible": sorted(summaries)})
        return

    if args.model is None:
        raise SystemExit("--model is required for Stages 2 and 3.")
    model = args.model
    first_doc = json.loads((args.results / "a6_stage1.json").read_text(encoding="utf-8"))
    if first_doc["design_sha256"] != design:
        raise SystemExit("Stage 1 output is from another design.")
    first = first_doc["decided"].get(model)
    rows, baselines = read(model, 1)
    if args.stage == 2:
        if first is None or first.get("vision_dependent") is not True:
            raise SystemExit(f"{model} did not pass Stage 1, so it has no Stage 2.")
        cells = Cells(items, rows)
        second = stage2(cells, partners, FULL_SETTINGS)
        small = an.below_token_floor({i["trial_id"]: i for i in items}, tokens_across_view(baselines), geometry["items"])
        write(f"a6_stage2_{model}.json", {
            "model": model, "stage2": second, "pattern": pattern(cells, human), "bridge": bridge(cells, FULL_SETTINGS),
            "text_primes": text_ranks(prime_table(baselines)), "directed_target": stage2_sensitivity(cells, FULL_SETTINGS),
            "clip_sensitivities": clip_sensitivities(cells, partners, None, None, small, FULL_SETTINGS),
            "verdict_if_stopped_here": None if second["reaches_stage3"] else verdict(first_doc["gates"][model], first, second, None),
        })
        return
    second_doc = json.loads((args.results / f"a6_stage2_{model}.json").read_text(encoding="utf-8"))
    if second_doc["design_sha256"] != design or second_doc["stage2"]["reaches_stage3"] is not True:
        raise SystemExit(f"{model} has no locked Stage 2 that reaches Stage 3.")
    second = second_doc["stage2"]
    third_rows, _ = read(model, 3)
    cells = Cells(items, rows + third_rows)
    across = tokens_across_view(baselines)
    third = stage3(cells, second["cue_margin"], across, FULL_SETTINGS)
    small = an.below_token_floor({i["trial_id"]: i for i in items}, across, geometry["items"])
    primes = text_ranks(prime_table(baselines))
    write(f"a6_stage3_{model}.json", {
        "model": model, "stage3": third, "reading": reading(third),
        "verdict": verdict(first_doc["gates"][model], first, second, third, primes),
        "repeat_check_max_abs": float(np.nanmax(np.abs(cells.mcq3[:, ORIGINAL] - cells.mcq1[:, list(EDIT_ROWS)]))),
        "sensitivities": stage3_sensitivities(cells, second["cue_margin"], across, FULL_SETTINGS),
        "clip_sensitivities": clip_sensitivities(cells, partners, second["cue_margin"], across, small, FULL_SETTINGS),
    })


if __name__ == "__main__":
    main()
