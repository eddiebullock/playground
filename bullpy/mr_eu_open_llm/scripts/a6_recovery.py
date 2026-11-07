"""
Recovery for draft amendment A6: synthetic observers with known truth through the real analysis.

Each observer writes cells in the format of scripts/a6_run.py, over the real 819 clips, 91
actors, partners and orders, with made-up option scores. Those rows go through Cells, stage1,
stage2, stage3 and verdict in scripts/a6_analyse.py. No model, clip or result is read. The noise
sizes are A2's recovery noise, not chosen for A6, and strengths are multiples of the human
directed rates in research/reproduction/a6_human_target.json. The observers are in the draft,
plus adversarial ones added in the audit. The pass rule is in the draft and the design block.

Usage:
    python -m scripts.a6_recovery --workers 3
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np  # noqa: E402

from scripts import a6_analyse as an  # noqa: E402
from scripts import a6_design as d  # noqa: E402
from scripts import a6_run as ar  # noqa: E402
from scripts import face_input_validity as fiv  # noqa: E402

D = d.DESIGN
R = D["recovery"]
OPTIONS = an.OPTIONS
K = len(OPTIONS)
INDEX = an.INDEX
A, B = (INDEX[x] for x in D["target_pair"])
NONE = INDEX["No Emotion"]
REGIONS = ("upper", "mid", "lower")
SETTINGS = {"flips": int(R["signflip_draws"]), "boots": int(R["bootstrap_draws"]), "seed": int(D["seed"])}
TOKENS_ACROSS = 16.0
SPAN_FRACTION = {"upper": 0.2, "mid": 0.2, "lower": 0.2, "midlower": 0.2, "uplow": 0.2, "all": 0.2}
TOO_SMALL = 0.01

# Expected verdict by observer. "forbidden" observers are judged on what they must not reach.
EXPECTED: Dict[str, Dict[str, Any]] = {
    "prior_only": {"expected": an.NO_FACE_USE},
    "no_sharing": {"expected": an.NOT_SHARED},
    "adjacency": {"expected": an.OPTION_DEPENDENT},
    "same_cue": {"expected": an.SAME_CUE, "no_false_null": True},
    "one_way_cue": {"expected": an.ONE_DIRECTION, "no_false_null": True},
    "reverse": {"expected": an.OPPOSES},
    "other_cue": {"expected": an.NOT_THIS_CUE},
    "one_way_other_cue": {"expected": an.NOT_THIS_CUE},
    "spread": {"expected": an.NOT_THIS_CUE, "rider": "label_neighbours_live"},
    "global_face": {"expected": an.NO_TESTED_FEATURE},
    "human_like": {"expected": an.SAME_CUE},
    "generic_mask": {"forbidden": [an.SAME_CUE]},
    "generic_pull": {"forbidden_match": True},
    # Added in the audit.
    "one_way_weak_cue": {"forbidden": [an.NOT_THIS_CUE]},
    "split_cue": {"forbidden": [an.NOT_THIS_CUE]},
    "signal_loss": {"forbidden": [an.SAME_CUE, an.ONE_DIRECTION]},
    "partner_pull": {"forbidden": [an.SAME_CUE, an.ONE_DIRECTION, an.OPPOSES]},
    "fill_artefact": {"forbidden": [an.SAME_CUE, an.ONE_DIRECTION]},
    "fill_artefact_up": {"forbidden": [an.OPPOSES]},
    "option_prior": {"expected": an.NOT_SHARED},
    "position_bias": {"expected": an.NOT_SHARED},
    "adjacency_all": {"expected": an.NOT_SHARED},
}
BOUNDARY = ("boundary_shared", "boundary_cue")
SCENARIOS = {
    "gate_failed": {"expected": an.NOT_TESTED, "reason": None},
    "clips_lost": {"expected": an.NOT_TESTED, "reason": None},
    "sham_differs": {"expected": an.INCONCLUSIVE, "reason": "sham_failed"},
    "region_small": {"expected": an.INCONCLUSIVE, "reason": "region_too_small"},
    "no_dose": {"expected": an.INCONCLUSIVE, "reason": "dose_failed"},
    "part_adjacent": {"expected": an.INCONCLUSIVE, "reason": "option_unresolved"},
    "actors_split": {"expected": an.INCONCLUSIVE, "reason": "shared_unresolved"},
}
GATES_OK = {key: True for key in an.GATE_KEYS}


def boost_for(prior: np.ndarray, a: int, b: int, target: float) -> float:
    """Nats added to B on clips of A so that X(A to B) is target, by bisection with no noise."""

    def excess(g: float) -> float:
        gaps = []
        for c in range(K):
            if c in (a, b):
                continue
            kept = [k for k in range(K) if k not in (a, c)]
            raw = np.exp(prior[kept])
            base = raw[kept.index(b)] / raw.sum()
            raw[kept.index(b)] *= np.exp(g)
            gaps.append(raw[kept.index(b)] / raw.sum() - base)
        return float(np.mean(gaps))

    low, high = -12.0, 12.0
    for _ in range(50):
        middle = (low + high) / 2.0
        low, high = (middle, high) if excess(middle) < target else (low, middle)
    return (low + high) / 2.0


class World:
    """The real clips, partners and orders, built once per worker."""

    def __init__(self) -> None:
        self.items = ar.load_items(d.SELECTION, Path("."))
        self.partners = ar.partners_for(self.items)
        self.cells = an.Cells(self.items, [])
        self.human = json.loads(d.HUMAN_TARGET.read_text(encoding="utf-8"))["directed"]
        self.emo = self.cells.emo
        self.actor = self.cells.actor
        self.stage3 = np.flatnonzero(np.isin(self.emo, [INDEX[x] for x in D["stage3_emotions"]]))
        self.n = len(self.emo)
        self.plans = {1: [], 3: []}
        for item in self.cells.items:
            for stage in (1, 3):
                self.plans[stage].extend(c for c in ar.stage_plan(item, stage))
        self.position = np.array([[list(o).index(label) for label in OPTIONS] for o in D["orders"]])
        self.adjacent = np.abs(self.position[:, :, None] - self.position[:, None, :]) == 1


WORLD: Optional[World] = None


def world() -> World:
    global WORLD
    if WORLD is None:
        WORLD = World()
    return WORLD


def visible(condition: str) -> Tuple[frozenset, str, float]:
    """Regions left visible by a condition, its fill, and the share of the face it covers."""
    spec = d.condition_spec(condition)
    if spec["kind"] == "none" or spec["fill"] == "self":
        return frozenset(REGIONS), "none", 0.0
    region = spec["region"]
    hidden = {name for name in REGIONS if name in ar_regions(region)}
    return frozenset(REGIONS) - hidden, spec["fill"], float(R["mask_area"][region])


def ar_regions(region: str) -> Tuple[str, ...]:
    return {"upper": ("upper",), "mid": ("mid",), "lower": ("lower",), "midlower": ("mid", "lower"),
            "uplow": ("upper", "lower"), "all": ("upper", "mid", "lower")}[region]


class Observer:
    """Additive scores: prior, noise, acted evidence by region, and pair excess by region."""

    def __init__(self, name: str, strength: float, seed: Sequence[int], noise: float = 1.0) -> None:
        self.name, self.strength = name, strength
        self.rng = np.random.default_rng(list(seed))
        self.w = world()
        self.noise = noise
        self.prior = self.rng.normal(0, R["prior_sd"], K)
        if name == "option_prior":
            self.prior[[A, B]] += 1.5
        rates = {(a, b): strength * self.w.human[f"{OPTIONS[a]}>{OPTIONS[b]}"]["X"] for a, b in ((A, B), (B, A))}
        self.gains = {pair: boost_for(self.prior, pair[0], pair[1], rate) for pair, rate in rates.items()}
        self.evidence = 0.0 if name == "prior_only" else float(R["evidence"])
        self.share = dict(R["evidence_share"])
        if name == "no_dose":
            self.share = {"upper": 0.0, "mid": 0.0, "lower": 0.0, "rest": 1.0}
        self.pair: Dict[Tuple[int, int], Dict[str, float]] = {pair: {} for pair in rates}
        self.adjacency = np.zeros((K, K))
        self.position_effect = np.zeros(K)
        self.actor_sign = np.ones(self.w.n)
        self.blind_pair: Dict[Tuple[int, int], float] = {}
        self.build(name)
        n = self.w.n
        scale = noise
        self.base = (self.prior + scale * self.rng.normal(0, R["actor_sd"], (len(self.w.cells.actors), K))[self.w.actor]
                     + scale * self.rng.normal(0, R["clip_sd"], (n, K)))
        self.order_noise = scale * self.rng.normal(0, R["order_sd"], (n, len(D["orders"]), K))

    def build(self, name: str) -> None:
        ab, ba = (A, B), (B, A)
        g, h = self.gains[ab], self.gains[ba]
        if name in ("same_cue",):
            self.pair = {ab: {"mid": g}, ba: {"mid": h}}
        elif name == "one_way_cue":
            self.pair = {ab: {"mid": g}, ba: {}}
        elif name == "one_way_other_cue":
            self.pair = {ab: {"upper": g}, ba: {}}
        elif name == "one_way_weak_cue":
            # Only the weaker human direction has an excess, and the mid-face carries it.
            self.pair = {ab: {}, ba: {"mid": h}}
        elif name == "split_cue":
            # The strong direction has another cue, and the weak one has the stated cue. If the
            # weak one were wrongly dropped, this would read as a false Not this cue.
            self.pair = {ab: {"upper": g}, ba: {"mid": h}}
        elif name == "reverse":
            self.pair = {ab: {"rest": 2 * g, "mid": -g}, ba: {"rest": 2 * h, "mid": -h}}
        elif name == "other_cue":
            self.pair = {ab: {"upper": g}, ba: {"upper": h}}
        elif name == "spread":
            self.pair = {ab: {"any": g}, ba: {"any": h}}
        elif name in ("global_face", "generic_mask", "generic_pull", "signal_loss", "partner_pull", "fill_artefact", "fill_artefact_up", "no_dose", "sham_differs", "gate_failed", "clips_lost"):
            self.pair = {ab: {"rest": g}, ba: {"rest": h}}
        elif name in ("region_small",):
            self.pair = {ab: {"rest": g}, ba: {"rest": h}}
        elif name == "human_like":
            lift = h * 2.0
            self.pair = {ab: {"mid": g, "rest": 0.5 * g, "upper": -0.5 * g},
                         ba: {"mid": h, "rest": h + lift, "lower": -h - lift}}
        elif name == "adjacency":
            third = boost_for(self.prior, A, B, min(0.7, 3 * self.strength * self.w.human["Anger>Disgust"]["X"]))
            back = boost_for(self.prior, B, A, min(0.7, 3 * self.strength * self.w.human["Disgust>Anger"]["X"]))
            self.adjacency[A, B], self.adjacency[B, A] = third, back
            self.pair = {ab: {}, ba: {}}
        elif name == "part_adjacent":
            # Adjacent in the MCQ, as Adjacency, while the blind readout shows the pair too.
            self.pair = {ab: {}, ba: {}}
            self.adjacency[A, B], self.adjacency[B, A] = 3 * g, 3 * h
            self.blind_pair = {ab: g, ba: h}
        elif name == "adjacency_all":
            self.adjacency[:] = 1.0
            self.pair = {ab: {}, ba: {}}
        elif name == "position_bias":
            self.position_effect = np.array([1.5, 0.75, 0.0, 0.0, -0.75, -1.5])
            self.pair = {ab: {}, ba: {}}
        elif name == "boundary_shared":
            # True X at the shared margin, in the human ratio of the two directions.
            forward, back = self.w.human["Anger>Disgust"]["X"], self.w.human["Disgust>Anger"]["X"]
            total = 2 * float(D["shared_margin"])
            self.pair = {ab: {"rest": boost_for(self.prior, A, B, total * forward / (forward + back))},
                         ba: {"rest": boost_for(self.prior, B, A, total * back / (forward + back))}}
        elif name == "actors_split":
            self.actor_sign = self.rng.choice((1.0, -1.0), len(self.w.cells.actors), p=(0.35, 0.65))[self.w.actor]
            self.pair = {ab: {"rest": 3.0}, ba: {"rest": 3.0}}
        elif name == "boundary_cue":
            self.pair = {ab: {"rest": g - 0.15, "mid": 0.15}, ba: {"rest": h - 0.15, "mid": 0.15}}
        elif name in ("prior_only", "no_sharing", "option_prior"):
            self.pair = {ab: {}, ba: {}}
        else:
            raise ValueError(name)

    def condition_shift(self, emo: np.ndarray, condition: str) -> np.ndarray:
        """The structured part of the scores of clips of emo under a condition, no noise."""
        seen, fill, area = visible(condition)
        count = len(emo)
        out = np.zeros((count, K))
        signs = self.actor_sign[self.w.stage3] if count != self.w.n else self.actor_sign
        kept = self.share["rest"] + sum(self.share[r] for r in seen)
        out[np.arange(count), emo] += self.evidence * kept
        for (a, b), parts in self.pair.items():
            total = parts.get("rest", 0.0) + sum(parts.get(r, 0.0) for r in seen)
            if seen:
                total += parts.get("any", 0.0)
            out[emo == a, b] += total * signs[emo == a]
        size = float(R["generic_nats"]) * area
        if area > 0:
            if self.name == "generic_mask":
                foils = np.ones((count, K), dtype=bool)
                foils[np.arange(count), emo] = False
                out[foils] -= size
                out[:, NONE] += 2.0 * size
            elif self.name == "generic_pull":
                out[:, B] += size * 3
            elif self.name == "partner_pull":
                out[:, A] += size * 3
            elif self.name == "signal_loss":
                out[np.arange(count), emo] -= size * 3
            elif self.name in ("fill_artefact", "fill_artefact_up") and fill == "neutral" and condition.startswith("mask_mid_"):
                sign = -1.0 if self.name == "fill_artefact" else 1.0
                out[emo == A, B] += sign * 0.6
                out[emo == B, A] += sign * 0.6
        return out

    def order_shift(self, row: int, emo: np.ndarray) -> np.ndarray:
        out = np.broadcast_to(self.position_effect[self.w.position[row]], (len(emo), K)).copy()
        if self.adjacency.any():
            near = self.w.adjacent[row]
            out += (self.adjacency * near)[emo]
        return out

    def stage1_scores(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        w = self.w
        original = self.base + self.condition_shift(w.emo, "original")
        mcq = np.stack([original + self.order_shift(row, w.emo) + self.order_noise[:, row] for row in range(len(D["orders"]))], axis=1)
        blind = original + self.noise * self.rng.normal(0, R["order_sd"], original.shape)
        for (a, b), gain in self.blind_pair.items():
            blind[w.emo == a, b] += gain
        bridge = mcq[:, list(an.EDIT_ROWS)] + self.noise * self.rng.normal(0, R["order_sd"], (w.n, len(an.EDIT_ROWS), K))
        return mcq, blind, bridge

    def stage3_scores(self, mcq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Scores of the Stage 3 clips by condition and edit order. The sham is the original."""
        w = self.w
        rows = list(an.EDIT_ROWS)
        clips = w.stage3
        emo = w.emo[clips]
        out = np.empty((len(clips), len(an.CONDITIONS), len(rows), K))
        blind = np.empty((len(clips), len(D["blind_conditions"]), K))
        names = list(D["blind_conditions"])
        for c, condition in enumerate(an.CONDITIONS):
            if condition in ("original", "sham_self_fill"):
                out[:, c] = mcq[clips][:, rows]
                continue
            shift = self.base[clips] + self.condition_shift(emo, condition) + self.noise * self.rng.normal(0, R["condition_sd"], (len(clips), K))
            for r, row in enumerate(rows):
                out[:, c, r] = shift + self.order_shift(row, emo) + self.order_noise[clips, row] + self.noise * self.rng.normal(0, R["condition_sd"], (len(clips), K))
            if condition in names:
                blind[:, names.index(condition)] = shift
        return out, blind

    def primes(self) -> np.ndarray:
        out = self.rng.normal(0, 1, (K, len(D["orders"]), K)) * 0.2 + self.rng.normal(0, 1, (K, 1, K))
        if self.name == "spread":
            out[A, :, B] += 4.0
            out[B, :, A] += 4.0
        return out


def rows_stage1(obs: Observer, mcq: np.ndarray, blind: np.ndarray, bridge: np.ndarray, *, lost: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
    w = obs.w
    edit = {r: i for i, r in enumerate(an.EDIT_ROWS)}
    rows = []
    for cell in w.plans[1]:
        i = w.cells.index[cell["trial_id"]]
        row = dict(cell)
        if lost is not None and lost[i]:
            row.update(status="failed", reason="no_face_detected")
        elif row["readout"] == "blind":
            row.update(status="ok", blind={"family_logprob": dict(zip(an.FAMILIES, blind[i].tolist()))})
        else:
            order = row["options"]
            source = bridge[i, edit[row["order"]]] if row["route"] == D["bridge_route"] else mcq[i, row["order"]]
            row.update(status="ok", joint_logprob=source[[INDEX[o] for o in order]].tolist())
        rows.append(row)
    return rows


def rows_stage3(obs: Observer, scores: np.ndarray, blind: np.ndarray, *, sham_differs: bool, span: float) -> List[Dict[str, Any]]:
    w = obs.w
    edit = {r: i for i, r in enumerate(an.EDIT_ROWS)}
    place = {int(c): n for n, c in enumerate(w.stage3)}
    conditions = {c: i for i, c in enumerate(an.CONDITIONS)}
    blinds = {c: i for i, c in enumerate(D["blind_conditions"])}
    rows = []
    for cell in w.plans[3]:
        i = place[w.cells.index[cell["trial_id"]]]
        row = dict(cell)
        condition = row["condition"]
        if row["readout"] == "blind":
            row.update(status="ok", blind={"family_logprob": dict(zip(an.FAMILIES, blind[i, blinds[condition]].tolist()))})
        else:
            order = row["options"]
            sha = "same"
            if condition == "sham_self_fill" and sham_differs and i == 0:
                sha = "different"
            row.update(status="ok", joint_logprob=scores[i, conditions[condition], edit[row["order"]]][[INDEX[o] for o in order]].tolist(),
                       media_sha256=[sha], edit={"span_view_fraction": span})
        rows.append(row)
    return rows


def one_replicate(task: Tuple[Any, ...]) -> Dict[str, Any]:
    """One observer, one strength, one replicate, through the real analysis.

    A fifth entry of True uses the draws of the real analysis, for the draw check.
    """
    name, strength, replicate, noise = task[:4]
    settings = an.FULL_SETTINGS if len(task) > 4 and task[4] else SETTINGS
    groups = sorted(set(EXPECTED) | set(BOUNDARY) | set(SCENARIOS))
    seed = [int(R["seed"]), groups.index(name), int(strength * 100), replicate, int(noise * 100)]
    obs = Observer(name, strength, seed, noise)
    w = obs.w
    mcq, blind, bridge = obs.stage1_scores()
    lost = None
    if name == "clips_lost":
        lost = np.zeros(w.n, dtype=bool)
        lost[obs.rng.choice(w.n, int(0.08 * w.n), replace=False)] = True
    first_rows = rows_stage1(obs, mcq, blind, bridge, lost=lost)
    cells = an.Cells(w.items, first_rows)
    first = an.stage1(cells, w.partners, settings, descriptive=False)
    # Holm's worst case across the two eligible models, as in A4's recovery.
    first["p_delta_holm"] = min(1.0, 2 * first["p_delta_actor_sign_flip"])
    first["vision_dependent"] = bool(first["p_delta_holm"] < an.ALPHA and first["delta_ci_actor_bootstrap"][0] > 0)
    gates = dict(GATES_OK)
    if name == "gate_failed":
        gates["rq0"] = False
    out: Dict[str, Any] = {"observer": name, "strength": strength, "noise": noise, "delta": first["delta_real_minus_mismatch"]}
    second = third = None
    if an.gates_passed(gates) and not first["over_lost_item_limit"] and first["vision_dependent"]:
        second = an.stage2(cells, w.partners, settings)
        target = second["pairs"][an.PAIR_NAMES[0]]
        out["direction_states"] = {k: v["state"] for k, v in second["directions"].items()}
        out.update(X=target["X"], O=target["O"], X_far=target["X_far"], state=second["state"], blind_X=target["blind_X"],
                   margin=second["cue_margin"]["C1"])
        if second["reaches_stage3"]:
            scores, blind3 = obs.stage3_scores(mcq)
            span = TOO_SMALL if name == "region_small" else SPAN_FRACTION["mid"]
            cells.ingest(rows_stage3(obs, scores, blind3, sham_differs=name == "sham_differs", span=span))
            third = an.stage3(cells, second["cue_margin"], TOKENS_ACROSS, settings)
            out["contrasts"] = {k: {f: v[f] for f in ("mean", "match", "opposes", "status")} for k, v in third["contrasts"].items()}
    result = an.verdict(gates, first, second, third, an.text_ranks(obs.primes()))
    out["verdict"] = result["verdict"]
    out["reason_code"] = result["reason_code"]
    out["label_neighbours_live"] = bool(result["riders"].get("label_neighbours_live"))
    out["sufficient_not_necessary"] = bool(result["riders"].get("sufficient_not_necessary"))
    return out


def summarise(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    counts: Dict[str, int] = {}
    reasons: Dict[str, int] = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
        if row["reason_code"]:
            reasons[row["reason_code"]] = reasons.get(row["reason_code"], 0) + 1

    def mean(key: str) -> Optional[float]:
        values = [r[key] for r in rows if r.get(key) is not None]
        return round(float(np.mean(values)), 4) if values else None

    out: Dict[str, Any] = {
        "n": n,
        "verdicts": {k: round(v / n, 4) for k, v in sorted(counts.items())},
        "reason_codes": {k: round(v / n, 4) for k, v in sorted(reasons.items())},
        "mean_delta": mean("delta"), "mean_X": mean("X"), "mean_O": mean("O"), "mean_X_far": mean("X_far"),
        "mean_cue_margin_C1": mean("margin"),
        "label_neighbours_live": round(float(np.mean([r["label_neighbours_live"] for r in rows])), 4),
        "sufficient_not_necessary": round(float(np.mean([r["sufficient_not_necessary"] for r in rows])), 4),
    }
    states = [r["direction_states"] for r in rows if "direction_states" in r]
    if states:
        out["direction_states"] = {name: {s: round(float(np.mean([x[name] == s for x in states])), 4) for s in ("carries", "absent", "unresolved")}
                                   for name in an.DIRECTIONS}
    with_stage3 = [r for r in rows if "contrasts" in r]
    if with_stage3:
        out["share_reaching_stage3"] = round(len(with_stage3) / n, 4)
        out["contrasts_given_stage3"] = {
            c: {"mean": round(float(np.mean([r["contrasts"][c]["mean"] for r in with_stage3])), 4),
                "match": round(float(np.mean([r["contrasts"][c]["match"] for r in with_stage3])), 4),
                "opposes": round(float(np.mean([r["contrasts"][c]["opposes"] for r in with_stage3])), 4)}
            for c in ("C1", "C2", "C3", "C4")}
        out["any_match"] = round(float(np.mean([any(v["match"] for v in r["contrasts"].values()) for r in with_stage3])) * len(with_stage3) / n, 4)
    return out


def judge(name: str, cell: Mapping[str, Any]) -> Dict[str, Any]:
    """The pass rule of the draft for one observer at one judged strength."""
    rule = EXPECTED[name]
    shares = cell["verdicts"]
    inconclusive = shares.get(an.INCONCLUSIVE, 0.0)
    if "expected" in rule:
        hit = shares.get(rule["expected"], 0.0)
        others = max([v for k, v in shares.items() if k not in (rule["expected"], an.INCONCLUSIVE)], default=0.0)
        checks = {"expected_verdict": hit >= R["expected_verdict_min"], "other_named": others <= R["other_named_verdict_max"],
                  "inconclusive": inconclusive <= R["inconclusive_max"]}
        if rule.get("rider"):
            checks["rider"] = cell[rule["rider"]] >= R["expected_verdict_min"]
        if rule.get("no_false_null"):
            checks["no_false_null"] = shares.get(an.NOT_THIS_CUE, 0.0) <= R["other_named_verdict_max"]
    elif rule.get("forbidden_match"):
        checks = {"no_match": cell.get("any_match", 0.0) <= R["other_named_verdict_max"]}
    else:
        worst = sum(shares.get(k, 0.0) for k in rule["forbidden"])
        checks = {"forbidden_verdict": worst <= R["other_named_verdict_max"]}
    return {"pass": all(checks.values()), "checks": checks}


def build_tasks(names: Sequence[str], strengths: Sequence[float], replicates: int, noise: float = 1.0) -> List[Tuple[Any, ...]]:
    return [(name, strength, k, noise) for name in names for strength in strengths for k in range(replicates)]


def run_tasks(tasks: Sequence[Tuple[Any, ...]], workers: int) -> List[Dict[str, Any]]:
    if workers <= 1:
        return [one_replicate(t) for t in tasks]
    with multiprocessing.Pool(workers) as pool:
        return pool.map(one_replicate, tasks, chunksize=8)


def table(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for key in sorted({(r["observer"], r["strength"]) for r in rows}):
        out.setdefault(key[0], {})[str(key[1])] = summarise([r for r in rows if (r["observer"], r["strength"]) == key])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--replicates", type=int, default=R["replicates"])
    parser.add_argument("--scenario_replicates", type=int, default=50)
    parser.add_argument("--noise_replicates", type=int, default=100)
    parser.add_argument("--draw_check_replicates", type=int, default=15)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--observers", nargs="*", default=None)
    parser.add_argument("--output", type=Path, default=d.RECOVERY_OUT)
    args = parser.parse_args()
    started = time.time()
    strengths = [float(s) for s in R["strengths"]]
    names = args.observers or list(EXPECTED) + list(BOUNDARY)
    rows = run_tasks(build_tasks(names, strengths, args.replicates), args.workers)
    tables = table(rows)
    checks: Dict[str, Any] = {}
    for name in names:
        if name in BOUNDARY:
            continue
        checks[name] = {str(s): judge(name, tables[name][str(s)]) for s in R["judged_strengths"]}
    scenario_rows = run_tasks(build_tasks(list(SCENARIOS), [1.0], args.scenario_replicates), args.workers)
    scenarios = table(scenario_rows)
    for name, rule in SCENARIOS.items():
        cell = scenarios[name]["1.0"]
        cell["reached"] = cell["verdicts"].get(rule["expected"], 0.0) > 0 and (rule["reason"] is None or cell["reason_codes"].get(rule["reason"], 0.0) > 0)
    noise_rows: Dict[str, Any] = {}
    for scale in R["noise_scales"]:
        noise_rows[str(scale)] = table(run_tasks(build_tasks(
            ["no_sharing", "same_cue", "other_cue", "global_face", "human_like"], [1.0], args.noise_replicates, float(scale)), args.workers))
    # The same replicates at the recovery's draws and at the real analysis's, verdict by verdict.
    check = build_tasks(["same_cue", "other_cue", "global_face", "no_sharing"], [1.0], args.draw_check_replicates)
    fewer = run_tasks(check, args.workers)
    full = run_tasks([(*task, True) for task in check], args.workers)
    draw_check = {"n": len(check), "verdicts_equal": sum(a["verdict"] == b["verdict"] for a, b in zip(fewer, full)),
                  "signflip_draws": an.FULL_SETTINGS["flips"], "bootstrap_draws": an.FULL_SETTINGS["boots"]}
    reached = sorted({k for t in tables.values() for c in t.values() for k in c["verdicts"]} | {k for t in scenarios.values() for c in t.values() for k in c["verdicts"]})
    reasons = sorted({k for t in tables.values() for c in t.values() for k in c["reason_codes"]} | {k for t in scenarios.values() for c in t.values() for k in c["reason_codes"]})
    out = {
        "design_sha256": d.block_hash(),
        "replicates": args.replicates,
        "scenario_replicates": args.scenario_replicates,
        "settings": {**R, "flips_used": SETTINGS["flips"], "boots_used": SETTINGS["boots"]},
        "human_rates": {k: world().human[k]["X"] for k in ("Anger>Disgust", "Disgust>Anger")},
        "tables": tables,
        "checks": checks,
        "scenarios": scenarios,
        "noise_sensitivity_not_judged": noise_rows,
        "draw_check": draw_check,
        "every_named_row_reached": sorted(set(an.VERDICTS) - set(reached)) == [],
        "rows_not_reached": sorted(set(an.VERDICTS) - set(reached)),
        "reason_codes_not_reached": sorted(set(an.REASON_CODES) - set(reasons)),
        "pass": all(c["pass"] for v in checks.values() for c in v.values()),
        "seconds": round(time.time() - started, 1),
    }
    args.output.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"pass": out["pass"], "seconds": out["seconds"], "rows_not_reached": out["rows_not_reached"],
                      "failed": {k: {s: c["checks"] for s, c in v.items() if not c["pass"]} for k, v in checks.items() if not all(c["pass"] for c in v.values())}}, indent=1))


if __name__ == "__main__":
    main()
