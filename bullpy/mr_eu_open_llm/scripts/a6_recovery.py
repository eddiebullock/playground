"""
Recovery for draft amendment A6: synthetic observers with known truth through the real analysis.

Each observer writes cells in the format of scripts/a6_run.py, over the real 819 clips, 91
actors, partners and orders, with made-up option scores. Those rows go through Cells, stage1,
stage2, stage3 and verdict in scripts/a6_analyse.py. No model, clip or result is read. Every
constant is in the recovery part of the design block. The noise sizes are A2's recovery noise,
not chosen for A6, plus a term at the level the Stage 3 test uses: each actor responds to each
masked region in its own way, per emotion and option, the same for every fill of that region.
Strengths are multiples of the human directed rates in research/reproduction/a6_human_target.json.

Usage:
    python -m scripts.a6_recovery --workers 3
    python -m scripts.a6_recovery --variant both_directions --output .../a6_recovery_both_directions_paired.json
    python -m scripts.a6_recovery --variant first_draft --output .../a6_recovery_draft_as_written.json
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

D = d.DESIGN
R = D["recovery"]
OPTIONS = an.OPTIONS
K = len(OPTIONS)
INDEX = an.INDEX
A, B = (INDEX[x] for x in D["target_pair"])
NONE = INDEX["No Emotion"]
HAPPY = INDEX["Happy"]
REGIONS = ("upper", "mid", "lower")
SETTINGS = {"flips": int(R["signflip_draws"]), "boots": int(R["bootstrap_draws"]), "seed": int(D["seed"])}
TOKENS_ACROSS = float(R["tokens_across"])
GATES_OK = {key: True for key in an.GATE_KEYS}
VARIANT_ENV = "A6_RECOVERY_VARIANT"

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
    "signal_loss": {"forbidden": [an.SAME_CUE, an.ONE_DIRECTION]},
    "partner_pull": {"forbidden": [an.SAME_CUE, an.ONE_DIRECTION, an.OPPOSES]},
    "fill_artefact": {"forbidden": [an.SAME_CUE, an.ONE_DIRECTION]},
    "fill_artefact_up": {"forbidden": [an.OPPOSES]},
    "option_prior": {"expected": an.NOT_SHARED},
    "position_bias": {"expected": an.NOT_SHARED},
    "adjacency_all": {"expected": an.NOT_SHARED},
    "one_way_weak_cue": {"forbidden": [an.NOT_THIS_CUE, an.NOT_SHARED]},
    "split_cue": {"forbidden": [an.NOT_THIS_CUE]},
    # Added after the second audit.
    "happy_up": {"forbidden": [an.SAME_CUE, an.ONE_DIRECTION]},
    "happy_down": {"forbidden": [an.OPPOSES]},
    "happy_cancel": {"forbidden": [an.NOT_THIS_CUE, an.NO_TESTED_FEATURE]},
    "negative_reverse": {"expected": an.ONE_DIRECTION, "forbidden_also": [an.NOT_SHARED]},
    "score_scale": {"expected": an.NOT_SHARED},
    "incomplete_first": {"expected": an.SAME_CUE},
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
GROUPS = sorted(set(EXPECTED) | set(BOUNDARY) | set(SCENARIOS))
LADDER_OBSERVERS = ("other_cue", "one_way_other_cue", "global_face", "same_cue", "human_like")


def standard_share(logs: np.ndarray, b: int) -> np.ndarray:
    """The standardised share of a.py, on log scores of four options, last axis."""
    centred = logs - logs.mean(-1, keepdims=True)
    spread = np.sqrt((centred**2).mean(-1, keepdims=True))
    z = np.where(spread > 0, centred / np.where(spread > 0, spread, 1.0), 0.0)
    weights = np.exp(z)
    return weights[..., b] / weights.sum(-1)


def boost_for(prior: np.ndarray, a: int, b: int, target: float, noise: float = 1.0) -> float:
    """Nats added to B on clips of A so that X(A to B) is target, at this noise, by bisection.

    The standardised share is flat in the boost when there is no noise, so it is computed on a
    fixed draw of score noise of the size the observers use, the same draw for every boost.
    """
    sd = noise * float(np.sqrt(R["actor_sd"] ** 2 + R["clip_sd"] ** 2 + R["order_sd"] ** 2))
    eps = np.random.default_rng(int(R["seed"])).normal(0, sd, (int(R["calibration_draws"]), K))

    def excess(g: float) -> float:
        gaps = []
        for c in range(K):
            if c in (a, b):
                continue
            kept = [k for k in range(K) if k not in (a, c)]
            logs = prior[kept] + eps[:, kept]
            base = standard_share(logs, kept.index(b)).mean()
            logs[:, kept.index(b)] += g
            gaps.append(standard_share(logs, kept.index(b)).mean() - base)
        return float(np.mean(gaps))

    low, high = -12.0, 12.0
    for _ in range(40):
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
                self.plans[stage].extend(ar.stage_plan(item, stage))
        self.position = np.array([[list(o).index(label) for label in OPTIONS] for o in D["orders"]])
        self.adjacent = np.abs(self.position[:, :, None] - self.position[:, None, :]) == 1


WORLD: Optional[World] = None


def world() -> World:
    global WORLD
    if WORLD is None:
        WORLD = World()
    return WORLD


def region_parts(region: str) -> Tuple[str, ...]:
    return {"upper": ("upper",), "mid": ("mid",), "lower": ("lower",), "midlower": ("mid", "lower"),
            "uplow": ("upper", "lower"), "all": ("upper", "mid", "lower")}[region]


def visible(condition: str) -> Tuple[frozenset, str, float, Optional[str]]:
    """Regions left visible by a condition, its fill, the share of the face it covers, its region."""
    spec = d.condition_spec(condition)
    if spec["kind"] == "none" or spec["fill"] == "self":
        return frozenset(REGIONS), "none", 0.0, None
    region = spec["region"]
    return frozenset(REGIONS) - set(region_parts(region)), spec["fill"], float(R["mask_area"][region]), region


class Observer:
    """Additive scores: prior, noise, acted evidence by region, and pair excess by region."""

    def __init__(self, name: str, strength: float, seed: Sequence[int], noise: float = 1.0,
                 overrides: Optional[Mapping[str, float]] = None) -> None:
        self.name, self.strength = name, strength
        self.options = dict(overrides or {})
        self.rng = np.random.default_rng(list(seed))
        self.w = world()
        self.noise = noise
        self.prior = self.rng.normal(0, R["prior_sd"], K)
        if name == "option_prior":
            self.prior[[A, B]] += float(R["option_prior_nats"])
        rates = {(a, b): strength * self.w.human[f"{OPTIONS[a]}>{OPTIONS[b]}"]["X"] for a, b in ((A, B), (B, A))}
        self.gains = {pair: boost_for(self.prior, pair[0], pair[1], rate, noise) for pair, rate in rates.items()}
        self.evidence = 0.0 if name == "prior_only" else float(R["evidence"])
        self.share = dict(R["evidence_share"])
        if name == "no_dose":
            self.share = {"upper": 0.0, "mid": 0.0, "lower": 0.0, "rest": 1.0}
        self.pair: Dict[Tuple[int, int], Dict[str, float]] = {pair: {} for pair in rates}
        self.adjacency = np.zeros((K, K))
        self.position_effect = np.zeros(K)
        self.actor_sign = np.ones(self.w.n)
        self.blind_pair: Dict[Tuple[int, int], float] = {}
        self.first_kept = float(self.options.get("first_kept", 0.0))
        self.build(name)
        n = self.w.n
        n_actors = len(self.w.cells.actors)
        self.base = (self.prior + noise * self.rng.normal(0, R["actor_sd"], (n_actors, K))[self.w.actor]
                     + noise * self.rng.normal(0, R["clip_sd"], (n, K)))
        self.order_noise = noise * self.rng.normal(0, R["order_sd"], (n, len(D["orders"]), K))
        region_sd = float(self.options.get("actor_region_sd", R["actor_region_sd"]))
        regions = sorted(D["regions"])
        self.actor_region = {r: region_sd * self.rng.normal(0, 1, (n_actors, K, K)) for r in regions}
        scale_sd = float(self.options.get("scale_sd", 0.0))
        self.scale = np.exp(self.rng.normal(0, scale_sd, K)) if scale_sd else np.ones(K)

    def build(self, name: str) -> None:
        ab, ba = (A, B), (B, A)
        g, h = self.gains[ab], self.gains[ba]
        rest_names = ("global_face", "generic_mask", "generic_pull", "signal_loss", "partner_pull", "fill_artefact",
                      "fill_artefact_up", "no_dose", "sham_differs", "gate_failed", "clips_lost", "region_small",
                      "happy_up", "happy_down")
        if name in ("same_cue", "happy_cancel", "incomplete_first"):
            self.pair = {ab: {"mid": g}, ba: {"mid": h}}
            if name == "incomplete_first" and "first_kept" not in self.options:
                self.first_kept = float(R["first_fill_kept"][0])
        elif name == "partial_cue":
            s = float(self.options["share"])
            self.pair = {ab: {"mid": s * g, "rest": (1 - s) * g}, ba: {"mid": s * h, "rest": (1 - s) * h}}
        elif name == "one_way_cue":
            self.pair = {ab: {"mid": g}, ba: {}}
        elif name == "one_way_other_cue":
            self.pair = {ab: {"upper": g}, ba: {}}
        elif name == "one_way_weak_cue":
            self.pair = {ab: {}, ba: {"mid": h}}
        elif name == "split_cue":
            self.pair = {ab: {"upper": g}, ba: {"mid": h}}
        elif name == "negative_reverse":
            # The forward direction at the human rate, the reverse as far below zero.
            back = boost_for(self.prior, B, A, -self.strength * self.w.human["Anger>Disgust"]["X"], self.noise)
            self.pair = {ab: {"mid": g}, ba: {"mid": back}}
        elif name == "reverse":
            self.pair = {ab: {"rest": 2 * g, "mid": -g}, ba: {"rest": 2 * h, "mid": -h}}
        elif name == "other_cue":
            self.pair = {ab: {"upper": g}, ba: {"upper": h}}
        elif name == "spread":
            self.pair = {ab: {"any": g}, ba: {"any": h}}
        elif name in rest_names:
            self.pair = {ab: {"rest": g}, ba: {"rest": h}}
        elif name == "human_like":
            lift = h * float(R["carbon_lift"])
            self.pair = {ab: {"mid": g, "rest": 0.5 * g, "upper": -0.5 * g},
                         ba: {"mid": h, "rest": h + lift, "lower": -h - lift}}
        elif name == "adjacency":
            cap = float(R["adjacency_cap"])
            forward = boost_for(self.prior, A, B, min(cap, 3 * self.strength * self.w.human["Anger>Disgust"]["X"]), self.noise)
            back = boost_for(self.prior, B, A, min(cap, 3 * self.strength * self.w.human["Disgust>Anger"]["X"]), self.noise)
            self.adjacency[A, B], self.adjacency[B, A] = forward, back
            self.pair = {ab: {}, ba: {}}
        elif name == "part_adjacent":
            self.pair = {ab: {}, ba: {}}
            self.adjacency[A, B], self.adjacency[B, A] = 3 * g, 3 * h
            self.blind_pair = {ab: g, ba: h}
        elif name == "adjacency_all":
            self.adjacency[:] = 1.0
            self.pair = {ab: {}, ba: {}}
        elif name == "position_bias":
            self.position_effect = np.array(R["position_effect"], dtype=np.float64)
            self.pair = {ab: {}, ba: {}}
        elif name == "boundary_shared":
            forward, back = self.w.human["Anger>Disgust"]["X"], self.w.human["Disgust>Anger"]["X"]
            total = 2 * float(D["shared_margin"])
            self.pair = {ab: {"rest": boost_for(self.prior, A, B, total * forward / (forward + back), self.noise)},
                         ba: {"rest": boost_for(self.prior, B, A, total * back / (forward + back), self.noise)}}
        elif name == "boundary_cue":
            step = float(R["boundary_cue_nats"])
            self.pair = {ab: {"rest": g - step, "mid": step}, ba: {"rest": h - step, "mid": step}}
        elif name == "actors_split":
            self.actor_sign = self.rng.choice((1.0, -1.0), len(self.w.cells.actors), p=tuple(R["split_actor_share"]))[self.w.actor]
            gain = float(R["split_gain"])
            self.pair = {ab: {"rest": gain}, ba: {"rest": gain}}
        elif name in ("prior_only", "no_sharing", "option_prior", "score_scale"):
            self.pair = {ab: {}, ba: {}}
        else:
            raise ValueError(name)

    def condition_shift(self, emo: np.ndarray, condition: str) -> np.ndarray:
        """The structured part of the scores of clips of emo under a condition, no noise."""
        seen, fill, area, region = visible(condition)
        count = len(emo)
        out = np.zeros((count, K))
        signs = self.actor_sign[self.w.stage3] if count != self.w.n else self.actor_sign
        hidden = [r for r in REGIONS if r not in seen]
        # An incomplete first fill: the first frame already shows part of what it should hide.
        kept_back = self.first_kept if fill == "first" else 0.0
        kept = self.share["rest"] + sum(self.share[r] for r in seen) + kept_back * sum(self.share[r] for r in hidden)
        out[np.arange(count), emo] += self.evidence * kept
        for (a, b), parts in self.pair.items():
            total = parts.get("rest", 0.0) + sum(parts.get(r, 0.0) for r in seen)
            total += kept_back * sum(parts.get(r, 0.0) for r in hidden)
            if seen:
                total += parts.get("any", 0.0)
            out[emo == a, b] += total * signs[emo == a]
        size = float(R["generic_nats"]) * area
        pull = float(R["pull_multiplier"])
        if area > 0:
            if self.name == "generic_mask":
                foils = np.ones((count, K), dtype=bool)
                foils[np.arange(count), emo] = False
                out[foils] -= size
                out[:, NONE] += 2.0 * size
            elif self.name == "generic_pull":
                out[:, B] += size * pull
            elif self.name == "partner_pull":
                out[:, A] += size * pull
            elif self.name == "signal_loss":
                out[np.arange(count), emo] -= size * pull
            elif self.name in ("fill_artefact", "fill_artefact_up") and fill == "neutral" and condition.startswith("mask_mid_"):
                sign = -1.0 if self.name == "fill_artefact" else 1.0
                out[emo == A, B] += sign * float(R["artefact_nats"])
                out[emo == B, A] += sign * float(R["artefact_nats"])
            elif self.name in ("happy_up", "happy_down") and condition.startswith("mask_mid_"):
                sign = 1.0 if self.name == "happy_up" else -1.0
                out[np.isin(emo, [A, B]), HAPPY] += sign * float(R["happy_nats"])
            elif self.name == "happy_cancel" and condition.startswith("mask_mid_"):
                # Happy falls as far as the partner does, so y does not move under the mask.
                for (a, b), parts in self.pair.items():
                    out[emo == a, HAPPY] -= parts.get("mid", 0.0)
        return out

    def actor_region_term(self, clips: np.ndarray, condition: str) -> np.ndarray:
        """Each actor's own response to a masked region, by emotion and option, shared across fills."""
        _seen, _fill, _area, region = visible(condition)
        if region is None:
            return np.zeros((len(clips), K))
        return self.actor_region[region][self.w.actor[clips], self.w.emo[clips]]

    def order_shift(self, row: int, emo: np.ndarray) -> np.ndarray:
        out = np.broadcast_to(self.position_effect[self.w.position[row]], (len(emo), K)).copy()
        if self.adjacency.any():
            out += (self.adjacency * self.w.adjacent[row])[emo]
        return out

    def scaled(self, scores: np.ndarray, emo: np.ndarray) -> np.ndarray:
        """A scale per acted emotion on the whole score vector, as a sharper or flatter model on it."""
        return scores * self.scale[emo].reshape(-1, *([1] * (scores.ndim - 1)))

    def stage1_scores(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        w = self.w
        original = self.base + self.condition_shift(w.emo, "original")
        mcq = np.stack([original + self.order_shift(row, w.emo) + self.order_noise[:, row] for row in range(len(D["orders"]))], axis=1)
        blind = original + self.noise * self.rng.normal(0, R["order_sd"], original.shape)
        for (a, b), gain in self.blind_pair.items():
            blind[w.emo == a, b] += gain
        bridge = mcq[:, list(an.EDIT_ROWS)] + self.noise * self.rng.normal(0, R["order_sd"], (w.n, len(an.EDIT_ROWS), K))
        return self.scaled(mcq, w.emo), self.scaled(blind, w.emo), self.scaled(bridge, w.emo)

    def stage3_scores(self, mcq: np.ndarray) -> np.ndarray:
        """Scores of the Stage 3 clips by condition and edit order. The sham is the original."""
        w = self.w
        rows = list(an.EDIT_ROWS)
        clips = w.stage3
        emo = w.emo[clips]
        out = np.empty((len(clips), len(an.CONDITIONS), len(rows), K))
        for c, condition in enumerate(an.CONDITIONS):
            if condition in ("original", "sham_self_fill"):
                out[:, c] = mcq[clips][:, rows]
                continue
            shift = (self.base[clips] + self.condition_shift(emo, condition) + self.actor_region_term(clips, condition)
                     + self.noise * self.rng.normal(0, R["condition_sd"], (len(clips), K)))
            for r, row in enumerate(rows):
                cell = (shift + self.order_shift(row, emo) + self.order_noise[clips, row]
                        + self.noise * self.rng.normal(0, R["condition_sd"], (len(clips), K)))
                out[:, c, r] = self.scaled(cell, emo)
        return out

    def primes(self) -> np.ndarray:
        out = self.rng.normal(0, 1, (K, len(D["orders"]), K)) * 0.2 + self.rng.normal(0, 1, (K, 1, K))
        if self.name == "spread":
            out[A, :, B] += float(R["spread_prime_nats"])
            out[B, :, A] += float(R["spread_prime_nats"])
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


def rows_stage3(obs: Observer, scores: np.ndarray, *, sham_differs: bool, span: float) -> List[Dict[str, Any]]:
    w = obs.w
    edit = {r: i for i, r in enumerate(an.EDIT_ROWS)}
    place = {int(c): n for n, c in enumerate(w.stage3)}
    conditions = {c: i for i, c in enumerate(an.CONDITIONS)}
    rows = []
    for cell in w.plans[3]:
        i = place[w.cells.index[cell["trial_id"]]]
        row = dict(cell)
        condition = row["condition"]
        order = row["options"]
        sha = "different" if condition == "sham_self_fill" and sham_differs and i == 0 else "same"
        row.update(status="ok", joint_logprob=scores[i, conditions[condition], edit[row["order"]]][[INDEX[o] for o in order]].tolist(),
                   media_sha256=[sha], edit={"span_view_fraction": span})
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Variants, for the audit trail only. Each replaces parts of the analysis in a worker.
# ---------------------------------------------------------------------------


APPLIED: Optional[str] = None


def apply_variant(name: Optional[str]) -> None:
    """both_directions: the run 1 rule, every direction in play. first_draft: the first draft's
    X, outcome reference and sign-only fill rule, with no Happy check and no direction rule."""
    global APPLIED
    if not name or APPLIED == name:
        return
    if APPLIED is not None:
        raise SystemExit(f"Variant {APPLIED} is already applied in this worker.")
    APPLIED = name
    original_states = an.direction_states

    def every_direction(cells: Any, settings: Mapping[str, Any], orders: Sequence[int]) -> Dict[str, Dict[str, Any]]:
        out = original_states(cells, settings, orders)
        for value in out.values():
            value["state"] = "carries"
        return out

    if name == "both_directions":
        an.direction_states = every_direction
        return
    if name == "happy_equivalence":
        an.D["reference_check"] = "happy_inside_margin"
        return
    if name != "first_draft":
        raise SystemExit(f"Unknown variant {name}.")

    def first_draft_excess(cells: Any, p: np.ndarray, a: int, b: int) -> np.ndarray:
        p = p.mean(1) if p.ndim == 3 else p
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(1 - p[:, a] > 0, p[:, b] / (1 - p[:, a]), np.nan)
        return cells.actor_mean(r, cells.emo == a) - cells.actor_mean(r, (cells.emo != a) & (cells.emo != b))

    an.directed_excess = first_draft_excess
    an.direction_states = every_direction
    an.D["outcome_reference"] = "two_out"
    an.beyond_zero = lambda ci, direction: bool(ci is not None and (ci[0] + ci[1]) * direction > 0)
    an.reference_change = lambda cells, contrast, condition, **kwargs: np.zeros(len(cells.actors))


def prior_bias_check(draws: int = 300) -> Dict[str, Any]:
    """The 'about 12%' in the draft: the share of observers with no pair excess whose X is over
    .05 under the simple fix r = p(B) / (1 - p(own label)), with random log priors of SD 1 and
    acted evidence 2 nats, against the standardised share, noise-free, on the real clips."""
    w = world()
    rng = np.random.default_rng(int(R["seed"]))
    holder = w.cells
    simple, standard = [], []
    for _ in range(draws):
        prior = rng.normal(0, R["prior_sd"], K)
        p = an.softmax(prior + float(R["evidence"]) * np.eye(K)[w.emo])
        rows = np.arange(len(p))
        share = p[:, B] / (1 - p[rows, w.emo])
        back = p[:, A] / (1 - p[rows, w.emo])

        def simple_x(r: np.ndarray, a: int, b: int) -> float:
            return float(np.nanmean(holder.actor_mean(r, w.emo == a) - holder.actor_mean(r, (w.emo != a) & (w.emo != b))))

        simple.append((simple_x(share, A, B) + simple_x(back, B, A)) / 2)
        standard.append(float(np.nanmean(an.pair_excess(holder, p, A, B))))
    return {"draws": draws, "simple_fix_share_over_shared_margin": float(np.mean(np.array(simple) > D["shared_margin"])),
            "standardised_share_max_abs": float(np.max(np.abs(standard)))}


# ---------------------------------------------------------------------------
# One replicate
# ---------------------------------------------------------------------------


def one_replicate(task: Tuple[Any, ...]) -> Dict[str, Any]:
    """One observer, one strength, one replicate, through the real analysis.

    task: name, strength, replicate, noise, use the real draws, overrides (a dict, or None).
    """
    apply_variant(os.environ.get(VARIANT_ENV))
    name, strength, replicate, noise = task[:4]
    full = bool(task[4]) if len(task) > 4 else False
    overrides = dict(task[5]) if len(task) > 5 and task[5] else {}
    settings = an.FULL_SETTINGS if full else SETTINGS
    tag = int(sum(v * 1000 * (i + 1) for i, v in enumerate(sorted(overrides.values()))))
    seed = [int(R["seed"]), GROUPS.index(name) if name in GROUPS else 99, int(strength * 100), replicate, int(noise * 100), tag]
    obs = Observer(name, strength, seed, noise, overrides)
    w = obs.w
    mcq, blind, bridge = obs.stage1_scores()
    lost = None
    if name == "clips_lost":
        lost = np.zeros(w.n, dtype=bool)
        lost[obs.rng.choice(w.n, int(float(R["lost_share"]) * w.n), replace=False)] = True
    cells = an.Cells(w.items, rows_stage1(obs, mcq, blind, bridge, lost=lost))
    first = an.stage1(cells, w.partners, settings, descriptive=False)
    # Holm's worst case across the two eligible models, as in A4's recovery.
    first["p_delta_holm"] = min(1.0, 2 * first["p_delta_actor_sign_flip"])
    first["vision_dependent"] = bool(first["p_delta_holm"] < an.ALPHA and first["delta_ci_actor_bootstrap"][0] > 0)
    gates = dict(GATES_OK)
    if name == "gate_failed":
        gates["rq0"] = False
    out: Dict[str, Any] = {"observer": name, "strength": strength, "noise": noise, "overrides": overrides,
                           "delta": first["delta_real_minus_mismatch"]}
    second = third = None
    if an.gates_passed(gates) and not first["over_lost_item_limit"] and first["vision_dependent"]:
        second = an.stage2(cells, w.partners, settings)
        target = second["pairs"][an.PAIR_NAMES[0]]
        out["direction_states"] = {k: v["state"] for k, v in second["directions"].items()}
        out.update(X=target["X"], O=target["O"], X_far=target["X_far"], state=second["state"], blind_X=target["blind_X"],
                   margin=second["cue_margin"]["C1"], reach_sd_limit=second["null_reach"]["C1"]["sd_limit"])
        if second["reaches_stage3"]:
            span = float(R["too_small_fraction"]) if name == "region_small" else float(R["span_fraction"])
            cells.ingest(rows_stage3(obs, obs.stage3_scores(mcq), sham_differs=name == "sham_differs", span=span))
            third = an.stage3(cells, second["cue_margin"], TOKENS_ACROSS, settings, dose_margins=second["dose_margin"])
            out["contrasts"] = {k: {f: v[f] for f in ("mean", "match", "opposes", "status", "share_of_excess_removed", "reference_stable")}
                                for k, v in third["contrasts"].items()}
            out["actor_sd_C1"] = float(np.nanstd(an.double_difference(cells, an.CONTRASTS[0], "mask_mid_neutral"), ddof=1))
    result = an.verdict(gates, first, second, third, an.text_ranks(obs.primes()))
    out["verdict"] = result["verdict"]
    out["reason_code"] = result["reason_code"]
    out["label_neighbours_live"] = bool(result["riders"].get("label_neighbours_live"))
    out["sufficient_not_necessary"] = bool(result["riders"].get("sufficient_not_necessary"))
    return out


# ---------------------------------------------------------------------------
# Tables, the pass rule, and the command line
# ---------------------------------------------------------------------------


def summarise(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    counts: Dict[str, int] = {}
    reasons: Dict[str, int] = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
        if row["reason_code"]:
            reasons[row["reason_code"]] = reasons.get(row["reason_code"], 0) + 1

    def mean(key: str) -> Optional[float]:
        values = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
        return round(float(np.mean(values)), 4) if values else None

    out: Dict[str, Any] = {
        "n": n,
        "verdicts": {k: round(v / n, 4) for k, v in sorted(counts.items())},
        "reason_codes": {k: round(v / n, 4) for k, v in sorted(reasons.items())},
        "mean_delta": mean("delta"), "mean_X": mean("X"), "mean_O": mean("O"), "mean_X_far": mean("X_far"),
        "mean_cue_margin_C1": mean("margin"), "mean_sd_limit_C1": mean("reach_sd_limit"), "mean_actor_sd_C1": mean("actor_sd_C1"),
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
                "opposes": round(float(np.mean([r["contrasts"][c]["opposes"] for r in with_stage3])), 4),
                "reference_stable": round(float(np.mean([r["contrasts"][c]["reference_stable"] for r in with_stage3])), 4),
                "share_of_excess_removed": round(float(np.nanmean([r["contrasts"][c]["share_of_excess_removed"] if r["contrasts"][c]["share_of_excess_removed"] is not None else np.nan for r in with_stage3])), 4)}
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
        for forbidden in rule.get("forbidden_also", []):
            checks[f"not_{forbidden}"] = shares.get(forbidden, 0.0) <= R["other_named_verdict_max"]
    elif rule.get("forbidden_match"):
        checks = {"no_match": cell.get("any_match", 0.0) <= R["other_named_verdict_max"]}
    else:
        worst = sum(shares.get(k, 0.0) for k in rule["forbidden"])
        checks = {"forbidden_verdict": worst <= R["other_named_verdict_max"]}
    return {"pass": all(checks.values()), "checks": checks}


def build_tasks(names: Sequence[str], strengths: Sequence[float], replicates: int, noise: float = 1.0,
                overrides: Optional[Mapping[str, float]] = None) -> List[Tuple[Any, ...]]:
    return [(name, strength, k, noise, False, dict(overrides or {})) for name in names for strength in strengths for k in range(replicates)]


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


def ladder(names: Sequence[str], key: str, values: Sequence[float], replicates: int, workers: int, base: str = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for value in values:
        rows = run_tasks(build_tasks(names, [1.0], replicates, overrides={key: value}), workers)
        out[str(value)] = {name: tab["1.0"] for name, tab in table(rows).items()}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--replicates", type=int, default=R["replicates"])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--observers", nargs="*", default=None)
    parser.add_argument("--variant", choices=("both_directions", "first_draft", "happy_equivalence"), default=None,
                        help="Audit trail only: rerun on the same seeds under an earlier rule.")
    parser.add_argument("--output", type=Path, default=d.RECOVERY_OUT)
    args = parser.parse_args()
    started = time.time()
    strengths = [float(s) for s in R["strengths"]]
    if args.variant:
        os.environ[VARIANT_ENV] = args.variant
        names = args.observers or ["split_cue", "one_way_weak_cue", "same_cue", "human_like"]
        chosen = [0.5, 1.0] if args.variant == "both_directions" else [1.0]
        replicates = {"both_directions": args.replicates, "first_draft": 30, "happy_equivalence": 100}[args.variant]
        rows = run_tasks(build_tasks(names, chosen, replicates), args.workers)
        out = {"variant": args.variant, "binding_sha256": d.binding_hash(), "design_sha256": d.block_hash(), "replicates": replicates,
               "note": "Audit trail only, on the seeds of a6_recovery.json, under an earlier rule. Not a result of the amended design.",
               "tables": table(rows), "seconds": round(time.time() - started, 1)}
        args.output.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(json.dumps({k: out[k] for k in ("variant", "replicates", "seconds")}))
        return

    names = args.observers or list(EXPECTED) + list(BOUNDARY)
    tasks = build_tasks([n for n in names if n not in ("score_scale",)], strengths, args.replicates)
    judged_scale = float(R["score_scale_log_sd"][1])
    if "score_scale" in names:
        tasks += build_tasks(["score_scale"], strengths, args.replicates, overrides={"scale_sd": judged_scale})
    rows = run_tasks(tasks, args.workers)
    tables = table(rows)
    checks = {name: {str(s): judge(name, tables[name][str(s)]) for s in R["judged_strengths"]} for name in names if name not in BOUNDARY}
    scenario_rows = run_tasks(build_tasks(list(SCENARIOS), [1.0], int(R["scenario_replicates"])), args.workers)
    scenarios = table(scenario_rows)
    for name, rule in SCENARIOS.items():
        cell = scenarios[name]["1.0"]
        cell["reached"] = cell["verdicts"].get(rule["expected"], 0.0) > 0 and (rule["reason"] is None or cell["reason_codes"].get(rule["reason"], 0.0) > 0)
    noise_rows: Dict[str, Any] = {}
    for scale in R["noise_scales"]:
        noise_rows[str(scale)] = table(run_tasks(build_tasks(["no_sharing", "same_cue", "other_cue", "global_face", "human_like"], [1.0],
                                                             int(R["noise_replicates"]), float(scale)), args.workers))
    ladders = {
        "actor_region_sd": ladder(list(LADDER_OBSERVERS) + ["no_sharing"], "actor_region_sd", R["actor_region_ladder"], int(R["ladder_replicates"]), args.workers),
        "score_scale_log_sd": ladder(["no_sharing", "same_cue"], "scale_sd", [v for v in R["score_scale_log_sd"] if v != judged_scale], int(R["ladder_replicates"]), args.workers),
        "first_fill_kept": ladder(["incomplete_first"], "first_kept", R["first_fill_kept"][1:], int(R["ladder_replicates"]), args.workers),
        "partial_share": ladder(["partial_cue"], "share", R["partial_share"], int(R["ladder_replicates"]), args.workers),
    }
    check = build_tasks(["same_cue", "other_cue", "global_face", "no_sharing"], [1.0], int(R["draw_check_replicates"]))
    fewer = run_tasks(check, args.workers)
    full = run_tasks([(t[0], t[1], t[2], t[3], True, t[5]) for t in check], args.workers)
    draw_check = {"n": len(check), "verdicts_equal": sum(a["verdict"] == b["verdict"] for a, b in zip(fewer, full)),
                  "signflip_draws": an.FULL_SETTINGS["flips"], "bootstrap_draws": an.FULL_SETTINGS["boots"]}
    reached = {k for t in tables.values() for c in t.values() for k in c["verdicts"]} | {k for t in scenarios.values() for c in t.values() for k in c["verdicts"]}
    reasons = {k for t in tables.values() for c in t.values() for k in c["reason_codes"]} | {k for t in scenarios.values() for c in t.values() for k in c["reason_codes"]}
    observers_pass = all(c["pass"] for v in checks.values() for c in v.values())
    scenarios_pass = all(scenarios[n]["1.0"]["reached"] for n in SCENARIOS)
    out = {
        "binding_sha256": d.binding_hash(),
        "design_sha256": d.block_hash(),
        "replicates": args.replicates,
        "settings": {**R, "flips_used": SETTINGS["flips"], "boots_used": SETTINGS["boots"]},
        "human_rates": {k: world().human[k]["X"] for k in an.DIRECTIONS},
        "tables": tables,
        "checks": checks,
        "scenarios": scenarios,
        "noise_sensitivity_not_judged": noise_rows,
        "ladders_not_judged": ladders,
        "draw_check": draw_check,
        "prior_bias_check": prior_bias_check(),
        "rows_not_reached": sorted(set(an.VERDICTS) - reached),
        "reason_codes_not_reached": sorted(set(an.REASON_CODES) - reasons),
        "observers_pass": observers_pass,
        "scenarios_pass": scenarios_pass,
        "seconds": round(time.time() - started, 1),
    }
    out["pass"] = bool(observers_pass and scenarios_pass and not out["rows_not_reached"] and not out["reason_codes_not_reached"])
    args.output.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"pass": out["pass"], "seconds": out["seconds"], "rows_not_reached": out["rows_not_reached"],
                      "failed": {k: {s: c["checks"] for s, c in v.items() if not c["pass"]} for k, v in checks.items() if not all(c["pass"] for c in v.values())}}, indent=1))


if __name__ == "__main__":
    main()
