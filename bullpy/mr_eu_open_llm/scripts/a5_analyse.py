"""
Analysis for draft amendment A5: prior and evidence on the blind readout (RQ4).

A reanalysis of A2's cells, with no new forward. The statistics and gates are A2's frozen
functions, imported and never edited. Per model and route, with no pooling. Every constant
comes from the design block in research/PREREG_A5_DRAFT.md, through scripts/a5_design.py.
scripts/a5_recovery.py runs this same code on synthetic observers. main refuses until A5
is frozen, because no A2 cell may be read before that.

Usage:
    python -m scripts.a5_analyse --a5_frozen --a2_frozen --cells_dir results/stimulus_interventions \
        --output results/a5/a5_analysis.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from scripts import a5_design
from scripts import analyse_stimulus_interventions as an
from scripts import run_stimulus_interventions as run
from scripts import stimulus_interventions as si
from scripts.run_stimulus_interventions import ABSTAIN, LEXICON, OPTION_TO_FAMILY, human_comparable

D = a5_design.load_design()
FAMILIES: Tuple[str, ...] = tuple(sorted(LEXICON))
ROUTES: Tuple[str, ...] = tuple(D["routes"])
MODELS: Tuple[str, ...] = tuple(D["models"])

NOT_TESTED, INCONCLUSIVE = "Not tested", "Inconclusive"
# Decision table: (evidence class, balance class) -> verdict. Unresolved goes to Inconclusive.
VERDICTS = {
    ("Shown", "Face"): "Evidence-led",
    ("Shown", "Default"): "Default-led with evidence",
    ("Shown", "Even"): "Evidence, neither part dominant",
    ("Unmapped", "Face"): "Face-led, unmapped",
    ("Unmapped", "Default"): "Default-led, unmapped",
    ("Unmapped", "Even"): "Unmapped, neither part dominant",
    ("Absent", "Default"): "No face evidence",  # the only row that can carry the word-knowledge reading
    ("Absent", "Even"): "No face evidence, neither part dominant",
    ("Absent", "Face"): "Face-dependent, no expression",
}
REASONS = ("evidence_unresolved", "balance_unresolved", "balance_small_family", "no_structure")
# A named balance class needs a B family of at least `min_balance_cells` cells, the smallest
# size at which recovery held the B error rate (added after recovery run 3; see the draft).
BALANCE_NAMED = ("Face", "Default", "Even")
PRECONDITION_GATES = ("rq0", "g2_g4", "delivery")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Judgement:
    """One route's `original` cells as family distributions, conditional on the lexicon."""

    trials: Tuple[str, ...]
    q: np.ndarray  # clips by 21 families
    actor: np.ndarray  # index into actors
    label: np.ndarray  # index into labels: the acted label, or any other grouping
    target: np.ndarray  # family index of the acted label
    actors: Tuple[str, ...]
    labels: Tuple[str, ...]
    mass: np.ndarray

    def where(self, keep: np.ndarray) -> "Judgement":
        return replace(self, trials=tuple(t for t, k in zip(self.trials, keep) if k), q=self.q[keep],
                       actor=self.actor[keep], label=self.label[keep], target=self.target[keep], mass=self.mass[keep])

    def grouped_by(self, names: Sequence[str]) -> "Judgement":
        labels = tuple(sorted(set(names)))
        return replace(self, label=np.array([labels.index(n) for n in names]), labels=labels)

    def tempered(self, power: float) -> "Judgement":
        q = self.q**power
        return replace(self, q=q / q.sum(1, keepdims=True))


def softmax_of_log(scores: np.ndarray) -> np.ndarray:
    top = scores.max(-1, keepdims=True)
    p = np.exp(scores - top)
    return p / p.sum(-1, keepdims=True)


def distribution(family_logprob: Mapping[str, float]) -> np.ndarray:
    """Joint family probabilities renormalised over the lexicon."""
    return softmax_of_log(np.array([family_logprob[f] for f in FAMILIES], dtype=np.float64))


def judgement(cells: an.Cells, route: str, actors: Sequence[str]) -> Judgement:
    """The `original` cells that ran. A failed cell is absent here and counted by the caller."""
    trials = tuple(sorted(t for t in cells.items if (t, route, "original") in cells.ok))
    rows = [cells.ok[(t, route, "original")] for t in trials]
    labels = tuple(sorted({cells.items[t]["label"] for t in trials}))
    return Judgement(
        trials=trials,
        q=np.array([distribution(r["blind"]["family_logprob"]) for r in rows]),
        actor=np.array([list(actors).index(cells.items[t]["actor"]) for t in trials]),
        label=np.array([labels.index(cells.items[t]["label"]) for t in trials]),
        target=np.array([FAMILIES.index(OPTION_TO_FAMILY[cells.items[t]["label"]]) for t in trials]),
        actors=tuple(actors),
        labels=labels,
        mass=np.array([r["blind"]["lexicon_mass"] for r in rows], dtype=np.float64),
    )


def baselines(reports: Sequence[Mapping[str, Any]], route: str, tolerance: float = D["baseline_agreement_nats"]) -> Dict[str, Any]:
    """The route's image-free and grey rows, if every shard has them and they agree.

    `ok` is False for a row missing from any shard, a failed row, or family scores that
    differ by more than the tolerance between shards. The first shard's row is the one
    used, as in A2's own Gate B call.
    """
    found: Dict[str, List[Mapping[str, Any]]] = {"image_free": [], "grey": []}
    complete = bool(reports)
    for report in reports:
        rows = {r["condition"]: r for r in report.get("baselines") or [] if r.get("input_name") == route}
        for condition, store in found.items():
            row = rows.get(condition)
            if row is None or row.get("status") != "ok":
                complete = False
            else:
                store.append(row)
    spread = None
    if complete:
        spread = 0.0
        for rows in found.values():
            matrix = np.array([[r["family_logprob"][f] for f in FAMILIES] for r in rows])
            spread = max(spread, float((matrix.max(0) - matrix.min(0)).max()))
    ok = complete and spread is not None and spread <= tolerance
    first = {c: (rows[0] if rows else None) for c, rows in found.items()}
    head = {r["condition"]: r for r in (reports[0].get("baselines") or []) if r.get("input_name") == route} if reports else {}
    return {
        "ok": ok,
        "spread_nats": spread,
        "w": distribution(first["image_free"]["family_logprob"]) if ok else None,
        "g": distribution(first["grey"]["family_logprob"]) if ok else None,
        # As A2: the first shard's row, whatever its status. A failed row has no scores, so B3 has no input.
        "image_free_family_logprob": (head.get("image_free") or {}).get("family_logprob"),
        "mass_image_free": first["image_free"].get("lexicon_mass") if ok else None,
        "mass_grey": first["grey"].get("lexicon_mass") if ok else None,
    }


# ---------------------------------------------------------------------------
# The identity
# ---------------------------------------------------------------------------


def flat_reference(n_families: int = len(FAMILIES)) -> np.ndarray:
    return np.full(n_families, 1.0 / n_families)


def size_reference(total_words: int = D["lexicon_words"]) -> np.ndarray:
    """u_f = words in family / lexicon size, the sensitivity that bounds family-size bias."""
    return np.array([len(LEXICON[f]) / total_words for f in FAMILIES], dtype=np.float64)


def actor_means(values: np.ndarray, actor: np.ndarray, n_actors: int) -> np.ndarray:
    """Mean within actor, ignoring NaN clips. NaN for an actor with no usable clip."""
    keep = ~np.isnan(values)
    totals = np.bincount(actor[keep], weights=values[keep], minlength=n_actors)
    counts = np.bincount(actor[keep], minlength=n_actors).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return totals / counts


def default_for_clips(q: np.ndarray, actor: np.ndarray, n_actors: int, leave_out: bool = True) -> np.ndarray:
    """q_def per clip: the mean of actor means, over the other actors if leave_out."""
    means = np.array([q[actor == a].mean(0) if (actor == a).any() else np.full(q.shape[1], np.nan) for a in range(n_actors)])
    present = ~np.isnan(means[:, 0])
    total, count = np.nansum(means, 0), present.sum()
    if not leave_out:
        return np.broadcast_to(total / count, q.shape)
    return ((total[None] - means[actor]) / (count - 1))


def templates(q: np.ndarray, actor: np.ndarray, group: np.ndarray, n_actors: int, n_groups: int) -> np.ndarray:
    """q_tpl for each (group, actor): the mean over that group's clips from other actors.

    NaN where no other actor holds that group.
    """
    key = group * n_actors + actor
    size = n_groups * n_actors
    sums = np.stack([np.bincount(key, weights=q[:, f], minlength=size) for f in range(q.shape[1])], 1)
    sums = sums.reshape(n_groups, n_actors, q.shape[1])
    counts = np.bincount(key, minlength=size).reshape(n_groups, n_actors).astype(np.float64)
    others = counts.sum(1, keepdims=True) - counts
    with np.errstate(invalid="ignore", divide="ignore"):
        return (sums.sum(1, keepdims=True) - sums) / others[..., None]


def weighted_log(q: np.ndarray, ratio_log: np.ndarray) -> np.ndarray:
    """sum q log(x), with 0 log(0) taken as 0, so that vote shares of zero stay finite."""
    with np.errstate(invalid="ignore"):
        return np.where(q > 0, q * ratio_log, 0.0).sum(-1)


def _log(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(x)


def decompose(
    j: Judgement,
    w: Optional[np.ndarray] = None,
    g: Optional[np.ndarray] = None,
    u: Optional[np.ndarray] = None,
    leave_out: bool = True,
) -> Dict[str, Optional[np.ndarray]]:
    """T = W + G + F + E + R for every clip, with P = W + G + F and D = E + R.

    W, G and F are None without baselines, since a route that fails its baseline check
    keeps P and D, which need neither w nor g. E and R are NaN on a clip with no template.
    """
    q, n_actors = j.q, len(j.actors)
    reference = flat_reference(q.shape[1]) if u is None else u
    default = default_for_clips(q, j.actor, n_actors, leave_out)
    tpl = templates(q, j.actor, j.label, n_actors, len(j.labels))[j.label, j.actor]
    log_q, log_def, log_tpl = _log(q), _log(default), _log(tpl)
    log_u = np.log(reference)
    out: Dict[str, Optional[np.ndarray]] = {
        "T": weighted_log(q, log_q - log_u),
        "P": weighted_log(q, log_def - log_u),
        "D": weighted_log(q, log_q - log_def),
        "E": weighted_log(q, log_tpl - log_def),
        "R": weighted_log(q, log_q - log_tpl),
        "W": None,
        "G": None,
        "F": None,
    }
    if w is not None and g is not None:
        out["W"] = weighted_log(q, np.log(w) - log_u)
        out["G"] = weighted_log(q, np.log(g) - np.log(w))
        out["F"] = weighted_log(q, log_def - np.log(g))
    return out


_SHUFFLE_CACHE: Dict[Tuple[bytes, int, int], np.ndarray] = {}


def shuffle_index(actor: np.ndarray, draws: int, seed: int) -> np.ndarray:
    """draws by clips: each row reorders clips within actor. Seeded, so fixed for a clip set."""
    key = (actor.tobytes(), draws, seed)
    if key not in _SHUFFLE_CACHE:
        rng = np.random.default_rng(seed)
        index = np.tile(np.arange(actor.size), (draws, 1))
        for a in np.unique(actor):
            members = np.where(actor == a)[0]
            index[:, members] = rng.permuted(np.tile(members, (draws, 1)), axis=1)
        _SHUFFLE_CACHE[key] = index
    return _SHUFFLE_CACHE[key]


def shuffled_e_rows(j: Judgement, draws: int, seed: int, leave_out: bool = True, batch: int = 50) -> np.ndarray:
    """Actor rows of E, averaged over label shuffles within actor.

    Each shuffle keeps each actor's label set, so the partial crossing is kept.
    """
    q, actor, n_actors = j.q, j.actor, len(j.actors)
    n_groups, n, families = len(j.labels), q.shape[0], q.shape[1]
    index = shuffle_index(actor, draws, seed)
    log_def = _log(default_for_clips(q, actor, n_actors, leave_out))
    fixed = weighted_log(q, log_def)
    size = n_groups * n_actors
    total = np.zeros(n_actors)
    used = np.zeros(n_actors)
    for start in range(0, draws, batch):
        groups = j.label[index[start:start + batch]]
        b = groups.shape[0]
        key = groups * n_actors + actor
        flat = (key + size * np.arange(b)[:, None]).ravel()
        counts = np.bincount(flat, minlength=b * size).reshape(b, n_groups, n_actors).astype(np.float64)
        sums = np.empty((b, n_groups, n_actors, families))
        for f in range(families):
            sums[..., f] = np.bincount(flat, weights=np.tile(q[:, f], b), minlength=b * size).reshape(b, n_groups, n_actors)
        others = counts.sum(2, keepdims=True) - counts
        with np.errstate(invalid="ignore", divide="ignore"):
            tpl = (sums.sum(2, keepdims=True) - sums) / others[..., None]
            log_tpl = np.log(tpl)
        rows = np.arange(b)[:, None]
        values = weighted_log(q[None], log_tpl[rows, groups, actor]) - fixed
        keep = ~np.isnan(values)
        for row in range(b):
            mask = keep[row]
            total += np.bincount(actor[mask], weights=values[row][mask], minlength=n_actors)
            used += np.bincount(actor[mask], minlength=n_actors)
    with np.errstate(invalid="ignore", divide="ignore"):
        return total / used


@dataclass
class Estimates:
    """A cell's clip values, actor rows and the margin. Rows are one entry per actor."""

    parts: Dict[str, Optional[np.ndarray]]
    rows: Dict[str, Optional[np.ndarray]]
    B_row: np.ndarray
    E_star_row: np.ndarray
    T: float
    margin: float  # for E* only. B has no plug-in margin: see balance_stats.


def estimate(
    j: Judgement,
    w: Optional[np.ndarray] = None,
    g: Optional[np.ndarray] = None,
    u: Optional[np.ndarray] = None,
    leave_out: bool = True,
    shuffles: int = D["label_shuffles"],
    seed: int = D["seed"],
) -> Estimates:
    parts = decompose(j, w, g, u, leave_out)
    n_actors = len(j.actors)
    rows = {k: (None if v is None else actor_means(v, j.actor, n_actors)) for k, v in parts.items()}
    T = float(np.nanmean(rows["T"]))
    return Estimates(
        parts=parts,
        rows=rows,
        B_row=rows["D"] - rows["P"],
        E_star_row=rows["E"] - shuffled_e_rows(j, shuffles, seed, leave_out),
        T=T,
        margin=D["margin_fraction"] * T,
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def bootstrap_t_maxt(matrix: np.ndarray, alternative: Sequence[int], draws: int = D["bootstrap_draws"],
                     seed: int = D["seed"]) -> Dict[str, Any]:
    """Actor bootstrap-t with Westfall-Young step-down maxT, for B.

    Whole actor rows are resampled, so dependence across columns is kept. Each draw's t is
    centred on the observed mean, so the null distribution of t is estimated from the data
    and does not assume that actor rows are symmetric, which the sign-flip does. Actor rows of
    B are skewed when actors differ in how strongly they portray (recovery run 2).
    alternative is +1, -1, or 0 for two-sided, per column. NaN marks an absent actor.
    """
    x = np.asarray(matrix, dtype=np.float64)
    keep = ~np.isnan(x).all(1)
    x = x[keep]
    n_actors, n_tests = x.shape
    signs = np.asarray(alternative, dtype=np.float64)
    observed_mean = np.nanmean(x, 0)
    present = (~np.isnan(x)).sum(0)
    observed_t = observed_mean / (np.nanstd(x, 0, ddof=1) / np.sqrt(present))
    index = np.random.default_rng(seed).integers(0, n_actors, size=(draws, n_actors))
    sample = x[index]  # draws, actors, tests
    with np.errstate(invalid="ignore", divide="ignore"):
        count = (~np.isnan(sample)).sum(1)
        mean = np.nanmean(sample, 1)
        sd = np.nanstd(sample, 1, ddof=1)
        null = (mean - observed_mean) / (sd / np.sqrt(count))
    null = np.nan_to_num(null, nan=0.0, posinf=0.0, neginf=0.0)

    def signed(t: np.ndarray) -> np.ndarray:
        return np.where(signs == 0, np.abs(t), t * signs)

    obs, nul = signed(observed_t), signed(null)
    order = np.argsort(-obs, kind="stable")
    running = np.full(draws, -np.inf)
    adjusted = np.empty(n_tests)
    for test in order[::-1]:
        running = np.maximum(running, nul[:, test])
        adjusted[test] = (int((running >= obs[test] - 1e-12).sum()) + 1) / (draws + 1)
    previous = 0.0
    for test in order:
        previous = max(previous, adjusted[test])
        adjusted[test] = previous
    raw = ((nul >= obs - 1e-12).sum(0) + 1) / (draws + 1)
    return {"t": observed_t.tolist(), "p_raw": raw.tolist(), "p_maxt": adjusted.tolist()}


def bootstrap_t_bounds(values: np.ndarray, levels: Sequence[float], draws: int = D["bootstrap_draws"],
                       seed: int = D["seed"]) -> Dict[float, Tuple[float, float]]:
    """Bootstrap-t interval of an actor mean, at each two-sided level."""
    x = values[~np.isnan(values)]
    n = x.size
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(n)
    sample = x[np.random.default_rng(seed).integers(0, n, size=(draws, n))]
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (sample.mean(1) - m) / (sample.std(1, ddof=1) / np.sqrt(n))
    t = t[np.isfinite(t)]
    out = {}
    for level in levels:
        low_q, high_q = np.quantile(t, [(1 - level) / 2, (1 + level) / 2])
        out[level] = (float(m - high_q * se), float(m - low_q * se))
    return out


def gain_heterogeneity(j: Judgement) -> Dict[str, Any]:
    """Estimated log-SD across actors of portrayal gain, from the acted family only, not from B.

    Per clip, d is the log-odds of the acted family under the judgement minus under the
    leave-one-out default. A gain that multiplies an additive push on the acted family's logit
    moves d by exactly that push, whatever the other families do. Per actor, m is the mean d
    and v its sampling variance (within-actor variance over clips / clips). Across actors, the
    variance of m less the mean v (method of moments, floored at 0) is the between-actor
    variance; for a lognormal gain, CV^2 = exp(s^2) - 1, so s = sqrt(log(1 + var / mean^2)).
    Estimable only where the mean push is clearly positive (mean m above
    `gain_heterogeneity_min_z` standard errors), since with no push there is no gain to vary,
    and skew from gain needs a push. Added after recovery run 5, as a descriptive report: a
    guard on it was tried and dropped, because at the validated gain it fired too often
    (see the draft, Changes after recovery run 5). It changes no verdict.
    """
    tiny = D["gain_heterogeneity_floor"]
    rows = np.arange(len(j.trials))
    q = np.clip(j.q[rows, j.target], tiny, 1 - tiny)
    default = default_for_clips(j.q, j.actor, len(j.actors), True)
    r = np.clip(default[rows, j.target], tiny, 1 - tiny)
    d = np.log(q / (1 - q)) - np.log(r / (1 - r))
    means, variances = [], []
    for a in range(len(j.actors)):
        x = d[(j.actor == a) & np.isfinite(d)]
        if x.size >= 2:
            means.append(x.mean())
            variances.append(x.var(ddof=1) / x.size)
    m, v = np.array(means), np.array(variances)
    if m.size < 3:
        return {"log_sd": None, "estimable": False, "mean_push": None, "actors": int(m.size)}
    centre, spread = float(m.mean()), float(m.var(ddof=1))
    estimable = centre > D["gain_heterogeneity_min_z"] * np.sqrt(spread / m.size)
    between = max(0.0, spread - float(v.mean()))
    log_sd = float(np.sqrt(np.log1p(between / centre ** 2))) if estimable else None
    return {"log_sd": log_sd, "estimable": bool(estimable), "mean_push": centre, "actors": int(m.size)}


def family_gain_log_sd(judgements: Sequence[Judgement]) -> Optional[float]:
    """The family's pooled estimate, reported beside its cells: the mean where it exists.

    Gain is a property of the actors, which every cell of a model and route shares, so the
    cells' estimates are pooled. None if no cell gives one."""
    values = [h["log_sd"] for h in (gain_heterogeneity(j) for j in judgements) if h["estimable"]]
    return float(np.mean(values)) if values else None


def family_tests(cells: Sequence[Estimates], flip_args: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    """Two step-down families across the eligible cells: B and E*.

    They answer different questions and play different parts in the decision table, so each
    is corrected only within itself. E* is one-sided above zero by A2's exact sign-flip maxT,
    whose false-positive rate held in recovery. B is two-sided and must pass two step-down
    tests: the sign-flip, exact if actor rows are symmetric, and the actor bootstrap-t, which
    does not need symmetry. Its p-value is the larger of the two, so it is valid wherever
    either test is.
    """
    if not cells:
        return []
    rows = np.column_stack([e.B_row for e in cells])
    boot = bootstrap_t_maxt(rows, [0] * len(cells))
    flip = si.signflip_maxt(rows, alternative=[0] * len(cells), **(flip_args or {}))
    e = si.signflip_maxt(np.column_stack([c.E_star_row for c in cells]), alternative=[1] * len(cells), **(flip_args or {}))
    return [{"p_B": max(boot["p_maxt"][i], flip["p_maxt"][i]), "p_E": e["p_maxt"][i],
             "p_B_raw": max(boot["p_raw"][i], flip["p_raw"][i]), "p_E_raw": e["p_raw"][i],
             "p_B_bootstrap_t": boot["p_maxt"][i], "p_B_sign_flip": flip["p_maxt"][i], "family_cells": len(cells)}
            for i in range(len(cells))]


def balance_stats(est: Estimates) -> Dict[str, Any]:
    """B with its bootstrap-t interval, and the direct test that neither part dominates.

    Even needs both ratio * P - D > 0 and ratio * D - P > 0, each with its one-sided 95%
    bootstrap-t lower bound above zero (two one-sided tests). This tests D / P between
    1 / ratio and ratio on the rows themselves, with no margin taken from an estimated T.
    """
    ratio = D["dominance_ratio"]
    P, Dv = est.rows["P"], est.rows["D"]
    b = bootstrap_t_bounds(est.B_row, [D["ci_decision"], D["ci_equivalence"]])
    lower = [bootstrap_t_bounds(ratio * P - Dv, [D["ci_equivalence"]])[D["ci_equivalence"]][0],
             bootstrap_t_bounds(ratio * Dv - P, [D["ci_equivalence"]])[D["ci_equivalence"]][0]]
    present = est.B_row[~np.isnan(est.B_row)]
    return {"mean": float(present.mean()), "ci95": list(b[D["ci_decision"]]), "ci90": list(b[D["ci_equivalence"]]),
            "neither_dominant_lower_bounds": lower}


def phi_interval(est: Estimates, draws: int = D["bootstrap_draws"], seed: int = D["seed"]) -> Optional[List[float]]:
    """90% actor-bootstrap percentile interval of Phi, a ratio of means. Descriptive."""
    top, bottom = _ratio_draws(est.rows["D"], est.rows["T"], draws, seed)
    tail = (1 - D["ci_equivalence"]) / 2
    return np.quantile(top / bottom, [tail, 1 - tail]).tolist()


def intervals(values: np.ndarray, draws: int = D["bootstrap_draws"], seed: int = D["seed"]) -> Dict[str, Any]:
    ci95 = an.actor_interval(values, D["ci_decision"], draws=draws, seed=seed)
    ci90 = an.actor_interval(values, D["ci_equivalence"], draws=draws, seed=seed)
    present = values[~np.isnan(values)]
    return {"mean": float(present.mean()) if present.size else float("nan"),
            "ci95": list(ci95) if ci95 else None, "ci90": list(ci90) if ci90 else None}


def inside(interval: Optional[Sequence[float]], margin: float) -> bool:
    return interval is not None and bool(margin > 0) and interval[0] > -margin and interval[1] < margin


def evidence_class(e: Mapping[str, Any], p: float, margin: float, gate: Mapping[str, Any], positive_control: bool) -> str:
    passes = p < D["alpha"] and e["ci95"] is not None and e["ci95"][0] > 0
    if passes:
        return "Shown" if gate.get("pass") is True else "Unmapped"
    if p >= D["alpha"] and inside(e["ci90"], margin) and gate.get("B4") is not True and positive_control:
        return "Absent"
    return "Unresolved"


def balance_class(b: Mapping[str, Any], p: float) -> str:
    """Face and Default need the test and the 95% interval. Even needs B not told from zero,
    by test and by interval, and the direct two one-sided tests of neither part dominating."""
    ci = b["ci95"]
    if p < D["alpha"] and b["mean"] > 0 and ci is not None and ci[0] > 0:
        return "Face"
    if p < D["alpha"] and b["mean"] < 0 and ci is not None and ci[1] < 0:
        return "Default"
    bounds = b.get("neither_dominant_lower_bounds") or []
    if p >= D["alpha"] and ci is not None and ci[0] <= 0 <= ci[1] and len(bounds) == 2 and min(bounds) > 0:
        return "Even"
    return "Unresolved"


def preconditions_met(gates: Mapping[str, Any], gate: Mapping[str, Any]) -> bool:
    """Missing means not passed. B3 to B5 are not preconditions; a failure there is a result."""
    return all(gates.get(k) is True for k in PRECONDITION_GATES) and gate.get("B1") is True and gate.get("B2") is True


def decide(
    gates: Mapping[str, Any],
    gate: Mapping[str, Any],
    T: float,
    margin: float,
    b: Mapping[str, Any],
    e: Mapping[str, Any],
    p_B: float,
    p_E: float,
    baseline_ok: bool,
    *,
    family_cells: int,
) -> Dict[str, Any]:
    """The decision table. Every output has a verdict, and Inconclusive has a reason code.

    family_cells is the size of the B step-down family, and is required. Below
    `min_balance_cells`, a named balance class is withheld: balance reads Unresolved with
    reason `balance_small_family`, and the class it would have had is kept for the record.
    """
    flags: List[str] = []
    if not baseline_ok:
        flags.append("no_baseline")
    if not preconditions_met(gates, gate):
        return check_row({"verdict": NOT_TESTED, "reasons": [], "flags": flags, "evidence": None, "balance": None})
    if gate.get("B3") is not True:
        flags.append("b3_failed")
    if gate.get("pass") is not True:
        flags.append("readout_dependence")
    if not np.isfinite(T) or T < D["min_total_nats"]:
        return check_row({"verdict": INCONCLUSIVE, "reasons": ["no_structure"], "flags": flags, "evidence": None, "balance": None})
    positive_control = gates.get("rq0") is True and gates.get("g2_g4") is True
    evidence = evidence_class(e, p_E, margin, gate, positive_control)
    balance = balance_class(b, p_B)
    withheld = None
    if family_cells < D["min_balance_cells"] and balance in BALANCE_NAMED:
        withheld, balance = balance, "Unresolved"
    reasons = []
    if evidence == "Unresolved":
        reasons.append("evidence_unresolved")
    if withheld is not None:
        reasons.append("balance_small_family")
    elif balance == "Unresolved":
        reasons.append("balance_unresolved")
    verdict = INCONCLUSIVE if reasons else VERDICTS[(evidence, balance)]
    return check_row({"verdict": verdict, "reasons": reasons, "flags": flags, "evidence": evidence, "balance": balance,
                      "balance_withheld": withheld})


def check_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """A row with no verdict, or Inconclusive with no valid code, is a code fault."""
    named = row.get("verdict") in set(VERDICTS.values()) | {NOT_TESTED}
    coded = row.get("verdict") == INCONCLUSIVE and row.get("reasons") and set(row["reasons"]) <= set(REASONS)
    if not (named or coded):
        raise RuntimeError(f"Output row has no verdict or reason code: {row}")
    return row


# ---------------------------------------------------------------------------
# Riders
# ---------------------------------------------------------------------------


def _ratio_draws(top: np.ndarray, bottom: np.ndarray, draws: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Bootstrap numerator and denominator means over actors, on the same draws."""
    keep = ~(np.isnan(top) | np.isnan(bottom))
    top, bottom = top[keep], bottom[keep]
    index = np.random.default_rng(seed).integers(0, top.size, size=(draws, top.size))
    return top[index].mean(1), bottom[index].mean(1)


def word_share(est: Estimates, draws: int = D["bootstrap_draws"], seed: int = D["seed"]) -> Optional[Dict[str, Any]]:
    """omega = W / P with an actor-bootstrap CI. Mixed if P is not reliably above zero."""
    if est.rows["W"] is None:
        return None
    W, P = est.rows["W"], est.rows["P"]
    w_mean, p_mean = float(np.nanmean(W)), float(np.nanmean(P))
    cut = D["word_share_cut"]
    top, bottom = _ratio_draws(W, P, draws, seed)
    valid = bottom > 0
    tail = (1 - D["ci_decision"]) / 2
    if p_mean <= 0 or valid.mean() < 1 - tail:
        return {"omega": None, "P": p_mean, "ci95": None, "reading": "mixed"}
    # A draw with P at or below zero has no finite omega, so it counts in the upper tail.
    ratios = np.where(valid, top / np.where(valid, bottom, 1.0), np.inf)
    low, high = np.quantile(ratios, [tail, 1 - tail], method="nearest")
    reading = "word_prior" if low > cut else "image_or_face" if np.isfinite(high) and high < cut else "mixed"
    return {"omega": w_mean / p_mean, "P": p_mean, "ci95": [float(low), None if not np.isfinite(high) else float(high)], "reading": reading}


def chain(est: Estimates, j: Judgement, base: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Means of W, G, F, E, R and lexicon mass, with G and F merged at low grey mass."""
    if not base["ok"]:
        return None
    rows = est.rows
    out: Dict[str, Any] = {k: float(np.nanmean(rows[k])) for k in ("W", "E", "R")}
    merged = base["mass_grey"] < D["grey_mass_floor"]
    if merged:
        out["G_plus_F"] = float(np.nanmean(rows["G"] + rows["F"]))
    else:
        out["G"], out["F"] = float(np.nanmean(rows["G"])), float(np.nanmean(rows["F"]))
    out.update({"G_F_merged": merged, "mass_real_median": float(np.median(j.mass)),
                "mass_grey": base["mass_grey"], "mass_image_free": base["mass_image_free"]})
    return out


def survival(j: Judgement, partners: Mapping[str, str], draws: int = D["bootstrap_draws"], seed: int = D["seed"]) -> Dict[str, Any]:
    """S: probability left on right answers when the face is wrong. Clips average within actor."""
    place = {t: i for i, t in enumerate(j.trials)}
    correct = [i for i, t in enumerate(j.trials) if j.q[i].argmax() == j.target[i] and partners.get(t) in place]
    actors_with = {int(j.actor[i]) for i in correct}
    out: Dict[str, Any] = {"n_correct_items": len(correct), "n_correct_actors": len(actors_with), "S": None, "ci95": None}
    if len(correct) < D["min_correct_items"] or len(actors_with) < D["min_correct_actors"]:
        return out
    chosen = np.array(correct)
    own = j.q[chosen, j.target[chosen]]
    swapped = j.q[[place[partners[j.trials[i]]] for i in correct], j.target[chosen]]
    n_actors = len(j.actors)
    top = actor_means(swapped, j.actor[chosen], n_actors)
    bottom = actor_means(own, j.actor[chosen], n_actors)
    a, b = _ratio_draws(top, bottom, draws, seed)
    tail = (1 - D["ci_decision"]) / 2
    low, high = np.quantile(a / b, [tail, 1 - tail])
    out.update({"S": float(np.nanmean(top) / np.nanmean(bottom)), "ci95": [float(low), float(high)]})
    return out


# ---------------------------------------------------------------------------
# One model and route
# ---------------------------------------------------------------------------


@dataclass
class Cell:
    """Everything a cell needs before the family test."""

    model: str
    route: str
    j: Judgement
    est: Estimates
    gates: Dict[str, Any]
    gate: Dict[str, Any]
    base: Dict[str, Any]
    partners: Mapping[str, str]

    @property
    def eligible(self) -> bool:
        return preconditions_met(self.gates, self.gate)


def build_cell(
    model: str,
    route: str,
    cells: an.Cells,
    actors: Sequence[str],
    partners: Mapping[str, str],
    base: Dict[str, Any],
    gates: Dict[str, Any],
    repeat: float,
    shuffles: int = D["label_shuffles"],
) -> Cell:
    j = judgement(cells, route, actors)
    gate = an.gate_b(cells, route, dict(partners), base["image_free_family_logprob"], repeat)
    return Cell(model, route, j, estimate(j, base["w"], base["g"], shuffles=shuffles), gates, gate, base, partners)


def finish_cell(cell: Cell, p: Optional[Mapping[str, float]], gain_log_sd: Optional[float]) -> Dict[str, Any]:
    """The cell's output row. p is its step-down result, or None for a Not tested cell.
    gain_log_sd is the family's pooled gain estimate (family_gain_log_sd), reported only."""
    est = cell.est
    b, e = balance_stats(est), intervals(est.E_star_row)
    share = word_share(est) if cell.base["ok"] else None
    alive = survival(cell.j, cell.partners)
    p_B = p["p_B"] if p else float("nan")
    p_E = p["p_E"] if p else float("nan")
    row = decide(cell.gates, cell.gate, est.T, est.margin, b, e, p_B, p_E, cell.base["ok"],
                 family_cells=p.get("family_cells", 0) if p else 0)
    if alive["S"] is not None and alive["S"] >= D["survival_cut"] and row["verdict"] == "Evidence-led":
        row["flags"].append("survival_conflict")
    return {
        **row,
        "model": cell.model,
        "route": cell.route,
        "n_clips": len(cell.j.trials),
        "T": est.T,
        "margin": est.margin,
        "Phi": float(np.nanmean(est.rows["D"])) / est.T if est.T else None,
        "Phi_ci90": phi_interval(est),
        "E_star_over_D": float(np.nanmean(est.E_star_row)) / float(np.nanmean(est.rows["D"])),
        "P": float(np.nanmean(est.rows["P"])),
        "D": float(np.nanmean(est.rows["D"])),
        "B": b,
        "E_star": e,
        "p": dict(p) if p else None,
        "gates": cell.gates,
        "gate_b": {k: cell.gate.get(k) for k in ("B1", "B2", "B3", "B4", "B5", "pass", "B3_rms_nats", "B4_mean", "B5_mean")},
        "baseline_ok": cell.base["ok"],
        "baseline_spread_nats": cell.base["spread_nats"],
        "word_share": share,
        "chain": chain(est, cell.j, cell.base),
        "survival": alive,
        "clips_without_template": int(np.isnan(est.parts["E"]).sum()),
        "gain_heterogeneity": gain_heterogeneity(cell.j),
        "family_gain_log_sd": gain_log_sd,
    }


def analyse_family(cells: Sequence[Cell], flip_args: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    """Step-down over eligible cells only, which gates settle before any contrast is read."""
    eligible = [c for c in cells if c.eligible]
    tests = dict(zip((id(c) for c in eligible), family_tests([c.est for c in eligible], flip_args)))
    spread = family_gain_log_sd([c.j for c in eligible])
    return [finish_cell(c, tests.get(id(c)), spread) for c in cells]


# ---------------------------------------------------------------------------
# Humans
# ---------------------------------------------------------------------------


def log_family(p: np.ndarray) -> Dict[str, float]:
    return dict(zip(FAMILIES, np.log(p).tolist()))


def human_divergence_rows(j: Judgement, human: Mapping[str, Mapping[str, float]], *, leave_out: bool = True) -> Dict[str, np.ndarray]:
    """H1 at clip level: JS for the default and for the template, minus JS for the judgement."""
    default = default_for_clips(j.q, j.actor, len(j.actors), leave_out)
    tpl = templates(j.q, j.actor, j.label, len(j.actors), len(j.labels))[j.label, j.actor]
    delta_def, delta_tpl = np.full(len(j.trials), np.nan), np.full(len(j.trials), np.nan)
    for i, trial in enumerate(j.trials):
        if trial not in human:
            continue
        own = si.js_divergence(*human_comparable(log_family(j.q[i]), human[trial]))
        delta_def[i] = si.js_divergence(*human_comparable(log_family(default[i]), human[trial])) - own
        if not np.isnan(tpl[i]).any():
            delta_tpl[i] = si.js_divergence(*human_comparable(log_family(tpl[i]), human[trial])) - own
    return {"default": delta_def, "template": delta_tpl}


def point_mass(distribution_of: Mapping[str, float], modal: str) -> Dict[str, float]:
    return {k: float(k == modal) for k in distribution_of}


def h1(cells: Sequence[Cell], human: Mapping[str, Mapping[str, float]], flip_args: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Delta J, with its own two-sided step-down family across the eligible cells."""
    if not cells:
        return {}
    rows = [human_divergence_rows(c.j, human) for c in cells]
    n_actors = len(cells[0].j.actors)
    matrix = np.column_stack([actor_means(r["default"], c.j.actor, n_actors) for r, c in zip(rows, cells)])
    tests = si.signflip_maxt(matrix, **(flip_args or {}))
    out = {}
    for index, (cell, row) in enumerate(zip(cells, rows)):
        column = matrix[:, index]
        out[f"{cell.model}|{cell.route}"] = {
            "delta_J": intervals(column),
            "p_maxt_two_sided": tests["p_maxt"][index],
            "template_form": intervals(actor_means(row["template"], cell.j.actor, n_actors)),
        }
    return out


def target_share(j: Judgement, human: Mapping[str, Mapping[str, float]], target_of: Mapping[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    """Model share of the target on the item's named options, and human p_target as in A1."""
    model, people = np.full(len(j.trials), np.nan), np.full(len(j.trials), np.nan)
    for i, trial in enumerate(j.trials):
        if trial not in human:
            continue
        names = [o for o in sorted(human[trial]) if o != ABSTAIN]
        shares, _ = human_comparable(log_family(j.q[i]), human[trial])
        model[i] = shares[names.index(target_of[trial])]
        people[i] = human[trial][target_of[trial]]
    return model, people


def within_label_r(x: np.ndarray, y: np.ndarray, group: np.ndarray, actor: np.ndarray, n_actors: int,
                   draws: int = D["bootstrap_draws"], seed: int = D["seed"]) -> Dict[str, Any]:
    """H2: pooled within-label r, as in A1, with an actor-bootstrap CI. Descriptive."""
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y, group, actor = x[keep], y[keep], group[keep], actor[keep]
    n_groups = int(group.max()) + 1
    key = group * n_actors + actor
    size = n_groups * n_actors

    def binned(values: np.ndarray) -> np.ndarray:
        return np.bincount(key, weights=values, minlength=size).reshape(n_groups, n_actors).astype(np.float64)

    n, sx, sy = binned(np.ones_like(x)), binned(x), binned(y)
    xx, yy, xy = binned(x * x), binned(y * y), binned(x * y)

    def r_of(weights: np.ndarray) -> np.ndarray:
        """weights is draws by actors: how often each actor is drawn."""
        def total(table: np.ndarray) -> np.ndarray:
            return weights @ table.T

        count, tx, ty = total(n), total(sx), total(sy)
        with np.errstate(invalid="ignore", divide="ignore"):
            cx = np.where(count > 0, tx * tx / count, 0.0)
            cy = np.where(count > 0, ty * ty / count, 0.0)
            cxy = np.where(count > 0, tx * ty / count, 0.0)
            ssx = (total(xx) - cx).sum(1)
            ssy = (total(yy) - cy).sum(1)
            sxy = (total(xy) - cxy).sum(1)
            return sxy / np.sqrt(ssx * ssy)

    observed = float(r_of(np.ones((1, n_actors)))[0])
    rng = np.random.default_rng(seed)
    counts = np.zeros((draws, n_actors))
    picks = rng.integers(0, n_actors, size=(draws, n_actors))
    np.add.at(counts, (np.arange(draws)[:, None], picks), 1.0)
    tail = (1 - D["ci_decision"]) / 2
    low, high = np.nanquantile(r_of(counts), [tail, 1 - tail])
    return {"r": observed, "ci95": [float(low), float(high)], "n_items": int(x.size), "status": "descriptive, no noise ceiling"}


def h2(j: Judgement, human: Mapping[str, Mapping[str, float]], target_of: Mapping[str, str], group: Optional[np.ndarray] = None) -> Dict[str, Any]:
    model, people = target_share(j, human, target_of)
    return within_label_r(model, people, j.label if group is None else group, j.actor, len(j.actors))


def lambda_share(q: np.ndarray, actor: np.ndarray, label: np.ndarray, n_actors: int, n_labels: int) -> Dict[str, np.ndarray]:
    """H3's E and P on a fixed option set, as actor rows. Each row is one actor's clip mean."""
    j = Judgement(tuple(map(str, range(len(q)))), q, actor, label, label, tuple(map(str, range(n_actors))),
                  tuple(map(str, range(n_labels))), np.ones(len(q)))
    parts = decompose(j)
    return {"E": actor_means(parts["E"], actor, n_actors), "P": actor_means(parts["P"], actor, n_actors)}


def h3_difference(human_q: np.ndarray, model_q: np.ndarray, actor: np.ndarray, label: np.ndarray,
                  n_actors: int, n_labels: int, draws: int = D["bootstrap_draws"], seed: int = D["seed"]) -> Dict[str, Any]:
    """Lambda of model minus Lambda of humans, paired by actor, against `lambda_margin`."""
    a = lambda_share(human_q, actor, label, n_actors, n_labels)
    b = lambda_share(model_q, actor, label, n_actors, n_labels)
    index = np.random.default_rng(seed).integers(0, n_actors, size=(draws, n_actors))

    def lam(e: np.ndarray, p: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return e / (e + p)

    diffs = lam(b["E"][index].mean(1), b["P"][index].mean(1)) - lam(a["E"][index].mean(1), a["P"][index].mean(1))
    human_lambda = float(lam(a["E"].mean(), a["P"].mean()))
    model_lambda = float(lam(b["E"].mean(), b["P"].mean()))
    ci95 = np.quantile(diffs, [(1 - D["ci_decision"]) / 2, (1 + D["ci_decision"]) / 2]).tolist()
    ci90 = np.quantile(diffs, [(1 - D["ci_equivalence"]) / 2, (1 + D["ci_equivalence"]) / 2]).tolist()
    return {"lambda_human": human_lambda, "lambda_model": model_lambda, "difference": model_lambda - human_lambda,
            "ci95": ci95, "ci90": ci90, "within_lambda_margin": inside(ci90, D["lambda_margin"])}


# ---------------------------------------------------------------------------
# Sensitivities: reported beside the primary, and none changes a verdict
# ---------------------------------------------------------------------------


def quick_row(est: Estimates) -> Dict[str, Any]:
    """One sensitivity: each test alone, with no family correction, as the primary tests run."""
    b = family_tests([est])[0]
    return {
        "T": est.T,
        "Phi": float(np.nanmean(est.rows["D"])) / est.T if est.T else None,
        "B": balance_stats(est),
        "E_star": intervals(est.E_star_row),
        "p_B_alone": b["p_B"],
        "p_E_alone": b["p_E"],
        "note": "each test alone, not corrected for a family",
    }


def label_tables(est: Estimates, j: Judgement) -> Dict[str, Any]:
    """W, E and R by label. Labels sit in partly crossed actor sets, so this is descriptive."""
    out: Dict[str, Any] = {"note": "descriptive; partial crossing of labels and actors"}
    for index, name in enumerate(j.labels):
        mask = j.label == index
        row = {"n_clips": int(mask.sum()), "n_actors": len(set(j.actor[mask].tolist()))}
        for key in ("W", "E", "R"):
            values = est.parts[key]
            row[key] = None if values is None else float(np.nanmean(actor_means(values[mask], j.actor[mask], len(j.actors))))
        out[name] = row
    return out


def sensitivities(
    cell: Cell,
    canonical: Judgement,
    modal_of: Mapping[str, str],
    small_clips: Optional[Sequence[str]],
    weak: Sequence[str],
    smoke: Sequence[str] = an.SMOKE_ITEMS,
    shuffles: int = D["label_shuffles"],
) -> Dict[str, Any]:
    """Every sensitivity the draft lists. small_clips None means geometry could not be read."""
    j, w, g = cell.j, cell.base["w"], cell.base["g"]

    def run_row(view: Judgement, **kwargs: Any) -> Optional[Dict[str, Any]]:
        if len(set(view.actor.tolist())) < 2:
            return None  # nothing left to compare, as when geometry says no clip is large enough
        return quick_row(estimate(view, w, g, shuffles=shuffles, **kwargs))

    def without(drop: Sequence[str]) -> Judgement:
        dropped = set(drop)
        return j.where(np.array([t not in dropped for t in j.trials]))

    basic = np.array([j.labels[k] in an.BASIC_LABELS for k in j.label])
    modal = j.grouped_by([modal_of[t] for t in j.trials])
    modal_est = estimate(modal, w, g, shuffles=shuffles)
    out: Dict[str, Any] = {
        "without_smoke_items": run_row(without(smoke)),
        "without_flagged_items": run_row(without(weak)),
        "templates_by_human_modal_label": {**quick_row(modal_est),
                                           "clips_without_template": int(np.isnan(modal_est.parts["E"]).sum())},
        "clips_at_token_floor_or_more": None if small_clips is None else run_row(without(small_clips)),
        "canonical_word_only": quick_row(estimate(canonical, shuffles=shuffles)),
        "flat_reference_by_family_size": run_row(j, u=size_reference()),
        "default_with_all_actors": run_row(j, leave_out=False),
        "temperature": {str(t): quick_row(estimate(j.tempered(t), shuffles=shuffles)) for t in D["temperature_sensitivity"]},
        "basic_labels_only_descriptive": run_row(j.where(basic)),
        "complex_labels_only_descriptive": run_row(j.where(~basic)),
        "tables_by_label": label_tables(cell.est, j),
    }
    return out


def human_repeats(cell: Cell, human: Mapping[str, Mapping[str, float]], modal_of: Mapping[str, str],
                  acted_of: Mapping[str, str], weak: Sequence[str]) -> Dict[str, Any]:
    """H1 and H2 without flagged items and against the human modal label, per A2."""
    j = cell.j
    n_actors = len(j.actors)
    dropped = set(weak)
    keep = np.array([t not in dropped for t in j.trials])
    trimmed = j.where(keep)
    pointed = {t: point_mass(human[t], modal_of[t]) for t in j.trials if t in human}

    def delta(view: Judgement, target: Mapping[str, Mapping[str, float]]) -> Dict[str, Any]:
        return intervals(actor_means(human_divergence_rows(view, target)["default"], view.actor, n_actors))

    return {
        "delta_J_without_flagged": delta(trimmed, human),
        "delta_J_against_modal_point_mass": delta(j, pointed),
        "h2_acted": h2(j, human, acted_of),
        "h2_without_flagged": h2(trimmed, human, acted_of),
        "h2_against_human_mode": h2(j, human, modal_of),
    }


def leakage_table(j: Judgement, human: Mapping[str, Mapping[str, float]], acted_of: Mapping[str, str], weak: Sequence[str]) -> Dict[str, Any]:
    """On flagged items: model accuracy on named options against mean human share. A flag only."""
    flagged = [t for t in j.trials if t in set(weak) and t in human]
    if not flagged:
        return {"n_items": 0}
    shares, _ = target_share(j, human, acted_of)
    place = {t: i for i, t in enumerate(j.trials)}
    hits = []
    for trial in flagged:
        i = place[trial]
        named = human_comparable(log_family(j.q[i]), human[trial])[0]
        hits.append(shares[i] == max(named))
    return {"n_items": len(flagged), "model_accuracy": float(np.mean(hits)),
            "mean_human_p_target": float(np.mean([human[t][acted_of[t]] for t in flagged]))}


# ---------------------------------------------------------------------------
# Real cells
# ---------------------------------------------------------------------------


def partners_for(items: Mapping[str, Mapping[str, Any]]) -> Dict[str, str]:
    from paper_v2.option_score import vision_mismatch_partners

    ids = sorted(items)
    return vision_mismatch_partners(ids, [items[i]["label"] for i in ids], [items[i]["actor"] for i in ids], seed=D["seed"])


def shard_paths(folder: Path, model: str) -> Tuple[List[Path], List[Path]]:
    """Full-run files only: a smoke run has `_smoke` in its stem and fits neither pattern."""
    cells = sorted([*folder.glob(f"cells_{model}.jsonl"), *folder.glob(f"cells_{model}_shard*.jsonl")])
    reports = sorted([*folder.glob(f"run_{model}.json"), *folder.glob(f"run_{model}_shard*.json")])
    return cells, reports


ROOT = a5_design.ROOT
A3_DIR = ROOT / an.A3_OUTPUT_DIR


def check_a3_pin() -> None:
    """A5 reads A3 only through A2's a3_summary, which accepts only the bytes A2 pins. This
    stops the run if A2's pin is not the one in A5's block."""
    if an.A3_SUMMARY_SHA256 != D["a3_summary_sha256"]:
        raise SystemExit(f"A2 pins A3 summary {an.A3_SUMMARY_SHA256}, not A5's {D['a3_summary_sha256']}.")


def require_gate_files(reproduction: Path, model: str) -> None:
    """A missing gate file is an error to fix, not a failed gate, so the run stops."""
    needed = [reproduction / "rotation_summary.json", reproduction / f"vision_gates_{model}.json",
              reproduction / f"input_delivery_{model}.json"]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise SystemExit(f"Gate files missing for {model}: {missing}")


def route_gates(reports: Sequence[Dict[str, Any]], reproduction: Path, model: str, route: str, a3_dir: Path = A3_DIR) -> Dict[str, Any]:
    outside = an.outside_gates(reproduction, model, route, a3_dir)
    return {"rq0": outside["rq0"], "g2_g4": outside["g2_g4"],
            "delivery": outside["delivery_file"] and an.delivery_passed(reports, route),
            "a1": outside["a1"], "a3": outside["a3"]}


def repeat_abs(reports: Sequence[Dict[str, Any]], route: str) -> float:
    rows = an.delivery_rows(reports, route)
    return max([d["repeat_forward_max_abs"] for d in rows if "failed" not in d], default=float("nan"))


def load_model(model: str, folder: Path, a2_design: str) -> Optional[Tuple[an.Cells, List[Dict[str, Any]]]]:
    cell_paths, report_paths = shard_paths(folder, model)
    if not cell_paths or not report_paths:
        return None
    rows, reports = an.load_shards(cell_paths, report_paths, model, a2_design)
    return an.Cells(rows), reports


def analyse_runs(
    runs: Mapping[str, Optional[Tuple[an.Cells, List[Dict[str, Any]]]]],
    reproduction: Path,
    human: Mapping[str, Mapping[str, float]],
    geometry: Optional[Mapping[str, Any]],
    a3_dir: Path = A3_DIR,
    shuffles: int = D["label_shuffles"],
) -> Dict[str, Any]:
    """Every cell of every model and route: a model with no A2 cells is Not tested."""
    built: List[Cell] = []
    rows_by_model: Dict[str, Any] = {}
    out_rows: List[Dict[str, Any]] = []
    for model in MODELS:
        loaded = runs.get(model)
        if loaded is None:
            out_rows += [{"model": model, "route": r, "verdict": NOT_TESTED, "reasons": [], "flags": ["no_a2_cells"]} for r in ROUTES]
            continue
        cells, reports = loaded
        require_gate_files(reproduction, model)
        rows_by_model[model] = cells
        actors = sorted({r["actor"] for r in cells.items.values()})
        partners = partners_for(cells.items)
        for route in ROUTES:
            built.append(build_cell(model, route, cells, actors, partners, baselines(reports, route),
                                    route_gates(reports, reproduction, model, route, a3_dir), repeat_abs(reports, route)))
    finished = analyse_family(built)
    eligible = [c for c in built if c.eligible]
    extra: Dict[str, Any] = {"h1": h1(eligible, human), "h3": {"status": "not tested", "why": "A6 has not reached Stage 1"}}
    for cell, row in zip(built, finished):
        cells = rows_by_model[cell.model]
        items = cells.items
        acted_of = {t: i["label"] for t, i in items.items()}
        modal_of = {t: i["human_modal_label"] for t, i in items.items()}
        weak = [t for t, i in items.items() if i.get("weak_label")]
        reports = runs[cell.model][1]
        across = [d["tokens_across_view"] for d in an.delivery_rows(reports, cell.route) if "failed" not in d]
        small = None
        if geometry is not None:
            small = an.below_token_floor(items, float(np.median(across)) if across else None, geometry)
        row["sensitivity"] = sensitivities(cell, judgement(an.Cells(an.canonical_rows(cells.rows)), cell.route, cell.j.actors),
                                           modal_of, small, weak, shuffles=shuffles)
        row["human"] = {"h2": h2(cell.j, human, acted_of), "repeats": human_repeats(cell, human, modal_of, acted_of, weak),
                        "flagged_item_accuracy": leakage_table(cell.j, human, acted_of, weak)}
    return {"cells": out_rows + finished, **extra}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cells_dir", type=Path, default=ROOT / "results/stimulus_interventions")
    parser.add_argument("--lookup", type=Path, default=ROOT / "data/eu_emotions_human_entropy.json")
    parser.add_argument("--geometry", type=Path, default=ROOT / "results/stimulus_interventions/geometry.json")
    parser.add_argument("--reproduction", type=Path, default=ROOT / "research/reproduction")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--a5_frozen", action="store_true")
    parser.add_argument("--a2_frozen", action="store_true")
    args = parser.parse_args()
    if not (args.a5_frozen and args.a2_frozen):
        raise SystemExit("A5 is not frozen. No A2 cell is read for A5 before that.")
    a5_design.check_a2_code(D)
    a5_design.check_code_pins()
    check_a3_pin()
    if run.design_hash() != D["a2_design_sha256"]:
        raise SystemExit("A2's design hash is not the one this amendment pins.")
    run.check_geometry(args.geometry)
    geometry = json.loads(args.geometry.read_text(encoding="utf-8"))["items"]
    human = {k: v["human_response_distribution"] for k, v in json.loads(args.lookup.read_text())["trials"].items()}
    runs = {m: load_model(m, args.cells_dir, D["a2_design_sha256"]) for m in MODELS}
    out = {"design_sha256": a5_design.block_hash(), **analyse_runs(runs, args.reproduction, human, geometry)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
