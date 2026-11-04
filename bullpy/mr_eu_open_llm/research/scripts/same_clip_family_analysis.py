#!/usr/bin/env python3
"""RQ1 same-clip analysis at action-unit family level, with a whole-matrix pattern score.

Follows the 2026-11-04 decision in research/DECISIONS.md. The estimator is the one in
same_clip_cue_analysis.py, unchanged: category intercepts, cues residualised on category
and actor, fractional ridge logistic per option, ridge strength by leave-one-actor-out.
That script is not edited and its outputs stay as they are. Three things differ here.

Cues. A family score is the mean of its members' z-scores, rescaled to unit variance. The
four families are brow_raise (AU01, AU02), brow_lower (AU04, AU09), smile (AU06, AU12)
and mouth_open (AU25, AU26). Remaining AUs stay single. Groupings are fixed in FAMILIES
and are not to be revised after a recovery result.

Inference. Bootstrap max-t flagged at least one cell on about 28 percent of null
observers, because 17 clusters and options seen by a few actors make resampled standard
errors unstable. Here every test is a permutation test. Cue rows are shuffled within
category, across actors, and residualised again. Responses stay in place, so the prior
and any actor effect stay in the null, and the shuffle is the same for all options.
  pointwise      |t| of a cell against its own permutation distribution, composite cues
  family-wise    |t| of a cell against the permutation maximum over all cells (max-T)
  two-detector   detectors agree in sign and min(|t_a|, |t_b|) exceeds its joint
                 permutation distribution, with the same shuffles on both detectors
A t here is a coefficient over the permutation SD of that cell. A flag needs p strictly
below the stated level. Ridge strength stays fixed across shuffles. Bootstrap intervals
are still reported for human matrices as a range, but make no decision.

Pattern score. The Pearson correlation over all cells between one observer's matrix and
a reference matrix on the same clips, cues and ridge strength. Its null shuffles cue
rows for the observer only, keeping the reference fixed. Recovery estimates it from pairs
of simulated observers, so noise in the human reference counts against it.

Second attempt. The within-category shuffle above is attempt 1 and its outputs stand as
the first result: it failed on actor-prior observers. `--attempt-2` is a post-hoc second
attempt prompted by that failure, not a replacement. It shuffles within actor, with new
seeds, the same thresholds, families, pattern score and conditions, and frozen ridge
strength per input as a model would take it. Its outputs have the prefix same_clip_family_a2.

Recovery criteria are fixed in FAMILY_RECOVERY_CRITERIA before any full recovery run, and
every recovery output carries their hash. No rotation result is read anywhere here. This
script takes no model responses; the frozen analysis runs them later.

Usage:
    python3 research/scripts/same_clip_family_analysis.py --write-criteria
    python3 research/scripts/same_clip_family_analysis.py --recovery --cue-set families \
        --au-csv research/reproduction/eu_au_measures.csv
    python3 research/scripts/same_clip_family_analysis.py --au-csv ... --graphau-csv ...
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
_name = "same_clip_cue_analysis"
if _name in sys.modules:
    scc = sys.modules[_name]
else:
    _spec = importlib.util.spec_from_file_location(_name, Path(__file__).with_name(f"{_name}.py"))
    scc = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = scc
    _spec.loader.exec_module(scc)

FAMILIES = {
    "brow_raise": ("AU01", "AU02"),
    "brow_lower": ("AU04", "AU09"),
    "smile": ("AU06", "AU12"),
    "mouth_open": ("AU25", "AU26"),
}
SINGLES_TWO_DETECTORS = ("AU07", "AU10", "AU14", "AU15", "AU17", "AU23", "AU24")
CUE_SETS = {
    "families": tuple(FAMILIES),
    "families_singles": tuple(FAMILIES) + SINGLES_TWO_DETECTORS,
    "au_exploratory": tuple(scc.SHARED_AUS) + SINGLES_TWO_DETECTORS,
}
SECOND_DETECTOR_SINGLES = "graphau"
# AU07, AU10, AU14, AU23 and AU24 have no published LibreFace intensity PCC, because
# LibreFace gives them as occurrence only. 0.50 is assumed. An empirical check is also run.
ASSUMED_PCC = {"AU07": 0.50, "AU10": 0.50, "AU14": 0.50, "AU23": 0.50, "AU24": 0.50}
AU_PCC = {**scc.PUBLISHED_PCC, **ASSUMED_PCC}

POINTWISE_ALPHA = 0.05
TWO_DETECTOR_ALPHA = 0.01
FAMILYWISE_ALPHA = 0.05
PATTERN_ALPHA = 0.05
HUMAN_SCALE_GRID = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
HALF_SHARED = 0.5  # true weight correlation of the partly matching observer

# Fixed before any full recovery run. The first two blocks repeat the earlier thresholds.
# The family-wise limit is new. Its allowance is Monte Carlo error only: the observed share
# may exceed 0.05 by at most two binomial standard errors at the run's n_sims. The pointwise
# and two-detector limits take no allowance. Thresholds are not to change after a run.
FAMILY_RECOVERY_CRITERIA = {
    "cue_observer": dict(scc.RECOVERY_CRITERIA["cue_observer"]),
    "null_observer": {
        **scc.RECOVERY_CRITERIA["null_observer"],
        "any_cell_familywise_max": 0.05,
        "any_cell_familywise_mc_se_allowance": 2.0,
    },
    "pattern_score": {
        "size_max": 0.05,
        "size_mc_se_allowance": 2.0,
        "power_min": 0.80,
    },
    "decision_rule": (
        "A condition passes only if it meets every limit for its observer type. Expected "
        "failures are not excused. Groupings, estimator, thresholds and the condition list "
        "stay fixed once a full run has started."
    ),
}


ATTEMPT_2 = {
    "label": "Post-hoc second attempt, prompted by the actor-prior failure of attempt 1. "
             "Not a replacement. Attempt 1 outputs stay unchanged and on record.",
    "scheme": "within_actor",
    "seed_offset": 100000,
    "frozen_lambda": "per input: composite, first and second detector each take the ridge "
                     "strength fixed on the human matrix for that input (attempt 1 used the "
                     "composite value on all three)",
    "unchanged": ["FAMILY_RECOVERY_CRITERIA thresholds", "family-wise limit .081 at 200 "
                  "observers", "FAMILIES", "pattern score", "estimator", "condition list",
                  "test levels"],
    "pilot_before_lock": {
        "cue_source": "synthetic cues only; no measured-cue recovery of this scheme ran first",
        "n_observers_per_condition": 60,
        "n_shuffles": 199,
        "within_actor": {"prior_only": [0.0424, 0.0034, 0.0167],
                         "actor_prior": [0.0413, 0.0047, 0.0833],
                         "sparse_0.5_tpr": [0.3333, 0.0526, 0.1327]},
        "within_cell": {"prior_only": [0.0718, 0.0225, 0.7667],
                        "actor_prior": [0.0642, 0.0208, 0.6833],
                        "sparse_0.5_tpr": [0.1431, 0.0, 0.037]},
        "actor_sign": {"prior_only": [0.0388, 0.0045, 0.0],
                       "actor_prior": [0.043, 0.0044, 0.0833],
                       "sparse_0.5_tpr": [0.2982, 0.0184, 0.0855]},
        "order_of_values": "pointwise, two-detector, family-wise",
        "choice": "within_actor: calibrated on both nulls in the pilot, with the most power "
                  "of the calibrated schemes",
    },
    "note_on_mechanism": "The between-actor share of cue variance on option clips is .3346 "
                         "observed, .2721 within category, .2714 within actor. Within-actor "
                         "shuffling does not restore that share, so the share alone does not "
                         "explain the pilot's improvement.",
}


def criteria_hash() -> str:
    text = json.dumps(FAMILY_RECOVERY_CRITERIA, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_criteria(path: Path):
    source = Path(__file__).read_bytes()
    record = {
        "criteria": FAMILY_RECOVERY_CRITERIA,
        "criteria_sha256": criteria_hash(),
        "script_sha256_at_lock": hashlib.sha256(source).hexdigest(),
        "families": {k: list(v) for k, v in FAMILIES.items()},
        "cue_sets": {k: list(v) for k, v in CUE_SETS.items()},
        "conditions": [c["name"] for c in family_conditions(lam_frozen=1.0)],
        "pattern_conditions": [c["name"] for c in pattern_conditions(1.0)],
        "levels": {"pointwise": POINTWISE_ALPHA, "two_detector": TWO_DETECTOR_ALPHA,
                   "familywise": FAMILYWISE_ALPHA, "pattern": PATTERN_ALPHA},
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return record


# --------------------------------------------------------------------------
# Cues
# --------------------------------------------------------------------------


def cue_members(cue):
    return FAMILIES.get(cue, (cue,))


def needed_aus(cue_names):
    return [au for cue in cue_names for au in cue_members(cue)]


def cue_matrix(au: np.ndarray, au_names, cue_names) -> np.ndarray:
    """Family or single scores, unit variance over clips, from an AU array."""
    z = (au - au.mean(0)) / au.std(0)
    where = {name: i for i, name in enumerate(au_names)}
    columns = []
    for cue in cue_names:
        score = z[:, [where[m] for m in cue_members(cue)]].mean(1)
        columns.append((score - score.mean()) / score.std())
    return np.column_stack(columns)


def cue_table(au: pd.DataFrame, cue_names) -> pd.DataFrame:
    """As cue_matrix, on a clip-indexed table. Rows with a missing member are dropped."""
    frame = au[needed_aus(cue_names)].dropna()
    matrix = cue_matrix(frame.to_numpy(float), list(frame.columns), cue_names)
    return pd.DataFrame(matrix, index=frame.index, columns=list(cue_names))


def detector_aus(au_csv, detector, *, frame_set="all", stat="mean", aus=None, graphau_csv=None):
    path = graphau_csv if detector == SECOND_DETECTOR_SINGLES else au_csv
    return scc.load_cues(path, detector=detector, frame_set=frame_set, stat=stat, cues=aus)


def pair_tables(au_csv, graphau_csv, cue_names, *, frame_set="all", stat="mean"):
    """The first and second detector for a cue set, and their composite.

    LibreFace is first throughout. The second is OpenFace 3.0 for AUs it measures, and
    the added detector for the rest, so each cue has two detectors of different lineage.
    """
    aus = needed_aus(cue_names)
    first = detector_aus(au_csv, "libreface", frame_set=frame_set, stat=stat, aus=aus)
    of3_aus = [a for a in aus if a in scc.SHARED_AUS]
    other = [a for a in aus if a not in scc.SHARED_AUS]
    second = detector_aus(au_csv, "openface3", frame_set=frame_set, stat=stat, aus=of3_aus)
    if other:
        if graphau_csv is None:
            raise ValueError(f"{other} need a second detector; pass --graphau-csv")
        added = detector_aus(au_csv, SECOND_DETECTOR_SINGLES, frame_set=frame_set, stat=stat,
                             aus=other, graphau_csv=graphau_csv)
        second = second.join(added, how="inner")
    a, b = cue_table(first, cue_names), cue_table(second, cue_names)
    shared = a.index.intersection(b.index)
    both = (a.loc[shared] + b.loc[shared]) / 2
    return a, b, both / both.std()


def within_cell_agreement(a: pd.Series, b: pd.Series, categories: pd.Series):
    pair = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    actors = pair.index.map(scc.actor_of).to_numpy()
    resid = scc.residualise(pair.to_numpy(float), categories.reindex(pair.index).to_numpy(), actors)
    return {"n_clips": len(pair), "pearson": round(scc._pearson(pair.a, pair.b), 4),
            "spearman": round(scc._spearman(pair.a, pair.b), 4),
            "pearson_within_category_actor": round(scc._pearson(resid[:, 0], resid[:, 1]), 4)}


# --------------------------------------------------------------------------
# Fitting and permutation
# --------------------------------------------------------------------------


class Fitter:
    """Options by cues slopes at fixed ridge strength for any cue matrix on one design.

    Same objective as same_clip_cue_analysis.fit_matrix, without weights. Newton steps
    start from the category-only fit, which every shuffle shares, so a permutation costs
    a few iterations. Convexity means the optimum is the same; a test checks it.
    """

    def __init__(self, design: scc.Design, lam):
        self.design, self.lam = design, float(lam)
        self.parts = []
        no_cues = np.zeros((design.X.shape[0], 0))
        for j in range(len(design.options)):
            rows, block = design.base[j]
            y = design.Y[rows, j]
            coef, _ = scc.fit_option(design, j, 0.0, X=no_cues)
            self.parts.append((j, rows, block, y, coef, np.ptp(y) == 0))

    def matrix(self, X=None):
        X = self.design.X if X is None else X
        k = X.shape[1]
        B = np.zeros((len(self.parts), k))
        for j, rows, block, y, start, constant in self.parts:
            if constant:
                continue
            B[j] = self._newton(block, X[rows], y, start)[-k:]
        return B

    def _newton(self, block, X, y, start):
        D = np.hstack([block, X])
        pen = np.r_[np.zeros(block.shape[1]), np.full(X.shape[1], self.lam)]
        beta = np.r_[start, np.zeros(X.shape[1])]
        k = X.shape[1]
        for _ in range(60):
            mu = scc.expit(D @ beta)
            v = np.maximum(mu * (1 - mu), 1e-10)
            grad = D.T @ (mu - y) + pen * beta
            step = np.linalg.solve((D.T * v) @ D + np.diag(pen + 1e-8), grad)
            size = np.abs(step).max()
            beta = beta - step * min(1.0, 4.0 / size) if size > 0 else beta
            # An intercept for a category with no use of an option drifts for ever, so
            # convergence is judged on the slopes.
            if np.abs(step[-k:]).max() < 1e-6:
                break
        if not np.isfinite(beta).all():
            return scc.fit_ridge_logit(D, y, pen=pen, max_iter=40, tol=1e-6)
        return beta


def permutation_indices(categories, n_perm, seed, spec="within"):
    """Row shuffles, within category unless the spec has no category control."""
    categories = np.asarray(categories)
    n = len(categories)
    rng = np.random.default_rng(seed)
    if spec == "pooled":
        groups = [np.arange(n)]
    else:
        groups = [np.flatnonzero(categories == c) for c in np.unique(categories)]
    out = np.tile(np.arange(n), (n_perm, 1))
    for r in range(n_perm):
        for group in groups:
            out[r, group] = rng.permutation(group)
    return out


class NullPlan:
    """The null draws for one observer: row shuffles, signs by row, and whether to centre."""

    def __init__(self, indices, signs=None, centre=False):
        self.indices, self.signs, self.centre = indices, signs, centre

    def __len__(self):
        return len(self.indices)


def as_plan(plan):
    return plan if isinstance(plan, NullPlan) else NullPlan(plan)


def null_plan(categories, actors, n_perm, seed, spec="within", scheme="within_category"):
    """Null draws under one scheme.

    within_category  attempt 1. Moves cues across actors.
    within_actor     moves cues across categories inside an actor.
    within_cell      inside one actor by category cell. Clips alone in a cell stay put, so
                     the null has a fixed part, and the statistic is centred on its mean.
    actor_sign       no row moves. Each actor's cue rows flip sign together, which keeps
                     how far each actor's cues sit from zero on any option's clips.
    """
    categories, actors = np.asarray(categories), np.asarray(actors)
    if scheme == "within_category":
        return NullPlan(permutation_indices(categories, n_perm, seed, spec))
    if scheme == "within_actor":
        return NullPlan(permutation_indices(actors, n_perm, seed, "within"))
    if scheme == "within_cell":
        cells = np.char.add(np.char.add(categories.astype(str), "|"), actors.astype(str))
        return NullPlan(permutation_indices(cells, n_perm, seed, "within"), centre=True)
    if scheme == "actor_sign":
        levels, codes = np.unique(actors, return_inverse=True)
        rng = np.random.default_rng(seed)
        flips = rng.choice([-1.0, 1.0], size=(n_perm, len(levels)))
        return NullPlan(np.tile(np.arange(len(actors)), (n_perm, 1)), signs=flips[:, codes])
    raise ValueError(f"Unknown scheme {scheme}")


def control_basis(design: scc.Design):
    if design.spec == "pooled":
        return None
    blocks = [pd.get_dummies(pd.Series(design.categories)).to_numpy(float)]
    if design.spec == "within":
        blocks.append(pd.get_dummies(pd.Series(design.actors)).to_numpy(float)[:, 1:])
    Q, R = np.linalg.qr(np.hstack(blocks))
    return Q[:, np.abs(np.diag(R)) > 1e-8]


def shuffled_cues(X, indices, basis, signs=None):
    out = X[indices]
    if signs is not None:
        out = out * signs[:, None]
    if basis is not None:
        out = out - basis @ (basis.T @ out)
    sd = out.std(0)
    return out / np.where(sd > 0, sd, 1.0)


def permutation_matrices(fitter: Fitter, plan):
    """Coefficient matrices under each null draw, with one draw shared across options."""
    plan = as_plan(plan)
    basis = control_basis(fitter.design)
    X = fitter.design.X
    signs = plan.signs if plan.signs is not None else [None] * len(plan)
    return np.stack([fitter.matrix(shuffled_cues(X, row, basis, sign))
                     for row, sign in zip(plan.indices, signs)])


def studentise(B, null):
    sd = null.std(0, ddof=1)
    safe = np.where(sd > 0, sd, np.inf)
    return B / safe, null / safe, sd


def exceedance_p(observed, null):
    return (1 + (null >= observed).sum(0)) / (len(null) + 1)


def cell_tests(T, T_null):
    """Pointwise and max-T family-wise p per cell, from permutation statistics."""
    size = np.abs(T)
    pointwise = exceedance_p(size, np.abs(T_null))
    peak = np.abs(T_null).max(axis=(1, 2))
    familywise = (1 + (peak[:, None, None] >= size).sum(0)) / (len(peak) + 1)
    return pointwise, familywise


def two_detector_stat(Ta, Tb):
    return np.where(np.sign(Ta) == np.sign(Tb), np.minimum(np.abs(Ta), np.abs(Tb)), 0.0)


def two_detector_p(Ta, Tb, Ta_null, Tb_null):
    """Joint test on matched shuffles. Sign disagreement or a zero cell gives p = 1."""
    observed = two_detector_stat(Ta, Tb)
    p = exceedance_p(observed, two_detector_stat(Ta_null, Tb_null))
    return np.where(observed > 0, p, 1.0)


def pattern_score(B, reference):
    return scc._pearson(B, reference)


def pattern_test(B, reference, null):
    """One-sided test of agreement with a fixed reference, against shuffled cues."""
    r = np.ravel(reference) - np.mean(reference)
    n = null.reshape(len(null), -1)
    n = n - n.mean(1, keepdims=True)
    norm = np.linalg.norm(n, axis=1) * np.linalg.norm(r)
    scores = (n @ r) / np.where(norm > 0, norm, np.inf)
    score = pattern_score(B, reference)
    return {"score": score, "p": float((1 + (scores >= score).sum()) / (len(scores) + 1)),
            "null_sd": float(scores.std(ddof=1))}


def observer_inference(design, lam, indices, *, with_null=True):
    fitter = Fitter(design, lam)
    B = fitter.matrix()
    out = {"lam": lam, "B": B, "fitter": fitter}
    if with_null:
        plan = as_plan(indices)
        null = permutation_matrices(fitter, plan)
        if plan.centre:
            middle = null.mean(0)
            T, T_null, sd = studentise(B - middle, null - middle)
        else:
            T, T_null, sd = studentise(B, null)
        pointwise, familywise = cell_tests(T, T_null)
        out.update({"null": null, "T": T, "T_null": T_null, "perm_sd": sd,
                    "p_pointwise": pointwise, "p_familywise": familywise})
    return out


# --------------------------------------------------------------------------
# Synthetic observers
# --------------------------------------------------------------------------


def write_attempt_2_criteria(path: Path):
    """Lock for the second attempt, with hashes of the attempt 1 outputs it leaves alone."""
    root = ROOT / "research/reproduction"
    first = {}
    for name in ("same_clip_family_criteria.json", "same_clip_family_recovery.json",
                 "same_clip_family_recovery_families_singles.json", "same_clip_family.json",
                 "same_clip_family_matrix.csv", "same_clip_family_verdicts.csv"):
        first[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    record = {
        "attempt": 2,
        **ATTEMPT_2,
        "criteria": FAMILY_RECOVERY_CRITERIA,
        "criteria_sha256": criteria_hash(),
        "familywise_limit_at_200_observers": 0.0808,
        "script_sha256_at_lock": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "conditions": [c["name"] for c in family_conditions(lam_frozen=(1.0, 1.0, 1.0))],
        "pattern_conditions": [c["name"] for c in pattern_conditions(1.0)],
        "attempt_1_outputs_sha256": first,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return record


def family_conditions(lam_frozen=None):
    """Same list as the earlier recovery, on family weights, plus the human-like observer."""
    base = {"noise": True, "shared_error": 0.0, "n_resp": 0, "spec": "within", "scale": 0.0,
            "with_gain": True, "lam": None, "pcc": "published"}
    conditions = []
    for observer, scale in (("sparse", 1.0), ("sparse", 0.5), ("sparse", 0.25), ("dense", 0.3)):
        conditions.append({**base, "name": f"{observer}_{scale}_published_noise",
                           "observer": observer, "scale": scale})
    conditions += [
        {**base, "name": "sparse_0.5_perfect_detectors", "observer": "sparse", "scale": 0.5,
         "noise": False},
        {**base, "name": "sparse_0.5_shared_detector_error", "observer": "sparse", "scale": 0.5,
         "shared_error": 0.5},
        {**base, "name": "sparse_0.5_model_sampling_n100", "observer": "sparse", "scale": 0.5,
         "n_resp": 100},
        {**base, "name": "sparse_0.5_empirical_detector_agreement", "observer": "sparse",
         "scale": 0.5, "pcc": "empirical"},
        {**base, "name": "human_like_published_noise", "observer": "human_like", "scale": 1.0},
        {**base, "name": "human_like_perfect_detectors", "observer": "human_like", "scale": 1.0,
         "noise": False},
        {**base, "name": "prior_only_within", "observer": "prior_only"},
        {**base, "name": "prior_only_perfect_detectors", "observer": "prior_only", "noise": False},
        {**base, "name": "prior_only_empirical_detector_agreement", "observer": "prior_only",
         "pcc": "empirical"},
        {**base, "name": "prior_only_within_category_spec", "observer": "prior_only",
         "spec": "within_category"},
        {**base, "name": "prior_only_pooled_spec", "observer": "prior_only", "spec": "pooled"},
        {**base, "name": "actor_prior_within", "observer": "actor_prior"},
        {**base, "name": "actor_prior_within_category_spec", "observer": "actor_prior",
         "spec": "within_category"},
    ]
    if lam_frozen is not None:
        conditions += [
            {**base, "name": "sparse_0.5_frozen_lambda", "observer": "sparse", "scale": 0.5,
             "lam": lam_frozen},
            {**base, "name": "prior_only_frozen_lambda", "observer": "prior_only",
             "lam": lam_frozen},
            {**base, "name": "actor_prior_frozen_lambda", "observer": "actor_prior",
             "lam": lam_frozen},
        ]
    for i, cond in enumerate(conditions):
        cond["seed"] = 2000 + i
    return conditions


def pattern_conditions(calibrated_scale):
    """A human-like observer at the calibrated strength, half of it and twice it."""
    conditions = []
    for i, factor in enumerate((0.5, 1.0, 2.0)):
        conditions.append({"name": f"pattern_{factor}x_human_effect", "factor": factor,
                           "scale": factor * calibrated_scale, "spec": "within", "noise": True,
                           "shared_error": 0.0, "pcc": "published", "n_resp_model": 100,
                           "seed": 3000 + i})
    return conditions


def observer_weights(cond, template, n_cues, rng):
    if cond["observer"] == "human_like":
        return cond["scale"] * cond["_W_ref"]
    return scc.planted_weights(cond["observer"], template.options, n_cues, cond["scale"], rng)


def measured_inputs(T, au_names, cue_names, cond, template, rng):
    """(composite, detector a, detector b) cue matrices, with noise at the AU level."""
    if not cond["noise"]:
        return [cue_matrix(T, au_names, cue_names)]
    table = cond["_pcc_tables"][cond["pcc"]]
    pcc = np.array([table[a] for a in au_names])
    cats, actors = template.categories, template.actors
    shared = rng.normal(0, 1, T.shape)
    a, b = [cue_matrix(scc.add_detector_noise(T, (cats, actors), pcc, rng, cond["shared_error"],
                                              shared), au_names, cue_names) for _ in range(2)]
    composite = (a + b) / 2
    return [composite / composite.std(0), a, b]


def template_args(template):
    return (template.trial_ids, template.actors, template.categories, template.options,
            template.cues, template.Y)


def run_family_sim(task):
    """One synthetic observer through the family pipeline."""
    template, T0, au_names, cond, sim, n_perm, fixed_matrix = task
    rng = np.random.default_rng([cond["seed"] + cond.get("_seed_offset", 0), sim])
    cats, actors, cues = template.categories, template.actors, template.cues
    T = T0 if T0 is not None else scc.synthetic_cues(cats, actors, au_names, rng).to_numpy()
    F = cue_matrix(T, au_names, cues)
    W = observer_weights(cond, template, len(cues), rng)
    n_resp = np.full(len(cats), cond["n_resp"]) if cond["n_resp"] else cond["_n_resp"]
    Y, clean = scc.simulate_observer(template, F, W, kind=cond["observer"], rng=rng, n_resp=n_resp)
    oracle = scc.oracle_matrix(template, F, clean)
    inputs = measured_inputs(T, au_names, cues, cond, template, rng)
    designs = [scc._rebuild(template_args(template), x, Y, cond["spec"]) for x in inputs]
    indices = null_plan(designs[0].categories, designs[0].actors, n_perm,
                        77 + sim + cond.get("_seed_offset", 0), cond["spec"],
                        cond.get("_scheme", "within_category"))
    frozen = cond["lam"]
    results = []
    for position, design in enumerate(designs):
        if frozen is None:
            lam = scc.choose_lambda(design)
        else:
            # A tuple is (composite, first, second), each as fixed on the human matrix.
            lam = frozen[position] if isinstance(frozen, tuple) else frozen
        results.append(observer_inference(design, lam, indices))
    gain = None
    if cond["with_gain"]:
        gain = scc.loao_gain(designs[0], lam=results[0]["lam"])["heldout_gain"]
    return score_family_sim(designs[0], results, oracle, W, fixed_matrix, template, gain)


def score_family_sim(design, results, oracle, W, fixed_matrix, template, gain):
    res = results[0]
    B = res["B"]
    flag = res["p_pointwise"] < POINTWISE_ALPHA
    flag_fw = res["p_familywise"] < FAMILYWISE_ALPHA
    both = both_old = flag
    if len(results) > 1:
        a, b = results[1], results[2]
        both = two_detector_p(a["T"], b["T"], a["T_null"], b["T_null"]) < TWO_DETECTOR_ALPHA
        both_old = ((a["p_pointwise"] < POINTWISE_ALPHA) & (b["p_pointwise"] < POINTWISE_ALPHA)
                    & (np.sign(a["B"]) == np.sign(b["B"])))
    powered = np.array([len(template.base[j][0]) >= scc.LOW_N_ITEMS for j in range(B.shape[0])])
    planted = W != 0
    null_cells = np.abs(oracle) < 0.02 if not planted.any() else (~planted) & (np.abs(oracle) < 0.05)
    out = {
        "lam": res["lam"], "gain": gain,
        "pointwise_fpr": float(flag[null_cells].mean()) if null_cells.any() else None,
        "familywise_cell_fpr": float(flag_fw[null_cells].mean()) if null_cells.any() else None,
        "two_detector_fpr": float(both[null_cells].mean()) if null_cells.any() else None,
        "two_detector_fpr_old_rule": float(both_old[null_cells].mean()) if null_cells.any() else None,
        "any_cell_familywise": bool(flag_fw.any()),
        "any_cell_two_detector": bool(both.any()),
    }
    if planted.any():
        strong = np.abs(oracle) >= 0.25 * np.abs(W[planted]).mean()
        right = np.sign(B) == np.sign(oracle)
        out.update({
            "pearson_with_oracle": scc._pearson(B, oracle),
            "pearson_with_oracle_powered": scc._pearson(B[powered], oracle[powered]),
            "pearson_single_detector": scc._pearson(results[-1]["B"], oracle),
            "pearson_with_planted_weights": scc._pearson(B, W),
            "attenuation_slope": float((B * oracle).sum() / (oracle**2).sum()),
            "sign_accuracy_strong": float(right[strong].mean()) if strong.any() else None,
            "tpr_pointwise": float((flag & right)[strong].mean()) if strong.any() else None,
            "tpr_familywise": float((flag_fw & right)[strong].mean()) if strong.any() else None,
            "tpr_two_detector": float((both & right)[strong].mean()) if strong.any() else None,
            "n_strong_cells": int(strong.sum()),
            "per_cue_sign_accuracy": {
                cue: float(right[:, k][strong[:, k]].mean())
                for k, cue in enumerate(design.cues) if strong[:, k].any()
            },
        })
    else:
        out["pearson_with_fixed_matrix"] = (
            scc._pearson(B, fixed_matrix) if fixed_matrix is not None else None)
    return out


def _agg(rows, key, how):
    values = [r[key] for r in rows if r.get(key) is not None and not np.isnan(r[key])]
    return round(float(how(values)), 4) if values else None


def summarise_family_condition(cond, rows):
    out = {k: cond[k] for k in ("name", "observer", "scale", "noise", "spec", "shared_error", "pcc")}
    n = len(rows)
    out["n_sims"] = n
    out["median_lambda"] = _agg(rows, "lam", np.median)
    out["median_heldout_gain"] = _agg(rows, "gain", np.median)
    for key in ("pointwise_fpr", "familywise_cell_fpr", "two_detector_fpr",
                "two_detector_fpr_old_rule"):
        out[key] = _agg(rows, key, np.mean)
    out["share_sims_any_familywise_cell"] = _agg(rows, "any_cell_familywise", np.mean)
    out["share_sims_any_two_detector_cell"] = _agg(rows, "any_cell_two_detector", np.mean)
    crit = FAMILY_RECOVERY_CRITERIA
    if "pearson_with_oracle" in rows[0]:
        for key in ("pearson_with_oracle", "pearson_with_oracle_powered", "pearson_single_detector",
                    "pearson_with_planted_weights", "attenuation_slope", "sign_accuracy_strong",
                    "tpr_pointwise", "tpr_familywise", "tpr_two_detector", "n_strong_cells"):
            out[key] = _agg(rows, key, np.median)
        per_cue = pd.DataFrame([r["per_cue_sign_accuracy"] for r in rows]).mean()
        out["per_cue_sign_accuracy"] = {k: round(float(v), 3) for k, v in per_cue.items()}
        c = crit["cue_observer"]
        out["pass_pearson"] = bool(out["pearson_with_oracle"] >= c["median_pearson_min"])
        out["pass_sign"] = bool(out["sign_accuracy_strong"] is not None
                                and out["sign_accuracy_strong"] >= c["median_sign_accuracy_min"])
        out["recovered"] = out["pass_pearson"] and out["pass_sign"]
    else:
        c = crit["null_observer"]
        fixed = [abs(r["pearson_with_fixed_matrix"]) for r in rows
                 if r.get("pearson_with_fixed_matrix") is not None]
        out["median_abs_pearson_with_fixed_matrix"] = (
            round(float(np.median(fixed)), 4) if fixed else None)
        limit = c["any_cell_familywise_max"]
        limit += c["any_cell_familywise_mc_se_allowance"] * np.sqrt(limit * (1 - limit) / n)
        out["familywise_limit_with_mc_allowance"] = round(float(limit), 4)
        out["pass_pointwise"] = bool(out["pointwise_fpr"] <= c["pointwise_fpr_max"])
        out["pass_two_detector"] = bool(
            not cond["noise"] or out["two_detector_fpr"] <= c["two_detector_fpr_max"])
        out["pass_familywise"] = bool(out["share_sims_any_familywise_cell"] <= limit)
        out["pass_fixed_matrix"] = bool(
            not fixed or out["median_abs_pearson_with_fixed_matrix"]
            <= c["abs_pearson_with_any_fixed_matrix_max"])
        out["recovered"] = (out["pass_pointwise"] and out["pass_two_detector"]
                            and out["pass_familywise"] and out["pass_fixed_matrix"])
    return out


def _sim_gain(task):
    """Held-out gain of a human-like observer at one strength; no permutations."""
    template, T0, au_names, W_ref, scale, n_resp, pcc, sim, offset = task
    rng = np.random.default_rng([4000 + offset, sim, int(scale * 100)])
    cues = template.cues
    F = cue_matrix(T0, au_names, cues)
    Y, _ = scc.simulate_observer(template, F, scale * W_ref, kind="human_like", rng=rng, n_resp=n_resp)
    cond = {"noise": True, "shared_error": 0.0, "pcc": "published",
            "_pcc_tables": {"published": pcc}}
    composite = measured_inputs(T0, au_names, cues, cond, template, rng)[0]
    design = scc._rebuild(template_args(template), composite, Y, "within")
    return scc.loao_gain(design, lam=scc.choose_lambda(design))["heldout_gain"]


def calibrate_human_scale(template, T0, au_names, W_ref, n_resp, pcc, target_gain, *, n_sims=24,
                          workers=1, seed_offset=0):
    """Multiple of the human matrix whose simulated held-out gain is nearest the human's.

    The human estimate is shrunk, so weights equal to it would understate the effect.
    Only held-out gain enters, never a recovery or pattern outcome.
    """
    grid = {}
    for scale in HUMAN_SCALE_GRID:
        tasks = [(template, T0, au_names, W_ref, scale, n_resp, pcc, s, seed_offset)
                 for s in range(n_sims)]
        if workers > 1:
            with ProcessPoolExecutor(workers) as pool:
                gains = list(pool.map(_sim_gain, tasks))
        else:
            gains = [_sim_gain(t) for t in tasks]
        grid[scale] = float(np.median(gains))
    best = min(grid, key=lambda s: (abs(grid[s] - target_gain), s))
    return best, {"target_human_gain": target_gain, "n_sims": n_sims,
                  "median_gain_by_scale": {str(k): round(v, 5) for k, v in grid.items()},
                  "chosen_scale": best}


def run_pattern_sim(task):
    """A simulated human, and five observers scored against it on the same measured cues."""
    template, T0, au_names, cond, sim, n_perm = task
    rng = np.random.default_rng([cond["seed"] + cond.get("_seed_offset", 0), sim])
    cats, actors, cues = template.categories, template.actors, template.cues
    T = T0 if T0 is not None else scc.synthetic_cues(cats, actors, au_names, rng).to_numpy()
    F = cue_matrix(T, au_names, cues)
    W_human = cond["scale"] * cond["_W_ref"]
    shuffled = lambda: rng.permutation(W_human.ravel()).reshape(W_human.shape)  # noqa: E731
    W_other, W_mix = shuffled(), shuffled()
    W_half = HALF_SHARED * W_human + np.sqrt(1 - HALF_SHARED**2) * W_mix
    composite = measured_inputs(T, au_names, cues, cond, template, rng)[0]

    Y_h, _ = scc.simulate_observer(template, F, W_human, kind="human_like", rng=rng,
                                   n_resp=cond["_n_resp"])
    design_h = scc._rebuild(template_args(template), composite, Y_h, cond["spec"])
    lam = scc.choose_lambda(design_h)
    reference = Fitter(design_h, lam).matrix()

    indices = null_plan(design_h.categories, design_h.actors, n_perm,
                        177 + sim + cond.get("_seed_offset", 0), cond["spec"],
                        cond.get("_scheme", "within_category"))
    n_model = np.full(len(cats), cond["n_resp_model"])
    zero = np.zeros_like(W_human)
    out = {"lam": lam}
    observers = {"human_like": (W_human, "human_like"), "half_shared": (W_half, "human_like"),
                 "different_cues": (W_other, "human_like"), "prior_only": (zero, "prior_only"),
                 "actor_prior": (zero, "actor_prior")}
    for name, (W, kind) in observers.items():
        Y, _ = scc.simulate_observer(template, F, W, kind=kind, rng=rng, n_resp=n_model)
        design = scc._rebuild(template_args(template), composite, Y, cond["spec"])
        fitter = Fitter(design, lam)
        B = fitter.matrix()
        test = pattern_test(B, reference, permutation_matrices(fitter, indices))
        out[f"{name}_score"], out[f"{name}_p"] = test["score"], test["p"]
    return out


def auc(positive, negative):
    pos, neg = np.asarray(positive), np.asarray(negative)
    wins = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
    return float(wins)


def summarise_pattern_condition(cond, rows):
    crit = FAMILY_RECOVERY_CRITERIA["pattern_score"]
    n = len(rows)
    size_limit = crit["size_max"] + crit["size_mc_se_allowance"] * np.sqrt(
        crit["size_max"] * (1 - crit["size_max"]) / n)
    out = {"name": cond["name"], "factor": cond["factor"], "scale": cond["scale"], "n_sims": n,
           "size_limit_with_mc_allowance": round(float(size_limit), 4)}
    frame = pd.DataFrame(rows)
    for name in ("human_like", "half_shared", "different_cues", "prior_only", "actor_prior"):
        scores = frame[f"{name}_score"].dropna()
        out[name] = {"median_score": round(float(scores.median()), 4),
                     "q05": round(float(scores.quantile(0.05)), 4),
                     "q95": round(float(scores.quantile(0.95)), 4),
                     "share_p_below_alpha": round(float((frame[f"{name}_p"] < PATTERN_ALPHA).mean()), 4)}
    prior = frame.prior_only_score.dropna()
    out["auc_human_like_vs_prior_only"] = round(auc(frame.human_like_score.dropna(), prior), 4)
    out["auc_half_shared_vs_prior_only"] = round(auc(frame.half_shared_score.dropna(), prior), 4)
    out["auc_human_like_vs_different_cues"] = round(
        auc(frame.human_like_score.dropna(), frame.different_cues_score.dropna()), 4)
    out["size_prior_only"] = out["prior_only"]["share_p_below_alpha"]
    out["size_actor_prior"] = out["actor_prior"]["share_p_below_alpha"]
    out["size_different_cues"] = out["different_cues"]["share_p_below_alpha"]
    out["power_human_like"] = out["human_like"]["share_p_below_alpha"]
    out["power_half_shared"] = out["half_shared"]["share_p_below_alpha"]
    out["pass_size"] = bool(max(out["size_prior_only"], out["size_actor_prior"]) <= size_limit)
    out["pass_power"] = bool(out["power_human_like"] >= crit["power_min"])
    out["separates"] = out["pass_size"] and out["pass_power"]
    return out


def _map(fn, tasks, workers, n_sims):
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            return list(pool.map(fn, tasks, chunksize=max(1, n_sims // (4 * workers))))
    return [fn(t) for t in tasks]


def empirical_pcc(au_csv, graphau_csv, au_names, categories):
    """Correlation with truth implied by between-detector agreement within cells.

    For two detectors with independent errors r(a, b) = rel, so a detector's correlation
    with truth is sqrt(rel). Shared error would make this too high.
    """
    out = {}
    a = detector_aus(au_csv, "libreface", aus=au_names)
    for au in au_names:
        if au in scc.SHARED_AUS:
            b = detector_aus(au_csv, "openface3", aus=[au])
        else:
            b = detector_aus(au_csv, SECOND_DETECTOR_SINGLES, aus=[au], graphau_csv=graphau_csv)
        r = within_cell_agreement(a[au], b[au], categories)["pearson_within_category_actor"]
        out[au] = round(float(np.sqrt(np.clip(r, 0.01, 0.99))), 4)
    return out


def run_recovery(args):
    human = scc.load_human(args.human)
    cue_names = list(CUE_SETS[args.cue_set])
    au_names = needed_aus(cue_names)
    categories = human.drop_duplicates("trial_id").set_index("trial_id").category
    pcc_tables = {"published": AU_PCC}
    fixed = lam_real = target_gain = W_ref = None
    if args.au_csv:
        a, b, composite = pair_tables(args.au_csv, args.graphau_csv, cue_names,
                                      frame_set=args.frame_set)
        T_frame = detector_aus(args.au_csv, "libreface", frame_set=args.frame_set,
                               aus=au_names)[au_names].dropna()
        ids = sorted(set(human.trial_id) & set(T_frame.index) & set(composite.index))
        placeholder = composite.loc[ids]
        real = scc.build_design(human[human.trial_id.isin(ids)], placeholder)
        lam_real = scc.choose_lambda(real)
        if args.attempt_2:
            lam_real = tuple([lam_real] + [
                scc.choose_lambda(scc.build_design(human[human.trial_id.isin(ids)], t.loc[ids]))
                for t in (a, b)])
        lam_composite = lam_real[0] if isinstance(lam_real, tuple) else lam_real
        fixed = W_ref = Fitter(real, lam_composite).matrix()
        target_gain = scc.loao_gain(real, lam=lam_composite)["heldout_gain"]
        template = real
        T = T_frame.loc[template.trial_ids].to_numpy(float)
        T0 = (T - T.mean(0)) / T.std(0)
        pcc_tables["empirical"] = empirical_pcc(args.au_csv, args.graphau_csv, au_names, categories)
    else:
        ids = sorted(human.trial_id.unique())
        rng = np.random.default_rng(0)
        placeholder = pd.DataFrame(rng.normal(0, 1, (len(ids), len(cue_names))), index=ids,
                                   columns=cue_names)
        template = scc.build_design(human, placeholder)
        T0 = None
        W_ref = rng.normal(0, 0.3, (len(template.options), len(cue_names)))
        pcc_tables["empirical"] = AU_PCC
    n_resp = human.drop_duplicates("trial_id").set_index("trial_id").loc[template.trial_ids, "n"].to_numpy()

    calibration, scale = None, 1.0
    if T0 is not None:
        scale, calibration = calibrate_human_scale(
            template, T0, au_names, W_ref, n_resp, pcc_tables["published"], target_gain,
            workers=args.workers, seed_offset=args.seed_offset)
        print(json.dumps(calibration), flush=True)
    shared = {"_n_resp": n_resp, "_pcc_tables": pcc_tables, "_W_ref": W_ref,
              "_scheme": args.scheme, "_seed_offset": args.seed_offset}

    conditions = family_conditions(lam_real)
    if args.only:
        conditions = [c for c in conditions if c["name"] in args.only]
    report = []
    for cond in conditions:
        cond = {**cond, **shared}
        if cond["observer"] == "human_like":
            cond["scale"] = scale
        tasks = [(template, T0, au_names, cond, s, args.n_perm, fixed) for s in range(args.n_sims)]
        started = time.time()
        rows = _map(run_family_sim, tasks, args.workers, args.n_sims)
        summary = summarise_family_condition(cond, rows)
        summary["seconds"] = round(time.time() - started, 1)
        report.append(summary)
        print(json.dumps({k: v for k, v in summary.items() if k != "per_cue_sign_accuracy"}), flush=True)
    pattern_report = []
    if not args.only:
        for cond in pattern_conditions(scale):
            cond = {**cond, **shared}
            tasks = [(template, T0, au_names, cond, s, args.pattern_perm) for s in range(args.n_sims)]
            started = time.time()
            rows = _map(run_pattern_sim, tasks, args.workers, args.n_sims)
            summary = summarise_pattern_condition(cond, rows)
            summary["seconds"] = round(time.time() - started, 1)
            pattern_report.append(summary)
            print(json.dumps(summary), flush=True)
    result = {
        "attempt": 2 if args.attempt_2 else 1,
        "attempt_label": ATTEMPT_2["label"] if args.attempt_2 else "First attempt",
        "scheme": args.scheme,
        "seed_offset": args.seed_offset,
        "frame_set": args.frame_set,
        "cue_set": args.cue_set,
        "cue_source": "measured EU-Emotion cues" if T0 is not None else "synthetic cues",
        "cues": cue_names,
        "aus": au_names,
        "n_sims": args.n_sims,
        "n_perm": args.n_perm,
        "pattern_perm": args.pattern_perm,
        "criteria": FAMILY_RECOVERY_CRITERIA,
        "criteria_sha256": criteria_hash(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pcc_used": pcc_tables,
        "assumed_pcc": ASSUMED_PCC,
        "human_scale_calibration": calibration,
        "real_human_lambda": lam_real,
        "notes": [
            "True cues in simulation are the measured LibreFace AUs, then detector noise is "
            "added at the AU level, so measured cues here are noisier than the real ones.",
            "Item and actor noise are independent between simulated observers, so a shared "
            "unmeasured cue that lifts real human-model agreement is not simulated.",
            "Oracle is the matrix this analysis returns on true family scores with no noise.",
            "human_like weights are a multiple of the estimated human family matrix, chosen "
            "only by held-out gain, so they carry that matrix's own estimation error.",
        ],
        "conditions": report,
        "pattern": pattern_report,
    }
    args.recovery_out.parent.mkdir(parents=True, exist_ok=True)
    args.recovery_out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return result


# --------------------------------------------------------------------------
# Human run
# --------------------------------------------------------------------------


def _inference_task(task):
    design, lam, plan = task
    res = observer_inference(design, lam, plan)
    res.pop("fitter")
    return res


def human_matrix_table(design, res, boot, *, detector, frame_set, cue_set):
    ci = scc.intervals(design, res["B"], boot)
    ame = scc.average_marginal_effects(design, res["B"], res["lam"])
    rows = []
    for j, option in enumerate(design.options):
        items, _ = design.base[j]
        for k, cue in enumerate(design.cues):
            rows.append({
                "observer": "human", "cue_set": cue_set, "detector": detector,
                "frame_set": frame_set, "spec": design.spec, "option": option, "cue": cue,
                "beta": round(res["B"][j, k], 4), "ame": round(ame[j, k], 5),
                "ci_lo": round(ci["lo"][j, k], 4), "ci_hi": round(ci["hi"][j, k], 4),
                "t_perm": round(res["T"][j, k], 3), "p_pointwise": round(res["p_pointwise"][j, k], 5),
                "p_familywise": round(res["p_familywise"][j, k], 5),
                "n_items": len(items), "n_categories": int(len(set(design.categories[items]))),
                "n_actors": int(len(set(design.actors[items]))),
                "low_n": len(items) < scc.LOW_N_ITEMS,
            })
    return pd.DataFrame(rows)


def verdict_table(design, res_a, res_b, res_c, *, frame_set, cue_set):
    p_two = two_detector_p(res_a["T"], res_b["T"], res_a["T_null"], res_b["T_null"])
    rows = []
    for j, option in enumerate(design.options):
        for k, cue in enumerate(design.cues):
            rows.append({
                "cue_set": cue_set, "frame_set": frame_set, "option": option, "cue": cue,
                "beta_first": round(res_a["B"][j, k], 4), "beta_second": round(res_b["B"][j, k], 4),
                "beta_composite": round(res_c["B"][j, k], 4),
                "same_sign": bool(np.sign(res_a["B"][j, k]) == np.sign(res_b["B"][j, k])),
                "p_two_detector": round(float(p_two[j, k]), 5),
                "flag_two_detector": bool(p_two[j, k] < TWO_DETECTOR_ALPHA),
                "p_pointwise_composite": round(res_c["p_pointwise"][j, k], 5),
                "flag_pointwise": bool(res_c["p_pointwise"][j, k] < POINTWISE_ALPHA),
                "p_familywise_composite": round(res_c["p_familywise"][j, k], 5),
                "flag_familywise": bool(res_c["p_familywise"][j, k] < FAMILYWISE_ALPHA),
            })
    return pd.DataFrame(rows)


def third_detector_check(au_csv, graphau_csv, cue_names, human, *, frame_set, stat, reference, lam,
                         n_perm, seed):
    """Added detector alone on the families it can score; a check, not a decision."""
    aus = [a for a in needed_aus(cue_names)
           if a in pd.read_csv(graphau_csv, usecols=["cue"]).cue.unique()]
    if not aus:
        return None
    table = detector_aus(au_csv, SECOND_DETECTOR_SINGLES, frame_set=frame_set, stat=stat, aus=aus,
                         graphau_csv=graphau_csv)
    usable = [c for c in cue_names if any(m in aus for m in cue_members(c))]
    columns = {}
    for cue in usable:
        members = [m for m in cue_members(cue) if m in aus]
        z = (table[members] - table[members].mean()) / table[members].std()
        columns[cue] = z.mean(1)
    frame = pd.DataFrame(columns)
    frame = (frame - frame.mean()) / frame.std()
    design = scc.build_design(human, frame.dropna())
    B = Fitter(design, lam).matrix()
    positions = [cue_names.index(c) for c in usable]
    aligned = reference[:, positions]
    return {
        "cues": usable,
        "partial_families": {c: [m for m in cue_members(c) if m in aus] for c in usable
                             if len([m for m in cue_members(c) if m in aus]) < len(cue_members(c))},
        "pattern_with_composite": round(pattern_score(B, aligned), 4),
        "sign_agreement": round(float((np.sign(B) == np.sign(aligned)).mean()), 4),
    }


def frozen_route(human: pd.DataFrame, responses: pd.DataFrame, cue_inputs: dict, *, n_perm,
                 seed, scheme, spec="within"):
    """The route a non-human observer takes through this analysis.

    `responses` is a table in the load_human format, built by the caller; nothing here
    reads a response file. `cue_inputs` holds the composite, first and second cue tables.
    Ridge strength is fixed on the human matrix for each input and never chosen on the
    observer. Clips, options and cues must match, and the null draws are the human's.
    """
    out = {}
    for key, table in cue_inputs.items():
        design_h = scc.build_design(human, table, spec=spec)
        design_o = scc.build_design(responses, table, spec=spec)
        if (design_h.trial_ids != design_o.trial_ids or design_h.options != design_o.options
                or design_h.cues != design_o.cues):
            raise ValueError("Frozen route needs the same clips, options and cues")
        lam = scc.choose_lambda(design_h)
        plan = null_plan(design_o.categories, design_o.actors, n_perm, seed, spec, scheme)
        reference = Fitter(design_h, lam).matrix()
        res = observer_inference(design_o, lam, plan)
        out[key] = {"lam": lam, "B": res["B"], "reference": reference, "res": res,
                    "pattern": pattern_test(res["B"], reference, res["null"])}
    return out


def run_real(args):
    if args.graphau_csv is None:
        raise SystemExit("The human rerun needs --graphau-csv")
    human = scc.load_human(args.human)
    categories = human.drop_duplicates("trial_id").set_index("trial_id").category
    out = {"attempt": 2 if args.attempt_2 else 1, "scheme": args.scheme,
           "seed_offset": args.seed_offset,
           "au_csv": str(args.au_csv), "graphau_csv": str(args.graphau_csv), "spec": args.spec,
           "stat": args.stat, "n_perm": args.n_perm, "n_boot": args.n_boot,
           "criteria_sha256": criteria_hash(),
           "note": "Human observer only. Observational. No model matrix is computed here.",
           "cue_sets": {}}
    tables, verdicts = [], []
    for cue_set in args.cue_sets:
        cue_names = list(CUE_SETS[cue_set])
        block = out["cue_sets"].setdefault(cue_set, {})
        for frame_set in ("all", "model_1fps"):
            a, b, composite = pair_tables(args.au_csv, args.graphau_csv, cue_names,
                                          frame_set=frame_set, stat=args.stat)
            ids = sorted(set(a.index) & set(b.index) & set(human.trial_id))
            frames = {"first": a.loc[ids], "second": b.loc[ids], "composite": composite.loc[ids]}
            designs = {k: scc.build_design(human, v, spec=args.spec) for k, v in frames.items()}
            first = designs["first"]
            indices = null_plan(first.categories, first.actors, args.n_perm,
                                args.seed + args.seed_offset, args.spec, args.scheme)
            lams = {k: scc.choose_lambda(d) for k, d in designs.items()}
            names = list(designs)
            tasks = [(designs[k], lams[k], indices) for k in names]
            if args.workers > 1:
                with ProcessPoolExecutor(min(3, args.workers)) as pool:
                    results = dict(zip(names, pool.map(_inference_task, tasks)))
            else:
                results = dict(zip(names, map(_inference_task, tasks)))
            stats = {}
            for key in names:
                design, res = designs[key], results[key]
                boot = scc.bootstrap(design, res["lam"], args.n_boot, args.seed)
                tables.append(human_matrix_table(design, res, boot, detector=key,
                                                 frame_set=frame_set, cue_set=cue_set))
                flagged = res["p_pointwise"] < POINTWISE_ALPHA
                stats[key] = {
                    "lam": res["lam"],
                    "n_cells": int(flagged.size),
                    "n_pointwise": int(flagged.sum()),
                    "expected_pointwise_under_null": round(flagged.size * POINTWISE_ALPHA, 1),
                    "n_familywise": int((res["p_familywise"] < FAMILYWISE_ALPHA).sum()),
                    **scc.loao_gain(design, lam=res["lam"]),
                }
            table = verdict_table(designs["first"], results["first"], results["second"],
                                  results["composite"], frame_set=frame_set, cue_set=cue_set)
            verdicts.append(table)
            stats["two_detector"] = {
                "n_flagged": int(table.flag_two_detector.sum()),
                "expected_under_null": round(len(table) * TWO_DETECTOR_ALPHA, 1),
                "sign_agreement_all_cells": round(float(table.same_sign.mean()), 4),
                "pattern_first_vs_second": round(
                    pattern_score(results["first"]["B"], results["second"]["B"]), 4),
            }
            stats["reliability_split_half_actors"] = scc.split_half(
                designs["composite"], lams["composite"], args.n_splits, args.seed)
            stats["added_detector_check"] = third_detector_check(
                args.au_csv, args.graphau_csv, cue_names, human, frame_set=frame_set,
                stat=args.stat, reference=results["composite"]["B"], lam=lams["composite"],
                n_perm=args.n_perm, seed=args.seed)
            block[frame_set] = stats
            print(cue_set, frame_set, json.dumps(stats, default=str)[:600], flush=True)
    agreement = {}
    for frame_set in ("all", "model_1fps"):
        first = detector_aus(args.au_csv, "libreface", frame_set=frame_set, stat=args.stat,
                             aus=list(scc.SHARED_AUS) + list(SINGLES_TWO_DETECTORS))
        added = pd.read_csv(args.graphau_csv)
        names = sorted(set(added.cue) & set(first.columns))
        second = detector_aus(args.au_csv, SECOND_DETECTOR_SINGLES, frame_set=frame_set,
                              stat=args.stat, aus=names, graphau_csv=args.graphau_csv)
        agreement[frame_set] = {
            au: {**within_cell_agreement(first[au], second[au], categories),
                 "singly_covered_before": au in SINGLES_TWO_DETECTORS}
            for au in names
        }
    out["added_detector_agreement_with_libreface"] = agreement
    prefix = args.out_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_csv(f"{prefix}_matrix.csv", index=False)
    pd.concat(verdicts, ignore_index=True).to_csv(f"{prefix}_verdicts.csv", index=False)
    Path(f"{prefix}.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--human", type=Path, default=ROOT / "data/eu_emotions_human_entropy.json")
    parser.add_argument("--au-csv", type=Path)
    parser.add_argument("--graphau-csv", type=Path)
    parser.add_argument("--stat", default="mean", choices=["mean", "p90", "max"])
    parser.add_argument("--spec", default="within", choices=scc.SPECS)
    parser.add_argument("--n-perm", type=int, default=None)
    parser.add_argument("--pattern-perm", type=int, default=199)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-splits", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-prefix", type=Path, default=ROOT / "research/reproduction/same_clip_family")
    parser.add_argument("--cue-sets", nargs="+", default=list(CUE_SETS), choices=list(CUE_SETS))
    parser.add_argument("--recovery", action="store_true")
    parser.add_argument("--scheme", default="within_category",
                        choices=["within_category", "within_actor", "within_cell", "actor_sign"])
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--attempt-2", action="store_true",
                        help="the disclosed second attempt: within-actor shuffle, new seeds")
    parser.add_argument("--frame-set", default="all", choices=["all", "model_1fps"],
                        help="cues behind a recovery run")
    parser.add_argument("--cue-set", default="families", choices=list(CUE_SETS))
    parser.add_argument("--n-sims", type=int, default=200)
    parser.add_argument("--only", nargs="*", help="condition names, for smoke runs only")
    parser.add_argument("--recovery-out", type=Path)
    parser.add_argument("--write-criteria", action="store_true")
    parser.add_argument("--criteria-out", type=Path,
                        default=ROOT / "research/reproduction/same_clip_family_criteria.json")
    parser.add_argument("--write-attempt-2-criteria", action="store_true")
    args = parser.parse_args(argv)
    if args.write_attempt_2_criteria:
        out = ROOT / "research/reproduction/same_clip_family_a2_criteria.json"
        print(json.dumps(write_attempt_2_criteria(out), indent=1))
        return
    if args.write_criteria:
        print(json.dumps(write_criteria(args.criteria_out), indent=1))
        return
    stem = "same_clip_family"
    if args.attempt_2:
        args.scheme, args.seed_offset = ATTEMPT_2["scheme"], ATTEMPT_2["seed_offset"]
        stem = "same_clip_family_a2"
        if args.out_prefix == ROOT / "research/reproduction/same_clip_family":
            args.out_prefix = ROOT / f"research/reproduction/{stem}"
    if args.recovery:
        args.n_perm = args.n_perm or 499
        if args.recovery_out is None:
            name = f"{stem}_recovery" if args.cue_set == "families" \
                else f"{stem}_recovery_{args.cue_set}"
            if args.frame_set != "all":
                name += "_model_frames"
            args.recovery_out = ROOT / f"research/reproduction/{name}.json"
        run_recovery(args)
        return
    if not args.au_csv:
        parser.error("--au-csv is required unless --recovery or --write-criteria is set")
    args.n_perm = args.n_perm or 4999
    print(json.dumps(run_real(args), indent=1))


if __name__ == "__main__":
    sys.exit(main())
