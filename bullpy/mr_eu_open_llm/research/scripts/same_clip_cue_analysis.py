#!/usr/bin/env python3
"""Same-clip cue analysis for RQ1 on the 243 EU-Emotion clips.

Regresses the probability of each response option on measured action units, and
returns one cue-by-option effect matrix per observer. Humans and models go through
the same code, the same ridge strength and the same bootstrap draws (rule 3).
This is observational on both sides: a human matrix here may be compared only with a
model matrix from this same analysis, never with an interventional estimate.

Primary model, one fractional ridge logistic per option on the clips offering it:

    logit P(option | clip) = category intercept + beta * cues

Each EU-Emotion category has one fixed option set, so a category intercept absorbs
the label prior and the foils offered. Cues are residualised on category and actor
before fitting. Cross-category and between-actor differences then cannot produce an
effect, which is what stops a purely prior-driven observer looking cue-driven. The
identifying variation is between clips of one category by one actor, plus the
crossing of actors and categories. Effects are log-odds per within-cell SD of the
cue, and `ame` is the same effect as average change in probability.

Inference: ridge strength by leave-one-actor-out log loss; held-out gain over a
category-only baseline by nested leave-one-actor-out; actor-cluster bootstrap for
pointwise (small-cluster t) and simultaneous (max-t) intervals. Actors are the resampling unit
throughout. All 17 are in the data, so claims stop at this actor sample.

Measurement error (rule 8): each detector is analysed on its own, on the same cues.
A cell counts as robust only when both detectors' intervals exclude zero with the
same sign, and the union of the intervals is the reported range.

Recovery (rule 7): `--recovery` sends synthetic observers with known weights, and
observers with no cue weights at all, through this pipeline.

Model responses must come from a complete clean rotation run. Target-first evals,
mismatch runs and smoke runs are refused by `load_rotation_run`.

Usage:
    python3 research/scripts/same_clip_cue_analysis.py --recovery
    python3 research/scripts/same_clip_cue_analysis.py \
        --au-csv research/reproduction/eu_au_measures.csv
    python3 research/scripts/same_clip_cue_analysis.py \
        --au-csv research/reproduction/eu_au_measures.csv --model qwen3vl \
        --model-evals 'results/baseline/eu_emotions/qwen3vl/*_rot?_real.json'
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.stats import t as t_dist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from paper_v2.option_score import ABSTAIN_LABEL, rotate_options  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_human_au_effects", Path(__file__).with_name("human_au_effects.py")
)
_hae = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hae)
fit_ridge_logit, expit = _hae.fit_ridge_logit, _hae.expit

ACTOR_RE = re.compile(r"([A-Za-z])F\s*\d+", re.IGNORECASE)
LABEL_ALIASES = {"Excitement": "Excited"}  # Kind items list this foil under another name
WEAK_TARGET_LABELS = ("Kind", "Unfriendly", "Jealous")  # see RESEARCH_QUESTIONS.md
SHARED_AUS = ["AU01", "AU02", "AU04", "AU06", "AU09", "AU12", "AU25", "AU26"]
LF_INTENSITY_AUS = ["AU01", "AU02", "AU04", "AU05", "AU06", "AU09", "AU12", "AU15", "AU17",
                    "AU20", "AU25", "AU26"]
LF_OCCURRENCE_ONLY = ["AU07", "AU10", "AU14", "AU23", "AU24"]
POSE_CUES = ["head_pitch", "head_yaw", "head_roll", "gaze_yaw", "gaze_pitch"]
# LibreFace DISFA per-AU PCC, from causal_methods.md section 2. OpenFace 3.0 publishes
# no per-AU intensity accuracy, so recovery assumes the same values for it.
PUBLISHED_PCC = {"AU01": 0.63, "AU02": 0.72, "AU04": 0.78, "AU05": 0.59, "AU06": 0.59,
                 "AU09": 0.62, "AU12": 0.85, "AU15": 0.36, "AU17": 0.50, "AU20": 0.34,
                 "AU25": 0.94, "AU26": 0.68}
LAMBDA_GRID = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
SPECS = ("within", "within_category", "pooled")
LOW_N_ITEMS = 20
ROTATIONS = (0, 1, 2, 3, 4)
ROTATION_FILE_RE = re.compile(r"_rot(\d)_real\.json$")
# Stated before any recovery run; a condition passes only if it meets these.
RECOVERY_CRITERIA = {
    "cue_observer": {"median_pearson_min": 0.70, "median_sign_accuracy_min": 0.90},
    "null_observer": {"pointwise_fpr_max": 0.05, "two_detector_fpr_max": 0.01,
                      "abs_pearson_with_any_fixed_matrix_max": 0.10},
}


def actor_of(trial_id: str) -> str:
    match = ACTOR_RE.search(trial_id)
    if not match:
        raise ValueError(f"No actor in {trial_id}")
    return match.group(1).upper()


def norm_label(label: str) -> str:
    return LABEL_ALIASES.get(label, label)


# --------------------------------------------------------------------------
# Observers
# --------------------------------------------------------------------------


def load_human(path) -> pd.DataFrame:
    """Long table: one row per clip and offered option."""
    trials = json.loads(Path(path).read_text(encoding="utf-8"))["trials"]
    rows = []
    for trial_id, rec in sorted(trials.items()):
        for option in rec["human_options"]:
            rows.append({
                "trial_id": trial_id,
                "actor": actor_of(trial_id),
                "category": rec["human_target_label"],
                "option": norm_label(option),
                "p": float(rec["human_response_distribution"].get(option, 0.0)),
                "n": int(rec["n_respondents"]),
            })
    return pd.DataFrame(rows)


def _rotation_of(path: Path):
    match = ROTATION_FILE_RE.search(path.name)
    return int(match.group(1)) if match else None


def load_rotation_run(paths, human_lookup, *, allow_partial=False) -> pd.DataFrame:
    """Mean response distribution over a clean rotation run, in `load_human` format.

    Refuses anything that is not a real-condition rotated eval. In particular the old
    target-first evals, whose answers are confounded with option position, can never
    enter: their names do not match, and their stored option order would not either.
    """
    paths = sorted(Path(p) for p in paths)
    if not paths:
        raise ValueError("No model eval files given")
    by_rotation = {}
    for path in paths:
        rotation = _rotation_of(path)
        if rotation is None or "smoke" in path.name:
            raise ValueError(f"Not a clean rotation run file: {path.name}")
        by_rotation[rotation] = path
    missing = [r for r in ROTATIONS if r not in by_rotation]
    if missing and not allow_partial:
        raise ValueError(f"Missing rotations {missing}; a partial run keeps position confounds")
    trials = json.loads(Path(human_lookup).read_text(encoding="utf-8"))["trials"]
    sums, parsed, counted = {}, {}, {}
    n_errors, n_samples = 0, 0
    for rotation, path in sorted(by_rotation.items()):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for trial in payload.get("trials") or payload["results"]:
            trial_id = trial["trial_id"]
            if trial_id not in trials:
                raise ValueError(f"{path.name}: unknown trial {trial_id}")
            if trial.get("vision_source_trial_id") not in (None, trial_id):
                raise ValueError(f"{path.name}: {trial_id} shows another clip's video")
            stage2 = trial.get("stage2") or {}
            if trial.get("error") or not stage2.get("response_distribution"):
                n_errors += 1
                continue
            expected = rotate_options(trials[trial_id]["human_options"], rotation)
            if list(stage2["options"]) != list(expected):
                raise ValueError(
                    f"{path.name}: option order of {trial_id} is not rotation {rotation}"
                )
            samples = int(stage2.get("fc_n_samples") or 0)
            n_samples += samples
            dist = stage2["response_distribution"]  # also holds __unparsed__
            total = sums.setdefault(trial_id, {o: 0.0 for o in expected})
            for option in expected:
                total[option] += float(dist.get(option, 0.0)) * samples
                parsed[trial_id] = parsed.get(trial_id, 0.0) + float(dist.get(option, 0.0)) * samples
            if samples > 0 and any(float(dist.get(option, 0.0)) > 0 for option in expected):
                counted[trial_id] = counted.get(trial_id, 0) + 1  # some answer named an option
    rows = []
    for trial_id, total in sorted(sums.items()):
        rec = trials[trial_id]
        if parsed[trial_id] <= 0:
            n_errors += 1
            continue
        for option, value in total.items():
            rows.append({
                "trial_id": trial_id,
                "actor": actor_of(trial_id),
                "category": rec["human_target_label"],
                "option": norm_label(option),
                "p": value / parsed[trial_id],  # share of answers that named an option
                "n": int(round(parsed[trial_id])),
            })
    table = pd.DataFrame(rows)
    table.attrs.update({
        "rotations": sorted(by_rotation),
        "n_trial_errors": n_errors,
        "unparsed_share": round(1 - sum(parsed.values()) / n_samples, 4) if n_samples else None,
        "partial": bool(missing),
        "rotations_by_trial": counted,
    })
    return table


# --------------------------------------------------------------------------
# Cues
# --------------------------------------------------------------------------


def load_cues(path, *, detector, frame_set="all", stat="mean", cues=None) -> pd.DataFrame:
    """Clip-by-cue table for one detector, or `composite` for the two-detector mean.

    The composite averages z-scored values, so it exists only for cues both measure.
    """
    table = pd.read_csv(path)
    table = table[table.frame_set == frame_set]

    def wide(name):
        part = table[table.detector == name]
        part = part[part.cue.isin(cues)] if cues is not None else part[part.kind != "occurrence"]
        part = part.sort_values("kind").drop_duplicates(["trial_id", "cue"])  # intensity first
        return part.pivot(index="trial_id", columns="cue", values=stat)

    if detector != "composite":
        return wide(detector)
    a, b = wide("libreface"), wide("openface3")
    shared = [c for c in a.columns if c in b.columns]
    if not shared:
        raise ValueError("No cue measured by both detectors")
    zscore = lambda x: (x - x.mean()) / x.std()  # noqa: E731
    return (zscore(a[shared]) + zscore(b[shared])) / 2


def residualise(X: np.ndarray, categories, actors=None) -> np.ndarray:
    """Remove category (and actor) means by least squares, column by column."""
    blocks = [pd.get_dummies(pd.Series(categories)).to_numpy(float)]
    if actors is not None:
        blocks.append(pd.get_dummies(pd.Series(actors)).to_numpy(float)[:, 1:])
    D = np.hstack(blocks)
    return X - D @ np.linalg.lstsq(D, X, rcond=None)[0]


@dataclass
class Design:
    trial_ids: list
    actors: np.ndarray
    categories: np.ndarray
    options: list
    cues: list
    spec: str
    X: np.ndarray  # n by k, standardised for the spec
    Y: np.ndarray  # n by m, NaN where not offered
    base: dict  # option -> (row indices, unpenalised block)

    def with_responses(self, Y):
        return Design(self.trial_ids, self.actors, self.categories, self.options, self.cues,
                      self.spec, self.X, Y, self.base)


def build_design(responses: pd.DataFrame, cues: pd.DataFrame, *, spec="within") -> Design:
    if spec not in SPECS:
        raise ValueError(f"Unknown spec {spec}")
    trial_ids = sorted(set(responses.trial_id) & set(cues.dropna().index))
    if not trial_ids:
        raise ValueError("No clip has both responses and cues")
    part = responses[responses.trial_id.isin(trial_ids)]
    meta = part.drop_duplicates("trial_id").set_index("trial_id").loc[trial_ids]
    actors, categories = meta.actor.to_numpy(), meta.category.to_numpy()
    options = sorted(part.option.unique())
    Y = (part.pivot(index="trial_id", columns="option", values="p")
         .reindex(index=trial_ids, columns=options).to_numpy(float))
    X = cues.loc[trial_ids].to_numpy(float)
    X = (X - X.mean(0)) / X.std(0)
    if spec == "within":
        X = residualise(X, categories, actors)
    elif spec == "within_category":
        X = residualise(X, categories)
    X = X / X.std(0)
    base = {}
    for j, _ in enumerate(options):
        rows = np.flatnonzero(~np.isnan(Y[:, j]))
        if spec == "pooled":
            block = np.ones((len(rows), 1))
        else:
            present = pd.Categorical(categories[rows])
            block = np.ones((len(rows), len(present.categories)))
            block[:, 1:] = pd.get_dummies(present).to_numpy(float)[:, 1:]
        base[j] = (rows, block)
    return Design(trial_ids, actors, categories, options, list(cues.columns), spec, X, Y, base)


# --------------------------------------------------------------------------
# Estimation
# --------------------------------------------------------------------------


def fit_option(design: Design, j, lam, weights=None, X=None):
    rows, block = design.base[j]
    X = design.X if X is None else X
    D = np.hstack([block, X[rows]])
    pen = np.r_[np.zeros(block.shape[1]), np.full(X.shape[1], float(lam))]
    w = np.ones(len(rows)) if weights is None else weights[rows]
    y = design.Y[rows, j]
    if w.sum() <= 0 or np.ptp(y[w > 0]) == 0:
        coef = np.zeros(D.shape[1])
        coef[0] = np.log(max(y.mean(), 1e-4) / max(1 - y.mean(), 1e-4))
        return coef, D
    return fit_ridge_logit(D, y, w=w, pen=pen, max_iter=40, tol=1e-6), D


def fit_matrix(design: Design, lam, weights=None):
    """Options by cues matrix of slopes, for fixed ridge strength."""
    B = np.zeros((len(design.options), design.X.shape[1]))
    for j in range(len(design.options)):
        coef, _ = fit_option(design, j, lam, weights)
        B[j] = coef[-design.X.shape[1]:]
    return B


def average_marginal_effects(design: Design, B, lam):
    A = np.zeros_like(B)
    for j in range(len(design.options)):
        coef, D = fit_option(design, j, lam)
        mu = expit(D @ coef)
        A[j] = np.mean(mu * (1 - mu)) * B[j]
    return A


def cross_entropy(y, mu):
    mu = np.clip(mu, 1e-6, 1 - 1e-6)
    return -(y * np.log(mu) + (1 - y) * np.log(1 - mu))


def heldout_loss(design: Design, lam, heldout, train_mask):
    """Sum of held-out cross-entropy over options. lam=None gives category-only."""
    no_cues = np.zeros((design.X.shape[0], 0))
    total, count = 0.0, 0
    for j in range(len(design.options)):
        rows, _ = design.base[j]
        test = heldout[rows]
        if not test.any():
            continue
        if lam is None:
            coef, D = fit_option(design, j, 0.0, train_mask, X=no_cues)
        else:
            coef, D = fit_option(design, j, lam, train_mask)
        if train_mask[rows].sum() == 0:
            continue
        # An intercept for a category absent from training is left at zero.
        total += cross_entropy(design.Y[rows, j][test], expit(D[test] @ coef)).sum()
        count += int(test.sum())
    return total, count


def choose_lambda(design: Design, grid=LAMBDA_GRID, keep=None):
    """Ridge strength by leave-one-actor-out log loss, optionally on a subset."""
    keep = np.ones(len(design.actors), bool) if keep is None else keep
    losses = {lam: 0.0 for lam in grid}
    for actor in np.unique(design.actors[keep]):
        heldout = keep & (design.actors == actor)
        train = (keep & ~heldout).astype(float)
        for lam in grid:
            losses[lam] += heldout_loss(design, lam, heldout, train)[0]
    return min(grid, key=lambda lam: (losses[lam], -lam))


def loao_gain(design: Design, lam=None, grid=LAMBDA_GRID, nested=True):
    """Held-out deviance reduction over the category-only baseline.

    With lam=None the ridge strength is rechosen inside each outer fold when nested,
    otherwise it is chosen once. A fixed lam is used as given (the model route).
    """
    if lam is None and not nested:
        lam = choose_lambda(design, grid)
    cue, base, n = 0.0, 0.0, 0
    per_actor = {}
    for actor in np.unique(design.actors):
        heldout = design.actors == actor
        train = (~heldout).astype(float)
        fold_lam = lam if lam is not None else choose_lambda(design, grid, keep=~heldout)
        loss, count = heldout_loss(design, fold_lam, heldout, train)
        loss0, _ = heldout_loss(design, None, heldout, train)
        cue, base, n = cue + loss, base + loss0, n + count
        per_actor[str(actor)] = round(1 - loss / loss0, 5) if loss0 > 0 else None
    return {
        "heldout_ce_cues": round(cue / n, 6),
        "heldout_ce_category_only": round(base / n, 6),
        "heldout_gain": round(1 - cue / base, 5),
        "gain_per_actor": per_actor,
    }


def actor_draws(actors, n_boot, seed):
    """Item weights per bootstrap draw. Same seed and actors, same draws."""
    levels, codes = np.unique(actors, return_inverse=True)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(levels), size=(n_boot, len(levels)))
    tallies = np.stack([np.bincount(p, minlength=len(levels)) for p in picks])
    return tallies[:, codes].astype(float)


def bootstrap(design: Design, lam, n_boot, seed):
    draws = actor_draws(design.actors, n_boot, seed)
    return np.stack([fit_matrix(design, lam, w) for w in draws])


def intervals(design: Design, B, boots, alpha=0.05):
    """Small-cluster interval per cell, and a max-t simultaneous interval over all cells.

    Percentile intervals ran at about 8 percent false positives on null observers. Many
    options occur in one category with about 8 actors, so each option uses t on its own
    actor count with the usual G/(G-1) correction to the bootstrap SE.
    """
    se = boots.std(0, ddof=1)
    n_actors = np.array([len(set(design.actors[design.base[j][0]])) for j in range(B.shape[0])])
    n_actors = np.maximum(n_actors, 3)
    factor = t_dist.ppf(1 - alpha / 2, n_actors - 1) * np.sqrt(n_actors / (n_actors - 1))
    half = factor[:, None] * se
    safe = np.where(se > 0, se, np.inf)
    z = np.abs(boots - B) / safe
    crit = float(np.quantile(z.max(axis=(1, 2)), 1 - alpha))
    return {"lo": B - half, "hi": B + half, "se": se, "sim_lo": B - crit * se,
            "sim_hi": B + crit * se, "crit": crit}


def sign_stability(design: Design, B, lam):
    """Share of leave-one-actor-out refits that keep each cell's sign."""
    same = np.zeros_like(B)
    actors = np.unique(design.actors)
    for actor in actors:
        same += np.sign(fit_matrix(design, lam, (design.actors != actor).astype(float))) == np.sign(B)
    return same / len(actors)


def estimate(design: Design, *, lam=None, n_boot=2000, seed=42, grid=LAMBDA_GRID, nested=True,
             with_gain=True):
    """The whole pipeline for one observer. Pass lam to freeze the ridge strength."""
    frozen = lam is not None
    lam = choose_lambda(design, grid) if lam is None else lam
    B = fit_matrix(design, lam)
    boots = bootstrap(design, lam, n_boot, seed)
    result = {"lam": lam, "lam_frozen": frozen, "B": B, "boots": boots, **intervals(design, B, boots)}
    if with_gain:
        result["gain"] = loao_gain(design, lam=lam if frozen else None, grid=grid, nested=nested)
    return result


def matrix_table(design: Design, result, observer, detector, frame_set):
    B = result["B"]
    ame = average_marginal_effects(design, B, result["lam"])
    stability = sign_stability(design, B, result["lam"])
    rows = []
    for j, option in enumerate(design.options):
        items, _ = design.base[j]
        for k, cue in enumerate(design.cues):
            rows.append({
                "observer": observer, "detector": detector, "frame_set": frame_set,
                "spec": design.spec, "option": option, "cue": cue,
                "beta": round(B[j, k], 4), "ame": round(ame[j, k], 5),
                "ci_lo": round(result["lo"][j, k], 4), "ci_hi": round(result["hi"][j, k], 4),
                "sim_lo": round(result["sim_lo"][j, k], 4),
                "sim_hi": round(result["sim_hi"][j, k], 4),
                "boot_se": round(result["se"][j, k], 4),
                "ci_excludes_zero": bool(result["lo"][j, k] > 0 or result["hi"][j, k] < 0),
                "sim_excludes_zero": bool(result["sim_lo"][j, k] > 0 or result["sim_hi"][j, k] < 0),
                "loao_sign_stability": round(stability[j, k], 3),
                "n_items": len(items),
                "n_categories": int(len(set(design.categories[items]))),
                "n_actors": int(len(set(design.actors[items]))),
                "low_n": len(items) < LOW_N_ITEMS,
            })
    return pd.DataFrame(rows)


def combine_detectors(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Cell verdict under rule 8. Disagreement widens the range, it is not averaged."""
    keys = ["observer", "frame_set", "spec", "option", "cue"]
    cols = keys + ["beta", "ci_lo", "ci_hi", "ci_excludes_zero", "loao_sign_stability"]
    merged = a[cols].merge(b[cols], on=keys, suffixes=("_a", "_b"))
    same_sign = np.sign(merged.beta_a) == np.sign(merged.beta_b)
    both = merged.ci_excludes_zero_a & merged.ci_excludes_zero_b
    merged["sign_agree"] = same_sign
    merged["range_lo"] = merged[["ci_lo_a", "ci_lo_b"]].min(axis=1)
    merged["range_hi"] = merged[["ci_hi_a", "ci_hi_b"]].max(axis=1)
    merged["verdict"] = np.select(
        [both & same_sign, both & ~same_sign,
         merged.ci_excludes_zero_a ^ merged.ci_excludes_zero_b],
        ["robust", "opposed", "one_detector"], default="neither",
    )
    return merged


# --------------------------------------------------------------------------
# Comparison of a model with humans
# --------------------------------------------------------------------------


def _pearson(a, b):
    a, b = np.ravel(a), np.ravel(b)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a, b):
    return _pearson(rankdata(np.ravel(a)), rankdata(np.ravel(b)))


def split_half(design: Design, lam, n_splits, seed):
    """Reliability of a matrix across random halves of the actors, Spearman-Brown."""
    rng = np.random.default_rng(seed)
    levels = np.unique(design.actors)
    rs = []
    for _ in range(n_splits):
        half = rng.permutation(levels)[: len(levels) // 2]
        in_half = np.isin(design.actors, half).astype(float)
        rs.append(_pearson(fit_matrix(design, lam, in_half), fit_matrix(design, lam, 1 - in_half)))
    r = float(np.nanmean(rs))
    return {"split_half_r": round(r, 4),
            "reliability": round(2 * r / (1 + r), 4) if r > 0 else 0.0,
            "n_splits": n_splits}


def compare(human: Design, model: Design, res_h, res_m, *, n_splits=200, seed=42):
    """Cell-wise alignment of two matrices estimated on identical clips and cues.

    Needs the same bootstrap draws, so that the interval is paired by actor.
    """
    if human.options != model.options or human.cues != model.cues:
        raise ValueError("Matrices differ in options or cues")
    if human.trial_ids != model.trial_ids or res_h["lam"] != res_m["lam"]:
        raise ValueError("Compare needs the same clips and the same ridge strength")
    a, b = res_h["B"], res_m["B"]
    paired = [_pearson(x, y) for x, y in zip(res_h["boots"], res_m["boots"])]
    rel_h = split_half(human, res_h["lam"], n_splits, seed)
    rel_m = split_half(model, res_m["lam"], n_splits, seed)
    r = _pearson(a, b)
    ceiling = float(np.sqrt(max(rel_h["reliability"], 0) * max(rel_m["reliability"], 0)))
    flagged = (res_h["lo"] > 0) | (res_h["hi"] < 0)
    return {
        "pearson": round(r, 4),
        "spearman": round(_spearman(a, b), 4),
        "pearson_ci_paired_actor_bootstrap": [round(float(q), 4) for q in np.nanquantile(paired, [0.025, 0.975])],
        "human_reliability": rel_h,
        "model_reliability": rel_m,
        "ceiling": round(ceiling, 4),
        "pearson_over_ceiling": round(r / ceiling, 4) if ceiling > 0 else None,
        "n_cells": int(a.size),
        "n_cells_human_ci_excludes_zero": int(flagged.sum()),
        "sign_agreement_on_those_cells": round(float((np.sign(a) == np.sign(b))[flagged].mean()), 4)
        if flagged.any() else None,
    }


def detector_agreement(path, frame_set="all", stat="mean", categories=None):
    """Clip-level agreement per shared AU: raw and after removing category and actor."""
    a = load_cues(path, detector="libreface", frame_set=frame_set, stat=stat)
    b = load_cues(path, detector="openface3", frame_set=frame_set, stat=stat)
    shared = sorted((set(a.columns) & set(b.columns)) - set(POSE_CUES))
    both = a.index.intersection(b.index)
    out = {}
    for cue in shared:
        pair = pd.concat([a.loc[both, cue], b.loc[both, cue]], axis=1, keys=["a", "b"]).dropna()
        actors = pair.index.map(actor_of).to_numpy()
        cats = (categories.reindex(pair.index).to_numpy() if categories is not None else actors)
        resid = residualise(pair.to_numpy(float), cats, actors)
        out[cue] = {
            "n_clips": len(pair),
            "pearson": round(_pearson(pair.a, pair.b), 4),
            "spearman": round(_spearman(pair.a, pair.b), 4),
            "pearson_within_category_actor": round(_pearson(resid[:, 0], resid[:, 1]), 4),
            "published_pcc_libreface": PUBLISHED_PCC.get(cue),
        }
    return out


# --------------------------------------------------------------------------
# Synthetic observers (rule 7)
# --------------------------------------------------------------------------


def synthetic_cues(categories, actors, cues, rng):
    """Cues with category and actor structure and correlated within-cell variation."""
    k = len(cues)
    n = len(categories)
    cat_levels, cat_codes = np.unique(categories, return_inverse=True)
    actor_levels, actor_codes = np.unique(actors, return_inverse=True)
    cat_mean = rng.normal(0, 0.8, (len(cat_levels), k))
    actor_mean = rng.normal(0, 0.6, (len(actor_levels), k))
    loadings = rng.normal(0, 0.5, (3, k))
    within = rng.normal(0, 1, (n, 3)) @ loadings + rng.normal(0, 0.8, (n, k))
    T = cat_mean[cat_codes] + actor_mean[actor_codes] + within
    return pd.DataFrame((T - T.mean(0)) / T.std(0), columns=cues)


def add_detector_noise(T, design_like, pcc, rng, shared_error=0.0, shared=None):
    """Add error so corr(measured, true) within cells equals the published PCC.

    Published values are frame-level against FACS coders, so applying them to clip
    means is conservative. `shared_error` is the share of error variance common to
    both detectors, since both trained on overlapping data.
    """
    cats, actors = design_like
    within_sd = residualise(T, cats, actors).std(0)
    scale = within_sd * np.sqrt(1 / pcc**2 - 1)
    noise = rng.normal(0, 1, T.shape)
    if shared is not None and shared_error > 0:
        noise = np.sqrt(shared_error) * shared + np.sqrt(1 - shared_error) * noise
    return T + noise * scale


def softmax_rows(eta):
    exp = np.exp(eta - eta.max(1, keepdims=True))
    return exp / exp.sum(1, keepdims=True)


def planted_weights(kind, options, n_cues, scale, rng):
    W = np.zeros((len(options), n_cues))
    if kind in ("prior_only", "actor_prior"):
        return W
    for j, option in enumerate(options):
        if option == ABSTAIN_LABEL:
            continue
        if kind == "sparse":
            picks = rng.choice(n_cues, size=min(2, n_cues), replace=False)
            W[j, picks] = rng.choice([-1.0, 1.0], size=len(picks)) * scale
        elif kind == "dense":
            W[j] = rng.normal(0, scale, n_cues)
        else:
            raise ValueError(kind)
    return W


def simulate_observer(template: Design, T, W, *, kind, rng, n_resp, item_sd=0.3, actor_sd=0.15):
    """Option probabilities for one observer. Returns (noisy Y, noise-free Y).

    Choice is softmax over the offered options of log prior + W * true cues, with
    actor and item noise, then multinomial sampling at n_resp. The prior is the human
    category mean, so a prior-only observer reproduces the human mean per category and
    has zero cue weights by construction.
    """
    n, m = template.Y.shape
    offered = ~np.isnan(template.Y)
    actors, codes = np.unique(template.actors, return_inverse=True)
    actor_noise = rng.normal(0, 0.8 if kind == "actor_prior" else actor_sd, (len(actors), m))
    item_noise = rng.normal(0, item_sd, (n, m))
    prior = np.zeros((n, m))
    frame = pd.DataFrame(template.Y).groupby(template.categories).transform("mean").to_numpy()
    prior[offered] = np.log(np.maximum(np.nan_to_num(frame)[offered], 0.005))
    drive = T @ W.T
    Y, clean = np.full((n, m), np.nan), np.full((n, m), np.nan)
    for i in range(n):
        keep = offered[i]
        p0 = softmax_rows((prior[i, keep] + drive[i, keep])[None])[0]
        p = softmax_rows((prior[i, keep] + drive[i, keep] + actor_noise[codes[i], keep]
                          + item_noise[i, keep])[None])[0]
        clean[i, keep] = p0
        Y[i, keep] = rng.multinomial(int(n_resp[i]), p) / int(n_resp[i])
    return Y, clean


def oracle_matrix(template: Design, T_true, clean_Y):
    """What this analysis would return with true cues, no noise, no shrinkage."""
    X = residualise(T_true, template.categories, template.actors)
    X = X / X.std(0)
    design = Design(template.trial_ids, template.actors, template.categories, template.options,
                    template.cues, template.spec, X, clean_Y, template.base)
    return fit_matrix(design, 1e-3)


def _rebuild(template_args, W_measured, Y, spec):
    trial_ids, actors, categories, options, cues, base_y = template_args
    frame = pd.DataFrame(W_measured, index=trial_ids, columns=cues)
    rows = []
    for j, option in enumerate(options):
        mask = ~np.isnan(base_y[:, j])
        rows.append(pd.DataFrame({
            "trial_id": np.array(trial_ids)[mask], "actor": actors[mask],
            "category": categories[mask], "option": option, "p": Y[mask, j],
        }))
    return build_design(pd.concat(rows, ignore_index=True), frame, spec=spec)


def run_one_sim(task):
    """One synthetic observer through the pipeline, for both detectors."""
    template, T0, cond, sim, n_boot, fixed_matrix = task
    rng = np.random.default_rng([cond["seed"], sim])
    cats, actors = template.categories, template.actors
    cues = template.cues
    T = T0 if T0 is not None else synthetic_cues(cats, actors, cues, rng).to_numpy()
    W = planted_weights(cond["observer"], template.options, len(cues), cond["scale"], rng)
    n_resp = np.full(len(cats), cond["n_resp"]) if cond["n_resp"] else cond["_n_resp"]
    Y, clean = simulate_observer(template, T, W, kind=cond["observer"], rng=rng, n_resp=n_resp)
    oracle = oracle_matrix(template, T, clean)

    pcc = np.array([PUBLISHED_PCC[c] for c in cues]) * cond["reliability_scale"]
    if cond["noise"]:
        shared_noise = rng.normal(0, 1, T.shape)
        measured = [
            add_detector_noise(T, (cats, actors), pcc, rng, cond["shared_error"], shared_noise)
            for _ in range(2)
        ]
    else:
        measured = [T, T]
    args = (template.trial_ids, actors, cats, template.options, cues, template.Y)
    if cond["noise"]:
        # The composite is what gets reported, so it is what gets scored; the two
        # detectors alone supply the rule 8 decision.
        zscored = [(m - m.mean(0)) / m.std(0) for m in measured]
        inputs = [(zscored[0] + zscored[1]) / 2, measured[0], measured[1]]
    else:
        inputs = [T]
    results, designs = [], []
    for W_measured in inputs:
        design = _rebuild(args, W_measured, Y, cond["spec"])
        designs.append(design)
        # Frozen ridge strength when given; that is the route a model takes.
        results.append(estimate(design, lam=cond.get("lam"), n_boot=n_boot, seed=42 + sim,
                                nested=False, with_gain=cond["with_gain"]))
    return score(designs[0], results, oracle, W, fixed_matrix, template)


def score(design, results, oracle, W, fixed_matrix, template):
    res = results[0]
    B = res["B"]
    excl = (res["lo"] > 0) | (res["hi"] < 0)
    excl_sim = (res["sim_lo"] > 0) | (res["sim_hi"] < 0)
    both = excl
    if len(results) > 1:
        one, other = results[1], results[2]
        flag_one = (one["lo"] > 0) | (one["hi"] < 0)
        flag_other = (other["lo"] > 0) | (other["hi"] < 0)
        both = flag_one & flag_other & (np.sign(one["B"]) == np.sign(other["B"]))
    powered = np.array([len(template.base[j][0]) >= LOW_N_ITEMS for j in range(B.shape[0])])
    planted = W != 0
    null_cells = np.abs(oracle) < 0.02 if not planted.any() else (~planted) & (np.abs(oracle) < 0.05)
    out = {
        "lam": res["lam"],
        "gain": res["gain"]["heldout_gain"] if "gain" in res else None,
        "pointwise_fpr": float(excl[null_cells].mean()) if null_cells.any() else None,
        "simultaneous_fpr": float(excl_sim[null_cells].mean()) if null_cells.any() else None,
        "any_cell_simultaneous": bool(excl_sim.any()),
        "two_detector_fpr": float(both[null_cells].mean()) if null_cells.any() else None,
    }
    if planted.any():
        cover = (res["lo"] <= oracle) & (oracle <= res["hi"])
        strong = np.abs(oracle) >= 0.25 * np.abs(W[planted]).mean()
        right = np.sign(B) == np.sign(oracle)
        simultaneous = excl_sim & right
        out.update({
            "pearson_with_oracle": _pearson(B, oracle),
            "pearson_with_oracle_powered": _pearson(B[powered], oracle[powered]),
            "pearson_single_detector": _pearson(results[-1]["B"], oracle),
            "pearson_with_planted_weights": _pearson(B, W),
            "attenuation_slope": float((B * oracle).sum() / (oracle**2).sum()),
            "sign_accuracy_strong": float(right[strong].mean()),
            "tpr_strong": float((excl & right)[strong].mean()),
            "tpr_simultaneous": float(simultaneous[strong].mean()),
            "tpr_two_detector": float((both & right)[strong].mean()),
            "ci_covers_oracle": float(cover.mean()),
            "per_cue_sign_accuracy": {
                cue: float(right[:, k][strong[:, k]].mean())
                for k, cue in enumerate(design.cues) if strong[:, k].any()
            },
        })
    else:
        out["pearson_with_fixed_matrix"] = _pearson(B, fixed_matrix) if fixed_matrix is not None else None
        out["sd_beta"] = float(B.std())
    return out


def default_conditions(lam_frozen=None):
    base = {"noise": True, "reliability_scale": 1.0, "shared_error": 0.0, "n_resp": 0,
            "spec": "within", "scale": 0.0, "with_gain": True, "lam": None}
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
        {**base, "name": "prior_only_within", "observer": "prior_only"},
        {**base, "name": "prior_only_perfect_detectors", "observer": "prior_only", "noise": False},
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
        ]
    for i, cond in enumerate(conditions):
        cond["seed"] = 1000 + i
    return conditions


def summarise_condition(cond, rows):
    def med(key):
        values = [r[key] for r in rows if r.get(key) is not None and not np.isnan(r[key])]
        return round(float(np.median(values)), 4) if values else None
    def mean(key):
        values = [r[key] for r in rows if r.get(key) is not None and not np.isnan(r[key])]
        return round(float(np.mean(values)), 4) if values else None
    out = {k: cond[k] for k in ("name", "observer", "scale", "noise", "spec", "shared_error")}
    out["n_sims"] = len(rows)
    out["median_lambda"] = med("lam")
    out["median_heldout_gain"] = med("gain")
    out["pointwise_fpr"] = mean("pointwise_fpr")
    out["simultaneous_fpr"] = mean("simultaneous_fpr")
    out["two_detector_fpr"] = mean("two_detector_fpr")
    out["share_sims_any_simultaneous_cell"] = mean("any_cell_simultaneous")
    if "pearson_with_oracle" in rows[0]:
        for key in ("pearson_with_oracle", "pearson_with_oracle_powered",
                    "pearson_single_detector", "pearson_with_planted_weights", "attenuation_slope",
                    "sign_accuracy_strong", "tpr_strong", "tpr_simultaneous", "tpr_two_detector",
                    "ci_covers_oracle"):
            out[key] = med(key)
        per_cue = pd.DataFrame([r["per_cue_sign_accuracy"] for r in rows]).mean()
        out["per_cue_sign_accuracy"] = {k: round(float(v), 3) for k, v in per_cue.items()}
        crit = RECOVERY_CRITERIA["cue_observer"]
        out["recovered"] = bool(
            out["pearson_with_oracle"] >= crit["median_pearson_min"]
            and out["sign_accuracy_strong"] >= crit["median_sign_accuracy_min"]
        )
    else:
        crit = RECOVERY_CRITERIA["null_observer"]
        out["median_abs_pearson_with_fixed_matrix"] = (
            round(float(np.median([abs(r["pearson_with_fixed_matrix"]) for r in rows])), 4)
            if rows[0].get("pearson_with_fixed_matrix") is not None else None
        )
        out["median_sd_beta"] = med("sd_beta")
        fixed = out["median_abs_pearson_with_fixed_matrix"]
        out["recovered"] = bool(
            out["pointwise_fpr"] <= crit["pointwise_fpr_max"]
            and (not cond["noise"] or out["two_detector_fpr"] <= crit["two_detector_fpr_max"])
            and (fixed is None or fixed <= crit["abs_pearson_with_any_fixed_matrix_max"])
        )
    return out


def run_recovery(human: pd.DataFrame, *, cues_table=None, cue_names=None, n_sims=100, n_boot=300,
                 workers=1, conditions=None, fixed_matrix=None, lam_frozen=None):
    """Every condition, by `n_sims` synthetic observers. Real cues are used if given."""
    cue_names = cue_names or SHARED_AUS
    ids = sorted(human.trial_id.unique())
    if cues_table is not None:
        cue_names = [c for c in cue_names if c in cues_table.columns]
        placeholder = cues_table[cue_names]
    else:
        placeholder = pd.DataFrame(
            np.random.default_rng(0).normal(0, 1, (len(ids), len(cue_names))),
            index=ids, columns=cue_names,
        )
    template = build_design(human, placeholder, spec="within")
    n_resp = human.drop_duplicates("trial_id").set_index("trial_id").loc[template.trial_ids, "n"].to_numpy()
    T0 = None
    if cues_table is not None:
        T = placeholder.loc[template.trial_ids].to_numpy(float)
        T0 = (T - T.mean(0)) / T.std(0)
    conditions = conditions or default_conditions(lam_frozen)
    report = []
    for cond in conditions:
        cond = {**cond, "_n_resp": n_resp}
        tasks = [(template, T0, cond, s, n_boot, fixed_matrix) for s in range(n_sims)]
        started = time.time()
        if workers > 1:
            with ProcessPoolExecutor(workers) as pool:
                rows = list(pool.map(run_one_sim, tasks, chunksize=max(1, n_sims // (4 * workers))))
        else:
            rows = [run_one_sim(t) for t in tasks]
        summary = summarise_condition(cond, rows)
        summary["seconds"] = round(time.time() - started, 1)
        report.append(summary)
        print(json.dumps({k: v for k, v in summary.items() if k != "per_cue_sign_accuracy"}), flush=True)
    return {
        "cue_source": "measured EU-Emotion cues" if T0 is not None else "synthetic with category and actor structure",
        "cues": cue_names,
        "n_sims": n_sims,
        "n_boot": n_boot,
        "criteria": RECOVERY_CRITERIA,
        "published_pcc": PUBLISHED_PCC,
        "notes": [
            "Second-detector per-AU reliability is unpublished, so the LibreFace values stand in.",
            "Published PCCs are frame-level, against FACS coders; they are applied here to the "
            "within-cell variation of clip aggregates, which is conservative.",
            "Simulated observers use nested selection off for speed: ridge strength is chosen "
            "once by leave-one-actor-out, so heldout_gain here is slightly optimistic.",
            "'Oracle' is the matrix this analysis returns on the true cues without noise, and it "
            "differs from planted weights because softmax choice makes options compete.",
        ],
        "conditions": report,
    }


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------


def analyse_observer(responses, cues_table, *, spec, lam, n_boot, seed):
    design = build_design(responses, cues_table, spec=spec)
    return design, estimate(design, lam=lam, n_boot=n_boot, seed=seed)


def json_ready(result):
    return {"lam": result["lam"], "lam_frozen": result["lam_frozen"],
            "max_t_critical": round(result["crit"], 3), **result.get("gain", {})}


def run_real(args):
    human = load_human(args.human)
    categories = human.drop_duplicates("trial_id").set_index("trial_id").category
    out = {"au_csv": str(args.au_csv), "spec": args.spec, "stat": args.stat, "n_boot": args.n_boot,
           "weak_target_labels": list(WEAK_TARGET_LABELS),
           "note": "Observational on both sides. Primary cue set is the eight AUs both detectors measure."}
    for frame_set in ("all", "model_1fps"):
        out.setdefault("detector_agreement", {})[frame_set] = detector_agreement(
            args.au_csv, frame_set=frame_set, stat=args.stat, categories=categories)
    frames_table = {}
    tables = []
    for frame_set in ("all", "model_1fps"):
        for detector in ("libreface", "openface3", "composite"):
            cues = load_cues(args.au_csv, detector=detector, frame_set=frame_set, stat=args.stat,
                             cues=SHARED_AUS)[SHARED_AUS]
            design, result = analyse_observer(human, cues, spec=args.spec, lam=None,
                                              n_boot=args.n_boot, seed=args.seed)
            tables.append(matrix_table(design, result, "human", detector, frame_set))
            out.setdefault("human", {}).setdefault(frame_set, {})[detector] = json_ready(result)
            frames_table[(frame_set, detector)] = (cues, design, result)
            print(frame_set, detector, json_ready(result), flush=True)

    # Secondary: every LibreFace cue, one detector only, so never robust under rule 8.
    full = load_cues(args.au_csv, detector="libreface", frame_set="all", stat=args.stat,
                     cues=LF_INTENSITY_AUS + LF_OCCURRENCE_ONLY)
    design, result = analyse_observer(human, full, spec=args.spec, lam=None, n_boot=args.n_boot,
                                      seed=args.seed)
    tables.append(matrix_table(design, result, "human", "libreface_all_cues", "all"))
    out["human"]["libreface_all_cues"] = json_ready(result)

    verdicts = []
    for frame_set in ("all", "model_1fps"):
        pair = [t for t in tables if t.detector.iloc[0] in ("libreface", "openface3")
                and t.frame_set.iloc[0] == frame_set]
        verdicts.append(combine_detectors(pair[0], pair[1]))
    verdict = pd.concat(verdicts, ignore_index=True)
    out["verdict_counts"] = {
        f: {str(k): int(v) for k, v in part.verdict.value_counts().items()}
        for f, part in verdict.groupby("frame_set")
    }
    B_all = frames_table[("all", "composite")][2]["B"]
    B_model = frames_table[("model_1fps", "composite")][2]["B"]
    out["human_matrix_all_vs_model_frames_pearson"] = round(_pearson(B_all, B_model), 4)

    if args.model_evals:
        paths = [p for pattern in args.model_evals for p in sorted(glob.glob(pattern))]
        responses = load_rotation_run(paths, args.human, allow_partial=args.allow_partial_rotations)
        model = args.model or "model"
        out.setdefault("model", {})[model] = {
            "rotations": responses.attrs["rotations"],
            "n_trial_errors": responses.attrs["n_trial_errors"],
            "unparsed_share": responses.attrs["unparsed_share"],
            "partial": responses.attrs["partial"],
        }
        for frame_set in ("all", "model_1fps"):
            for detector in ("libreface", "openface3", "composite"):
                cues, d_h, r_h = frames_table[(frame_set, detector)]
                d_m, r_m = analyse_observer(responses, cues, spec=args.spec, lam=r_h["lam"],
                                            n_boot=args.n_boot, seed=args.seed)
                tables.append(matrix_table(d_m, r_m, model, detector, frame_set))
                out["model"][model].setdefault(frame_set, {})[detector] = {
                    **json_ready(r_m), "alignment_with_human": compare(d_h, d_m, r_h, r_m)}
            pair = [t for t in tables if t.observer.iloc[0] == model and t.frame_set.iloc[0] == frame_set
                    and t.detector.iloc[0] in ("libreface", "openface3")]
            verdict = pd.concat([verdict, combine_detectors(pair[0], pair[1])], ignore_index=True)

    table = pd.concat(tables, ignore_index=True)
    prefix = args.out_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(f"{prefix}_matrix.csv", index=False)
    verdict.to_csv(f"{prefix}_detector_verdicts.csv", index=False)
    primary = table[(table.detector == "composite") & (table.frame_set == "all")]
    for observer, part in primary.groupby("observer"):
        part.pivot(index="cue", columns="option", values="beta").to_csv(f"{prefix}_{observer}_beta_wide.csv")
    Path(f"{prefix}.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--human", type=Path, default=ROOT / "data/eu_emotions_human_entropy.json")
    parser.add_argument("--au-csv", type=Path)
    parser.add_argument("--stat", default="mean", choices=["mean", "p90", "max"])
    parser.add_argument("--spec", default="within", choices=SPECS)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", help="name used in output files")
    parser.add_argument("--model-evals", nargs="*", help="rotation eval JSONs or globs")
    parser.add_argument("--allow-partial-rotations", action="store_true")
    parser.add_argument("--out-prefix", type=Path, default=ROOT / "research/reproduction/same_clip_cue")
    parser.add_argument("--recovery", action="store_true")
    parser.add_argument("--n-sims", type=int, default=100)
    parser.add_argument("--recovery-boot", type=int, default=300)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--recovery-out", type=Path,
                        default=ROOT / "research/reproduction/same_clip_cue_recovery.json")
    args = parser.parse_args(argv)
    if args.recovery:
        human = load_human(args.human)
        cues_table = fixed = lam = None
        if args.au_csv:
            cues_table = load_cues(args.au_csv, detector="libreface", cues=SHARED_AUS)[SHARED_AUS]
            design = build_design(human, cues_table)
            lam = choose_lambda(design)
            fixed = fit_matrix(design, lam)
        report = run_recovery(human, cues_table=cues_table, n_sims=args.n_sims,
                              n_boot=args.recovery_boot, workers=args.workers,
                              fixed_matrix=fixed, lam_frozen=lam)
        args.recovery_out.parent.mkdir(parents=True, exist_ok=True)
        args.recovery_out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        return
    if not args.au_csv:
        parser.error("--au-csv is required unless --recovery is set")
    print(json.dumps(run_real(args), indent=1))


if __name__ == "__main__":
    sys.exit(main())
