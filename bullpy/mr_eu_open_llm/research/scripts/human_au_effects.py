#!/usr/bin/env python3
"""Human action-unit effects on emotion and mental-state judgements (RQ1).

Estimates, for each answer option and cultural group in the Snoek AU + emotion
data (DOI 10.21942/uva.21261885, CC BY 4.0), the effect of each action unit on
the probability that a human chooses that option.

Method: one ridge logistic per subject and label (one-vs-rest), with the 33 AU
amplitudes and face gender as predictors; subjects are averaged, and confidence
intervals come from a subject-level bootstrap. Ridge strength is chosen per
dataset by held-out log loss only.

The zip is treated as untrusted data. It is read in place, never extracted, and
only TSV members matching the expected names are parsed as text.

Usage:
    python3 -I research/scripts/human_au_effects.py \
        --zip data/external/snoek_au/au_emotion_data.zip \
        --out-json research/reproduction/human_au_effects.json \
        --out-csv research/reproduction/human_au_effects.csv
"""
from __future__ import annotations

import argparse
import io
import json
import re
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

AU_COLS = [
    "AU01", "AU02L", "AU02R", "AU04", "AU05", "AU06L", "AU06R", "AU07L", "AU07R",
    "AU09", "AU10L", "AU10R", "AU11L", "AU11R", "AU12L", "AU12R", "AU13", "AU14L",
    "AU14R", "AU15", "AU16", "AU17", "AU20L", "AU20R", "AU22", "AU23", "AU24",
    "AU25", "AU26", "AU27", "AU38", "AU39", "AU43",
]
# Lateralised AUs. The composite is the bilateral effect, L plus R.
BILATERAL = ["AU02", "AU06", "AU07", "AU10", "AU11", "AU12", "AU14", "AU20"]
FAMILIES = sorted({c[:4] for c in AU_COLS})  # 25 AU families

DATASETS = {
    "basic_WC": ("basic_emotions/WC/", "emotion"),
    "basic_EA": ("basic_emotions/EA/", "emotion"),
    "conversational_WC": ("conversational_emotions/WC/", "state"),
    "conversational_EA": ("conversational_emotions/EA/", "state"),
}
LABEL_ORDER = {
    "emotion": ["anger", "disgust", "fear", "happy", "sadness", "surprise", "other"],
    "state": ["bored", "confused", "interested", "thinking", "other"],
}
# Prototype AUs written down before looking at any estimate: Ekman-Friesen
# EMFACS-style prototypes as commonly cited. Not altered after seeing results.
CLASSIC = {
    "happy": ["AU06", "AU12"],
    "sadness": ["AU01", "AU04", "AU15"],
    "surprise": ["AU01", "AU02", "AU05", "AU26"],
    "fear": ["AU01", "AU02", "AU04", "AU05", "AU20", "AU26"],
    "anger": ["AU04", "AU05", "AU07", "AU23"],
    "disgust": ["AU09", "AU10", "AU15", "AU16"],
}
LAMBDA_GRID = (0.01, 0.1, 1.0, 3.0, 10.0, 30.0)
MEMBER_RE = re.compile(
    r"^(basic_emotions|conversational_emotions)/(WC|EA)/sub-\d{2}(WC|EA)_ratings\.tsv$"
)
MAX_UNCOMPRESSED_BYTES = 200_000_000


def expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# --------------------------------------------------------------------------
# Estimation
# --------------------------------------------------------------------------


def _penalised_nll(X, y, w, beta, pen):
    eta = np.clip(X @ beta, -30, 30)
    return float(np.sum(w * (np.logaddexp(0, eta) - y * eta)) + 0.5 * np.sum(pen * beta**2))


def fit_ridge_logit(X, y, w=None, pen=None, max_iter=100, tol=1e-7):
    """Weighted ridge logistic by Newton steps with step halving.

    y may be fractional (a proportion with weight w). pen is the diagonal
    penalty per column; a zero leaves that column unpenalised. Returns beta.
    An outcome that never varies gets zero slopes and a clipped intercept in
    column 0, so subjects who never use a label add an honest zero.
    """
    n, p = X.shape
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    pen = np.zeros(p) if pen is None else np.asarray(pen, dtype=float)
    ybar = float(np.clip(np.sum(w * y) / np.sum(w), 1e-4, 1 - 1e-4))
    beta = np.zeros(p)
    beta[0] = np.log(ybar / (1 - ybar))
    if np.all(y == y[0]):
        return beta
    loss = _penalised_nll(X, y, w, beta, pen)
    for _ in range(max_iter):
        mu = expit(X @ beta)
        v = np.maximum(w * mu * (1 - mu), 1e-10)
        grad = X.T @ (w * (mu - y)) + pen * beta
        hess = (X.T * v) @ X + np.diag(pen + 1e-8)
        step = np.linalg.solve(hess, grad)
        t = 1.0
        while True:
            new_beta = beta - t * step
            new_loss = _penalised_nll(X, y, w, new_beta, pen)
            if new_loss <= loss + 1e-10 or t < 1e-4:
                break
            t *= 0.5
        converged = abs(loss - new_loss) < tol * (abs(loss) + 1.0)
        beta, loss = new_beta, new_loss
        if converged:
            break
    return beta


def effect_index(au_cols=AU_COLS):
    """Names and column positions of the 33 AUs plus bilateral composites."""
    names, members = [], []
    for j, col in enumerate(au_cols):
        names.append(col)
        members.append([j])
    for fam in BILATERAL:
        cols = [au_cols.index(fam + side) for side in "LR" if fam + side in au_cols]
        if len(cols) == 2:
            names.append(fam + "_bilateral")
            members.append(cols)
    return names, members


def family_positions(effect_names):
    """Index of one effect per AU family, using the bilateral composite if any."""
    out = []
    for fam in FAMILIES:
        for candidate in (fam + "_bilateral", fam):
            if candidate in effect_names:
                out.append(effect_names.index(candidate))
                break
    return out


def do_effects(X_au, X_cov, beta_by_label, members):
    """G-computation, P(label | do(AU=1)) minus P(label | do(AU=0)).

    beta_by_label is (labels, covariates + AUs). Other inputs stay at their
    observed values, averaged over the subject's own trials. Composites set
    both sides together. One-vs-rest probabilities are renormalised over labels
    on each trial, because options are exclusive, so effects sum to zero across
    labels. Returns (labels, effects).
    """
    n_cov = X_cov.shape[1]
    b_cov, b_au = beta_by_label[:, :n_cov], beta_by_label[:, n_cov:]
    eta = X_cov @ b_cov.T + X_au @ b_au.T
    out = np.empty((beta_by_label.shape[0], len(members)))

    def normalised(values):
        p = expit(values)
        return p / p.sum(axis=1, keepdims=True)

    for k, cols in enumerate(members):
        rest = eta - X_au[:, cols] @ b_au[:, cols].T
        out[:, k] = (normalised(rest + b_au[:, cols].sum(axis=1)) - normalised(rest)).mean(axis=0)
    return out


def fit_subject_effects(frame, labels, label_col, lam, au_cols=AU_COLS):
    """Per-subject fits. Returns subject ids and (S, L, E) beta and do arrays."""
    names, members = effect_index(au_cols)
    subs = sorted(frame["sub"].unique())
    betas = np.zeros((len(subs), len(labels), len(names)))
    dos = np.zeros_like(betas)
    never = np.zeros((len(subs), len(labels)), dtype=bool)
    for s, sub in enumerate(subs):
        d = frame[frame["sub"] == sub]
        X_au = d[au_cols].to_numpy(float)
        gender = (d["face_gender"].to_numpy() == "M").astype(float) - 0.5
        X_cov = np.column_stack([np.ones(len(d)), gender])
        X = np.column_stack([X_cov, X_au])
        pen = np.r_[0.0, 0.0, np.full(len(au_cols), lam)]
        y_all = d[label_col].to_numpy()
        fitted = np.zeros((len(labels), X.shape[1]))
        for l, label in enumerate(labels):
            y = (y_all == label).astype(float)
            never[s, l] = y.sum() == 0
            fitted[l] = fit_ridge_logit(X, y, pen=pen)
            betas[s, l] = [fitted[l, 2:][c].sum() for c in members]
        dos[s] = do_effects(X_au, X_cov, fitted, members)
    return subs, betas, dos, never


def choose_lambda(frame, labels, label_col, au_cols=AU_COLS, grid=LAMBDA_GRID, seed=0):
    """Pick the ridge strength by held-out log loss, on unseen stimuli.

    Stimuli, not trials, are split, so a stimulus is never in both halves.
    """
    rng = np.random.default_rng(seed)
    stim = frame["cfg"].astype(str) + "|" + frame["face_gender"].astype(str) + "|" + frame[
        "face_id"
    ].astype(str)
    uniq = stim.unique()
    fold = dict(zip(uniq, rng.integers(0, 2, len(uniq))))
    is_test = stim.map(fold).to_numpy() == 1
    losses = {}
    null_total, null_count = 0.0, 0
    for _, d in frame.assign(_t=is_test).groupby("sub"):
        te = d["_t"].to_numpy()
        y_all = d[label_col].to_numpy()
        for label in labels:
            y = (y_all == label).astype(float)
            p = float(np.clip(y[~te].mean(), 1e-6, 1 - 1e-6))
            null_total += float(-(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p)).sum())
            null_count += int(te.sum())
    for lam in grid:
        total, count = 0.0, 0
        for _, d in frame.assign(_t=is_test).groupby("sub"):
            te = d["_t"].to_numpy()
            X = np.column_stack(
                [
                    np.ones(len(d)),
                    (d["face_gender"].to_numpy() == "M") - 0.5,
                    d[au_cols].to_numpy(float),
                ]
            )
            pen = np.r_[0.0, 0.0, np.full(len(au_cols), lam)]
            y_all = d[label_col].to_numpy()
            for label in labels:
                y = (y_all == label).astype(float)
                b = fit_ridge_logit(X[~te], y[~te], pen=pen)
                p = np.clip(expit(X[te] @ b), 1e-6, 1 - 1e-6)
                total += float(-(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p)).sum())
                count += int(te.sum())
        losses[lam] = total / count
    best = min(losses, key=losses.get)
    report = {str(k): round(v, 5) for k, v in losses.items()}
    report["intercept_only"] = round(null_total / null_count, 5)
    return best, report


# --------------------------------------------------------------------------
# Inference and reliability
# --------------------------------------------------------------------------


def stratified_bootstrap_ci(arr, strata, n_boot, seed, alpha=0.05):
    """Percentile CI of the subject mean, resampling subjects within strata."""
    rng = np.random.default_rng(seed)
    strata = np.asarray(strata)
    groups = [np.where(strata == s)[0] for s in np.unique(strata)]
    boots = np.empty((n_boot,) + arr.shape[1:])
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        boots[b] = arr[idx].mean(axis=0)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2], axis=0)
    return lo, hi, boots.std(axis=0, ddof=1)


def pearson(a, b):
    a, b = np.ravel(a), np.ravel(b)
    a, b = a - a.mean(), b - b.mean()
    denom = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / denom) if denom > 0 else float("nan")


def spearman_brown(r):
    return 2 * r / (1 + r)


def split_half_reliability(arr, strata, n_splits, seed):
    """Random half-split of subjects, within strata.

    arr is (subjects, labels, effects). Returns correlations between half means,
    overall and per label, Spearman-Brown projected to the full sample, and
    the implied ceiling on correlation with the full-sample mean,
    sqrt(reliability), for a noise-free comparison vector.

    Both halves saw the same stimuli, so this is the ceiling for a model tested
    on those stimuli. It does not cover stimulus sampling.
    """
    rng = np.random.default_rng(seed)
    strata = np.asarray(strata)
    groups = [np.where(strata == s)[0] for s in np.unique(strata)]
    n_labels = arr.shape[1]
    overall = np.empty(n_splits)
    per_label = np.empty((n_splits, n_labels))
    for k in range(n_splits):
        a_idx, b_idx = [], []
        for g in groups:
            perm = rng.permutation(g)
            mid = len(perm) // 2
            a_idx.extend(perm[:mid])
            b_idx.extend(perm[mid:])
        a, b = arr[a_idx].mean(axis=0), arr[b_idx].mean(axis=0)
        overall[k] = pearson(a, b)
        per_label[k] = [pearson(a[l], b[l]) for l in range(n_labels)]

    def summarise(values):
        r = float(np.mean(values))
        rel = spearman_brown(r)
        return {
            "split_half_r": round(r, 4),
            "split_half_r_sd": round(float(np.std(values, ddof=1)), 4),
            "spearman_brown": round(rel, 4),
            "ceiling_r_noise_free_vs_full_sample": round(float(np.sqrt(max(rel, 0.0))), 4),
        }

    return summarise(overall), [summarise(per_label[:, l]) for l in range(n_labels)]


def leave_one_subject_out(arr):
    """Mean correlation of each subject's effect matrix with the rest's mean."""
    n = arr.shape[0]
    rs = [
        pearson(arr[i], np.delete(arr, i, axis=0).mean(axis=0))
        for i in range(n)
        if np.ptp(arr[i]) > 0
    ]
    return round(float(np.mean(rs)), 4)


class StimulusTable:
    """Counts of each label by stimulus and subject, for fast pooled fits."""

    def __init__(self, frame, labels, label_col, au_cols=AU_COLS):
        keys = list(au_cols) + ["face_gender", "face_id", "stim_set"]
        grouped = frame.groupby(keys, sort=False, dropna=False).ngroup().to_numpy()
        first = pd.DataFrame({"g": grouped}).drop_duplicates("g").sort_values("g").index
        self.subs = sorted(frame["sub"].unique())
        sub_pos = frame["sub"].map({s: i for i, s in enumerate(self.subs)}).to_numpy()
        n_stim = grouped.max() + 1
        self.stim = frame.iloc[first]
        self.sub_set = frame.drop_duplicates("sub").set_index("sub").loc[self.subs, "stim_set"].to_numpy()
        self.N = np.zeros((len(self.subs), n_stim))
        np.add.at(self.N, (sub_pos, grouped), 1)
        self.C = np.zeros((len(labels), len(self.subs), n_stim))
        y = frame[label_col].to_numpy()
        for l, label in enumerate(labels):
            m = y == label
            np.add.at(self.C[l], (sub_pos[m], grouped[m]), 1)
        sets = sorted(frame["stim_set"].unique())
        cols = [np.ones(n_stim), (self.stim["face_gender"].to_numpy() == "M") - 0.5]
        cols += [(self.stim["stim_set"].to_numpy() == s).astype(float) for s in sets[1:]]
        self.n_cov = len(cols)
        self.X = np.column_stack(cols + [self.stim[au_cols].to_numpy(float)])
        self.stim_strata = self.stim["stim_set"].to_numpy()

    def fit(self, subject_mask, stim_index, lam=1.0):
        """Pooled effects, one row per label, using the given subjects and stimuli."""
        names, members = effect_index()
        N = self.N[subject_mask].sum(axis=0)[stim_index]
        keep = N > 0
        X = self.X[stim_index][keep]
        pen = np.r_[np.zeros(self.n_cov), np.full(X.shape[1] - self.n_cov, lam)]
        out = np.zeros((self.C.shape[0], len(names)))
        for l in range(self.C.shape[0]):
            y = self.C[l][subject_mask].sum(axis=0)[stim_index][keep] / N[keep]
            b = fit_ridge_logit(X, y, w=N[keep], pen=pen)[self.n_cov :]
            out[l] = [b[c].sum() for c in members]
        return out


def stimulus_bootstrap_se(table, n_boot, seed):
    """SE of pooled log-odds effects from resampling stimuli, all subjects kept."""
    rng = np.random.default_rng(seed)
    everyone = np.ones(len(table.subs), dtype=bool)
    groups = [np.where(table.stim_strata == s)[0] for s in np.unique(table.stim_strata)]
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        draws.append(table.fit(everyone, idx))
    return np.std(np.stack(draws), axis=0, ddof=1)


def cross_stimulus_reliability(table, n_splits, seed):
    """Reliability with subjects AND stimuli split, using pooled fits.

    The halves share neither. This is the ceiling for a model tested on
    different configurations from those used for the human estimate.
    """
    rng = np.random.default_rng(seed)
    names, _ = effect_index()
    fam = family_positions(names)
    groups_sub = [np.where(table.sub_set == s)[0] for s in np.unique(table.sub_set)]
    groups_stim = [np.where(table.stim_strata == s)[0] for s in np.unique(table.stim_strata)]
    overall, per_label = [], []
    for _ in range(n_splits):
        mask = np.zeros(len(table.subs), dtype=bool)
        for g in groups_sub:
            mask[rng.permutation(g)[: len(g) // 2]] = True
        stim_a, stim_b = [], []
        for g in groups_stim:
            perm = rng.permutation(g)
            stim_a.extend(perm[: len(perm) // 2])
            stim_b.extend(perm[len(perm) // 2 :])
        a = table.fit(mask, np.array(stim_a))[:, fam]
        b = table.fit(~mask, np.array(stim_b))[:, fam]
        overall.append(pearson(a, b))
        per_label.append([pearson(a[l], b[l]) for l in range(a.shape[0])])
    r = float(np.mean(overall))
    return {
        "split_half_r": round(r, 4),
        "spearman_brown": round(spearman_brown(r), 4),
        "ceiling_r_noise_free_vs_full_sample": round(float(np.sqrt(max(spearman_brown(r), 0))), 4),
        "per_label_split_half_r": [round(float(x), 4) for x in np.mean(per_label, axis=0)],
    }


# --------------------------------------------------------------------------
# Loading and description
# --------------------------------------------------------------------------


def load_zip(path):
    """Read the four datasets from the zip without extracting anything."""
    frames = {}
    file_counts = {}
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        if sum(i.file_size for i in infos) > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("archive larger than expected, refusing to read it")
        for key, (prefix, label_col) in DATASETS.items():
            members = sorted(i.filename for i in infos if i.filename.startswith(prefix))
            members = [m for m in members if m.endswith(".tsv")]
            rejected = [m for m in members if not MEMBER_RE.match(m)]
            if rejected:
                raise ValueError(f"unexpected member names: {rejected[:3]}")
            parts, empty = [], 0
            for member in members:
                d = pd.read_csv(io.BytesIO(z.read(member)), sep="\t", engine="c")
                if d.empty:
                    empty += 1
                    continue
                parts.append(d.rename(columns={"Unnamed: 0": "cfg"}))
            frame = pd.concat(parts, ignore_index=True)
            frame = frame[frame[label_col].notna()].reset_index(drop=True)
            missing = [c for c in AU_COLS if c not in frame.columns]
            if missing:
                raise ValueError(f"{key}: missing AU columns {missing}")
            # In the Western basic study subjects 01-15 and 16-60 are separate
            # stimulus sets; everywhere else there is one set.
            if key == "basic_WC":
                num = frame["sub"].str[:2].astype(int)
                frame["stim_set"] = np.where(num <= 15, "A_sub01_15", "B_sub16_60")
            else:
                frame["stim_set"] = "single"
            frames[key] = (frame, label_col)
            file_counts[key] = {"tsv_files": len(members), "empty_tsv_files": empty}
    return frames, file_counts


def describe(frame, label_col, labels):
    au = frame[AU_COLS].to_numpy(float)
    per_sub = frame.groupby("sub").size()
    uniq = frame.drop_duplicates("cfg")[AU_COLS].to_numpy(float)
    corr = np.corrcoef(au.T)
    off = corr - np.eye(len(AU_COLS))
    i, j = np.unravel_index(np.argmax(np.abs(off)), off.shape)
    vif = np.diag(np.linalg.inv(np.corrcoef(uniq.T)))
    stim_keys = ["cfg", "face_gender", "face_id"]
    use = frame.groupby("sub")[label_col].value_counts().unstack(fill_value=0)
    use = use.reindex(columns=labels, fill_value=0)
    active = au > 0
    return {
        "n_subjects": int(frame["sub"].nunique()),
        "n_trials": int(len(frame)),
        "n_trials_excluding_other": int((frame[label_col] != "other").sum()),
        "trials_per_subject": {
            "min": int(per_sub.min()),
            "median": float(per_sub.median()),
            "max": int(per_sub.max()),
        },
        "subjects_with_under_half_median_trials": {
            k: int(v) for k, v in per_sub[per_sub < per_sub.median() / 2].items()
        },
        "n_au_columns": len(AU_COLS),
        "n_unique_au_configurations": int(frame["cfg"].nunique()),
        "n_unique_stimuli_config_plus_face": int(frame.drop_duplicates(stim_keys).shape[0]),
        "n_distinct_face_ids": int(frame["face_id"].nunique()),
        "stimulus_sets": {k: int(v) for k, v in frame.groupby("stim_set")["sub"].nunique().items()},
        "stimulus_set_config_counts": {
            k: int(v) for k, v in frame.groupby("stim_set")["cfg"].nunique().items()
        },
        "label_counts": {k: int(v) for k, v in frame[label_col].value_counts().items()},
        "amplitude_values_in_columns": [round(float(v), 2) for v in np.unique(au)],
        "active_aus_per_trial_mean": round(float(active.sum(axis=1).mean()), 3),
        "au_active_rate_min_max": [
            round(float(active.mean(axis=0).min()), 3),
            round(float(active.mean(axis=0).max()), 3),
        ],
        "largest_au_correlation": {
            "pair": [AU_COLS[i], AU_COLS[j]],
            "r": round(float(corr[i, j]), 3),
        },
        "max_vif_over_configurations": round(float(vif.max()), 2),
        "subject_label_pairs_never_used": int((use == 0).sum().sum()),
    }


def au_active(frame, au):
    """True where any column of the AU family is active."""
    cols = [c for c in AU_COLS if c[:4] == au]
    return frame[cols].to_numpy(float).max(axis=1) > 0


def raw_rate(frame, mask, label):
    """Unadjusted choice rate of a label within a set of trials."""
    return {
        "rate": round(float((frame.loc[mask, "emotion"] == label).mean()), 3),
        "n_trials": int(mask.sum()),
    }


def anger_au4_diagnostic(frame):
    """Rates of 'anger' for AU4 alone and with partners, unadjusted.

    An additive model cannot express a conjunction, so a negative AU4 main
    effect may coexist with AU4 raising anger alongside another AU. This is
    reported only, and feeds no estimate.
    """
    au4 = au_active(frame, "AU04")
    out = {"au4_on": raw_rate(frame, au4, "anger"), "au4_off": raw_rate(frame, ~au4, "anger")}
    for partner in ("AU05", "AU07", "AU23", "AU24"):
        other = au_active(frame, partner)
        out[f"au4_on_{partner}_on"] = raw_rate(frame, au4 & other, "anger")
        out[f"au4_off_{partner}_on"] = raw_rate(frame, ~au4 & other, "anger")
    return out


def round_nested(arr, digits=4):
    return np.round(arr, digits).tolist()


def sanity_check(frame, emo_labels, effect_names, beta, lo, hi, do_mean):
    """Do the prototype AUs raise the prototype label? Reports, never adjusts."""
    fam_pos = family_positions(effect_names)
    fam_names = [effect_names[p] for p in fam_pos]
    out = {}
    for label, aus in CLASSIC.items():
        l = emo_labels.index(label)
        order = np.argsort(-beta[l, fam_pos])
        ranks = {fam_names[p]: int(r + 1) for r, p in enumerate(order)}
        rows, supported = [], 0
        for au in aus:
            name = au + "_bilateral" if au + "_bilateral" in effect_names else au
            k = effect_names.index(name)
            ok = bool(lo[l, k] > 0)
            supported += ok
            active = au_active(frame, au)
            rows.append(
                {
                    "au": name,
                    "beta": round(float(beta[l, k]), 3),
                    "ci95": [round(float(lo[l, k]), 3), round(float(hi[l, k]), 3)],
                    "do_effect": round(float(do_mean[l, k]), 3),
                    "rank_of_25_by_beta": ranks[name],
                    "ci_excludes_zero_positive": ok,
                    "raw_choice_rate_au_on_off": [
                        raw_rate(frame, active, label)["rate"],
                        raw_rate(frame, ~active, label)["rate"],
                    ],
                }
            )
        out[label] = {
            "prototype_aus_supported": f"{supported}/{len(aus)}",
            "prototype": rows,
            "top5_by_beta": [fam_names[p] for p in order[:5]],
        }
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def analyse(frame, label_col, labels, args, seed):
    names, members = effect_index()
    fam = family_positions(names)
    lam, losses = choose_lambda(frame, labels, label_col, seed=seed)
    subs, betas, dos, never = fit_subject_effects(frame, labels, label_col, lam)
    strata = frame.drop_duplicates("sub").set_index("sub").loc[subs, "stim_set"].to_numpy()

    beta_mean = betas.mean(axis=0)
    do_mean = dos.mean(axis=0)
    b_lo, b_hi, b_se = stratified_bootstrap_ci(betas, strata, args.n_boot, seed)
    d_lo, d_hi, d_se = stratified_bootstrap_ci(dos, strata, args.n_boot, seed + 1)

    table = StimulusTable(frame, labels, label_col)
    everyone = np.ones(len(table.subs), dtype=bool)
    all_stim = np.arange(table.X.shape[0])
    pooled = table.fit(everyone, all_stim)
    stim_se = stimulus_bootstrap_se(table, args.n_stim_boot, seed + 2)

    lam_low = min(LAMBDA_GRID)
    _, betas_low, _, _ = fit_subject_effects(frame, labels, label_col, lam_low)
    beta_low = betas_low.mean(axis=0)

    reliability = {}
    for basis, idx in {"family_25": fam, "column_33": list(range(33))}.items():
        for metric, arr in {"beta": betas, "do_effect": dos}.items():
            sub = arr[:, :, idx]
            overall, per_label = split_half_reliability(sub, strata, args.n_splits, seed + 3)
            reliability[f"{metric}_{basis}"] = {
                "overall": overall,
                "per_label": dict(zip(labels, per_label)),
                "leave_one_subject_out_mean_r": leave_one_subject_out(sub),
            }
    cross = cross_stimulus_reliability(table, args.n_cross_splits, seed + 4)
    cross["per_label_split_half_r"] = dict(zip(labels, cross["per_label_split_half_r"]))
    reliability["beta_family_25_subjects_and_stimuli_split_pooled_fit"] = cross

    result = {
        "ridge_lambda_selected": lam,
        "heldout_log_loss_by_lambda": losses,
        "effect_names": names,
        "labels": labels,
        "subjects_with_no_use_of_label": dict(zip(labels, never.sum(axis=0).tolist())),
        "beta_log_odds_per_unit_amplitude": round_nested(beta_mean),
        "do_effect_probability_au_full_vs_none": round_nested(do_mean),
        "reliability": reliability,
        "checks": {
            "sum_do_effect_over_labels_abs_max_33_columns": round(
                float(np.abs(do_mean[:, :33].sum(axis=0)).max()), 4
            ),
            "r_per_subject_mean_vs_pooled_fit": round(pearson(beta_mean[:, fam], pooled[:, fam]), 4),
            "r_selected_lambda_vs_lowest_lambda": round(
                pearson(beta_mean[:, fam], beta_low[:, fam]), 4
            ),
            "slope_selected_on_lowest_lambda": round(
                float(np.polyfit(beta_low[:, fam].ravel(), beta_mean[:, fam].ravel(), 1)[0]), 4
            ),
            "median_stimulus_se_over_subject_se_pooled_vs_per_subject": round(
                float(np.median(stim_se[:, fam] / b_se[:, fam])), 3
            ),
        },
    }
    if len(set(strata)) > 1:
        a, b = sorted(set(strata))
        result["checks"]["stim_set_agreement_beta_family_25"] = {
            "sets": [a, b],
            "r": round(
                pearson(betas[strata == a].mean(axis=0)[:, fam], betas[strata == b].mean(axis=0)[:, fam]),
                4,
            ),
        }

    rows = []
    for l, label in enumerate(labels):
        for k, name in enumerate(names):
            rows.append(
                {
                    "label": label,
                    "effect": name,
                    "beta": beta_mean[l, k],
                    "beta_lo": b_lo[l, k],
                    "beta_hi": b_hi[l, k],
                    "beta_se_subject": b_se[l, k],
                    "beta_se_stimulus": stim_se[l, k],
                    "do_effect": do_mean[l, k],
                    "do_lo": d_lo[l, k],
                    "do_hi": d_hi[l, k],
                }
            )
    arrays = {"beta": beta_mean, "lo": b_lo, "hi": b_hi, "do": do_mean, "fam": fam}
    return result, pd.DataFrame(rows), arrays


TRIAL_COUNT_NOTE = {
    "notes_claim": "119,387 (own count) versus 121,902 (thesis) for Western basic, excluding 'other'",
    "public_file_count_excluding_other": None,
    "published_rows": None,
    "design_rows": "60 subjects x 2400 trials = 144,000 (thesis and Snoek preprocessing code)",
    "rows_missing_from_public_files": None,
    "implied_other_among_missing_rows_if_thesis_used_all_144000": None,
    "other_rate_in_public_rows": None,
    "thesis_mean_sd_per_subject": [2031.7, 311.5],
    "public_mean_sd_per_subject_excluding_other": None,
    "conclusion": (
        "Likely but not proven. Snoek's determine_train_and_test.py drops from the test "
        "split every row whose configuration also occurs in train, then rewrites the "
        "subject TSVs. That explains why public subjects have 2342 or 2368 rows, not 2400. "
        "If the thesis count was taken before that step, the 3,090 dropped rows must hold "
        "575 'other' responses, 18.6 percent against a 15.3 percent base rate, and the "
        "mean per subject fits. The raw .mat files are not public, so this cannot be "
        "confirmed. The removal depends on the stimulus, not the response, so it should "
        "not bias AU effects."
    ),
}


DATA_CAVEATS = [
    "Conversational data have 20 subjects per group, not 60. The zip holds 60 TSVs per group, "
    "of which 40 are header-only, an artefact of a range(1, 61) loop in Snoek's split script.",
    "Stimuli were randomised once, then shared. Every subject in a set saw the same "
    "configuration-plus-face stimuli, and row order is identical across subjects, so trial "
    "order is not available. Subject bootstrap CIs are conditional on that set.",
    "Western basic is two cohorts: subjects 01-15 saw blended identities (959 face codes) "
    "and subjects 16-60 saw 8 base faces. Their configurations share 3 strings.",
    "East Asian basic and conversational use nearly the same configurations. Western basic "
    "and conversational do not. See shared_au_configurations_between_datasets.",
    "The 33 columns are recoded from 42 raw AUs: bilateral AUs become L and R together, and "
    "compounds AU1-2, AU25-12 and AU12-6 become their parts, taking the maximum. So AU "
    "columns are correlated by construction (up to r 0.79) and estimates assume additivity.",
    "AU02R has no unilateral condition: it is active only for bilateral AU2. Read AU02L and "
    "AU02R through AU02_bilateral.",
    "Column amplitudes are rounded to one decimal. The EA and conversational sixths "
    "(0.16, 0.33, 0.5, 0.66, 0.83, 1) survive in the row-label strings; Western basic labels "
    "were rebuilt after rounding. The thesis says steps of 0.25, which the files do not show.",
    "Temporal parameters were randomised but are absent, so effects are averaged over time "
    "course and say nothing about dynamics. Effects hold for GFG faces in this design.",
    "sub_split and trial_split were not used for estimation. They were built for Snoek's "
    "prediction models, and trial_split is unbalanced.",
    "face_id 1-4 repeats across genders in the conversational files, so a face is gender plus id.",
    "Subject 43EA has 783 trials; 01-60 all carry equal weight in the group mean.",
    "Ridge shrinks magnitudes, and more so where a larger lambda was selected (see "
    "slope_selected_on_lowest_lambda). Compare patterns, not sizes, across tasks.",
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-stim-boot", type=int, default=500)
    parser.add_argument("--n-splits", type=int, default=500)
    parser.add_argument("--n-cross-splits", type=int, default=200)
    args = parser.parse_args(argv)

    warnings.simplefilter("ignore", FutureWarning)
    started = time.time()
    frames, file_counts = load_zip(args.zip)

    output = {
        "source": {
            "doi": "10.21942/uva.21261885.v1",
            "licence": "CC BY 4.0",
            "file": "au_emotion_data.zip",
            "md5": "1c8938c50b7702022b03d66987c79402",
            "read_from": "zip members in place, nothing extracted or executed",
        },
        "method": {
            "model": (
                "One ridge logistic per subject and label, one-vs-rest, on 33 AU amplitudes "
                "plus centred face gender. Intercept and gender unpenalised."
            ),
            "aggregate": "Equal-weight mean of subject estimates.",
            "ci": (
                "95 percent percentile, resampling subjects within stimulus set. Conditional "
                "on the fixed stimulus set; see beta_se_stimulus in the CSV."
            ),
            "beta_units": "log odds per unit amplitude (0 to 1); composites are L plus R",
            "do_effect": (
                "Difference in P(label) for do(AU=1) minus do(AU=0), others at observed "
                "values, averaged over the subject's trials. One-vs-rest probabilities are "
                "renormalised over labels per trial, so effects sum to zero across labels. "
                "do(AU=1) is full amplitude, at the top of the tested range."
            ),
            "ceiling": (
                "family_25 is the recommended basis. Use the same-stimuli ceiling if a model "
                "sees these configurations; use the subjects_and_stimuli ceiling if it sees "
                "others. Keep beta and do_effect separate."
            ),
            "seed": args.seed,
            "n_boot": args.n_boot,
            "n_stimulus_boot": args.n_stim_boot,
            "n_splits": args.n_splits,
            "n_cross_splits": args.n_cross_splits,
        },
        "datasets": {},
    }

    csv_parts, arrays = [], {}
    for i, (key, (frame, label_col)) in enumerate(frames.items()):
        labels = LABEL_ORDER[label_col]
        print(f"{key}: {len(frame)} trials", flush=True)
        result, table, arr = analyse(frame, label_col, labels, args, args.seed + 100 * i)
        structure = {**file_counts[key], **describe(frame, label_col, labels)}
        output["datasets"][key] = {"structure": structure, **result}
        csv_parts.append(table.assign(dataset=key))
        arrays[key] = arr
        print(f"  done at {time.time() - started:.0f}s", flush=True)

    names, _ = effect_index()
    output["sanity_check_classic_patterns"] = {
        key: sanity_check(
            frames[key][0], LABEL_ORDER["emotion"], names, arrays[key]["beta"], arrays[key]["lo"],
            arrays[key]["hi"], arrays[key]["do"],
        )
        for key in ("basic_WC", "basic_EA")
    }
    output["anger_au4_diagnostic"] = {
        key: anger_au4_diagnostic(frames[key][0]) for key in ("basic_WC", "basic_EA")
    }
    fam = family_positions(names)
    cross_culture = {}
    for task, wc, ea in (
        ("basic", "basic_WC", "basic_EA"),
        ("conversational", "conversational_WC", "conversational_EA"),
    ):
        labels = output["datasets"][wc]["labels"]
        a, b = arrays[wc]["beta"][:, fam], arrays[ea]["beta"][:, fam]
        cross_culture[task] = {
            "overall": round(pearson(a, b), 4),
            "per_label": {lab: round(pearson(a[l], b[l]), 4) for l, lab in enumerate(labels)},
        }
    output["wc_ea_correlation_beta_family_25"] = cross_culture

    empty = frames["basic_WC"][0].query("cfg == 'empty'")
    output["no_au_control"] = {
        "basic_WC_rows_with_no_active_au": int(len(empty)),
        "response_counts": {k: int(v) for k, v in empty["emotion"].value_counts().items()},
    }

    configs = {key: set(frame["cfg"]) for key, (frame, _) in frames.items()}
    keys = list(configs)
    output["shared_au_configurations_between_datasets"] = {
        f"{a}|{b}": len(configs[a] & configs[b])
        for i, a in enumerate(keys)
        for b in keys[i + 1 :]
    }
    output["data_caveats"] = DATA_CAVEATS

    wc = frames["basic_WC"][0]
    non_other = wc[wc["emotion"] != "other"].groupby("sub").size()
    note = dict(TRIAL_COUNT_NOTE)
    missing = 144000 - len(wc)
    note["public_file_count_excluding_other"] = int(non_other.sum())
    note["published_rows"] = int(len(wc))
    note["rows_missing_from_public_files"] = int(missing)
    note["implied_other_among_missing_rows_if_thesis_used_all_144000"] = int(
        144000 - 121902 - (wc["emotion"] == "other").sum()
    )
    note["other_rate_in_public_rows"] = round(float((wc["emotion"] == "other").mean()), 4)
    note["public_mean_sd_per_subject_excluding_other"] = [
        round(float(non_other.mean()), 1),
        round(float(non_other.std()), 1),
    ]
    output["trial_count_reconciliation"] = note
    output["runtime_seconds"] = round(time.time() - started)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, indent=1) + "\n")
    cols = ["dataset", "label", "effect"]
    table = pd.concat(csv_parts, ignore_index=True)
    table = table[cols + [c for c in table.columns if c not in cols]]
    table.round(4).to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_json} and {args.out_csv}")


if __name__ == "__main__":
    main()
