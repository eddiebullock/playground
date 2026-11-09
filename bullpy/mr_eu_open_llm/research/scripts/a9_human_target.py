#!/usr/bin/env python3
"""Draft amendment A9: human cue effects and ceilings from Liu et al. 2022, Experiment II.

The fit is A4's human fit (research/scripts/human_au_effects.py), imported, not copied: one ridge
logistic per subject and label, one-vs-rest, on A4's 33 AU columns, with ridge strength chosen by
held-out log loss on unseen stimuli over the same grid, then G-computation of do(AU=1) minus
do(AU=0) with probabilities renormalised over labels, on the 25-family basis. The one change is
forced by the data: Liu does not release face gender per stimulus, so the only unpenalised
covariate is the intercept. The loops that build design matrices are written here for that
reason; every estimating step (fit_ridge_logit, do_effects, effect_index, family_positions,
bootstrap, split-half, Spearman-Brown) is human_au_effects' own.

Data files are read in place (scipy.io.loadmat, no code execution), checked against pinned sha256,
and never copied. Outputs are small numeric JSON files.

Usage:
    python -I research/scripts/a9_human_target.py target \
        --out research/reproduction/a9_human_target.json \
        --ceiling-out research/reproduction/a9_human_ceiling.json
    python -I research/scripts/a9_human_target.py chen-overlap \
        --chen data/external/chen2024/WE_resp_data.mat --out research/reproduction/a9_chen_overlap.json
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "scripts"))
sys.path.insert(0, str(ROOT))

import human_au_effects as hae  # noqa: E402

from scripts import a4_analyse as an  # noqa: E402
from scripts import a9_design as a9  # noqa: E402

SNOEK_ZIP = ROOT / "data" / "external" / "snoek_au" / "au_emotion_data.zip"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _labels(params: np.ndarray) -> List[str]:
    entry = params[0, 0]
    return [str(np.ravel(x)[0]) for x in np.ravel(entry["labels"])]


def load_liu(root: Path = ROOT, check: bool = True) -> Dict[str, Any]:
    """Responses, peak amplitudes and frame-wise amplitudes, with structural checks."""
    from scipy.io import loadmat

    if check:
        a9.check_liu_files(root)
    folder = root / a9.LIU_DIR
    resp = loadmat(folder / "resp_threetask_20par.mat")
    par = loadmat(folder / "parameters_dynamic_precise.mat")
    labels = _labels(par["params"])
    if tuple(labels) != a9.LIU_LABELS:
        raise SystemExit(f"params.labels differ from a9_design.LIU_LABELS: {labels}")
    all_same = all(tuple(str(np.ravel(x)[0]) for x in np.ravel(par["params"][0, i]["labels"])) == a9.LIU_LABELS
                   for i in range(par["params"].shape[1]))
    p = np.asarray(par["p"], dtype=float)
    presence = np.asarray(resp["stimuli_raw"]) > 0
    codes = np.asarray(resp["resp_emo"], dtype=int)
    if p.shape != (42, a9.N_STIMULI) or codes.shape != (a9.N_STIMULI, a9.N_SUBJECTS):
        raise SystemExit(f"unexpected shapes p {p.shape}, resp_emo {codes.shape}")
    if not np.array_equal(p > 0, presence):
        raise SystemExit("p > 0 does not match stimuli_raw")
    if set(np.unique(codes)) - set(a9.RESPONSE_CODES):
        raise SystemExit(f"unexpected response codes {np.unique(codes)}")
    return {"labels": labels, "labels_same_for_all_stimuli": bool(all_same), "p": p, "presence": presence,
            "pseq": np.asarray(par["pseq"], dtype=float), "codes": codes}


def design_matrix(p: np.ndarray, decimals: int = a9.AMPLITUDE_DECIMALS) -> np.ndarray:
    """Stimuli by 33 columns, through a9_design.liu_to_columns."""
    X = np.zeros((p.shape[1], len(hae.AU_COLS)))
    for i in range(p.shape[1]):
        cols = a9.liu_to_columns(dict(zip(a9.LIU_LABELS, p[:, i])), decimals)
        for c, v in cols.items():
            X[i, hae.AU_COLS.index(c)] = v
    return X


def response_labels(codes: np.ndarray) -> np.ndarray:
    """Codes 1 to 7 to A4's label names (stimuli by subjects)."""
    lookup = np.array([""] + [a9.RESPONSE_CODES[k] for k in range(1, 8)], dtype=object)
    return lookup[codes]


# --------------------------------------------------------------------------
# Estimation: human_au_effects' fit, without the gender column
# --------------------------------------------------------------------------


def _X(X_au: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X_cov = np.ones((len(X_au), 1))
    return X_cov, np.column_stack([X_cov, X_au])


def choose_lambda(Y: np.ndarray, X_au: np.ndarray, labels: Sequence[str], grid=a9.LAMBDA_GRID, seed: int = 0):
    """hae.choose_lambda's rule: two folds of unseen stimuli, held-out log loss summed over
    subjects and labels, intercept-only loss reported beside it. Every subject saw every stimulus,
    so a stimulus is its index."""
    rng = np.random.default_rng(seed)
    is_test = rng.integers(0, 2, len(X_au)) == 1
    _, X = _X(X_au)
    losses, null_total, null_count = {}, 0.0, 0
    for s in range(Y.shape[1]):
        for label in labels:
            y = (Y[:, s] == label).astype(float)
            q = float(np.clip(y[~is_test].mean(), 1e-6, 1 - 1e-6))
            null_total += float(-(y[is_test] * np.log(q) + (1 - y[is_test]) * np.log(1 - q)).sum())
            null_count += int(is_test.sum())
    for lam in grid:
        pen = np.r_[0.0, np.full(X_au.shape[1], lam)]
        total, count = 0.0, 0
        for s in range(Y.shape[1]):
            for label in labels:
                y = (Y[:, s] == label).astype(float)
                b = hae.fit_ridge_logit(X[~is_test], y[~is_test], pen=pen)
                q = np.clip(hae.expit(X[is_test] @ b), 1e-6, 1 - 1e-6)
                total += float(-(y[is_test] * np.log(q) + (1 - y[is_test]) * np.log(1 - q)).sum())
                count += int(is_test.sum())
        losses[lam] = total / count
    best = min(losses, key=losses.get)
    report = {str(k): round(v, 5) for k, v in losses.items()}
    report["intercept_only"] = round(null_total / null_count, 5)
    return best, report


def fit_subject_effects(Y: np.ndarray, X_au: np.ndarray, labels: Sequence[str], lam: float):
    """hae.fit_subject_effects without gender. Returns (S, L, E) beta and do arrays, never-used flags."""
    names, members = hae.effect_index()
    X_cov, X = _X(X_au)
    pen = np.r_[0.0, np.full(X_au.shape[1], lam)]
    S = Y.shape[1]
    betas = np.zeros((S, len(labels), len(names)))
    dos = np.zeros_like(betas)
    never = np.zeros((S, len(labels)), dtype=bool)
    for s in range(S):
        fitted = np.zeros((len(labels), X.shape[1]))
        for l, label in enumerate(labels):
            y = (Y[:, s] == label).astype(float)
            never[s, l] = y.sum() == 0
            fitted[l] = hae.fit_ridge_logit(X, y, pen=pen)
            betas[s, l] = [fitted[l, 1:][c].sum() for c in members]
        dos[s] = hae.do_effects(X_au, X_cov, fitted, members)
    return betas, dos, never


def pooled_do(Y: np.ndarray, X_au: np.ndarray, labels: Sequence[str], subjects: np.ndarray, stimuli: np.ndarray,
              lam: float) -> np.ndarray:
    """an.pooled_do's fit for a design where every subject saw every stimulus: the share of the
    chosen subjects giving each label, weighted by their number. Family basis."""
    X_cov, X = _X(X_au[stimuli])
    pen = np.r_[0.0, np.full(X_au.shape[1], lam)]
    n = float(subjects.sum())
    fitted = []
    for label in labels:
        y = (Y[np.ix_(stimuli, np.where(subjects)[0])] == label).mean(axis=1)
        fitted.append(hae.fit_ridge_logit(X, y, w=np.full(len(stimuli), n), pen=pen))
    return hae.do_effects(X_au[stimuli], X_cov, np.vstack(fitted), an.FAMILY_MEMBERS)


def do_ceiling(Y: np.ndarray, X_au: np.ndarray, labels: Sequence[str], splits: int, seed: int,
               lam: float = a9.CEILING_LAMBDA) -> Dict[str, Any]:
    """an.human_do_ceiling's subjects-and-stimuli split: halves share neither, pooled fits, do
    effects, family basis, raw and residual, Spearman-Brown."""
    rng = np.random.default_rng(seed)
    S, N = Y.shape[1], Y.shape[0]
    rs, plain = [], []
    for _ in range(splits):
        mask = np.zeros(S, dtype=bool)
        mask[rng.permutation(S)[: S // 2]] = True
        perm = rng.permutation(N)
        a, b = perm[: N // 2], perm[N // 2:]
        first, second = pooled_do(Y, X_au, labels, mask, a, lam), pooled_do(Y, X_au, labels, ~mask, b, lam)
        rs.append(an.cosine(first, second))
        plain.append(an.cosine(an.remove_common(first), an.remove_common(second)))
    r, q = float(np.mean(rs)), float(np.mean(plain))
    brown, brown_q = hae.spearman_brown(r), hae.spearman_brown(q)
    return {"split_half_r": round(r, 4), "spearman_brown": round(brown, 4),
            "ceiling": round(float(np.sqrt(max(brown, 0))), 4),
            "split_half_r_residual": round(q, 4), "ceiling_residual": round(float(np.sqrt(max(brown_q, 0))), 4),
            "split_half_r_sd": round(float(np.std(rs, ddof=1)), 4),
            "split_half_r_residual_sd": round(float(np.std(plain, ddof=1)), 4),
            "splits": splits, "ridge": lam}


def subjects_only_ceiling(dos: np.ndarray, splits: int, seed: int) -> Dict[str, Any]:
    """Same-stimuli split-half of the per-subject do matrices (subjects split, stimuli shared).
    Not A4's ceiling for a model on other stimuli; reported for comparison only."""
    rng = np.random.default_rng(seed)
    fam = an.FAM
    rs, plain = [], []
    S = dos.shape[0]
    for _ in range(splits):
        perm = rng.permutation(S)
        a, b = dos[perm[: S // 2]].mean(axis=0)[:, fam], dos[perm[S // 2:]].mean(axis=0)[:, fam]
        rs.append(an.cosine(a, b))
        plain.append(an.cosine(an.remove_common(a), an.remove_common(b)))
    r, q = float(np.mean(rs)), float(np.mean(plain))
    return {"split_half_r": round(r, 4), "ceiling": round(float(np.sqrt(max(hae.spearman_brown(r), 0))), 4),
            "split_half_r_residual": round(q, 4),
            "ceiling_residual": round(float(np.sqrt(max(hae.spearman_brown(q), 0))), 4), "splits": splits}


# --------------------------------------------------------------------------
# Checks on the map and labels
# --------------------------------------------------------------------------


def snoek_token_check(zip_path: Path = SNOEK_ZIP) -> Dict[str, Any]:
    """Do Snoek's raw tokens equal Liu's stripped labels, and does the A9 map reproduce Snoek's 33
    columns from the raw tokens in Snoek's own labels (EA and conversational sets keep them)?"""
    if not zip_path.exists():
        return {"checked": False, "reason": f"{zip_path} missing"}
    frames, _ = hae.load_zip(zip_path)
    sixths = {"16": 1 / 6, "33": 1 / 3, "50": 0.5, "66": 2 / 3, "83": 5 / 6, "100": 1.0}
    tokens, n, mismatched, collisions = set(), 0, 0, 0
    for key in ("basic_EA", "conversational_EA", "conversational_WC"):
        table = frames[key][0].drop_duplicates("cfg")
        for _, row in table.iterrows():
            if row["cfg"] == "empty":
                continue
            raw: Dict[str, float] = {}
            parts = []
            for item in row["cfg"].split("_"):
                token, amp = item.rsplit("-", 1)
                tokens.add(token)
                parts.append(token)
                raw[token] = max(raw.get(token, 0.0), sixths[amp])
            built: Dict[str, float] = {}
            for token, value in raw.items():
                for column in a9.token_columns(token):
                    if column in built:
                        collisions += 1
                    built[column] = max(built.get(column, 0.0), value)
            n += 1
            if any(abs(round(built.get(c, 0.0), 1) - float(row[c])) > 1e-9 for c in hae.AU_COLS):
                mismatched += 1
    stripped = {a9.snoek_token(label) for label in a9.LIU_LABELS}
    return {"checked": True, "configurations": n, "mismatched": mismatched, "column_collisions_resolved_by_max": collisions,
            "snoek_tokens": len(tokens), "liu_stripped_labels": len(stripped),
            "token_sets_equal": tokens == stripped, "only_in_snoek": sorted(tokens - stripped),
            "only_in_liu": sorted(stripped - tokens)}


def label_corroboration(beta: np.ndarray, labels: Sequence[str]) -> Dict[str, List[str]]:
    """Top three families by mean log-odds weight per label. Reported to corroborate the code order
    taken from Liu's released code; it is never used to set that order."""
    names, _ = hae.effect_index()
    fam = hae.family_positions(names)
    fam_names = [names[p].replace("_bilateral", "") for p in fam]
    return {lab: [fam_names[k] for k in np.argsort(-beta[l, fam])[:3]] for l, lab in enumerate(labels)}


def long_frame(Y: np.ndarray, X_au: np.ndarray):
    import pandas as pd

    S = Y.shape[1]
    frame = pd.DataFrame(np.repeat(X_au, S, axis=0), columns=hae.AU_COLS)
    frame["emotion"] = Y.reshape(-1)
    return frame


# --------------------------------------------------------------------------
# Main analysis
# --------------------------------------------------------------------------


def analyse(data: Mapping[str, Any], seed: int, n_boot: int, n_splits: int, ceiling_splits: int,
            store_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    labels = list(a9.HUMAN_LABELS)
    names, _ = hae.effect_index()
    fam = hae.family_positions(names)
    X_au = design_matrix(data["p"])
    Y = response_labels(data["codes"])

    lam, losses = choose_lambda(Y, X_au, labels, seed=seed)
    betas, dos, never = fit_subject_effects(Y, X_au, labels, lam)
    beta_mean, do_mean = betas.mean(axis=0), dos.mean(axis=0)
    strata = np.zeros(betas.shape[0])
    b_lo, b_hi, _ = hae.stratified_bootstrap_ci(betas, strata, n_boot, seed)
    d_lo, d_hi, _ = hae.stratified_bootstrap_ci(dos, strata, n_boot, seed + 1)

    reliability = {}
    for basis, idx in {"family_25": fam, "column_33": list(range(33))}.items():
        for metric, arr in {"beta": betas, "do_effect": dos}.items():
            overall, per_label = hae.split_half_reliability(arr[:, :, idx], strata, n_splits, seed + 3)
            reliability[f"{metric}_{basis}"] = {"overall": overall, "per_label": dict(zip(labels, per_label)),
                                                 "leave_one_subject_out_mean_r": hae.leave_one_subject_out(arr[:, :, idx])}

    ceiling = do_ceiling(Y, X_au, labels, ceiling_splits, a9.SEED)
    ceiling_selected = do_ceiling(Y, X_au, labels, ceiling_splits, a9.SEED, lam) if lam != a9.CEILING_LAMBDA else ceiling
    same_stimuli = subjects_only_ceiling(dos, ceiling_splits, a9.SEED)

    # Presence-coded sensitivity, through the same rule.
    X_bin = (X_au > 0).astype(float)
    lam_bin, losses_bin = choose_lambda(Y, X_bin, labels, seed=seed)
    _, dos_bin, _ = fit_subject_effects(Y, X_bin, labels, lam_bin)
    human = do_mean[:, fam]
    human_bin = dos_bin.mean(axis=0)[:, fam]

    # Fixed lowest-lambda check, as hae reports.
    betas_low = fit_subject_effects(Y, X_au, labels, min(a9.LAMBDA_GRID))[0]

    store = json.loads(store_path.read_text(encoding="utf-8"))
    yardstick = {}
    for ds in ("basic_WC", "basic_EA"):
        other = an.human_matrix(store, ds)[0]
        yardstick[ds] = {"raw": round(an.cosine(human, other), 4),
                         "residual": round(an.cosine(an.remove_common(human), an.remove_common(other)), 4)}
    wc, ea = an.human_matrix(store, "basic_WC")[0], an.human_matrix(store, "basic_EA")[0]
    yardstick["snoek_basic_WC_to_EA"] = {"raw": round(an.cosine(wc, ea), 4),
                                         "residual": round(an.cosine(an.remove_common(wc), an.remove_common(ea)), 4)}

    total = float(np.sum(human ** 2))
    plain = an.remove_common(human)
    approx = [j for j, f in enumerate(an.FAMILIES) if f in a9.APPROXIMATED_FAMILIES]
    confirmed = an.family_columns(an.d.CONFIRMED_FAMILIES)
    frame = long_frame(Y, X_au)
    sanity = hae.sanity_check(frame, labels, names, beta_mean, b_lo, b_hi, do_mean)

    counts = {lab: int((Y == lab).sum()) for lab in labels}
    active = X_au > 0
    corr = np.corrcoef(X_au.T)
    off = corr - np.eye(33)
    i, j = np.unravel_index(np.nanargmax(np.abs(off)), off.shape)
    fam_X = np.column_stack([X_au[:, m].max(axis=1) for m in an.FAMILY_MEMBERS])
    eig = np.linalg.eigvalsh(np.corrcoef(fam_X.T))
    small = int(np.sum((data["p"] > 0) & (np.round(data["p"], a9.AMPLITUDE_DECIMALS) == 0)))
    block = {
        "structure": {
            "n_subjects": int(Y.shape[1]), "n_stimuli": int(Y.shape[0]), "n_trials": int(Y.size),
            "label_counts": counts,
            "subjects_label_pairs_never_used": int(never.sum()),
            "active_raw_aus_per_stimulus": {"min": int(data["presence"].sum(0).min()),
                                            "mean": round(float(data["presence"].sum(0).mean()), 3),
                                            "max": int(data["presence"].sum(0).max())},
            "raw_au_presence_count_min_max": [int(data["presence"].sum(1).min()), int(data["presence"].sum(1).max())],
            "column_active_rate_min_max": [round(float(active.mean(0).min()), 3), round(float(active.mean(0).max()), 3)],
            "family_active_count_min_max": [int((fam_X > 0).sum(0).min()), int((fam_X > 0).sum(0).max())],
            "largest_column_correlation": {"pair": [hae.AU_COLS[i], hae.AU_COLS[j]], "r": round(float(corr[i, j]), 3)},
            "family_basis_lambda_min": round(float(eig.min()), 3),
            "family_basis_condition_number": round(float(eig.max() / eig.min()), 2),
            "amplitude_quantiles_nonzero": [round(float(x), 4) for x in np.quantile(data["p"][data["p"] > 0], [0, .25, .5, .75, 1])],
            "raw_amplitudes_rounding_to_zero": small,
            "labels_same_for_all_stimuli": data["labels_same_for_all_stimuli"],
        },
        "ridge_lambda_selected": lam,
        "heldout_log_loss_by_lambda": losses,
        "effect_names": names,
        "labels": labels,
        "subjects_with_no_use_of_label": dict(zip(labels, never.sum(axis=0).tolist())),
        "beta_log_odds_per_unit_amplitude": hae.round_nested(beta_mean),
        "beta_ci95": [hae.round_nested(b_lo), hae.round_nested(b_hi)],
        "do_effect_probability_au_full_vs_none": hae.round_nested(do_mean),
        "do_ci95": [hae.round_nested(d_lo), hae.round_nested(d_hi)],
        "reliability": reliability,
        "checks": {
            "sum_do_effect_over_labels_abs_max_33_columns": round(float(np.abs(do_mean[:, :33].sum(axis=0)).max()), 4),
            "r_selected_lambda_vs_lowest_lambda": round(hae.pearson(beta_mean[:, fam], betas_low.mean(axis=0)[:, fam]), 4),
        },
        "presence_sensitivity": {
            "ridge_lambda_selected": lam_bin, "heldout_log_loss_by_lambda": losses_bin,
            "do_family_25": hae.round_nested(human_bin),
            "cfa_vs_amplitude_target": {"raw": round(an.cosine(human_bin, human), 4),
                                        "residual": round(an.cosine(an.remove_common(human_bin), plain), 4)},
        },
        "human_energy": {"AU22_AU27_share": round(float(np.sum(human[:, approx] ** 2)) / total, 4),
                         "common_profile_share": round(1 - float(np.sum(plain ** 2)) / total, 4),
                         "confirmed_share_of_residual": round(float(np.sum(plain[:, confirmed] ** 2) / np.sum(plain ** 2)), 4)},
        "label_corroboration_top3_families_by_beta": label_corroboration(beta_mean, labels),
        "sanity_check_classic_patterns": sanity,
    }
    ceiling_out = {"datasets": {a9.DATASET: ceiling}, "splits": ceiling_splits, "seed": a9.SEED,
                   "note": "Do effects, family_25 basis, subjects and stimuli split, pooled fits at ridge 1.0 as "
                           "a4_analyse.human_do_ceiling. Human data only.",
                   "at_selected_ridge": ceiling_selected, "subjects_only_same_stimuli": same_stimuli,
                   "human_reference_cfa": yardstick}
    return block, ceiling_out


def chen_overlap(chen_path: Path, zip_path: Path = SNOEK_ZIP) -> Dict[str, Any]:
    """Is Chen et al. 2024's Western file the Snoek basic WC experiment? Trial-level match.

    Within each Snoek cohort every subject saw the same stimuli in the same row order, and within
    each Chen stimulus group likewise, so a stimulus is identified by the vector of all subjects'
    responses. Chen participant j is paired with Snoek subject j + 1 (the response counts point to
    this pairing; the match below tests it)."""
    from scipy.io import loadmat

    m = loadmat(chen_path)
    S, R = m["stimuli_AUs"], m["resp_emotion"].astype(int)
    chen_labels = [str(np.ravel(x)[0]) for x in np.ravel(m["label_AUs"])]
    chen_short = [re.search(r"\((AU[^)]*)\)", x).group(1) for x in chen_labels]
    order = [str(np.ravel(x)[0]).lower() for x in np.ravel(m["label_responses"])]
    frames, _ = hae.load_zip(zip_path)
    wc = frames["basic_WC"][0]
    code = {"happy": 1, "surprise": 2, "fear": 3, "disgust": 4, "anger": 5, "sadness": 6, "other": 7}
    groups = [0 if np.array_equal(S[:, :, j], S[:, :, 0]) else 1 for j in range(S.shape[2])]
    out: Dict[str, Any] = {"chen_file": chen_path.name, "chen_shape_aus_trials_participants": list(S.shape),
                           "chen_label_responses": order, "chen_au_labels": chen_short,
                           "chen_stimulus_groups": dict(collections.Counter(groups)), "cohorts": {}}
    for cohort, idx in (("A_sub01_15", list(range(0, 15))), ("B_sub16_60", list(range(15, 60)))):
        subs = [f"{j + 1:02d}WC" for j in idx]
        parts = [wc[wc["sub"] == s] for s in subs]
        sr = np.stack([p_["emotion"].map(code).to_numpy() for p_ in parts], axis=1)
        cr = R[:, idx]
        ham = (sr[:, None, :] != cr[None, :, :]).sum(axis=2)
        best, bestd = ham.argmin(axis=1), ham.min(axis=1)
        second = np.partition(ham, 1, axis=1)[:, 1]
        index = collections.defaultdict(list)
        for t in range(cr.shape[0]):
            index[cr[t].tobytes()].append(t)
        exact = [index.get(sr[r].astype(cr.dtype).tobytes(), []) for r in range(len(sr))]
        per_sub = (sr == cr[best]).mean(axis=0)
        cols = hae.AU_COLS
        snoek_fam = (np.column_stack([parts[0][[c for c in cols if c[:4] == f]].to_numpy().max(axis=1)
                                      for f in an.FAMILIES]) > 0)
        chen_fam = np.zeros_like(snoek_fam)
        for a, lab in enumerate(chen_short):
            nums = re.findall(r"\d+", lab)
            for num in nums:
                fam = f"AU{int(num):02d}"
                fam = "AU43" if fam in ("AU41", "AU42") else fam
                chen_fam[:, an.FAMILIES.index(fam)] |= S[a, best, idx[0]].astype(bool)
        others = [f"{k:02d}WC" for k in range(1, 61) if f"{k:02d}WC" not in subs]
        out["cohorts"][cohort] = {
            "pairing": "Chen participant j with Snoek subject j + 1",
            "snoek_rows_per_subject": int(len(sr)), "chen_trials": int(cr.shape[0]),
            "snoek_row_order_shared_across_subjects": bool(all(
                np.array_equal(p_["cfg"].to_numpy(), parts[0]["cfg"].to_numpy()) for p_ in parts)),
            "rows_with_exact_response_vector_in_chen": int(sum(bool(e) for e in exact)),
            "rows_with_unique_exact_match": int(sum(len(e) == 1 for e in exact)),
            "median_hamming_to_best_trial": float(np.median(bestd)),
            "median_margin_to_second_best": float(np.median(second - bestd)),
            "distinct_chen_trials_matched": int(len(set(best))),
            "per_subject_response_agreement_min_median_max": [round(float(per_sub.min()), 4),
                                                              round(float(np.median(per_sub)), 4),
                                                              round(float(per_sub.max()), 4)],
            "aligned_rows_identical_au_family_set_with_AU41_AU42_as_AU43": int((snoek_fam == chen_fam).all(axis=1).sum()),
            "snoek_subjects_not_in_cohort": len(others),
        }
    a, b = out["cohorts"]["A_sub01_15"], out["cohorts"]["B_sub16_60"]
    out["verdict"] = ("same data" if b["rows_with_unique_exact_match"] == b["snoek_rows_per_subject"]
                      and a["per_subject_response_agreement_min_median_max"][1] > 0.99 else "undetermined")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("target")
    t.add_argument("--out", type=Path, default=ROOT / "research/reproduction/a9_human_target.json")
    t.add_argument("--ceiling-out", type=Path, default=ROOT / "research/reproduction/a9_human_ceiling.json")
    t.add_argument("--human", type=Path, default=ROOT / "research/reproduction/human_au_effects.json")
    t.add_argument("--n-boot", type=int, default=2000)
    t.add_argument("--n-splits", type=int, default=500)
    t.add_argument("--ceiling-splits", type=int, default=a9.CEILING_SPLITS)
    c = sub.add_parser("chen-overlap")
    c.add_argument("--chen", type=Path, default=ROOT / "data/external/chen2024/WE_resp_data.mat")
    c.add_argument("--out", type=Path, default=ROOT / "research/reproduction/a9_chen_overlap.json")
    args = parser.parse_args(argv)
    if args.command == "chen-overlap":
        result = chen_overlap(args.chen)
        args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        print(json.dumps({"verdict": result["verdict"]}))
        return 0
    started = time.time()
    data = load_liu()
    block, ceiling = analyse(data, a9.SEED, args.n_boot, args.n_splits, args.ceiling_splits, args.human)
    if block["ridge_lambda_selected"] != a9.HUMAN_RIDGE:
        raise SystemExit(f"Selected ridge {block['ridge_lambda_selected']} differs from the block's {a9.HUMAN_RIDGE}.")
    out = {
        "amendment": "A9",
        "source": {**a9.LIU_SOURCE, "files_sha256": a9.LIU_FILES,
                   "read_from": "scipy.io.loadmat in place; nothing copied"},
        "method": {
            "model": "One ridge logistic per subject and label, one-vs-rest, on A4's 33 AU columns built from "
                     "Liu's peak amplitudes by a9_design.liu_to_columns. Intercept unpenalised. No gender column: "
                     "Liu does not release face gender per stimulus.",
            "aggregate": "Equal-weight mean of subject estimates.",
            "ci": "95 percent percentile, resampling subjects; conditional on the 1,200 stimuli.",
            "response_codes": {str(k): v for k, v in a9.RESPONSE_CODES.items()},
            "response_code_source": "PMI_exp2.m, Mendeley Data 10.17632/dr853shk56.3",
            "seed": a9.SEED, "n_boot": args.n_boot, "n_splits": args.n_splits,
        },
        "au_map": {k: list(v) for k, v in a9.AU_MAP.items()},
        "au_map_check_against_snoek": snoek_token_check(),
        "datasets": {a9.DATASET: block},
    }
    # No runtime field: the file is pinned by sha256 in the design block, so it must be reproducible.
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    args.ceiling_out.write_text(json.dumps(ceiling, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"lambda": block["ridge_lambda_selected"], "ceiling": ceiling["datasets"][a9.DATASET],
                      "reference": ceiling["human_reference_cfa"], "runtime": round(time.time() - started)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
