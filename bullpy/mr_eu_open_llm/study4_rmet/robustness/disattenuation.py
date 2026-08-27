"""
Reliability and Spearman–Brown disattenuation for item-level H1 correlations.

Model side: uses existing k sampled completions per item (item × k binary matrix).
Human side: bootstrap subjects and recompute per-item EQ point-biserial / coef proxy.

If sample matrices are missing, run_stub() prints the required shape and returns.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

try:
    from .data_io import (
        load_human_item_sensitivity,
        load_human_trials,
        model_sample_matrix,
        paired_item_vectors,
    )
except ImportError:
    from data_io import (  # type: ignore
        load_human_item_sensitivity,
        load_human_trials,
        model_sample_matrix,
        paired_item_vectors,
    )


def split_half_reliability(item_by_k: np.ndarray, n_splits: int = 200, seed: int = 42) -> Dict[str, float]:
    """
    Mean Spearman–Brown corrected split-half reliability of the item-accuracy profile.

    For each split, randomly partition the k sample columns into two halves,
    compute per-item mean accuracy in each half, correlate across items, correct
    with Spearman–Brown: r_tt = 2 r / (1 + r).
    """
    x = np.asarray(item_by_k, dtype=float)
    if x.ndim != 2 or x.shape[0] < 5 or x.shape[1] < 2:
        return {"reliability": float("nan"), "n_items": float(x.shape[0] if x.ndim == 2 else 0), "k": float(x.shape[1] if x.ndim == 2 else 0)}
    n_items, k = x.shape
    rng = np.random.default_rng(seed)
    half = k // 2
    rs = []
    for _ in range(n_splits):
        perm = rng.permutation(k)
        a = np.nanmean(x[:, perm[:half]], axis=1)
        b = np.nanmean(x[:, perm[half : half + half]], axis=1)
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 5:
            continue
        r, _ = pearsonr(a[mask], b[mask])
        if np.isfinite(r):
            rs.append(2 * r / (1 + r) if r < 1 else 1.0)
    return {
        "reliability": float(np.nanmean(rs)) if rs else float("nan"),
        "reliability_sd": float(np.nanstd(rs)) if rs else float("nan"),
        "n_items": float(n_items),
        "k": float(k),
        "n_splits": float(len(rs)),
    }


def human_sensitivity_reliability(
    n_boot: int = 50,
    seed: int = 42,
    max_subjects: Optional[int] = None,
) -> Dict[str, float]:
    """
    Bootstrap subjects; per bootstrap, compute item-wise point-biserial(EQ, correct);
    reliability = mean correlation between bootstrap sensitivity vectors (split-half style
    via odd/even bootstrap replicates), Spearman–Brown corrected.

    Uses point-biserial (fast) as a proxy for the logistic EQ coef profile.
    """
    trials = load_human_trials()
    if max_subjects is not None:
        ids = trials["VolunteerID"].drop_duplicates().sample(n=min(max_subjects, trials["VolunteerID"].nunique()), random_state=seed)
        trials = trials[trials["VolunteerID"].isin(ids)]
    subjects = trials["VolunteerID"].unique()
    items = np.sort(trials["item"].unique())
    rng = np.random.default_rng(seed)
    profiles = []
    for _ in range(n_boot):
        samp = rng.choice(subjects, size=len(subjects), replace=True)
        sub = trials[trials["VolunteerID"].isin(samp)]
        vec = []
        for it in items:
            g = sub[sub["item"] == it]
            if len(g) < 20 or g["correct"].nunique() < 2:
                vec.append(np.nan)
                continue
            r, _ = pearsonr(g["eq_total"].to_numpy(float), g["correct"].to_numpy(float))
            vec.append(r)
        profiles.append(vec)
    P = np.asarray(profiles, dtype=float)  # boot × items
    # correlate odd vs even bootstrap means as a stability proxy
    odd = np.nanmean(P[0::2], axis=0)
    even = np.nanmean(P[1::2], axis=0)
    mask = np.isfinite(odd) & np.isfinite(even)
    r, _ = pearsonr(odd[mask], even[mask])
    r_tt = 2 * r / (1 + r) if np.isfinite(r) and r < 1 else (1.0 if np.isfinite(r) else float("nan"))
    return {
        "reliability": float(r_tt),
        "raw_odd_even_r": float(r) if np.isfinite(r) else float("nan"),
        "n_boot": float(n_boot),
        "n_subjects": float(len(subjects)),
        "n_items": float(len(items)),
        "method": "bootstrap_subjects_pointbiserial_odd_even_SB",
    }


def disattenuate(r_obs: float, rel_x: float, rel_y: float) -> float:
    denom = np.sqrt(max(rel_x, 1e-12) * max(rel_y, 1e-12))
    return float(r_obs / denom)


def disattenuate_with_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    rel_x: float,
    rel_y: float,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap CI on disattenuated r (resample items; hold reliabilities fixed)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    r_obs, _ = pearsonr(x, y)
    r_true = disattenuate(float(r_obs), rel_x, rel_y)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ri, _ = pearsonr(x[idx], y[idx])
        boots.append(disattenuate(float(ri), rel_x, rel_y))
    boots = np.asarray(boots, float)
    return {
        "r_obs": float(r_obs),
        "rel_x": float(rel_x),
        "rel_y": float(rel_y),
        "r_disattenuated": float(r_true),
        "ci_low": float(np.nanpercentile(boots, 2.5)),
        "ci_high": float(np.nanpercentile(boots, 97.5)),
    }


def run_stub() -> Dict[str, Any]:
    msg = (
        "disattenuation needs model sample matrices of shape (n_items, k) with k>=2 "
        "binary correctness columns per item, plus human trial-level data. "
        "Expected locations: results/model/<model>/rmet_eval_*_full_*.json "
        "(trials[].samples.predictions) and data/processed/card_rmet_item_level.csv."
    )
    print(msg)
    return {"status": "stub", "message": msg}


def run_disattenuation_for_model(
    model: str,
    metric: str = "sample_accuracy",
    human_n_boot: int = 40,
    seed: int = 42,
    human_rel_cache: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    mat, items, meta = model_sample_matrix(model)
    if mat.size == 0 or mat.shape[1] < 2:
        out = run_stub()
        out["model"] = model
        return out
    rel_y = split_half_reliability(mat, seed=seed)
    rel_x = human_rel_cache if human_rel_cache is not None else human_sensitivity_reliability(n_boot=human_n_boot, seed=seed)
    human = load_human_item_sensitivity()
    # rebuild model table means from matrix for pairing
    model_df = pd.DataFrame({"item": items, "sample_accuracy": np.nanmean(mat, axis=1), "det_correct": mat[:, 0]})
    x, y, n = paired_item_vectors(human, model_df, metric=metric)
    diss = disattenuate_with_bootstrap(x, y, rel_x["reliability"], rel_y["reliability"], seed=seed)
    return {
        "status": "ok",
        "model": model,
        "model_meta": meta,
        "rel_model_item_profile": rel_y,
        "rel_human_sensitivity": rel_x,
        "disattenuation": diss,
        "note": (
            f"Model k={meta.get('k')} samples/item (pipeline default 10; "
            "protocol ideal was 20–50 — treat reliability as lower-bound precision)."
        ),
    }
