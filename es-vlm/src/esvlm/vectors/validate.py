"""Held-out projection AUROC and a matched-random null."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from esvlm.steering.hooks import unit_vector
from esvlm.vectors.extract import difference_of_means


def project(acts: np.ndarray, direction: np.ndarray) -> np.ndarray:
    v = unit_vector(direction)
    return acts @ v


def projection_auroc(pos: np.ndarray, neg: np.ndarray, direction: np.ndarray) -> float:
    scores = np.concatenate([project(pos, direction), project(neg, direction)])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return float(roc_auc_score(labels, scores))


def held_out_auroc(
    pos: np.ndarray,
    neg: np.ndarray,
    *,
    hold_frac: float = 0.25,
    seed: int = 0,
) -> float:
    rng = np.random.default_rng(seed)
    n_pos = max(1, int(len(pos) * (1.0 - hold_frac)))
    n_neg = max(1, int(len(neg) * (1.0 - hold_frac)))
    pos_idx = rng.permutation(len(pos))
    neg_idx = rng.permutation(len(neg))
    direction = difference_of_means(pos[pos_idx[:n_pos]], neg[neg_idx[:n_neg]])
    return projection_auroc(pos[pos_idx[n_pos:]], neg[neg_idx[n_neg:]], direction)


def matched_random_null(
    pos: np.ndarray,
    neg: np.ndarray,
    *,
    n_splits: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """Shuffle labels, rebuild the direction, record held-out AUROC."""

    rng = np.random.default_rng(seed)
    all_acts = np.concatenate([pos, neg], axis=0)
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    scores = []
    for i in range(n_splits):
        shuffled = rng.permutation(labels)
        p = all_acts[shuffled == 1]
        n = all_acts[shuffled == 0]
        if len(p) < 4 or len(n) < 4:
            continue
        scores.append(held_out_auroc(p, n, seed=seed + i + 1))
    return np.asarray(scores, dtype=np.float64)
