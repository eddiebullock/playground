"""RQ1: E vs S vs controls vs matched-random null."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from esvlm.vectors.validate import held_out_auroc, matched_random_null, projection_auroc


def separability_matrix(directions: Mapping[str, np.ndarray]) -> np.ndarray:
    names = list(directions)
    mat = np.zeros((len(names), len(names)), dtype=np.float64)
    for i, a in enumerate(names):
        ua = directions[a] / np.linalg.norm(directions[a])
        for j, b in enumerate(names):
            ub = directions[b] / np.linalg.norm(directions[b])
            mat[i, j] = float(ua @ ub)
    return mat


def residualize(vector: np.ndarray, remove: np.ndarray) -> np.ndarray:
    """Unit vector of ``vector`` after removing the component along ``remove``."""

    u = remove / np.linalg.norm(remove)
    leftover = vector - float(vector @ u) * u
    norm = float(np.linalg.norm(leftover))
    if norm < 1e-12:
        raise ValueError("vector is collinear with the removed direction")
    return leftover / norm


def unique_fraction(vector: np.ndarray, other: np.ndarray) -> float:
    """||vector_⊥other|| / ||vector||. 0 if parallel, 1 if orthogonal."""

    u = other / np.linalg.norm(other)
    leftover = vector - float(vector @ u) * u
    denom = float(np.linalg.norm(vector))
    if denom < 1e-12:
        return float("nan")
    return float(np.linalg.norm(leftover) / denom)


def summarize_direction(
    pos: np.ndarray,
    neg: np.ndarray,
    direction: np.ndarray,
    *,
    n_null: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    auroc = projection_auroc(pos, neg, direction)
    held = held_out_auroc(pos, neg, seed=seed)
    null = matched_random_null(pos, neg, n_splits=n_null, seed=seed)
    return {
        "auroc": auroc,
        "held_out_auroc": held,
        "null_mean": float(null.mean()) if len(null) else float("nan"),
        "null_std": float(null.std()) if len(null) else float("nan"),
    }
