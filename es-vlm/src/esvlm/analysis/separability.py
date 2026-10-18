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
