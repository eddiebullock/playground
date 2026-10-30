"""Metric primitives with known-answer unit tests (entropy, RDM, Spearman, LOO)."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler


def shannon_entropy(probs: Sequence[float], *, log_base: str = "e") -> float:
    arr = np.asarray(probs, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    total = float(arr.sum())
    if total <= 0:
        return 0.0
    p = arr / total
    if log_base == "e":
        return float(-np.sum(p * np.log(p)))
    return float(-np.sum(p * np.log(p) / np.log(float(log_base))))


def binomial_gt_chance(n_correct: int, n: int, p0: float) -> float:
    return float(binomtest(k=n_correct, n=n, p=p0, alternative="greater").pvalue)


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def cosine_rdm(X: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance RDM; diagonal zero."""
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    U = X / norms
    sim = U @ U.T
    sim = np.clip(sim, -1.0, 1.0)
    rdm = 1.0 - sim
    np.fill_diagonal(rdm, 0.0)
    return rdm


def upper_triangle(rdm: np.ndarray) -> np.ndarray:
    n = rdm.shape[0]
    i, j = np.triu_indices(n, k=1)
    return rdm[i, j]


def rsa_spearman(rdm_a: np.ndarray, rdm_b: np.ndarray) -> float:
    a = upper_triangle(rdm_a)
    b = upper_triangle(rdm_b)
    rho, _ = spearmanr(a, b)
    return float(rho)


def loo_ridge_rho(X: np.ndarray, y: Iterable[float]) -> float:
    """Leave-one-out ridge; alpha chosen by CV inside each training fold via RidgeCV."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(list(y), dtype=float)
    mask = np.isfinite(y)
    X, y = X[mask], y[mask]
    n = len(y)
    if n < 8:
        return float("nan")
    preds = np.zeros(n)
    loo = LeaveOneOut()
    for train, test in loo.split(X):
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X[train])
        Xv = scaler.transform(X[test])
        model = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0))
        model.fit(Xt, y[train])
        preds[test] = model.predict(Xv)
    rho, _ = spearmanr(y, preds)
    return float(rho)
