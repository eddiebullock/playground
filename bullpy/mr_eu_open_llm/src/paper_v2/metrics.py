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


RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def loo_ridge_rho(X: np.ndarray, y: Iterable[float]) -> float:
    """Leave-one-out ridge; alpha chosen by CV inside each training fold via RidgeCV."""
    preds = nested_ridge_preds(X, y, groups=None)
    y_arr = np.asarray(list(y), dtype=float)
    mask = np.isfinite(y_arr) & np.isfinite(preds)
    if mask.sum() < 8:
        return float("nan")
    rho, _ = spearmanr(y_arr[mask], preds[mask])
    return float(rho)


def _dual_ridge_predict(Xt: np.ndarray, yt: np.ndarray, Xv: np.ndarray, alpha: float) -> np.ndarray:
    """Predict with ||Xw-y||^2 + α||w||^2 via the dual (n x n) system."""
    n = Xt.shape[0]
    K = Xt @ Xt.T
    K.flat[:: n + 1] += float(alpha)
    try:
        dual = np.linalg.solve(K, yt)
    except np.linalg.LinAlgError:
        dual, *_ = np.linalg.lstsq(K, yt, rcond=None)
    return Xv @ (Xt.T @ dual)


def grouped_ridge_preds_fixed_alpha(
    X: np.ndarray,
    y: Iterable[float],
    groups: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Leave-one-group-out Ridge with a frozen alpha (used for permutation nulls)."""
    from sklearn.model_selection import LeaveOneGroupOut

    X = np.asarray(X, dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    n = len(y_arr)
    preds = np.full(n, np.nan, dtype=float)
    finite = np.isfinite(y_arr) & np.isfinite(X).all(axis=1)
    if finite.sum() < 8:
        return preds
    X_use, y_use = X[finite], y_arr[finite]
    g = np.asarray(groups)[finite]
    idx = np.flatnonzero(finite)
    splitter = LeaveOneGroupOut()
    for train, test in splitter.split(X_use, y_use, g):
        if len(train) < 5:
            continue
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X_use[train])
        Xv = scaler.transform(X_use[test])
        preds[idx[test]] = _dual_ridge_predict(Xt, y_use[train], Xv, alpha)
    return preds


def choose_ridge_alpha(X: np.ndarray, y: Iterable[float], *, alphas: Sequence[float] = RIDGE_ALPHAS) -> float:
    """GCV alpha on the full sample; used only to freeze the permutation regulariser."""
    X = np.asarray(X, dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    mask = np.isfinite(y_arr) & np.isfinite(X).all(axis=1)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X[mask])
    model = RidgeCV(alphas=tuple(alphas))
    model.fit(Xs, y_arr[mask])
    return float(model.alpha_)


def nested_ridge_preds(
    X: np.ndarray,
    y: Iterable[float],
    *,
    groups: np.ndarray | None = None,
    alphas: Sequence[float] = RIDGE_ALPHAS,
) -> np.ndarray:
    """Held-out Ridge predictions. LOO if groups is None, else Leave-One-Group-Out.

    Alpha is chosen on the training fold only (RidgeCV GCV). Features are
    standardised on the training fold only.
    """
    from sklearn.model_selection import LeaveOneGroupOut

    X = np.asarray(X, dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    n = len(y_arr)
    preds = np.full(n, np.nan, dtype=float)
    finite = np.isfinite(y_arr) & np.isfinite(X).all(axis=1)
    if finite.sum() < 8:
        return preds
    X_use, y_use = X[finite], y_arr[finite]
    idx = np.flatnonzero(finite)
    if groups is None:
        splitter = LeaveOneOut()
        splits = splitter.split(X_use)
    else:
        g = np.asarray(groups)[finite]
        splitter = LeaveOneGroupOut()
        splits = splitter.split(X_use, y_use, g)
    for train, test in splits:
        if len(train) < 5:
            continue
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X_use[train])
        Xv = scaler.transform(X_use[test])
        model = RidgeCV(alphas=tuple(alphas))
        model.fit(Xt, y_use[train])
        preds[idx[test]] = model.predict(Xv)
    return preds


def spearman_from_preds(y: Iterable[float], preds: np.ndarray) -> float:
    y_arr = np.asarray(list(y), dtype=float)
    preds = np.asarray(preds, dtype=float)
    mask = np.isfinite(y_arr) & np.isfinite(preds)
    if mask.sum() < 8:
        return float("nan")
    rho, _ = spearmanr(y_arr[mask], preds[mask])
    return float(rho)


def residualise(y: Iterable[float], Z: np.ndarray) -> np.ndarray:
    """OLS residuals of y on Z with intercept. Rank-deficient Z is ok (lstsq)."""
    y_arr = np.asarray(list(y), dtype=float)
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z[:, None]
    n = len(y_arr)
    A = np.column_stack([np.ones(n), Z])
    finite = np.isfinite(y_arr) & np.isfinite(A).all(axis=1)
    resid = np.full(n, np.nan, dtype=float)
    if finite.sum() < 3:
        return resid
    beta, *_ = np.linalg.lstsq(A[finite], y_arr[finite], rcond=None)
    resid[finite] = y_arr[finite] - A[finite] @ beta
    return resid


def one_hot(labels: Sequence[str]) -> np.ndarray:
    labels = [str(x) for x in labels]
    uniq = sorted(set(labels))
    idx = {u: i for i, u in enumerate(uniq)}
    Z = np.zeros((len(labels), len(uniq)), dtype=float)
    for i, lab in enumerate(labels):
        Z[i, idx[lab]] = 1.0
    return Z


def holm_bonferroni(pvalues: Sequence[float]) -> np.ndarray:
    """Holm–Bonferroni adjusted p-values, monotonic, clipped to [0, 1]."""
    pvals = np.asarray(list(pvalues), dtype=float)
    m = len(pvals)
    adjusted = np.ones(m, dtype=float)
    if m == 0:
        return adjusted
    order = np.argsort(pvals)
    running = 0.0
    for rank, idx in enumerate(order):
        p = pvals[idx]
        if not np.isfinite(p):
            adjusted[idx] = 1.0
            continue
        adj = (m - rank) * p
        running = max(running, adj)
        adjusted[idx] = min(1.0, running)
    return adjusted


def bootstrap_spearman_ci(
    y: Iterable[float],
    pred: Iterable[float],
    *,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap CI of Spearman rho from paired (y, pred) item resamples."""
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(list(y), dtype=float)
    p_arr = np.asarray(list(pred), dtype=float)
    mask = np.isfinite(y_arr) & np.isfinite(p_arr)
    y_arr, p_arr = y_arr[mask], p_arr[mask]
    n = len(y_arr)
    obs = float("nan")
    if n >= 8:
        obs, _ = spearmanr(y_arr, p_arr)
        obs = float(obs)
    rhos = np.full(n_boot, np.nan, dtype=float)
    if n >= 8:
        for i in range(n_boot):
            idx = rng.integers(0, n, n)
            r, _ = spearmanr(y_arr[idx], p_arr[idx])
            rhos[i] = r
    finite = rhos[np.isfinite(rhos)]
    if len(finite) == 0:
        lo = hi = float("nan")
    else:
        lo, hi = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "rho": obs,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_boot": int(n_boot),
        "n": int(n),
        "excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)),
    }


def bootstrap_delta_rho_ci(
    y: Iterable[float],
    pred_a: Iterable[float],
    pred_b: Iterable[float],
    *,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap CI of rho(y, pred_a) - rho(y, pred_b)."""
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(list(y), dtype=float)
    a = np.asarray(list(pred_a), dtype=float)
    b = np.asarray(list(pred_b), dtype=float)
    mask = np.isfinite(y_arr) & np.isfinite(a) & np.isfinite(b)
    y_arr, a, b = y_arr[mask], a[mask], b[mask]
    n = len(y_arr)

    def _delta(yy, aa, bb) -> float:
        ra, _ = spearmanr(yy, aa)
        rb, _ = spearmanr(yy, bb)
        return float(ra - rb)

    obs = float("nan") if n < 8 else _delta(y_arr, a, b)
    deltas = np.full(n_boot, np.nan, dtype=float)
    if n >= 8:
        for i in range(n_boot):
            idx = rng.integers(0, n, n)
            deltas[i] = _delta(y_arr[idx], a[idx], b[idx])
    finite = deltas[np.isfinite(deltas)]
    if len(finite) == 0:
        lo = hi = float("nan")
    else:
        lo, hi = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "delta_rho": obs,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_boot": int(n_boot),
        "n": int(n),
        "excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)),
    }


def permutation_p(
    observed: float,
    null: Sequence[float],
    *,
    alternative: str = "two-sided",
) -> float:
    """(1 + #{null as or more extreme}) / (1 + n)."""
    null_arr = np.asarray(list(null), dtype=float)
    null_arr = null_arr[np.isfinite(null_arr)]
    n = len(null_arr)
    if n == 0 or not np.isfinite(observed):
        return float("nan")
    if alternative == "greater":
        hits = np.sum(null_arr >= observed)
    elif alternative == "less":
        hits = np.sum(null_arr <= observed)
    else:
        hits = np.sum(np.abs(null_arr) >= abs(observed))
    return float((hits + 1) / (n + 1))


def partial_spearman(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    """Spearman of a vs b after residualising each on c (linear, on the given values)."""
    a_arr = np.asarray(list(a), dtype=float)
    b_arr = np.asarray(list(b), dtype=float)
    c_arr = np.asarray(list(c), dtype=float)
    a_res = residualise(a_arr, c_arr)
    b_res = residualise(b_arr, c_arr)
    return spearman_from_preds(a_res, b_res)
