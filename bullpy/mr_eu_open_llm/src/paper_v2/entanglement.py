"""Phase 5 entanglement null: target correlation, probe-cosine, partial RSA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from paper_v2.datasets.eu_emotion import load_eu_emotion_items
from paper_v2.metrics import (
    RIDGE_ALPHAS,
    cosine_rdm,
    partial_spearman,
    rsa_spearman,
    upper_triangle,
)
from paper_v2.probe import ACT_DIRS, align_X, list_activation_layers, load_trial_ids

ROOT = Path(__file__).resolve().parents[2]
SEED = 42


def _ridge_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = RidgeCV(alphas=RIDGE_ALPHAS)
    model.fit(Xs, y)
    w = np.asarray(model.coef_, dtype=float)
    nrm = np.linalg.norm(w)
    return w / nrm if nrm > 0 else w


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def correlated_targets(n: int, rho: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    cov = np.array([[1.0, rho], [rho, 1.0]])
    y = rng.multivariate_normal([0.0, 0.0], cov, size=n)
    return y[:, 0], y[:, 1]


def simulate_probe_cosine(
    X: np.ndarray,
    target_rho: float,
    *,
    n_sim: int = 200,
    seed: int = SEED,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    cosines = np.empty(n_sim, dtype=float)
    for i in range(n_sim):
        y1, y2 = correlated_targets(n, target_rho, rng)
        w1 = _ridge_weights(X, y1)
        w2 = _ridge_weights(X, y2)
        cosines[i] = cosine(w1, w2)
    return {
        "target_rho": float(target_rho),
        "n_sim": int(n_sim),
        "cosine_mean": float(np.nanmean(cosines)),
        "cosine_std": float(np.nanstd(cosines)),
        "cosine_ci_lo": float(np.nanquantile(cosines, 0.025)),
        "cosine_ci_hi": float(np.nanquantile(cosines, 0.975)),
    }


def difficulty_rdm(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    rdm = np.abs(v[:, None] - v[None, :])
    np.fill_diagonal(rdm, 0.0)
    return rdm


def run_model(model: str, *, n_sim: int, seed: int) -> Dict[str, Any]:
    items = load_eu_emotion_items()
    act_dir = ACT_DIRS[model]
    layers = list_activation_layers(act_dir)
    trial_ids = load_trial_ids(act_dir, layers[0][1])
    item_by_id = {it["item_id"]: it for it in items}
    y_ent = np.array([float(item_by_id[t]["metadata"]["human_entropy"]) for t in trial_ids])
    y_pt = np.array([float(item_by_id[t]["metadata"]["p_target"]) for t in trial_ids])
    y_conf = 1.0 - y_pt
    target_rho, target_p = spearmanr(y_conf, y_ent)
    human_path = ROOT / "data" / "human_confusion_rdm.npy"
    human_rdm = np.load(human_path) if human_path.is_file() else None

    layer_rows = []
    for layer_idx, path in layers:
        X = align_X(path, act_dir, trial_ids)
        w_ent = _ridge_weights(X, y_ent)
        w_conf = _ridge_weights(X, y_conf)
        obs_cos = cosine(w_ent, w_conf)
        sim = simulate_probe_cosine(X, float(target_rho), n_sim=n_sim, seed=seed + layer_idx)
        model_rdm = cosine_rdm(X)
        diff_rdm = difficulty_rdm(y_ent)
        row: Dict[str, Any] = {
            "layer": int(layer_idx),
            "observed_probe_cosine": obs_cos,
            "null_probe_cosine": sim,
            "cosine_minus_null_mean": float(obs_cos - sim["cosine_mean"]),
        }
        if human_rdm is not None:
            n = min(model_rdm.shape[0], human_rdm.shape[0])
            raw = rsa_spearman(model_rdm[:n, :n], human_rdm[:n, :n])
            partial = partial_spearman(
                upper_triangle(model_rdm[:n, :n]),
                upper_triangle(human_rdm[:n, :n]),
                upper_triangle(diff_rdm[:n, :n]),
            )
            row["rsa_raw"] = raw
            row["rsa_partial_entropy_rdm"] = partial
        layer_rows.append(row)

    return {
        "model": model,
        "n_trials": len(trial_ids),
        "target_spearman_1minus_p_vs_entropy": float(target_rho),
        "target_spearman_p": float(target_p),
        "layers": layer_rows,
    }


def run_all(
    *,
    models: Sequence[str] = ("qwen3vl", "gemma4", "molmo2"),
    n_sim: int = 200,
    seed: int = SEED,
    out_path: Optional[Path] = None,
) -> Dict[str, Any]:
    payload = {
        "seed": int(seed),
        "n_sim": int(n_sim),
        "models": {m: run_model(m, n_sim=n_sim, seed=seed) for m in models},
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["qwen3vl", "gemma4", "molmo2"])
    ap.add_argument("--n_sim", type=int, default=200)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=ROOT / "research/reproduction/phase5_entanglement.json")
    args = ap.parse_args()
    payload = run_all(models=args.models, n_sim=args.n_sim, seed=args.seed, out_path=args.out)
    print(f"wrote {args.out}")
    for name, block in payload["models"].items():
        print(
            f"{name}: corr(1-p, H)={block['target_spearman_1minus_p_vs_entropy']:.3f} "
            f"layers={[(r['layer'], round(r['observed_probe_cosine'], 3), round(r['null_probe_cosine']['cosine_mean'], 3)) for r in block['layers']]}"
        )


if __name__ == "__main__":
    main()
