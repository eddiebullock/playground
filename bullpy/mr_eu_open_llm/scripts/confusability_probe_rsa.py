#!/usr/bin/env python3
"""Confusability RSA + LOO Ridge probes for study3 EU (243 trials).

Compares layer-wise activation RDMs to the human confusion RDM and probes whether
confusability scalars (human entropy, 1-p_target) are linearly decodable.

Requires:
  - data/human_confusion_rdm.npy + human_confusion_meta.json (build_human_confusion.py)
  - results/activations/<condition>/<model>/layer*_eu_emotions_seed*.npy

Usage:

  python -m scripts.build_human_confusion

  python -m scripts.confusability_probe_rsa \\
    --model qwen3vl \\
    --activations_dir results/activations/baseline_qwen3vl_6afc/qwen3vl \\
    --out results/mech/qwen3vl_confusability_probe_rsa.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

from config import LOCAL_DATA_DIR, LOCAL_RESULTS_DIR, SEED
from scripts.probing import layer_index_from_path, list_activation_layers
from scripts.rsa import compute_rdm, rsa_spearman


def loo_ridge_rho(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    mask = np.isfinite(y)
    X, y = X[mask], y[mask]
    n = len(y)
    if n < 8:
        return {"rho": float("nan"), "n": float(n)}
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
    return {"rho": float(rho), "n": float(n)}


def align_activations_to_human_order(
    act_path: Path,
    act_dir: Path,
    human_trial_ids: List[str],
) -> np.ndarray:
    X = np.load(act_path)
    layer_idx = layer_index_from_path(act_path)
    sidecar = act_dir / f"layer{layer_idx}_trial_ids.json"
    if not sidecar.exists():
        if X.shape[0] == len(human_trial_ids):
            return X
        raise ValueError(f"Activation rows {X.shape[0]} != human trials {len(human_trial_ids)}")
    act_ids = json.loads(sidecar.read_text(encoding="utf-8"))
    index = {tid: i for i, tid in enumerate(act_ids)}
    rows = []
    for tid in human_trial_ids:
        if tid not in index:
            raise KeyError(f"human trial_id missing from activations: {tid}")
        rows.append(X[index[tid]])
    return np.stack(rows, axis=0)


def perm_rsa_pvalue(model_rdm: np.ndarray, human_rdm: np.ndarray, *, n_perm: int, seed: int) -> Dict[str, float]:
    n = min(model_rdm.shape[0], human_rdm.shape[0])
    a = model_rdm[:n, :n]
    b = human_rdm[:n, :n]
    rho_obs = rsa_spearman(a, b)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        perm = rng.permutation(n)
        null[i] = rsa_spearman(a, b[np.ix_(perm, perm)])
    p = float((np.sum(np.abs(null) >= abs(rho_obs)) + 1) / (n_perm + 1))
    return {"rho": float(rho_obs), "p_perm": p, "n_perm": float(n_perm)}


def run_model(
    model: str,
    activations_dir: Path,
    human_rdm: np.ndarray,
    meta: Dict[str, Any],
    *,
    n_perm: int = 2000,
    seed: int = SEED,
) -> Dict[str, Any]:
    human_trial_ids = list(meta["trial_ids"])
    per_item = {row["trial_id"]: row for row in meta["per_item"]}
    y_entropy = np.array([per_item[tid]["human_entropy"] for tid in human_trial_ids], float)
    y_confusability = np.array(
        [per_item[tid]["confusability_1_minus_p_target"] for tid in human_trial_ids], float
    )

    layers = list_activation_layers(activations_dir)
    if not layers:
        return {"status": "missing_activations", "model": model, "activations_dir": str(activations_dir)}

    layer_rows: List[Dict[str, Any]] = []
    for layer_idx, path in layers:
        X = align_activations_to_human_order(path, activations_dir, human_trial_ids)
        model_rdm = compute_rdm(X)
        rsa = perm_rsa_pvalue(model_rdm, human_rdm, n_perm=n_perm, seed=seed + layer_idx)
        probe_ent = loo_ridge_rho(X, y_entropy)
        probe_conf = loo_ridge_rho(X, y_confusability)
        layer_rows.append(
            {
                "layer": layer_idx,
                "rsa_human_confusion_rho": rsa["rho"],
                "rsa_human_confusion_p_perm": rsa["p_perm"],
                "probe_human_entropy_rho": probe_ent["rho"],
                "probe_confusability_rho": probe_conf["rho"],
                "n_trials": int(X.shape[0]),
            }
        )

    df = pd.DataFrame(layer_rows)
    peak_rsa = df.iloc[df["rsa_human_confusion_rho"].abs().idxmax()].to_dict()
    peak_probe_ent = df.iloc[df["probe_human_entropy_rho"].abs().idxmax()].to_dict()
    peak_probe_conf = df.iloc[df["probe_confusability_rho"].abs().idxmax()].to_dict()

    return {
        "status": "ok",
        "model": model,
        "activations_dir": str(activations_dir),
        "n_trials": len(human_trial_ids),
        "n_layers": len(layer_rows),
        "peak_rsa_layer": peak_rsa,
        "peak_probe_entropy_layer": peak_probe_ent,
        "peak_probe_confusability_layer": peak_probe_conf,
        "layers": layer_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--activations_dir", type=Path, required=True)
    ap.add_argument("--human_rdm", type=Path, default=LOCAL_DATA_DIR / "human_confusion_rdm.npy")
    ap.add_argument("--human_meta", type=Path, default=LOCAL_DATA_DIR / "human_confusion_meta.json")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    if not args.human_rdm.exists() or not args.human_meta.exists():
        raise FileNotFoundError(
            "Run python -m scripts.build_human_confusion first "
            f"(missing {args.human_rdm} or {args.human_meta})"
        )

    human_rdm = np.load(args.human_rdm)
    meta = json.loads(args.human_meta.read_text(encoding="utf-8"))
    result = run_model(
        args.model,
        args.activations_dir,
        human_rdm,
        meta,
        n_perm=args.n_perm,
        seed=args.seed,
    )

    out = args.out or (
        LOCAL_RESULTS_DIR / "mech" / f"{args.model}_confusability_probe_rsa.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_path = out.with_suffix(".csv")
    if result.get("layers"):
        pd.DataFrame(result["layers"]).to_csv(csv_path, index=False)

    print(json.dumps({k: v for k, v in result.items() if k != "layers"}, indent=2))
    print(f"wrote {out}")
    if result.get("layers"):
        print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
