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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

from config import LOCAL_DATA_DIR, LOCAL_RESULTS_DIR, SEED
from scripts.causal_eu_confusion_axes import (
    default_pair_specs,
    label_pair_tuple,
    pair_trial_indices,
    select_difficulty_matched_pairs,
    trial_indices_for_label_pairs,
)
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


def rsa_difficulty_matched(
    X: np.ndarray,
    human_rdm: np.ndarray,
    rows: Sequence[Dict[str, Any]],
    pair_specs: Sequence[Tuple[str, str]],
    *,
    n_perm: int,
    seed: int,
) -> Dict[str, Any]:
    """RSA on confused-pair items plus difficulty-matched non-confused pair items."""
    confused_keys = [label_pair_tuple(la, lb) for la, lb in pair_specs]
    per_pair: List[Dict[str, Any]] = []
    rhos: List[float] = []
    for la, lb in pair_specs:
        match_info = select_difficulty_matched_pairs(rows, la, lb, confused_keys)
        confused_idx = pair_trial_indices(rows, la, lb)
        control_keys = [
            label_pair_tuple(m["label_a"], m["label_b"]) for m in match_info["matched_pairs"]
        ]
        control_idx = trial_indices_for_label_pairs(rows, control_keys)
        subset_idx = sorted(set(confused_idx + control_idx))
        if len(subset_idx) < 8:
            per_pair.append(
                {
                    "pair": f"{la}/{lb}",
                    "n_subset": len(subset_idx),
                    "status": "insufficient_subset",
                    **match_info,
                }
            )
            continue
        model_sub = compute_rdm(X[subset_idx])
        human_sub = human_rdm[np.ix_(subset_idx, subset_idx)]
        rsa = perm_rsa_pvalue(model_sub, human_sub, n_perm=n_perm, seed=seed)
        rhos.append(rsa["rho"])
        per_pair.append(
            {
                "pair": f"{la}/{lb}",
                "n_subset": len(subset_idx),
                "n_confused": len(confused_idx),
                "n_control_items": len(control_idx),
                "rsa_rho": rsa["rho"],
                "rsa_p_perm": rsa["p_perm"],
                **match_info,
            }
        )
    valid = [r for r in rhos if r == r]
    return {
        "note": "Difficulty-controlled RSA: confused-pair items vs matched low-confusion pairs.",
        "rho_mean": float(np.mean(valid)) if valid else float("nan"),
        "n_pairs": len(per_pair),
        "per_pair": per_pair,
    }


def run_model(
    model: str,
    activations_dir: Path,
    human_rdm: np.ndarray,
    meta: Dict[str, Any],
    *,
    n_perm: int = 2000,
    seed: int = SEED,
    pairs_json: Optional[Path] = None,
    top_pairs: int = 3,
) -> Dict[str, Any]:
    human_trial_ids = list(meta["trial_ids"])
    per_item = {row["trial_id"]: row for row in meta["per_item"]}
    y_entropy = np.array([per_item[tid]["human_entropy"] for tid in human_trial_ids], float)
    y_confusability = np.array(
        [per_item[tid]["confusability_1_minus_p_target"] for tid in human_trial_ids], float
    )
    meta_rows = list(meta["per_item"])
    pair_specs = default_pair_specs(
        pairs_json or LOCAL_DATA_DIR / "human_confused_pairs.json", top_k=top_pairs
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
        rsa_matched = rsa_difficulty_matched(
            X, human_rdm, meta_rows, pair_specs, n_perm=n_perm, seed=seed + layer_idx + 1000
        )
        layer_rows.append(
            {
                "layer": layer_idx,
                "rsa_human_confusion_rho": rsa["rho"],
                "rsa_human_confusion_p_perm": rsa["p_perm"],
                "rsa_human_confusion_note": "raw full-set RSA; not difficulty-controlled",
                "rsa_difficulty_matched": rsa_matched,
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
    ap.add_argument("--pairs_json", type=Path, default=LOCAL_DATA_DIR / "human_confused_pairs.json")
    ap.add_argument("--top_pairs", type=int, default=3)
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
        pairs_json=args.pairs_json,
        top_pairs=args.top_pairs,
    )

    out = args.out or (
        LOCAL_RESULTS_DIR / "mech" / f"{args.model}_confusability_probe_rsa.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_path = out.with_suffix(".csv")
    if result.get("layers"):
        flat = []
        for row in result["layers"]:
            flat_row = {k: v for k, v in row.items() if k != "rsa_difficulty_matched"}
            matched = row.get("rsa_difficulty_matched") or {}
            flat_row["rsa_difficulty_matched_rho_mean"] = matched.get("rho_mean")
            flat.append(flat_row)
        pd.DataFrame(flat).to_csv(csv_path, index=False)

    print(json.dumps({k: v for k, v in result.items() if k != "layers"}, indent=2))
    print(f"wrote {out}")
    if result.get("layers"):
        print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
