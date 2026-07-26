"""
Sparse feature analysis on peak-layer activations (Study 2 B2).

Lightweight alternative to full SAE training: NMF basis on baseline activations,
then measure how finetuned activations project onto those bases and how bases
predict emotion labels. Surfaces whether finetune reorganizes sparse features
vs shifts readout weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.decomposition import NMF
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from config import LOCAL_RESULTS_DIR, SEED
from scripts.probing import (
    _effective_n_splits,
    entropy_tertiles,
    layer_index_from_path,
    list_activation_layers,
    load_labels_from_eval,
    load_trial_ids,
)


def _peak_layer_from_json(probe_dir: Path, model: str) -> int:
    peak_path = probe_dir / model / "peak_layer.json"
    if peak_path.is_file():
        return int(json.loads(peak_path.read_text()).get("peak_layer", 0))
    summary = probe_dir / model / "probes_summary.json"
    if summary.is_file():
        return int(json.loads(summary.read_text()).get("peak_layer", 0))
    raise FileNotFoundError(f"No peak layer under {probe_dir}")


def _load_layer_matrix(act_dir: Path, layer: int) -> tuple[np.ndarray, List[str]]:
    path = act_dir / f"layer{layer}_eu_emotions_seed{SEED}.npy"
    if not path.is_file():
        layers = list_activation_layers(act_dir)
        path = layers[0][1]
        layer = layer_index_from_path(path)
    trial_ids = load_trial_ids(act_dir, path)
    return np.load(path), trial_ids


def run_activation_sae_analysis(
    *,
    model: str,
    baseline_act_dir: Path,
    finetuned_act_dir: Path,
    eval_json: Path,
    probe_dir: Path,
    n_components: int = 16,
    output: Path,
    seed: int = SEED,
) -> Dict[str, Any]:
    peak_layer = _peak_layer_from_json(probe_dir, model)
    X_base, trial_ids = _load_layer_matrix(baseline_act_dir, peak_layer)
    X_ft, trial_ids_ft = _load_layer_matrix(finetuned_act_dir, peak_layer)
    if trial_ids_ft != trial_ids:
        raise ValueError("Baseline and finetuned trial_id order mismatch")

    y = load_labels_from_eval(eval_json, trial_ids)
    low_ids, high_ids = entropy_tertiles(eval_json)

    scaler = StandardScaler()
    Xb = scaler.fit_transform(X_base)
    Xf = scaler.transform(X_ft)

    n_comp = max(2, min(n_components, Xb.shape[1] // 4, Xb.shape[0] - 1))
    nmf = NMF(n_components=n_comp, init="nndsvda", random_state=seed, max_iter=400)
    W_base = nmf.fit_transform(np.maximum(Xb, 0))
    W_ft = nmf.transform(np.maximum(Xf, 0))

    def _basis_probe(W: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        Xp = W if mask is None else W[mask]
        yp = y if mask is None else y[mask]
        if len(np.unique(yp)) < 2 or Xp.shape[0] < 8:
            return float("nan")
        n_splits = _effective_n_splits(len(Xp), yp, n_splits=5)
        if n_splits is None:
            return float("nan")
        clf = LogisticRegression(multi_class="multinomial", max_iter=2000, random_state=seed)
        scores = cross_val_score(clf, Xp, yp, cv=n_splits, scoring="accuracy")
        return float(np.mean(scores))

    low_mask = np.array([tid in low_ids for tid in trial_ids]) if low_ids else None
    high_mask = np.array([tid in high_ids for tid in trial_ids]) if high_ids else None

    delta_norm = np.linalg.norm(W_ft - W_base, axis=1)
    results: Dict[str, Any] = {
        "model": model,
        "peak_layer": peak_layer,
        "n_components": n_comp,
        "reconstruction_error_baseline": float(nmf.reconstruction_err_),
        "basis_probe_accuracy_baseline": _basis_probe(W_base),
        "basis_probe_accuracy_finetuned": _basis_probe(W_ft),
        "basis_probe_low_ambiguity_baseline": _basis_probe(W_base, low_mask) if low_mask is not None else None,
        "basis_probe_high_ambiguity_baseline": _basis_probe(W_base, high_mask) if high_mask is not None else None,
        "basis_probe_low_ambiguity_finetuned": _basis_probe(W_ft, low_mask) if low_mask is not None else None,
        "basis_probe_high_ambiguity_finetuned": _basis_probe(W_ft, high_mask) if high_mask is not None else None,
        "mean_feature_delta_l2": float(np.mean(delta_norm)),
        "median_feature_delta_l2": float(np.median(delta_norm)),
        "interpretation": (
            "Large feature deltas + stable basis probe => readout/routing shift; "
            "large deltas + collapsed basis probe => representational reorganization."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="NMF sparse-feature analysis at peak probe layer.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--baseline_act_dir", type=Path, required=True)
    ap.add_argument("--finetuned_act_dir", type=Path, required=True)
    ap.add_argument("--eval_json", type=Path, required=True)
    ap.add_argument("--probe_dir", type=Path, required=True)
    ap.add_argument("--n_components", type=int, default=16)
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = ap.parse_args()
    out = args.output or LOCAL_RESULTS_DIR / "sae" / f"{args.model}_peak_nmf.json"
    run_activation_sae_analysis(
        model=args.model,
        baseline_act_dir=args.baseline_act_dir,
        finetuned_act_dir=args.finetuned_act_dir,
        eval_json=args.eval_json,
        probe_dir=args.probe_dir,
        n_components=args.n_components,
        output=out,
    )


if __name__ == "__main__":
    main()
