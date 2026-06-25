"""
Representational similarity analysis (Study 2/3).

Model-only: save per-layer RDMs and compare baseline vs finetuned geometry.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

from config import LOCAL_DATA_DIR, LOCAL_RESULTS_DIR, SEED
from scripts.probing import layer_index_from_path, list_activation_layers, load_trial_ids


def compute_rdm(activations: np.ndarray) -> np.ndarray:
    """Cosine distance RDM from trial activation vectors."""
    x = activations.astype(np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(norms, 1e-12)
    sim = x @ x.T
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return dist.astype(np.float32)


def rdm_upper_triangle(rdm: np.ndarray) -> np.ndarray:
    n = rdm.shape[0]
    iu = np.triu_indices(n, k=1)
    return rdm[iu]


def rsa_spearman(model_rdm: np.ndarray, other_rdm: np.ndarray) -> float:
    n = min(model_rdm.shape[0], other_rdm.shape[0])
    a = rdm_upper_triangle(model_rdm[:n, :n])
    b = rdm_upper_triangle(other_rdm[:n, :n])
    rho, _ = spearmanr(a, b)
    return float(rho)


def run_rsa_layer(
    activations_path: Path,
    output_dir: Path,
    human_rdm_path: Optional[Path] = None,
) -> Dict:
    acts = np.load(activations_path)
    model_rdm = compute_rdm(acts)
    layer_idx = layer_index_from_path(activations_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    rdm_path = output_dir / f"rdm_layer{layer_idx}.npy"
    np.save(rdm_path, model_rdm)
    result: Dict = {
        "layer_index": layer_idx,
        "model_rdm_shape": list(model_rdm.shape),
        "rdm_path": str(rdm_path),
    }
    if human_rdm_path is not None and human_rdm_path.exists():
        human = np.load(human_rdm_path)
        result["human_rdm_source"] = str(human_rdm_path)
        result["spearman_rho_vs_human"] = rsa_spearman(model_rdm, human)
    else:
        result["human_rdm_source"] = "pending"
        result["spearman_rho_vs_human"] = None
    return result


def run_rsa_sweep(
    activations_dir: Path,
    output_dir: Path,
    human_rdm_path: Optional[Path] = None,
) -> Dict:
    layers = list_activation_layers(activations_dir)
    per_layer: List[Dict] = []
    for layer_idx, path in layers:
        per_layer.append(run_rsa_layer(path, output_dir, human_rdm_path=human_rdm_path))
    summary = {
        "activations_dir": str(activations_dir),
        "output_dir": str(output_dir),
        "layers": per_layer,
        "human_rdm_source": str(human_rdm_path) if human_rdm_path and human_rdm_path.exists() else "pending",
    }
    summary_path = output_dir / "rsa_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def compare_baseline_finetuned_rdms(
    baseline_dir: Path,
    finetuned_dir: Path,
    output: Path,
) -> Dict:
    """Spearman correlation between baseline and FT trial RDMs per layer."""
    base_layers = dict(list_activation_layers(baseline_dir))
    ft_layers = dict(list_activation_layers(finetuned_dir))
    common = sorted(set(base_layers) & set(ft_layers))
    if not common:
        raise ValueError("No common layers between baseline and finetuned activation dirs")

    comparisons: List[Dict] = []
    for layer_idx in common:
        rdm_b = compute_rdm(np.load(base_layers[layer_idx]))
        rdm_f = compute_rdm(np.load(ft_layers[layer_idx]))
        rho = rsa_spearman(rdm_b, rdm_f)
        comparisons.append(
            {
                "layer_index": layer_idx,
                "spearman_rho_baseline_vs_finetuned": rho,
                "interpretation": "high=r preserved geometry; low=r representational reorganization",
            }
        )
    result = {
        "baseline_dir": str(baseline_dir),
        "finetuned_dir": str(finetuned_dir),
        "layers": comparisons,
        "mean_rho": float(np.mean([c["spearman_rho_baseline_vs_finetuned"] for c in comparisons])),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_rsa(
    activations_path: Path,
    human_rdm_path: Optional[Path],
    output: Path,
) -> Dict:
    acts = np.load(activations_path)
    model_rdm = compute_rdm(acts)
    result: Dict = {"model_rdm_shape": list(model_rdm.shape)}
    if human_rdm_path is None or not human_rdm_path.exists():
        result["human_rdm_source"] = "pending"
        result["spearman_rho"] = None
    else:
        human = np.load(human_rdm_path)
        result["human_rdm_source"] = str(human_rdm_path)
        result["spearman_rho"] = rsa_spearman(model_rdm, human)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="RSA: per-layer RDMs and baseline vs FT comparison.")
    ap.add_argument("--activations_dir", type=Path, default=None, help="Sweep all layers in directory.")
    ap.add_argument("--activations", type=Path, default=None, help="Single layer .npy (legacy).")
    ap.add_argument("--output_dir", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=LOCAL_RESULTS_DIR / "rsa" / "rsa_summary.json")
    ap.add_argument("--human_rdm", type=Path, default=LOCAL_DATA_DIR / "human_rdm.npy")
    ap.add_argument("--compare_ft_dir", type=Path, default=None, help="Finetuned activations dir for RDM correlation.")
    ap.add_argument("--compare_output", type=Path, default=LOCAL_RESULTS_DIR / "rsa" / "baseline_vs_finetuned.json")
    args = ap.parse_args()

    if args.activations_dir is not None:
        out_dir = args.output_dir or args.output.parent
        summary = run_rsa_sweep(args.activations_dir, out_dir, human_rdm_path=args.human_rdm)
        if args.compare_ft_dir is not None:
            compare_baseline_finetuned_rdms(args.activations_dir, args.compare_ft_dir, args.compare_output)
        print(json.dumps(summary, indent=2))
        return

    if args.activations is None:
        ap.error("Provide --activations_dir or --activations")
    run_rsa(args.activations, args.human_rdm, args.output)


if __name__ == "__main__":
    main()
