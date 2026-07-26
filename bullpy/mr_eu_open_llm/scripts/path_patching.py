"""
Multi-layer path patching (Study 2 B3).

Patches baseline activations at each probed layer separately to test whether
causal leverage concentrates at the peak probe layer (late readout) vs early layers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import LOCAL_RESULTS_DIR, MODELS, SEED
from scripts.activation_patching import run_patching_experiment, select_ft_incorrect_trials
from scripts.evaluate import resolve_dataset_root
from scripts.probing import list_activation_layers


def run_path_patching(
    model_key: str,
    *,
    baseline_eval: Path,
    finetuned_eval: Path,
    baseline_activations_dir: Path,
    peak_layer_json: Path,
    checkpoint: Path,
    modality: str,
    layers: Optional[List[int]] = None,
    max_trials: int = 30,
    output: Path,
    seed: int = SEED,
) -> Dict[str, Any]:
    if layers is None:
        layers = [idx for idx, _ in list_activation_layers(baseline_activations_dir)]

    trials = select_ft_incorrect_trials(finetuned_eval, max_trials=max_trials)
    dataset_root = resolve_dataset_root("eu_emotions")

    per_layer: List[Dict[str, Any]] = []
    peak_layer = None
    if peak_layer_json.is_file():
        peak_layer = int(json.loads(peak_layer_json.read_text()).get("peak_layer"))

    for layer in layers:
        layer_out = output.parent / f"path_patch_{model_key}_L{layer}.json"
        res = run_patching_experiment(
            model_key,
            trials,
            peak_layer=int(layer),
            baseline_activations_dir=baseline_activations_dir,
            dataset_root=dataset_root,
            output=layer_out,
            seed=seed,
            checkpoint=checkpoint,
            condition_modality=modality,
            selection_mode="ft_incorrect_same_trial",
        )
        per_layer.append(
            {
                "layer_index": layer,
                "n_trials": res.get("n_trials_requested"),
                "fix_rate": res.get("fix_rate"),
                "change_rate": res.get("prediction_change_rate"),
                "accuracy_before": res.get("accuracy_before"),
                "accuracy_after": res.get("accuracy_after"),
                "is_peak_layer": layer == peak_layer,
            }
        )

    peak_fix = next((r["fix_rate"] for r in per_layer if r["is_peak_layer"]), None)
    early_fix = [r["fix_rate"] for r in per_layer if not r["is_peak_layer"]]
    summary = {
        "model": model_key,
        "modality": modality,
        "peak_layer": peak_layer,
        "max_trials": max_trials,
        "layers": per_layer,
        "peak_fix_rate": peak_fix,
        "mean_non_peak_fix_rate": float(sum(early_fix) / len(early_fix)) if early_fix else None,
        "interpretation": (
            "Peak fix_rate >> non-peak supports late-layer readout hijack; "
            "uniform fix rates suggest distributed or shallow effect."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Path patching across probed layers.")
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--baseline_eval", type=Path, required=True)
    ap.add_argument("--finetuned_eval", type=Path, required=True)
    ap.add_argument("--baseline_activations_dir", type=Path, required=True)
    ap.add_argument("--peak_layer_json", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--modality", default="multimodal")
    ap.add_argument("--max_trials", type=int, default=30)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    out = args.output or LOCAL_RESULTS_DIR / "patching" / f"path_patching_{args.model}_4afc.json"
    run_path_patching(
        args.model,
        baseline_eval=args.baseline_eval,
        finetuned_eval=args.finetuned_eval,
        baseline_activations_dir=args.baseline_activations_dir,
        peak_layer_json=args.peak_layer_json,
        checkpoint=args.checkpoint,
        modality=args.modality,
        max_trials=args.max_trials,
        output=out,
    )


if __name__ == "__main__":
    main()
