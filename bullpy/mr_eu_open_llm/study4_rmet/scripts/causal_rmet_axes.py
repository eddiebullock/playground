"""
Phase 4 — Causal axes (diagnosticity vs entropy) from existing activations.

Offline (CPU): build unit axes as mean(high_class) − mean(low_class) activations;
compute cosine alignment between diagnosticity and entropy axes; project each
item onto both axes; report reuse signature (do high-entropy items load on the
diagnosticity axis?).

GPU steer/patch: optional --run_steer requires parent hooks on HPC; this script
always writes axis artifacts + planned intervention protocol so C1 is reported
even when GPU is unavailable (axis geometry + class projections as interim C1).

Controls: random directions; shuffled class labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

STUDY4_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
DEFAULT_STRUCT = STUDY4_ROOT / "results" / "card_structure"
DEFAULT_OUT = STUDY4_ROOT / "results" / "mech"


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def mean_diff_axis(X: np.ndarray, high_idx: List[int], low_idx: List[int]) -> np.ndarray:
    return _unit(X[high_idx].mean(axis=0) - X[low_idx].mean(axis=0))


def project(X: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return X @ axis


def load_X(model: str, layer: int) -> np.ndarray:
    path = STUDY4_ROOT / "results" / "activations" / model / "full" / f"layer{layer}_rmet_seed42.npy"
    if not path.exists():
        # find any matching layer file
        cands = list((STUDY4_ROOT / "results" / "activations" / model / "full").glob(f"layer{layer}_*.npy"))
        if not cands:
            raise FileNotFoundError(path)
        path = cands[0]
    return np.load(path)


def item_indices(items: List[int]) -> List[int]:
    """Items are 1..36; activation rows assumed ordered by item."""
    return [i - 1 for i in items]


def analyze_model_layer(
    model: str,
    layer: int,
    classes: Dict[str, Any],
    seed: int = 42,
) -> Dict[str, Any]:
    X = load_X(model, layer)
    if X.shape[0] != 36:
        raise ValueError(f"{model} layer{layer} shape {X.shape}")

    high_d = item_indices(classes["trait_diagnosticity"]["high"])
    low_d = item_indices(classes["trait_diagnosticity"]["low"])
    high_e = item_indices(classes["human_entropy"]["high"])
    low_e = item_indices(classes["human_entropy"]["low"])

    axis_d = mean_diff_axis(X, high_d, low_d)
    axis_e = mean_diff_axis(X, high_e, low_e)
    align = float(np.dot(axis_d, axis_e))

    rng = np.random.default_rng(seed)
    # Random-direction control: mean |proj| contrast for true vs random axes
    proj_d = project(X, axis_d)
    proj_e = project(X, axis_e)

    # Reuse signature: mean projection of high-entropy items onto diagnosticity axis
    # vs low-entropy items (if shared/generic, high-entropy also separates on diag axis)
    reuse_diag_on_entropy = float(proj_d[high_e].mean() - proj_d[low_e].mean())
    reuse_entropy_on_diag = float(proj_e[high_d].mean() - proj_e[low_d].mean())
    own_diag = float(proj_d[high_d].mean() - proj_d[low_d].mean())
    own_ent = float(proj_e[high_e].mean() - proj_e[low_e].mean())

    # Shuffled class control
    shuffle_gaps = []
    for _ in range(200):
        perm = rng.permutation(36)
        h = perm[: len(high_d)].tolist()
        l = perm[len(high_d) : len(high_d) + len(low_d)].tolist()
        ax = mean_diff_axis(X, h, l)
        pr = project(X, ax)
        shuffle_gaps.append(float(pr[h].mean() - pr[l].mean()))
    shuffle_gaps = np.asarray(shuffle_gaps)

    # Random axes: gap for true high/low diagnosticity
    rand_gaps = []
    for _ in range(200):
        ax = _unit(rng.normal(size=X.shape[1]))
        pr = project(X, ax)
        rand_gaps.append(float(pr[high_d].mean() - pr[low_d].mean()))
    rand_gaps = np.asarray(rand_gaps)

    return {
        "model": model,
        "layer": layer,
        "axis_alignment_diag_vs_entropy": align,
        "own_effect_diagnosticity": own_diag,
        "own_effect_entropy": own_ent,
        "reuse_diag_axis_on_entropy_classes": reuse_diag_on_entropy,
        "reuse_entropy_axis_on_diag_classes": reuse_entropy_on_diag,
        "shuffled_diag_gap_mean": float(shuffle_gaps.mean()),
        "shuffled_diag_gap_p": float((np.sum(shuffle_gaps >= own_diag) + 1) / (len(shuffle_gaps) + 1)),
        "random_axis_diag_gap_mean": float(rand_gaps.mean()),
        "random_axis_diag_gap_p": float((np.sum(np.abs(rand_gaps) >= abs(own_diag)) + 1) / (len(rand_gaps) + 1)),
        "C1_interim_interpretation": (
            "If |reuse_diag_axis_on_entropy_classes| ≈ |own_effect_diagnosticity|, "
            "axes behave generically across classes; if reuse << own and axis_alignment "
            "is low, diagnosticity and entropy are dissociable in activation space "
            "(geometry-only interim C1; steer/patch still required for causality)."
        ),
        "steer_patch_status": "not_run_cpu_geometry_only",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--structure_dir", type=Path, default=DEFAULT_STRUCT)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--models", default="qwen3vl,gemma4,molmo2")
    ap.add_argument(
        "--layers",
        default="",
        help="Comma layer indices; default = all available per model",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--run_steer",
        action="store_true",
        help="Reserved: run GPU steer/patch (HPC). Not executed unless parent hooks wired.",
    )
    args = ap.parse_args(argv)

    classes = json.loads(
        (args.structure_dir / "item_classes_preregistered.json").read_text(encoding="utf-8")
    )
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    summary: Dict[str, Any] = {"C1_mode": "activation_geometry_reuse", "models": {}}
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        act_dir = STUDY4_ROOT / "results" / "activations" / model / "full"
        if not act_dir.is_dir():
            summary["models"][model] = {"status": "missing_activations"}
            continue
        layer_files = sorted(act_dir.glob("layer*_rmet_seed*.npy"))
        layers = []
        for p in layer_files:
            # layerN_
            try:
                layers.append(int(p.name.split("_")[0].replace("layer", "")))
            except ValueError:
                continue
        if args.layers.strip():
            want = {int(x) for x in args.layers.split(",") if x.strip()}
            layers = [L for L in layers if L in want]
        model_rows = []
        for L in layers:
            print(f"causal axes: {model} L{L}", flush=True)
            r = analyze_model_layer(model, L, classes, seed=args.seed)
            # persist axes
            X = load_X(model, L)
            high_d = item_indices(classes["trait_diagnosticity"]["high"])
            low_d = item_indices(classes["trait_diagnosticity"]["low"])
            high_e = item_indices(classes["human_entropy"]["high"])
            low_e = item_indices(classes["human_entropy"]["low"])
            np.save(
                args.outdir / f"axis_diagnosticity_{model}_layer{L}.npy",
                mean_diff_axis(X, high_d, low_d),
            )
            np.save(
                args.outdir / f"axis_entropy_{model}_layer{L}.npy",
                mean_diff_axis(X, high_e, low_e),
            )
            model_rows.append(r)
            rows.append(r)
        summary["models"][model] = {"layers": model_rows}
        if model_rows:
            pd.DataFrame(model_rows).to_csv(
                args.outdir / f"{model}_causal_axis_geometry.csv", index=False
            )

    if rows:
        pd.DataFrame(rows).to_csv(args.outdir / "causal_axis_geometry_all.csv", index=False)
    summary["run_steer_requested"] = bool(args.run_steer)
    if args.run_steer:
        # Delegate to study4 steer wrapper (GPU if available; else protocol-only).
        from steer_rmet_axes import main as steer_main

        peak_layer = 4
        for model, payload in summary["models"].items():
            layers = payload.get("layers") or []
            if not layers:
                continue
            # Prefer layer with strongest |own_effect_diagnosticity| among available
            best = max(layers, key=lambda r: abs(float(r.get("own_effect_diagnosticity", 0))))
            peak_layer = int(best["layer"])
            print(f"C1 steer delegate: {model} L{peak_layer}", flush=True)
            steer_main(
                [
                    "--model",
                    model,
                    "--layer",
                    str(peak_layer),
                    "--outdir",
                    str(args.outdir),
                    "--mech_dir",
                    str(args.outdir),
                    "--smoke",
                ]
            )
        summary["steer_note"] = (
            "Delegated to steer_rmet_axes.py (last_token + all_tokens; "
            "±alpha diagnosticity/entropy/random; JS to low-EQ/ASC). "
            "Without CUDA writes protocol only."
        )
    else:
        summary["steer_note"] = (
            "GPU steer deferred. Run: python study4_rmet/scripts/steer_rmet_axes.py "
            "--model qwen3vl --layer 4 --smoke"
        )
    (args.outdir / "causal_axes_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({"n_rows": len(rows), "models": list(summary["models"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
