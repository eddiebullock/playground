#!/usr/bin/env python3
"""Activation patching (necessity ablation) along EU confusion axes (study3 v2).

Primary v2 causal test: mean-projection ablation replaces each item's projection onto
a unit axis with the dataset mean projection (orthogonal component unchanged).
This is activation patching / ablation — NOT additive steering (see steer_eu_confusion_axes.py).

Readout: delta accuracy and delta confusion rate (chose human top foil) vs unablated baseline.
Double-dissociation: pair-axis ablation should degrade own-pair trials more than other
confused pairs; entropy-axis ablation should degrade broadly. Random axis is the control.

Requires causal_eu_confusion_axes.py outputs in results/mech/.

Usage (HPC GPU):
  python -m scripts.causal_eu_confusion_axes --model qwen3vl --layer 4
  python -m scripts.ablate_eu_confusion_axes --model qwen3vl --layer 4 --smoke
  python -m scripts.ablate_eu_confusion_axes --model qwen3vl --layer 4 \\
      --max_items 36 --top_pairs 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from config import LOCAL_DATA_DIR, LOCAL_RESULTS_DIR, SEED
from scripts.causal_eu_confusion_axes import (
    default_pair_specs,
    load_activations,
    pair_key,
)
from scripts.steer_eu_confusion_axes import load_trial_table, trial_pair_membership

AblationMethod = Literal["mean_ablation", "zero_projection"]
DEFAULT_MECH = LOCAL_RESULTS_DIR / "mech"


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


class AxisMeanAblator:
    """Mean-projection ablation along a 1-D axis (activation patching, not steering)."""

    def __init__(
        self,
        direction: np.ndarray,
        mean_projection: float,
        *,
        method: AblationMethod = "mean_ablation",
        patch_mode: str = "last_token",
    ) -> None:
        self.direction = _unit(np.asarray(direction, dtype=np.float32).reshape(-1))
        self.mean_projection = float(mean_projection)
        self.method = method
        self.patch_mode = patch_mode
        self.hook_calls = 0
        self._handle = None
        self._dir_t = None

    def attach(self, module: Any, device: Any, dtype: Any) -> None:
        import torch

        ablator = self
        ablator._dir_t = torch.tensor(ablator.direction, device=device, dtype=dtype)

        def hook(_mod, _inp, out):
            ablator.hook_calls += 1
            h = out[0] if isinstance(out, tuple) else out
            if not hasattr(h, "shape") or h.ndim < 2:
                return out
            d = ablator._dir_t
            if d is None or d.shape[-1] != h.shape[-1]:
                return out
            h = h.clone()
            if ablator.patch_mode == "last_token":
                token = h[:, -1, :]
            else:
                token = h
            proj = (token * d).sum(dim=-1, keepdim=True)
            if ablator.method == "mean_ablation":
                token = token - (proj - ablator.mean_projection) * d
            else:
                token = token - proj * d
            if ablator.patch_mode == "last_token":
                h[:, -1, :] = token
            else:
                h = token
            if isinstance(out, tuple):
                return (h,) + out[1:]
            return h

        self._handle = module.register_forward_hook(hook)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def load_axis(mech_dir: Path, kind: str, model: str, layer: int) -> np.ndarray:
    path = mech_dir / f"axis_{kind}_{model}_layer{layer}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing axis {path}. Run causal_eu_confusion_axes.py first.")
    return _unit(np.load(path))


def mean_projection_for_axis(
    act_dir: Path,
    layer: int,
    trial_ids: Sequence[str],
    axis: np.ndarray,
) -> float:
    X = load_activations(act_dir, layer, trial_ids)
    return float((X @ _unit(axis)).mean())


def run_ablate(
    *,
    model_key: str,
    layer: int,
    mech_dir: Path,
    act_dir: Path,
    manifest_path: Path,
    data_root: Path,
    human_path: Path,
    outdir: Path,
    axis_names: List[str],
    pair_specs: Sequence[Tuple[str, str]],
    max_items: Optional[int],
    max_frames: int,
    seed: int,
    smoke: bool,
    sample_temperature: float,
    ablation_method: AblationMethod,
) -> Dict[str, Any]:
    import torch
    from transformers import AutoProcessor

    from scripts.activation_forward import find_layer_module, find_layer_module_name, prepare_trial_media
    from scripts.emotion_parse import parse_emotion
    from scripts.evaluate import load_hf_model_for_key, resolve_model_path, resolve_trial_media
    from scripts.model_inference import generate_model_response, seed_generation
    from scripts.prompts import build_4afc_prompt

    trials, _human_lookup, meta_rows = load_trial_table(
        manifest_path,
        human_path,
        data_root,
        max_items=max_items,
        smoke=smoke,
        pair_specs=pair_specs,
    )

    device_s = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device_s == "cuda" else torch.float32
    device = torch.device(device_s)

    meta = json.loads((LOCAL_DATA_DIR / "human_confusion_meta.json").read_text(encoding="utf-8"))
    all_trial_ids = list(meta["trial_ids"])

    if device_s != "cuda":
        proto = {
            "status": "planned_only_no_cuda",
            "model": model_key,
            "layer": layer,
            "n_trials": len(trials),
            "intervention": "activation_patching_mean_ablation",
        }
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / f"ablate_protocol_{model_key}_layer{layer}.json").write_text(
            json.dumps(proto, indent=2) + "\n", encoding="utf-8"
        )
        return proto

    axes: Dict[str, np.ndarray] = {}
    mean_projs: Dict[str, float] = {}
    for name in axis_names:
        vec = load_axis(mech_dir, name, model_key, layer)
        axes[name] = vec
        mean_projs[name] = mean_projection_for_axis(act_dir, layer, all_trial_ids, vec)

    model_path = resolve_model_path(model_key)
    if model_key != "gemma4":
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    else:
        try:
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            from transformers import Gemma4Processor  # type: ignore

            processor = Gemma4Processor.from_pretrained(model_path)

    model = load_hf_model_for_key(model_key, model_path, device_s, dtype)
    model.eval()

    module = find_layer_module(model, layer)
    if module is None:
        raise RuntimeError(f"Could not resolve layer module for {model_key} L{layer}")
    module_name = find_layer_module_name(model, layer)
    tokenizer = getattr(processor, "tokenizer", None)
    pipe_cache: Dict[str, Any] = {}

    meta_by_tid = {r["trial_id"]: r for r in meta_rows if "trial_id" in r}
    pair_tags: Dict[str, str] = {}
    for la, lb in pair_specs:
        pk = pair_key(la, lb)
        for tr in trials:
            row = meta_by_tid.get(tr["trial_id"], tr)
            if trial_pair_membership(row, la, lb):
                pair_tags[tr["trial_id"]] = pk

    rows: List[Dict[str, Any]] = []

    def generate_once(
        trial: Dict[str, Any],
        *,
        ablator: Optional[AxisMeanAblator],
        tag: str,
    ) -> Optional[str]:
        options = list(trial["options"])
        trial_copy = dict(trial)
        trial_copy["candidate_labels"] = options
        video_path, audio_path, _ = resolve_trial_media(
            trial_copy,
            dataset_key="eu_emotions",
            dataset_root=data_root,
            condition="video_only",
            seed=seed,
        )
        path = Path(trial["stimulus_path"])
        if not path.is_file():
            path = (data_root / path).resolve()
        if video_path is not None and Path(video_path).is_file():
            path = Path(video_path)
        images, images_for_proc, _ = prepare_trial_media(
            path, model_key=model_key, fps=1.0, max_frames=max_frames
        )
        prompt = build_4afc_prompt(options, condition="video_only")
        seed_generation(seed, model_key, f"eu_ablate_{trial['trial_id']}_{tag}", 0)
        if ablator is not None:
            ablator.hook_calls = 0
            ablator.attach(module, device, dtype)
        try:
            text = generate_model_response(
                model_key=model_key,
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                model_path=model_path,
                prompt=prompt,
                images=images,
                images_for_processor=images_for_proc,
                device=device_s,
                dtype=dtype,
                temperature=float(sample_temperature),
                max_new_tokens=128,
                pipe_cache=pipe_cache,
                condition="video_only",
                prefer_loaded_model=True,
            )
        finally:
            if ablator is not None:
                ablator.remove()
        pred, _ = parse_emotion(text, options)
        return pred

    conditions: List[Tuple[str, str, Optional[str]]] = [("baseline", "baseline", None)]
    for axis_name in axis_names:
        conditions.append((f"ablate_{axis_name}", axis_name, axis_name))

    for trial in trials:
        tid = trial["trial_id"]
        options = list(trial["options"])
        correct = trial.get("correct_label") or trial.get("label")
        meta_row = meta_by_tid.get(tid, {})
        top_foil = str(meta_row.get("top_foil_label", ""))
        membership = pair_tags.get(tid, "other")
        print(f"ablate {tid}", flush=True)

        for cond_name, axis_label, axis_key in conditions:
            ablator = None
            if axis_key is not None:
                ablator = AxisMeanAblator(
                    axes[axis_key],
                    mean_projs[axis_key],
                    method=ablation_method,
                )
            pred = generate_once(trial, ablator=ablator, tag=cond_name)
            confused = bool(pred is not None and top_foil and pred == top_foil)
            rows.append(
                {
                    "trial_id": tid,
                    "condition": cond_name,
                    "axis": axis_label,
                    "correct": bool(pred is not None and correct and pred == correct),
                    "confused_with_top_foil": confused,
                    "prediction": pred,
                    "correct_label": correct,
                    "top_foil_label": top_foil,
                    "pair_membership": membership,
                    "hook_module": module_name,
                    "ablation_method": ablation_method if axis_key else "none",
                }
            )

    import pandas as pd

    df = pd.DataFrame(rows)
    csv_path = outdir / f"ablate_trials_{model_key}_layer{layer}.csv"
    df.to_csv(csv_path, index=False)

    base = df[df["condition"] == "baseline"][
        ["trial_id", "correct", "confused_with_top_foil"]
    ].rename(
        columns={
            "correct": "correct_base",
            "confused_with_top_foil": "confused_base",
        }
    )
    merged = df[df["condition"] != "baseline"].merge(base, on="trial_id", how="left")
    merged["delta_correct"] = merged["correct"].astype(float) - merged["correct_base"].astype(float)
    merged["delta_confusion_rate"] = (
        merged["confused_with_top_foil"].astype(float) - merged["confused_base"].astype(float)
    )

    summary_rows: List[Dict[str, Any]] = []
    for axis_name, g in merged.groupby("axis"):
        if axis_name == "baseline":
            continue
        row: Dict[str, Any] = {
            "axis": axis_name,
            "ablation_method": ablation_method,
            "mean_delta_accuracy": float(g["delta_correct"].mean()),
            "mean_delta_confusion_rate": float(g["delta_confusion_rate"].mean()),
            "n_trials": int(g["trial_id"].nunique()),
        }
        for la, lb in pair_specs:
            pk = pair_key(la, lb)
            own = g[g["pair_membership"] == pk]
            other = g[(g["pair_membership"] != pk) & (g["pair_membership"] != "other")]
            if len(own):
                row[f"own_{pk}_mean_delta_accuracy"] = float(own["delta_correct"].mean())
                row[f"own_{pk}_mean_delta_confusion_rate"] = float(
                    own["delta_confusion_rate"].mean()
                )
            if len(other):
                row["reuse_other_pairs_mean_delta_accuracy"] = float(other["delta_correct"].mean())
                row["reuse_other_pairs_mean_delta_confusion_rate"] = float(
                    other["delta_confusion_rate"].mean()
                )
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = outdir / f"ablate_summary_{model_key}_layer{layer}.csv"
    summary_df.to_csv(summary_csv, index=False)

    result = {
        "status": "ok",
        "model": model_key,
        "layer": layer,
        "hook_module": module_name,
        "intervention": "activation_patching_mean_ablation",
        "ablation_method": ablation_method,
        "n_trials": len(trials),
        "axes": axis_names,
        "pair_specs": [{"label_a": a, "label_b": b} for a, b in pair_specs],
        "mean_projections": mean_projs,
        "csv": str(csv_path),
        "summary_csv": str(summary_csv),
        "readout_note": (
            "Negative mean_delta_accuracy = ablation hurt accuracy (necessity)."
            " Positive mean_delta_confusion_rate = more top-foil choices after ablation."
            " Compare own pair vs other pairs vs random axis."
        ),
    }
    json_path = outdir / f"{model_key}_eu_ablation_layer{layer}.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--mech_dir", type=Path, default=DEFAULT_MECH)
    ap.add_argument(
        "--activations_dir",
        type=Path,
        default=None,
        help="default: results/activations/baseline_{model}_6afc/{model}",
    )
    ap.add_argument("--manifest", type=Path, default=LOCAL_DATA_DIR / "eu_emotions_full_manifest.json")
    ap.add_argument("--data_root", type=Path, default=LOCAL_DATA_DIR / "eu_emotions")
    ap.add_argument("--human", type=Path, default=LOCAL_DATA_DIR / "eu_emotions_human_entropy.json")
    ap.add_argument("--pairs_json", type=Path, default=LOCAL_DATA_DIR / "human_confused_pairs.json")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_MECH)
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--max_frames", type=int, default=4)
    ap.add_argument("--top_pairs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--sample_temperature", type=float, default=0.0)
    ap.add_argument(
        "--ablation_method",
        choices=("mean_ablation", "zero_projection"),
        default="mean_ablation",
    )
    args = ap.parse_args()

    pair_specs = default_pair_specs(args.pairs_json, top_k=args.top_pairs)
    axis_names = ["entropy", "random"]
    for la, lb in pair_specs:
        pk = pair_key(la, lb)
        if (args.mech_dir / f"axis_{pk}_{args.model}_layer{args.layer}.npy").exists():
            axis_names.append(pk)

    act_dir = args.activations_dir or (
        LOCAL_RESULTS_DIR / "activations" / f"baseline_{args.model}_6afc" / args.model
    )

    result = run_ablate(
        model_key=args.model,
        layer=args.layer,
        mech_dir=args.mech_dir,
        act_dir=act_dir,
        manifest_path=args.manifest,
        data_root=args.data_root,
        human_path=args.human,
        outdir=args.outdir,
        axis_names=axis_names,
        pair_specs=pair_specs,
        max_items=args.max_items,
        max_frames=args.max_frames,
        seed=args.seed,
        smoke=bool(args.smoke),
        sample_temperature=args.sample_temperature,
        ablation_method=args.ablation_method,  # type: ignore[arg-type]
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
