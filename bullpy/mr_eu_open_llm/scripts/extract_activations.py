"""
Extract hidden states at protocol layer fractions (Study 2 / Study 3).

Uses real stimulus-conditioned forwards (same frames + prompts as evaluate.py).
Supports free-response or per-trial 4AFC prompts and mean vs last-token pooling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch
from transformers import AutoProcessor

from config import DATASETS, FRAME_POLICY, LOCAL_RESULTS_DIR, MODELS, MODALITY_CONDITIONS, SEED
from scripts.activation_forward import (
    build_forward_inputs,
    default_extraction_prompt,
    find_layer_module,
    prepare_trial_media,
    run_forward,
)
from scripts.evaluate import (
    load_eu_emotions_manifest,
    load_hf_model_for_key,
    resolve_dataset_root,
    resolve_model_path,
    resolve_trial_media,
)
from scripts.layer_map import get_layer_indices, n_layers
from scripts.prompts import build_4afc_prompt
from scripts.trial_foils import resolve_candidate_labels, resolve_eu_emotion_pool

PromptMode = Literal["free_response", "4afc"]
PoolingMode = Literal["mean", "last_token"]


def default_pooling_for_prompt_mode(prompt_mode: PromptMode) -> PoolingMode:
    return "last_token" if prompt_mode == "4afc" else "mean"


def resolve_trial_extraction_prompt(
    trial: Dict[str, Any],
    *,
    prompt_mode: PromptMode,
    condition_modality: str,
    seed: int,
    trial_index: int,
    global_prompt: Optional[str] = None,
    n_options: int = 4,
) -> str:
    """Return the forward prompt for one trial."""
    if global_prompt is not None:
        return global_prompt
    if prompt_mode == "4afc":
        pool = resolve_eu_emotion_pool()
        options = resolve_candidate_labels(
            dict(trial), pool, seed=seed, trial_index=trial_index, n_options=n_options
        )
        return build_4afc_prompt(options, condition=condition_modality)
    return default_extraction_prompt(condition=condition_modality)


def _pool_hidden(h: torch.Tensor, pooling: PoolingMode) -> np.ndarray:
    hf = h.detach().float()
    if pooling == "last_token":
        return hf[:, -1, :].cpu().numpy()
    return hf.mean(dim=1).cpu().numpy()


def _register_hook(
    module: torch.nn.Module,
    storage: List[np.ndarray],
    *,
    pooling: PoolingMode,
) -> Any:
    def hook(_mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if hasattr(h, "detach"):
            storage.append(_pool_hidden(h, pooling))

    return module.register_forward_hook(hook)


def _load_model_and_processor(
    model_key: str,
    *,
    device: str,
    dtype: torch.dtype,
    checkpoint: Optional[Path] = None,
) -> tuple[Any, Any]:
    model_path = resolve_model_path(model_key)
    processor = None
    if model_key != "gemma4":
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    else:
        try:
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            from transformers import Gemma4Processor  # type: ignore

            processor = Gemma4Processor.from_pretrained(model_path)

    model = load_hf_model_for_key(model_key, model_path, device, dtype)
    if checkpoint is not None:
        from peft import PeftModel

        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        model = PeftModel.from_pretrained(model, str(checkpoint), is_trainable=False)
    if device != "cuda":
        model = model.to(device)
    model.eval()
    return model, processor


def extract_activations(
    model_key: str,
    condition: str,
    dataset_root: Path,
    manifest: Path,
    output_dir: Path,
    seed: int = SEED,
    checkpoint: Optional[Path] = None,
    max_trials: Optional[int] = None,
    condition_modality: str = "video_only",
    fps: Optional[float] = None,
    max_frames: Optional[int] = None,
    prompt: Optional[str] = None,
    prompt_mode: PromptMode = "free_response",
    pooling: Optional[PoolingMode] = None,
    n_options: Optional[int] = None,
) -> Dict[str, Any]:
    trials, _labels = load_eu_emotions_manifest(manifest, dataset_root)
    if max_trials is not None:
        trials = trials[: int(max_trials)]

    modality = (condition_modality or "video_only").strip().lower()
    if modality not in MODALITY_CONDITIONS:
        raise ValueError(f"Invalid modality {modality}")
    if prompt_mode not in {"free_response", "4afc"}:
        raise ValueError(f"Invalid prompt_mode={prompt_mode}")

    if n_options is None:
        from scripts.evaluate import manifest_n_options

        n_options = manifest_n_options(manifest)
    if n_options is None:
        n_options = 4
    n_options = int(n_options)

    pool_mode: PoolingMode = pooling or default_pooling_for_prompt_mode(prompt_mode)

    fps_val = float(fps if fps is not None else FRAME_POLICY["fps"])
    max_frames_val = int(max_frames if max_frames is not None else FRAME_POLICY["max_frames"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model, processor = _load_model_and_processor(
        model_key, device=device, dtype=dtype, checkpoint=checkpoint
    )

    layer_map = get_layer_indices(model_key, model=model)
    nl = n_layers(model_key, model)
    hooks = []
    storage: Dict[str, List[np.ndarray]] = {k: [] for k in layer_map}

    for frac_key, layer_idx in layer_map.items():
        mod = find_layer_module(model, layer_idx)
        if mod is not None:
            hooks.append(_register_hook(mod, storage[frac_key], pooling=pool_mode))

    trial_ids: List[str] = []
    errors: List[Dict[str, str]] = []

    for trial_idx, t in enumerate(trials):
        trial_id = str(t.get("trial_id", ""))
        trial_ids.append(trial_id)
        trial_copy = dict(t)
        try:
            trial_prompt = resolve_trial_extraction_prompt(
                trial_copy,
                prompt_mode=prompt_mode,
                condition_modality=modality,
                seed=seed,
                trial_index=trial_idx,
                global_prompt=prompt,
                n_options=n_options,
            )
            video_path, audio_path, _audio_rule = resolve_trial_media(
                trial_copy,
                dataset_key="eu_emotions",
                dataset_root=dataset_root,
                condition=modality,
                seed=seed,
            )
            path = video_path or Path(t["stimulus_path"])
            images: List[Any] = []
            images_for_model: Any = None
            if video_path is not None and modality != "audio_only":
                images, images_for_model, _fi = prepare_trial_media(
                    path, model_key=model_key, fps=fps_val, max_frames=max_frames_val
                )
            elif modality == "audio_only" and audio_path is None:
                raise FileNotFoundError(f"audio_only requires audio for trial {trial_id}")
            inputs = build_forward_inputs(
                model_key,
                model,
                processor,
                prompt=trial_prompt,
                images=images,
                images_for_processor=images_for_model,
                device=device,
                dtype=dtype,
                audio_path=audio_path,
                condition=modality,
            )
            run_forward(model, inputs, model_key=model_key)
        except Exception as e:
            errors.append({"trial_id": trial_id, "error": f"{type(e).__name__}: {e}"})
            for frac_key in storage:
                if storage[frac_key]:
                    dim = storage[frac_key][0].shape[-1]
                else:
                    dim = 4096
                storage[frac_key].append(np.zeros((1, dim), dtype=np.float32))

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_layers: List[int] = []
    for frac_key, arrs in storage.items():
        if not arrs:
            continue
        acts = np.concatenate(arrs, axis=0)
        layer_idx = layer_map[frac_key]
        saved_layers.append(layer_idx)
        out_path = output_dir / f"layer{layer_idx}_eu_emotions_seed{seed}.npy"
        np.save(out_path, acts)
        sidecar = output_dir / f"layer{layer_idx}_trial_ids.json"
        sidecar.write_text(json.dumps(trial_ids, indent=2) + "\n", encoding="utf-8")

    meta = {
        "model": model_key,
        "condition": condition,
        "modality": modality,
        "prompt_mode": prompt_mode,
        "pooling": pool_mode,
        "n_options": n_options,
        "n_layers": nl,
        "layer_map": layer_map,
        "n_trials": len(trials),
        "seed": seed,
        "fps": fps_val,
        "max_frames": max_frames_val,
        "prompt": prompt if prompt is not None else None,
        "prompt_note": "per_trial_4afc" if prompt_mode == "4afc" and prompt is None else "fixed",
        "checkpoint": str(checkpoint) if checkpoint else None,
        "saved_layers": saved_layers,
        "n_errors": len(errors),
        "errors_head": errors[:5],
    }

    if len(trials) >= 2 and saved_layers and len(errors) < len(trials):
        p0 = output_dir / f"layer{saved_layers[0]}_eu_emotions_seed{seed}.npy"
        if p0.is_file():
            arr = np.load(p0)
            if arr.shape[0] >= 2:
                same = np.allclose(arr[0], arr[1])
                meta["smoke_trials_identical"] = bool(same)
                if same:
                    meta["smoke_warning"] = "first two trial activations are identical"

    if len(errors) == len(trials) and trials:
        meta["status"] = "failed_all_trials"

    (output_dir / "extract_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    for h in hooks:
        h.remove()

    if len(errors) == len(trials) and trials:
        raise RuntimeError(
            f"All {len(trials)} trials failed during activation extraction. "
            f"First error: {errors[0].get('error')}"
        )

    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract transformer activations for Study 2/3.")
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--condition", default="baseline", help="e.g. baseline_gemma4_4afc")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--data_root", type=Path, default=None)
    ap.add_argument("--output_dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--checkpoint", type=Path, default=None, help="PEFT LoRA adapter path")
    ap.add_argument("--max_trials", type=int, default=None)
    ap.add_argument("--modality", default="video_only", choices=list(MODALITY_CONDITIONS))
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument(
        "--prompt_mode",
        default="free_response",
        choices=["free_response", "4afc"],
        help="free_response (stage-1 style) or per-trial 4AFC (aligned with patching)",
    )
    ap.add_argument(
        "--pooling",
        default=None,
        choices=["mean", "last_token"],
        help="Token pooling for saved vectors (default: last_token for 4afc, mean otherwise)",
    )
    ap.add_argument(
        "--prompt",
        default=None,
        help="Optional fixed prompt for all trials (overrides prompt_mode templates)",
    )
    ap.add_argument(
        "--n_options",
        type=int,
        default=None,
        help="Forced-choice size when prompt_mode=4afc (default: manifest n_options or 4).",
    )
    args = ap.parse_args()

    dataset_root = args.data_root or resolve_dataset_root("eu_emotions")
    manifest = args.manifest
    if manifest is None:
        manifest = Path(
            DATASETS["eu_emotions"].get(
                "manifest_local", dataset_root.parent / "eu_emotions_118_manifest.json"
            )
        )
    if args.output_dir is None:
        args.output_dir = LOCAL_RESULTS_DIR / "activations" / args.condition / args.model

    meta = extract_activations(
        model_key=args.model,
        condition=args.condition,
        dataset_root=dataset_root,
        manifest=manifest,
        output_dir=args.output_dir,
        seed=args.seed,
        checkpoint=args.checkpoint,
        max_trials=args.max_trials,
        condition_modality=args.modality,
        fps=args.fps,
        max_frames=args.max_frames,
        prompt=args.prompt,
        prompt_mode=args.prompt_mode,
        pooling=args.pooling,
        n_options=args.n_options,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
