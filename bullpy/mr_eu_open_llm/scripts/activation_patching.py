"""
Activation patching at peak probe layer (Study 3).

Injects baseline hidden states into finetuned model during 4AFC generation.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from transformers import AutoProcessor

from config import EVAL, FRAME_POLICY, LOCAL_RESULTS_DIR, MODELS, SEED, STAGE2_MAX_NEW_TOKENS
from scripts.activation_forward import (
    build_forward_inputs,
    find_layer_module,
    find_layer_module_name,
    prepare_trial_media,
    run_forward,
)
from scripts.emotion_parse import parse_emotion
from scripts.evaluate import (
    load_eu_emotions_manifest,
    load_hf_model_for_key,
    resolve_dataset_root,
    resolve_model_path,
    resolve_trial_media,
)
from scripts.model_compat import apply_llavanext_compat
from scripts.model_inference import _generate_gemma4, generate_model_response
from scripts.probing import load_trial_ids
from scripts.prompts import build_4afc_prompt
from scripts.trial_foils import resolve_candidate_labels, resolve_eu_emotion_pool

PatchMode = Literal["last_token", "all_tokens"]


@dataclass
class HookTarget:
    module: Optional[torch.nn.Module] = None
    name: Optional[str] = None
    generate_counts: Dict[str, int] = field(default_factory=dict)
    forward_counts: Dict[str, int] = field(default_factory=dict)
    resolved_via: str = "unresolved"


def _layer_block_candidates(model: Any, layer_index: int) -> List[Tuple[str, torch.nn.Module]]:
    """Decoder blocks only (exact layers.N), excluding vision towers."""
    suffix = rf"\.layers\.{layer_index}$"
    pattern = re.compile(suffix)
    out: List[Tuple[str, torch.nn.Module]] = []
    for name, mod in model.named_modules():
        if "vision" in name.lower():
            continue
        if pattern.search(name):
            out.append((name, mod))
    out.sort(key=lambda x: len(x[0]))
    return out


def _count_hook_fires(
    model: Any,
    layer_index: int,
    run_fn: Callable[[], None],
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    handles = []
    for name, mod in _layer_block_candidates(model, layer_index):
        def _hook(_m, _i, _o, n=name):
            counts[n] = counts.get(n, 0) + 1

        handles.append(mod.register_forward_hook(_hook))
    run_fn()
    for h in handles:
        h.remove()
    return counts


def resolve_hook_target_for_generation(
    model_key: str,
    model: Any,
    processor: Any,
    trial: Dict[str, Any],
    *,
    model_path: Path,
    peak_layer: int,
    dataset_root: Path,
    seed: int,
    condition_modality: str,
    trial_index: int,
    n_options: int = 4,
) -> HookTarget:
    """Pick the layer block that actually runs during generate (and forward as fallback)."""

    def _run_generate() -> None:
        _generate_4afc(
            model_key,
            model,
            processor,
            model_path,
            trial,
            dataset_root=dataset_root,
            seed=seed,
            condition_modality=condition_modality,
            trial_index=trial_index,
            patcher=None,
            hook_target=None,
            n_options=n_options,
        )

    def _run_forward() -> None:
        trial_copy = dict(trial)
        pool = resolve_eu_emotion_pool()
        options = resolve_candidate_labels(
            trial_copy, pool, seed=seed, trial_index=trial_index, n_options=n_options
        )
        prompt = build_4afc_prompt(options, condition=condition_modality)
        video_path, audio_path, _ = resolve_trial_media(
            trial_copy,
            dataset_key="eu_emotions",
            dataset_root=dataset_root,
            condition=condition_modality,
            seed=seed,
        )
        images: List[Any] = []
        images_for_processor: Any = None
        if video_path is not None and condition_modality != "audio_only":
            images, images_for_processor, _ = prepare_trial_media(
                Path(video_path), model_key=model_key, fps=float(FRAME_POLICY["fps"]), max_frames=int(FRAME_POLICY["max_frames"])
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        inputs = build_forward_inputs(
            model_key,
            model,
            processor,
            prompt=prompt,
            images=images,
            images_for_processor=images_for_processor,
            device=device,
            dtype=dtype,
            audio_path=audio_path,
            condition=condition_modality,
        )
        run_forward(model, inputs, model_key=model_key)

    gen_counts = _count_hook_fires(model, peak_layer, _run_generate)
    fwd_counts = _count_hook_fires(model, peak_layer, _run_forward)

    target = HookTarget(generate_counts=gen_counts, forward_counts=fwd_counts)
    pick_counts = gen_counts if gen_counts else fwd_counts
    if pick_counts:
        best_name = max(pick_counts, key=pick_counts.get)
        target.module = dict(model.named_modules())[best_name]
        target.name = best_name
        target.resolved_via = "generate_probe" if gen_counts else "forward_probe_fallback"
        return target

    mod = find_layer_module(model, peak_layer)
    target.module = mod
    target.name = find_layer_module_name(model, peak_layer)
    target.resolved_via = "find_layer_module_fallback"
    return target


def _raw_snippet(text: Any, limit: int = 400) -> str:
    if text is None:
        return ""
    if isinstance(text, str):
        return text[:limit]
    try:
        return json.dumps(text, ensure_ascii=False)[:limit]
    except Exception:
        return str(text)[:limit]


class ActivationPatcher:
    """Inject stored activations during forward at a hooked layer."""

    def __init__(self, *, patch_mode: PatchMode = "last_token") -> None:
        self.source_activation: Optional[torch.Tensor] = None
        self.patch_mode = patch_mode
        self.hook_calls = 0
        self.hook_layer_found = False
        self.hook_module_name: Optional[str] = None
        self._handle = None

    def set_source(self, activation: np.ndarray, device: torch.device, dtype: torch.dtype) -> None:
        arr = activation.reshape(1, -1) if activation.ndim == 1 else activation
        self.source_activation = torch.tensor(arr, device=device, dtype=dtype)

    def attach(self, module: torch.nn.Module, *, module_name: Optional[str] = None) -> None:
        patcher = self

        def hook(_mod, _inp, out):
            patcher.hook_calls += 1
            if patcher.source_activation is None:
                return out
            h = out[0] if isinstance(out, tuple) else out
            if not hasattr(h, "shape") or h.ndim < 2:
                return out
            src = patcher.source_activation
            if src.shape[-1] != h.shape[-1]:
                return out
            if patcher.patch_mode == "last_token":
                h = h.clone()
                pos = -1
                h[:, pos, :] = src.expand(h.shape[0], -1)
                patched = h
            else:
                patched = src.expand(h.shape[0], h.shape[1], -1)
            if isinstance(out, tuple):
                return (patched,) + out[1:]
            return patched

        self._handle = module.register_forward_hook(hook)
        self.hook_layer_found = True
        self.hook_module_name = module_name

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def load_activation_row(
    activations_dir: Path,
    peak_layer: int,
    trial_id: str,
    seed: int = SEED,
) -> np.ndarray:
    npy = activations_dir / f"layer{peak_layer}_eu_emotions_seed{seed}.npy"
    if not npy.is_file():
        raise FileNotFoundError(f"Missing activations: {npy}")
    trial_ids = load_trial_ids(activations_dir, npy)
    idx = trial_ids.index(trial_id)
    return np.load(npy)[idx]


def _load_model_bundle(
    model_key: str,
    checkpoint: Optional[Path],
) -> Tuple[Any, Any, str, torch.dtype, Path]:
    device_s = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device_s == "cuda" else torch.float32
    model_path = resolve_model_path(model_key)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = load_hf_model_for_key(model_key, model_path, device_s, dtype)
    if checkpoint is not None and checkpoint.exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(checkpoint), is_trainable=False)
    apply_llavanext_compat(model, model_key)
    model.eval()
    return model, processor, device_s, dtype, model_path


def _generate_4afc(
    model_key: str,
    model: Any,
    processor: Any,
    model_path: Path,
    trial: Dict[str, Any],
    *,
    dataset_root: Path,
    seed: int,
    condition_modality: str,
    trial_index: int = 0,
    patcher: Optional[ActivationPatcher] = None,
    hook_target: Optional[HookTarget] = None,
    n_options: int = 4,
) -> Tuple[Optional[str], str, Optional[str]]:
    trial_copy = dict(trial)
    pool = resolve_eu_emotion_pool()
    options = resolve_candidate_labels(
        trial_copy, pool, seed=seed, trial_index=trial_index, n_options=n_options
    )
    prompt = build_4afc_prompt(options, condition=condition_modality)

    video_path, audio_path, _ = resolve_trial_media(
        trial_copy,
        dataset_key="eu_emotions",
        dataset_root=dataset_root,
        condition=condition_modality,
        seed=seed,
    )
    images: List[Any] = []
    images_for_processor: Any = None
    fps = float(FRAME_POLICY["fps"])
    max_frames = int(FRAME_POLICY["max_frames"])
    if video_path is not None and condition_modality != "audio_only":
        images, images_for_processor, _ = prepare_trial_media(
            Path(video_path), model_key=model_key, fps=fps, max_frames=max_frames
        )

    hook_module_name: Optional[str] = None
    if patcher is not None and hook_target is not None and hook_target.module is not None:
        hook_module_name = hook_target.name
        patcher.attach(hook_target.module, module_name=hook_module_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    cond = (condition_modality or "video_only").strip().lower()
    use_gemma4_native = model_key == "gemma4" and cond in {"audio_only", "multimodal"}

    try:
        if use_gemma4_native:
            out = _generate_gemma4(
                model,
                processor,
                prompt,
                images_for_processor=images_for_processor if cond == "multimodal" else None,
                audio_path=audio_path,
                condition=cond,
                device=device,
                dtype=dtype,
                temperature=float(EVAL["temperature"]),
                max_new_tokens=STAGE2_MAX_NEW_TOKENS,
            )
        else:
            out = generate_model_response(
                model_key=model_key,
                model=model,
                processor=processor,
                tokenizer=None,
                model_path=model_path,
                prompt=prompt,
                images=images,
                images_for_processor=images_for_processor,
                device=device,
                dtype=dtype,
                temperature=float(EVAL["temperature"]),
                max_new_tokens=STAGE2_MAX_NEW_TOKENS,
                pipe_cache={},
                audio_path=audio_path,
                condition=condition_modality,
                prefer_loaded_model=True,
            )
    finally:
        if patcher is not None:
            patcher.remove()

    if not isinstance(out, str):
        out = _raw_snippet(out)
    pred, _ = parse_emotion(out, options)
    return pred, out, hook_module_name


def select_ft_incorrect_trials(
    finetuned_eval_json: Path,
    *,
    max_trials: Optional[int] = None,
    require_baseline_correct: bool = False,
    baseline_eval_json: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    ft_obj = json.loads(finetuned_eval_json.read_text(encoding="utf-8"))
    baseline_correct: set = set()
    if require_baseline_correct and baseline_eval_json is not None:
        base_obj = json.loads(baseline_eval_json.read_text(encoding="utf-8"))
        for t in base_obj.get("trials", []):
            s2 = t.get("stage2") or {}
            if s2.get("correct") is True and t.get("trial_id"):
                baseline_correct.add(str(t["trial_id"]))

    selected: List[Dict[str, Any]] = []
    for t in ft_obj.get("trials", []):
        s2 = t.get("stage2") or {}
        if s2.get("correct") is True:
            continue
        tid = t.get("trial_id")
        if require_baseline_correct and tid and str(tid) not in baseline_correct:
            continue
        selected.append(t)
        if max_trials is not None and len(selected) >= max_trials:
            break
    return selected


def run_patching_trial_same_stimulus(
    model_key: str,
    *,
    trial: Dict[str, Any],
    trial_index: int,
    peak_layer: int,
    baseline_activations_dir: Path,
    dataset_root: Path,
    seed: int,
    condition_modality: str = "multimodal",
    checkpoint: Optional[Path] = None,
    patch_mode: PatchMode = "last_token",
    model_bundle: Optional[Tuple[Any, Any, str, torch.dtype, Path]] = None,
    hook_target: Optional[HookTarget] = None,
    n_options: int = 4,
) -> Dict[str, Any]:
    if model_bundle is None:
        model_bundle = _load_model_bundle(model_key, checkpoint)
    model, processor, _device, dtype, model_path = model_bundle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trial_id = str(trial.get("trial_id"))
    true_label = trial.get("label")
    src_act = load_activation_row(baseline_activations_dir, peak_layer, trial_id, seed=seed)

    pred_before, raw_before, _ = _generate_4afc(
        model_key,
        model,
        processor,
        model_path,
        trial,
        dataset_root=dataset_root,
        seed=seed,
        condition_modality=condition_modality,
        trial_index=trial_index,
        n_options=n_options,
    )

    patcher = ActivationPatcher(patch_mode=patch_mode)
    patcher.set_source(src_act, device=device, dtype=dtype)
    pred_after, raw_after, hook_module_name = _generate_4afc(
        model_key,
        model,
        processor,
        model_path,
        trial,
        dataset_root=dataset_root,
        seed=seed,
        condition_modality=condition_modality,
        trial_index=trial_index,
        n_options=n_options,
        patcher=patcher,
        hook_target=hook_target,
    )

    return {
        "trial_id": trial_id,
        "true_label": true_label,
        "peak_layer": peak_layer,
        "patch_mode": patch_mode,
        "hook_module_name": hook_module_name,
        "hook_resolved_via": hook_target.resolved_via if hook_target else None,
        "pred_before_patch": pred_before,
        "pred_after_patch": pred_after,
        "correct_before": pred_before == true_label,
        "correct_after": pred_after == true_label,
        "prediction_changed": pred_before != pred_after,
        "fixed": pred_before != true_label and pred_after == true_label,
        "broken": pred_before == true_label and pred_after != true_label,
        "hook_calls": patcher.hook_calls,
        "hook_layer_found": patcher.hook_layer_found,
        "raw_before": _raw_snippet(raw_before),
        "raw_after": _raw_snippet(raw_after),
    }


def run_patching_experiment(
    model_key: str,
    trials: List[Dict[str, Any]],
    peak_layer: int,
    *,
    baseline_activations_dir: Path,
    dataset_root: Path,
    output: Path,
    seed: int = SEED,
    checkpoint: Optional[Path] = None,
    condition_modality: str = "multimodal",
    patch_mode: PatchMode = "last_token",
    selection_mode: str = "ft_incorrect_same_trial",
    n_options: int = 4,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {
        "model": model_key,
        "seed": seed,
        "peak_layer": peak_layer,
        "patch_mode": patch_mode,
        "selection_mode": selection_mode,
        "n_options": n_options,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "baseline_activations_dir": str(baseline_activations_dir),
        "n_trials_requested": len(trials),
        "trials": [],
        "n_prediction_changed": 0,
        "n_fixed": 0,
        "n_broken": 0,
        "n_hook_never_fired": 0,
    }

    if not trials:
        results["error"] = "no_trials_selected"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        return results

    if not torch.cuda.is_available():
        results["error"] = "cuda_required"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        return results

    model_bundle = _load_model_bundle(model_key, checkpoint)
    manifest_trials, _ = load_eu_emotions_manifest(
        dataset_root.parent / "eu_emotions_118_manifest.json", dataset_root
    )
    trial_index_by_id = {str(t.get("trial_id")): i for i, t in enumerate(manifest_trials)}

    model, processor, _device, _dtype, model_path = model_bundle
    probe_trial = trials[0]
    for candidate in trials:
        trial_copy = dict(candidate)
        _vp, ap, _ = resolve_trial_media(
            trial_copy,
            dataset_key="eu_emotions",
            dataset_root=dataset_root,
            condition=condition_modality,
            seed=seed,
        )
        if condition_modality != "multimodal" or ap is not None or _vp is not None:
            probe_trial = candidate
            break
    hook_target = resolve_hook_target_for_generation(
        model_key,
        model,
        processor,
        probe_trial,
        model_path=model_path,
        peak_layer=peak_layer,
        dataset_root=dataset_root,
        seed=seed,
        condition_modality=condition_modality,
        trial_index=trial_index_by_id.get(str(probe_trial.get("trial_id")), 0),
        n_options=n_options,
    )
    results["hook_target"] = {
        "name": hook_target.name,
        "resolved_via": hook_target.resolved_via,
        "generate_counts": hook_target.generate_counts,
        "forward_counts": hook_target.forward_counts,
    }

    for trial in trials:
        tid = str(trial.get("trial_id"))
        trial_result = run_patching_trial_same_stimulus(
            model_key,
            trial=trial,
            trial_index=trial_index_by_id.get(tid, 0),
            peak_layer=peak_layer,
            baseline_activations_dir=baseline_activations_dir,
            dataset_root=dataset_root,
            seed=seed,
            condition_modality=condition_modality,
            checkpoint=checkpoint,
            patch_mode=patch_mode,
            model_bundle=model_bundle,
            hook_target=hook_target,
            n_options=n_options,
        )
        if trial_result.get("prediction_changed"):
            results["n_prediction_changed"] += 1
        if trial_result.get("fixed"):
            results["n_fixed"] += 1
        if trial_result.get("broken"):
            results["n_broken"] += 1
        if int(trial_result.get("hook_calls") or 0) == 0:
            results["n_hook_never_fired"] += 1
        results["trials"].append(trial_result)

    n = len(trials)
    results["prediction_change_rate"] = results["n_prediction_changed"] / n
    results["fix_rate"] = results["n_fixed"] / n
    results["break_rate"] = results["n_broken"] / n
    results["accuracy_before"] = sum(1 for t in results["trials"] if t.get("correct_before")) / n
    results["accuracy_after"] = sum(1 for t in results["trials"] if t.get("correct_after")) / n

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


# Legacy confused-pair API (kept for compatibility)
def find_trial_for_confusion(
    eval_json: Path,
    *,
    true_label: str,
    pred_label: Optional[str] = None,
    require_correct: bool = False,
    require_incorrect: bool = False,
) -> Optional[Dict[str, Any]]:
    obj = json.loads(eval_json.read_text(encoding="utf-8"))
    fallback: Optional[Dict[str, Any]] = None
    for t in obj.get("trials", []):
        if t.get("label") != true_label:
            continue
        s2 = t.get("stage2") or {}
        pred = s2.get("prediction")
        correct = s2.get("correct")
        if require_correct and not correct:
            continue
        if require_incorrect and correct:
            continue
        if pred_label is not None and pred != pred_label:
            if fallback is None:
                fallback = t
            continue
        return t
    return fallback


def load_confused_pairs(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    obj = json.loads(path.read_text(encoding="utf-8"))
    pairs = obj.get("confused_pairs", obj if isinstance(obj, list) else [])
    return pairs if isinstance(pairs, list) else []


def main() -> None:
    ap = argparse.ArgumentParser(description="Activation patching: baseline->FT at peak layer.")
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--baseline_eval", type=Path, required=True)
    ap.add_argument("--finetuned_eval", type=Path, required=True)
    ap.add_argument(
        "--baseline_activations_dir",
        type=Path,
        default=LOCAL_RESULTS_DIR / "activations" / "baseline_gemma4" / "gemma4",
    )
    ap.add_argument("--peak_layer", type=int, default=None)
    ap.add_argument(
        "--peak_layer_json",
        type=Path,
        default=LOCAL_RESULTS_DIR / "probes" / "baseline_gemma4" / "gemma4" / "peak_layer.json",
    )
    ap.add_argument("--data_root", type=Path, default=None)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--modality", default="multimodal")
    ap.add_argument(
        "--output",
        type=Path,
        default=LOCAL_RESULTS_DIR / "patching" / "patching_results_gemma4_v2.json",
    )
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--max_trials",
        type=int,
        default=0,
        help="FT-incorrect trials to patch (0 = all incorrect, ~115; 30 is a quick smoke).",
    )
    ap.add_argument(
        "--patch_mode",
        choices=["last_token", "all_tokens"],
        default="last_token",
    )
    ap.add_argument(
        "--require_baseline_correct",
        action="store_true",
        help="Only patch trials baseline got right (stronger causal contrast).",
    )
    ap.add_argument(
        "--n_options",
        type=int,
        default=4,
        help="Forced-choice size (use 6 for study3 full-EU).",
    )
    args = ap.parse_args()

    peak = args.peak_layer
    if peak is None and args.peak_layer_json.is_file():
        peak = int(json.loads(args.peak_layer_json.read_text()).get("peak_layer", 12))

    max_trials = None if args.max_trials == 0 else args.max_trials
    trials = select_ft_incorrect_trials(
        args.finetuned_eval,
        max_trials=max_trials,
        require_baseline_correct=args.require_baseline_correct,
        baseline_eval_json=args.baseline_eval if args.require_baseline_correct else None,
    )
    dataset_root = args.data_root or resolve_dataset_root("eu_emotions")

    run_patching_experiment(
        args.model,
        trials,
        peak_layer=int(peak or 12),
        baseline_activations_dir=args.baseline_activations_dir,
        dataset_root=dataset_root,
        output=args.output,
        seed=args.seed,
        checkpoint=args.checkpoint,
        condition_modality=args.modality,
        patch_mode=args.patch_mode,
        n_options=args.n_options,
    )


if __name__ == "__main__":
    main()
