"""
Stimulus-conditioned forward passes for activation extraction and patching.

Reuses the same frame sampling, multi-frame policy, and prompts as evaluate.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from config import FRAME_POLICY
from scripts.frame_sampling import load_stimulus_as_images
from scripts.multi_frame import prepare_images_for_model
from scripts.prompts import build_free_response_prompt


def resolve_model_device(model: Any, fallback: str = "cpu") -> torch.device:
    dev = getattr(model, "device", None)
    if dev is not None:
        return torch.device(dev) if isinstance(dev, str) else dev
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(fallback)


def prepare_trial_media(
    stimulus_path: Path,
    *,
    model_key: str,
    fps: float,
    max_frames: int,
) -> Tuple[List[Any], Any, List[int]]:
    images, frame_indices = load_stimulus_as_images(
        stimulus_path, fps=fps, max_frames=max_frames
    )
    images_for_model, _meta = prepare_images_for_model(
        model_key,
        images,
        enforce_multi_frame=FRAME_POLICY.get("enforce_multi_frame", True),
    )
    return images, images_for_model, frame_indices


def build_forward_inputs(
    model_key: str,
    model: Any,
    processor: Any,
    *,
    prompt: str,
    images: List[Any],
    images_for_processor: Any,
    device: str,
    dtype: torch.dtype,
    audio_path: Optional[Path] = None,
    condition: str = "video_only",
) -> Dict[str, Any]:
    """Build tensor inputs for a single forward pass (no generation)."""
    cond = (condition or "video_only").strip().lower()

    if model_key == "gemma4":
        content: List[Dict[str, Any]] = []
        if cond in {"audio_only", "multimodal"} and audio_path is not None:
            content.append({"type": "audio", "audio": str(audio_path.resolve())})
        if cond in {"video_only", "multimodal"} and images_for_processor is not None:
            content.append({"type": "image", "image": images_for_processor})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        try:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        except TypeError:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if not isinstance(inputs, dict):
                inputs = {"input_ids": inputs}
        model_device = getattr(model, "device", None) or device
        model_dtype = getattr(model, "dtype", None) or dtype
        return {
            k: v.to(model_device, dtype=model_dtype if hasattr(v, "is_floating_point") and v.is_floating_point() else None)
            if hasattr(v, "to")
            else v
            for k, v in inputs.items()
        }

    # Qwen2-VL / LLaVA-style
    n_img = len(images) if isinstance(images, list) else 1
    if model_key in {"qwen2vl", "qwen3vl"}:
        content_q: List[Dict[str, Any]] = [{"type": "image"} for _ in range(n_img)]
        content_q.append({"type": "text", "text": prompt})
        messages_q = [{"role": "user", "content": content_q}]
        try:
            text = processor.apply_chat_template(messages_q, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = "\n".join(["<image>"] * n_img) + f"\n{prompt}"
    elif model_key == "llavanext":
        conversation = [
            {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image"}]}
        ]
        try:
            text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = f"{prompt}\n<image>"
    elif model_key == "molmo2":
        messages_m = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        ]
        try:
            text = processor.apply_chat_template(messages_m, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = f"{prompt}\n<image>"
    else:
        text = f"{prompt}\n<image>"

    try:
        inputs = processor(images=images_for_processor, text=text, return_tensors="pt")
    except Exception:
        inputs = processor(text=text, images=images_for_processor, return_tensors="pt")
    if inputs is None:
        raise RuntimeError("Processor returned None for forward inputs.")

    if model_key == "llavanext":
        inputs = dict(inputs)
        inputs.pop("image_sizes", None)

    if model_key == "molmo2":
        # Molmo is trained with causal attention only; HF emits token_type_ids to switch
        # image tokens to bidirectional attention, which changes the model's outputs.
        inputs = dict(inputs)
        inputs.pop("token_type_ids", None)

    model_device = resolve_model_device(model)

    # Do not pass image_sizes for llavanext — Siglip-backed LLaVA errors on patch_size.
    return {
        k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()
    }


def run_forward(
    model: Any,
    inputs: Dict[str, Any],
    *,
    model_key: str,
) -> None:
    """Single forward pass; hooks on registered modules capture activations."""
    with torch.inference_mode():
        try:
            model(**inputs)
        except TypeError:
            # Some VLMs need input_ids only on inner LM
            if model_key == "gemma4" and hasattr(model, "language_model"):
                model.language_model(**{k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")})
            else:
                raise


def default_extraction_prompt(condition: str = "video_only") -> str:
    return build_free_response_prompt(condition=condition)


def find_layer_module(model: Any, layer_index: int) -> Optional[torch.nn.Module]:
    """Locate transformer block *layer_index* for hooking (PEFT / multimodal-safe)."""
    import re

    roots: List[Any] = []
    for candidate in (model, getattr(model, "base_model", None), getattr(model, "model", None)):
        if candidate is not None and candidate not in roots:
            roots.append(candidate)
    if hasattr(model, "get_base_model"):
        try:
            base = model.get_base_model()
            if base not in roots:
                roots.append(base)
        except Exception:
            pass

    layer_chains = (
        ("model", "layers"),
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
        ("layers",),
    )
    for root in roots:
        for chain in layer_chains:
            try:
                mod: Any = root
                for part in chain:
                    mod = getattr(mod, part)
                return mod[layer_index]
            except (AttributeError, IndexError, TypeError, KeyError):
                continue

    pattern = re.compile(rf"(?:^|\.)layers\.{layer_index}(?:\.|$)")
    best_name: Optional[str] = None
    best_mod: Optional[torch.nn.Module] = None
    for name, mod in model.named_modules():
        if "vision" in name.lower():
            continue
        if pattern.search(name):
            if best_name is None or len(name) < len(best_name):
                best_name, best_mod = name, mod
    if best_mod is not None:
        return best_mod

    needles = (
        f".layers.{layer_index}.",
        f".h.{layer_index}.",
        f".blocks.{layer_index}.",
        f".layer.{layer_index}.",
    )
    for name, mod in dict(model.named_modules()).items():
        for nd in needles:
            if nd in name:
                return mod
    for name, mod in dict(model.named_modules()).items():
        if name.endswith(f".{layer_index}") or name.endswith(f"layers.{layer_index}"):
            return mod
    return None


def find_layer_module_name(model: Any, layer_index: int) -> Optional[str]:
    """Return module path string for logging (best-effort)."""
    import re

    pattern = re.compile(rf"(?:^|\.)layers\.{layer_index}(?:\.|$)")
    names = [
        name
        for name, _ in model.named_modules()
        if pattern.search(name) and "vision" not in name.lower()
    ]
    return min(names, key=len) if names else None
