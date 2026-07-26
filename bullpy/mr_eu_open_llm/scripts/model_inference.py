"""
Shared multimodal generation for Stage 1 (free response) and Stage 2 (4AFC).

Gemma 4 E4B-it supports native audio (audio_only / multimodal). Other open VLMs in this
repo are vision-only; audio ablations are gated via config.MODEL_AUDIO_CAPABILITIES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from config import EVAL
from scripts.activation_forward import build_forward_inputs
from scripts.model_compat import is_peft_model


def _gen_kwargs(temperature: float, max_new_tokens: int) -> Dict[str, Any]:
    gen_kwargs: Dict[str, Any] = {"max_new_tokens": int(max_new_tokens)}
    if float(temperature) > 0:
        gen_kwargs.update(
            {"do_sample": True, "temperature": float(temperature), "top_p": float(EVAL["top_p"])}
        )
    else:
        gen_kwargs["do_sample"] = False
    return gen_kwargs


def _decode_generated_ids(processor: Any, out_ids: Any, inputs: Dict[str, Any]) -> str:
    in_len = 0
    input_ids = inputs.get("input_ids")
    if input_ids is not None:
        in_len = int(input_ids.shape[1])

    # Some VLMs / Peft wrappers return only new tokens (len <= prompt length).
    if out_ids.shape[1] <= in_len:
        gen_ids = out_ids
    else:
        gen_ids = out_ids[:, in_len:]

    try:
        text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    except Exception:
        text = ""
    if not (text or "").strip():
        tok = getattr(processor, "tokenizer", None)
        if tok is not None:
            try:
                text = tok.decode(gen_ids[0], skip_special_tokens=True)
            except Exception:
                text = str(gen_ids)
    return text or ""


def _build_gemma4_content(
    prompt: str,
    *,
    condition: str,
    images_for_processor: Any,
    audio_path: Optional[Path],
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    cond = (condition or "video_only").strip().lower()

    if cond in {"audio_only", "multimodal"} and audio_path is not None:
        content.append({"type": "audio", "audio": str(audio_path.resolve())})

    if cond in {"video_only", "multimodal"} and images_for_processor is not None:
        content.append({"type": "image", "image": images_for_processor})

    content.append({"type": "text", "text": prompt})
    return content


def _generate_gemma4(
    model: Any,
    processor: Any,
    prompt: str,
    *,
    images_for_processor: Any,
    audio_path: Optional[Path],
    condition: str,
    device: str,
    dtype: torch.dtype,
    temperature: float,
    max_new_tokens: int,
) -> str:
    content = _build_gemma4_content(
        prompt,
        condition=condition,
        images_for_processor=images_for_processor,
        audio_path=audio_path,
    )
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
    inputs = {
        k: v.to(model_device, dtype=model_dtype if v.is_floating_point() else None)
        if hasattr(v, "to")
        else v
        for k, v in inputs.items()
    }

    gen_kwargs = _gen_kwargs(temperature, max_new_tokens)
    with torch.inference_mode():
        out_ids = model.generate(**inputs, **gen_kwargs)

    return _decode_generated_ids(processor, out_ids, inputs)


def _generate_on_loaded_model(
    model_key: str,
    model: Any,
    processor: Any,
    *,
    prompt: str,
    images: List[Any],
    images_for_processor: Any,
    audio_path: Optional[Path],
    condition: str,
    device: str,
    dtype: torch.dtype,
    temperature: float,
    max_new_tokens: int,
) -> str:
    """Generate with the in-memory model (required for PeftModel + patching hooks)."""
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
        condition=condition,
    )
    gen_kwargs = _gen_kwargs(temperature, max_new_tokens)
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        pad_id = getattr(tok, "pad_token_id", None)
        eos_id = getattr(tok, "eos_token_id", None)
        if pad_id is not None:
            gen_kwargs.setdefault("pad_token_id", pad_id)
        if eos_id is not None:
            gen_kwargs.setdefault("eos_token_id", eos_id)

    with torch.inference_mode():
        out_ids = model.generate(**inputs, **gen_kwargs)
    return _decode_generated_ids(processor, out_ids, inputs)


def _should_use_loaded_model(
    model_key: str,
    model: Any,
    *,
    prefer_loaded_model: bool,
    use_audio: bool,
) -> bool:
    if prefer_loaded_model or is_peft_model(model):
        return True
    if model_key == "qwen2vl":
        return True
    if model_key == "llavanext":
        return True
    if model_key == "gemma4" and use_audio:
        return True
    return False


def generate_model_response(
    model_key: str,
    model: Any,
    processor: Any,
    tokenizer: Any,
    model_path: Path,
    prompt: str,
    images: List[Any],
    images_for_processor: Any,
    device: str,
    dtype: torch.dtype,
    temperature: float,
    max_new_tokens: int,
    pipe_cache: Dict[str, Any],
    audio_path: Optional[Path] = None,
    condition: str = "video_only",
    prefer_loaded_model: bool = False,
) -> str:
    """
    Run one forward/generate pass and return decoded text (completion only where possible).
    """
    cond = (condition or "video_only").strip().lower()
    use_audio = audio_path is not None and cond in {"audio_only", "multimodal"}

    if model_key == "gemma4" and use_audio:
        return _generate_gemma4(
            model,
            processor,
            prompt,
            images_for_processor=images_for_processor if cond == "multimodal" else None,
            audio_path=audio_path,
            condition=cond,
            device=device,
            dtype=dtype,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    if _should_use_loaded_model(
        model_key, model, prefer_loaded_model=prefer_loaded_model, use_audio=use_audio
    ):
        return _generate_on_loaded_model(
            model_key,
            model,
            processor,
            prompt=prompt,
            images=images,
            images_for_processor=images_for_processor,
            audio_path=audio_path,
            condition=condition,
            device=device,
            dtype=dtype,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    out_text: Optional[str] = None

    if model_key in {"gemma4", "llavanext"}:
        from transformers import pipeline  # type: ignore

        pipe_key = f"{model_key}:{'peft' if is_peft_model(model) else 'base'}"
        pipe = pipe_cache.get(pipe_key)
        if pipe is None:
            pipe_kwargs: Dict[str, Any] = {
                "task": "image-text-to-text",
                "device": 0 if device == "cuda" else -1,
                "torch_dtype": dtype if device == "cuda" else None,
            }
            if is_peft_model(model):
                pipe_kwargs["model"] = model
                pipe_kwargs["processor"] = processor
            else:
                pipe_kwargs["model"] = model_path
            pipe = pipeline(**pipe_kwargs)
            pipe_cache[pipe_key] = pipe
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": images_for_processor},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        out = pipe(text=messages, max_new_tokens=int(max_new_tokens))
        if isinstance(out, list) and out:
            out0 = out[0]
            out_text = out0.get("generated_text") if isinstance(out0, dict) else None
            if out_text is None:
                out_text = str(out0)
        else:
            out_text = str(out)

    else:
        raise RuntimeError(f"Unhandled model_key in pipeline path: {model_key}")

    if not isinstance(out_text, str):
        try:
            out_text = json.dumps(out_text, ensure_ascii=False)
        except Exception:
            out_text = str(out_text)
    return out_text
