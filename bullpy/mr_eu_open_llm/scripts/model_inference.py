"""
Shared multimodal generation for Stage 1 (free response) and Stage 2 (4AFC).

Gemma 4 E4B-it supports native audio (audio_only / multimodal). Other open VLMs in this
repo are vision-only; audio ablations are gated via config.MODEL_AUDIO_CAPABILITIES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from config import EVAL


def _gen_kwargs(temperature: float, max_new_tokens: int) -> Dict[str, Any]:
    gen_kwargs: Dict[str, Any] = {"max_new_tokens": int(max_new_tokens)}
    if float(temperature) > 0:
        gen_kwargs.update(
            {"do_sample": True, "temperature": float(temperature), "top_p": float(EVAL["top_p"])}
        )
    else:
        gen_kwargs["do_sample"] = False
    return gen_kwargs


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

    input_ids = inputs.get("input_ids")
    in_len = int(input_ids.shape[1]) if input_ids is not None else 0
    gen_ids = out_ids[:, in_len:] if in_len > 0 else out_ids
    return processor.batch_decode(gen_ids, skip_special_tokens=True)[0]


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

    out_text: Optional[str] = None

    if model_key in {"gemma4", "llavanext"}:
        from transformers import pipeline  # type: ignore

        pipe = pipe_cache.get(model_key)
        if pipe is None:
            pipe = pipeline(
                "image-text-to-text",
                model=model_path,
                device=0 if device == "cuda" else -1,
                torch_dtype=dtype if device == "cuda" else None,
            )
            pipe_cache[model_key] = pipe
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
        text: str
        if model_key == "llavanext":
            conversation = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image"}]}]
            try:
                text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            except Exception:
                text = f"{prompt}\n<image>"
        elif model_key == "qwen2vl":
            n_img = len(images) if isinstance(images, list) else 1
            content_q: List[Dict[str, Any]] = [{"type": "image"} for _ in range(n_img)]
            content_q.append({"type": "text", "text": prompt})
            messages_q = [{"role": "user", "content": content_q}]
            try:
                text = processor.apply_chat_template(messages_q, tokenize=False, add_generation_prompt=True)
            except Exception:
                image_placeholder = "<image>"
                joined = "\n".join([image_placeholder] * n_img)
                text = f"{joined}\n{prompt}"
        else:
            image_placeholder = "<image>"
            joined = "\n".join([image_placeholder] * (len(images) if isinstance(images, list) else 1))
            text = f"{joined}\n{prompt}"

        try:
            inputs = processor(images=images_for_processor, text=text, return_tensors="pt")
        except Exception:
            try:
                inputs = processor(images=images_for_processor, text=[text], return_tensors="pt")
            except Exception:
                inputs = processor(text=text, images=images_for_processor, return_tensors="pt")
        if inputs is None:
            raise RuntimeError("Processor returned None.")

        if model_key == "llavanext":
            try:
                image_sizes = inputs.get("image_sizes") if hasattr(inputs, "get") else inputs["image_sizes"]
            except Exception:
                image_sizes = None
            if image_sizes is None and isinstance(images, list) and images:
                h = int(images[0].size[1])
                w = int(images[0].size[0])
                try:
                    inputs["image_sizes"] = torch.tensor([[h, w]], device=model.device)
                except Exception:
                    pass

        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        gen_kwargs = _gen_kwargs(temperature, max_new_tokens)

        with torch.inference_mode():
            out_ids = model.generate(**inputs, **gen_kwargs)

        in_len = 0
        try:
            input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else inputs["input_ids"]
            if input_ids is not None:
                in_len = int(input_ids.shape[1])
        except Exception:
            in_len = 0
        try:
            gen_ids = out_ids[:, in_len:] if in_len > 0 else out_ids
            out_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        except Exception:
            out_text = str(out_ids)

    if not isinstance(out_text, str):
        try:
            out_text = json.dumps(out_text, ensure_ascii=False)
        except Exception:
            out_text = str(out_text)
    return out_text
