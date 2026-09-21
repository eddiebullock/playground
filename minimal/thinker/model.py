"""Thinker inference backends: mock (laptop) and Qwen-VL (HPC GPU)."""

from __future__ import annotations

import hashlib
import time
from io import BytesIO
from typing import Protocol

from PIL import Image

from schemas import (
    PROMPT_V1,
    Advisory,
    format_line,
    parse_advisory_text,
    safe_default,
)


class Inferencer(Protocol):
    model_id: str

    def infer(self, image_bytes: bytes) -> Advisory:
        ...


class MockInferencer:
    """Deterministic stand-in so the HTTP contract can be tested without a GPU.

    Hashes the JPEG so the same frame always yields the same advisory. This is
    NOT a vision model. Replace by starting the server without --mock on HPC.
    """

    def __init__(self) -> None:
        self.model_id = "mock-thinker"

    def infer(self, image_bytes: bytes) -> Advisory:
        t0 = time.time()
        digest = hashlib.sha256(image_bytes).digest()
        bucket = digest[0] % 10
        if bucket == 0:
            hazard, action, cap = "PEDESTRIAN", "STOP", 0.5
        elif bucket in (1, 2):
            hazard, action, cap = "VEHICLE", "SLOW", 3.0
        elif bucket == 3:
            hazard, action, cap = "NONE", "NUDGE_LEFT", None
        else:
            hazard, action, cap = "NONE", "KEEP", None
        raw = format_line(hazard, action, cap)
        return Advisory(
            hazard=hazard,
            action=action,
            speed_cap_mps=cap,
            raw_text=raw,
            model_ts=time.time(),
            inference_ms=(time.time() - t0) * 1000.0,
            model_id=self.model_id,
            parse_ok=True,
        )


class QwenInferencer:
    """Loads Qwen2-VL once. Intended for the HPC GPU session."""

    def __init__(self, model_id: str = "Qwen/Qwen2-VL-2B-Instruct") -> None:
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        print(f"[thinker] loading {model_id} on {self.device} ...", flush=True)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()
        print("[thinker] model ready", flush=True)

    def infer(self, image_bytes: bytes) -> Advisory:
        import torch

        t0 = time.time()
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            image.thumbnail((640, 360))
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": PROMPT_V1},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=48,
                    do_sample=False,
                )
            trimmed = generated[:, inputs.input_ids.shape[1] :]
            raw = self.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            hazard, action, cap, ok = parse_advisory_text(raw)
            return Advisory(
                hazard=hazard,
                action=action,
                speed_cap_mps=cap,
                raw_text=raw,
                model_ts=time.time(),
                inference_ms=(time.time() - t0) * 1000.0,
                model_id=self.model_id,
                parse_ok=ok,
            )
        except Exception as exc:
            adv = safe_default(self.model_id, time.time())
            adv.raw_text = f"INFER_ERROR: {exc}"
            adv.inference_ms = (time.time() - t0) * 1000.0
            return adv
