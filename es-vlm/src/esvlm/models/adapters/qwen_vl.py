"""Qwen2-VL / Qwen2.5-VL / Qwen3-VL / text-only Qwen3."""

from __future__ import annotations

from typing import Any


def load(
    source: str,
    *,
    revision: str | None,
    device: Any,
    dtype: Any,
    trust_remote_code: bool,
) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "dtype": dtype,
    }
    if revision:
        kwargs["revision"] = revision
    proc_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if revision:
        proc_kwargs["revision"] = revision
    try:
        processor = AutoProcessor.from_pretrained(source, **proc_kwargs)
    except Exception:
        processor = AutoTokenizer.from_pretrained(source, **proc_kwargs)
    try:
        model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    except Exception:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(source, **kwargs)
    model = model.to(device)
    model.eval()
    return model, processor


def residual_modules(model: Any) -> list[Any]:
    for path in (
        "model.language_model.layers",
        "model.layers",
        "language_model.layers",
        "transformer.h",
        "model.model.layers",
    ):
        cur: Any = model
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok:
            return list(cur)
    raise AttributeError(f"could not find residual layers on {type(model)}")
