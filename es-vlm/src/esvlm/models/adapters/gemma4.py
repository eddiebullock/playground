"""Gemma 4 (Transformers 5.x)."""

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
    from transformers import AutoModelForCausalLM, AutoProcessor

    kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code, "dtype": dtype}
    if revision:
        kwargs["revision"] = revision
    try:
        processor = AutoProcessor.from_pretrained(source, trust_remote_code=trust_remote_code, revision=revision)
    except Exception:
        from transformers import AutoTokenizer

        processor = AutoTokenizer.from_pretrained(source, trust_remote_code=trust_remote_code, revision=revision)
    try:
        model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    except Exception:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(source, **kwargs)
    model = model.to(device)
    model.eval()
    return model, processor


def residual_modules(model: Any) -> list[Any]:
    for path in ("model.language_model.layers", "model.layers", "language_model.layers"):
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
