"""Molmo2. Must run under env-molmo (Transformers 4.57)."""

from __future__ import annotations

from typing import Any

import transformers


def load(
    source: str,
    *,
    revision: str | None,
    device: Any,
    dtype: Any,
    trust_remote_code: bool,
) -> tuple[Any, Any]:
    ver = transformers.__version__
    if not ver.startswith("4.57"):
        raise RuntimeError(
            f"Molmo2 requires Transformers 4.57 (processor optional_attributes); "
            f"got {ver}. Activate env-molmo."
        )
    from transformers import AutoModelForCausalLM, AutoProcessor

    kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if revision:
        kwargs["revision"] = revision
    processor = AutoProcessor.from_pretrained(source, trust_remote_code=True, revision=revision)
    try:
        model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    except Exception:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(source, **kwargs)
    model = model.to(device)
    model.eval()
    return model, processor


def residual_modules(model: Any) -> list[Any]:
    for path in ("model.transformer.blocks", "model.layers", "transformer.blocks"):
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
