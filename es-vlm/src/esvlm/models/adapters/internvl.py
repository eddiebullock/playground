"""InternVL3.5 — optional, throwaway env only.

InternVL2-8B failed to load on Transformers 5.10 and was dropped. If this
adapter fails to load twice, drop InternVL3.5 and record the decision.
"""

from __future__ import annotations

from typing import Any

_FAILURES = 0


def load(
    source: str,
    *,
    revision: str | None,
    device: Any,
    dtype: Any,
    trust_remote_code: bool,
) -> tuple[Any, Any]:
    global _FAILURES
    try:
        from transformers import AutoModel, AutoProcessor

        kwargs: dict[str, Any] = {"trust_remote_code": True, "dtype": dtype}
        if revision:
            kwargs["revision"] = revision
        processor = AutoProcessor.from_pretrained(source, trust_remote_code=True, revision=revision)
        model = AutoModel.from_pretrained(source, **kwargs)
        model = model.to(device)
        model.eval()
        return model, processor
    except Exception as exc:
        _FAILURES += 1
        raise RuntimeError(
            f"InternVL load failure {_FAILURES}/2 from {source}: {exc}. "
            "Try only in a throwaway env; drop after two failures."
        ) from exc


def residual_modules(model: Any) -> list[Any]:
    for path in ("language_model.model.layers", "model.language_model.layers", "model.layers"):
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
