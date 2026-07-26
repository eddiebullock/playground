"""Model-family compatibility shims shared by evaluate, patching, and extraction."""

from __future__ import annotations

from typing import Any


def is_peft_model(model: Any) -> bool:
  if getattr(model, "peft_config", None):
    return True
  try:
    from peft import PeftModel

    if isinstance(model, PeftModel):
      return True
  except ImportError:
    pass
  return "Peft" in model.__class__.__name__


def apply_llavanext_compat(model: Any, model_key: str) -> None:
    """SigLIP-backed LLaVA towers may lack `.patch_size` on the vision module."""
    if model_key != "llavanext":
        return
    try:
        vt = getattr(model, "vision_tower", None)
        if vt is None and hasattr(model, "get_vision_tower"):
            vt = model.get_vision_tower()
        if vt is None and hasattr(model, "base_model"):
            inner = model.base_model
            vt = getattr(inner, "vision_tower", None)
            if vt is None and hasattr(inner, "get_vision_tower"):
                vt = inner.get_vision_tower()
        if vt is not None and not hasattr(vt, "patch_size"):
            cfg = getattr(vt, "config", None)
            ps = getattr(cfg, "patch_size", None) if cfg is not None else None
            if ps is None:
                vision_cfg = getattr(cfg, "vision_config", None) if cfg is not None else None
                ps = getattr(vision_cfg, "patch_size", None) if vision_cfg is not None else None
            if ps is not None:
                setattr(vt, "patch_size", ps)
    except Exception:
        pass
