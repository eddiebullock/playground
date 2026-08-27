"""
Map depth fractions to transformer layer indices per model family.
"""

from __future__ import annotations

from typing import Dict, List

from config import LAYER_DEPTH_FRACTIONS, MODELS


# Documented defaults when live model.config is unavailable (7B-class ~28-32 layers).
# Confirm against each checkpoint's config.json; gemma4 is 42 per text_config.
_DEFAULT_N_LAYERS: Dict[str, int] = {
    "qwen2vl": 28,
    "llavanext": 28,
    "gemma4": 42,
    "qwen3vl": 36,
    "molmo2": 32,
}


def n_layers(model_key: str, model: object = None) -> int:
    if model is not None:
        cfg = getattr(model, "config", None)
        if cfg is not None:
            for attr in ("num_hidden_layers", "n_layer", "num_layers"):
                val = getattr(cfg, attr, None)
                if isinstance(val, int) and val > 0:
                    return int(val)
        lm = getattr(model, "language_model", None)
        if lm is not None:
            return n_layers(model_key, lm)
    return _DEFAULT_N_LAYERS.get(model_key, 28)


def fraction_to_layer_index(model_key: str, fraction: float, model: object = None) -> int:
    nl = n_layers(model_key, model)
    idx = int(round(fraction * (nl - 1)))
    return max(0, min(nl - 1, idx))


def get_layer_indices(
    model_key: str,
    fractions: List[float] = None,
    model: object = None,
) -> Dict[str, int]:
    fractions = fractions if fractions is not None else list(LAYER_DEPTH_FRACTIONS)
    out: Dict[str, int] = {}
    for frac in fractions:
        key = f"frac_{frac:.3f}".replace(".", "p")
        out[key] = fraction_to_layer_index(model_key, frac, model)
    return out


def layer_label(layer_index: int, n: int) -> str:
    return f"layer_{layer_index}_of_{n}"


__all__ = ["n_layers", "fraction_to_layer_index", "get_layer_indices", "layer_label", "MODELS"]
