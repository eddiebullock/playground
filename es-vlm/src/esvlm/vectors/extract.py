"""Difference of means of final-token residuals per layer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from esvlm.steering.hooks import unit_vector


def difference_of_means(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """``pos`` / ``neg`` are [N, D] final-token residuals."""

    if pos.ndim != 2 or neg.ndim != 2:
        raise ValueError("expected [N, D] arrays")
    if pos.shape[1] != neg.shape[1]:
        raise ValueError("dimension mismatch")
    vec = pos.mean(axis=0) - neg.mean(axis=0)
    return unit_vector(vec)


def text_residuals(
    model: Any,
    processor: Any,
    modules: Sequence[Any],
    texts: Sequence[str],
    device: Any,
) -> np.ndarray:
    """Final-token residuals for each text at every hooked layer. [N, L, D]."""

    import torch

    acts: list[np.ndarray] = []
    for text in texts:
        inputs = processor(text=text, return_tensors="pt")
        items = dict(inputs)
        for key in list(items):
            low = key.lower()
            if any(part in low for part in ("image", "pixel", "grid")):
                items.pop(key)
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in items.items()}
        catcher = ResidualCatcher(modules)
        with torch.no_grad(), catcher:
            model(**inputs)
        acts.append(catcher.stacked())
    return np.stack(acts, axis=0)


def directions_for_layers(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """``pos`` / ``neg`` are [N, L, D]. Returns unit vectors [L, D]."""

    if pos.ndim != 3 or neg.ndim != 3:
        raise ValueError("expected [N, L, D] arrays")
    if pos.shape[1:] != neg.shape[1:]:
        raise ValueError("layer/dim mismatch")
    out = [difference_of_means(pos[:, i, :], neg[:, i, :]) for i in range(pos.shape[1])]
    return np.stack(out, axis=0)


def capture_final_token(module: Any, hidden_out: Any) -> np.ndarray:
    hidden = hidden_out[0] if isinstance(hidden_out, tuple) else hidden_out
    token = hidden[0, -1, :].detach().float().cpu().numpy()
    return np.asarray(token)


class ResidualCatcher:
    """Grab the last-token residual from a list of layer modules."""

    def __init__(self, modules: Sequence[Any]) -> None:
        self.modules = list(modules)
        self.captures: list[np.ndarray | None] = [None] * len(self.modules)
        self._handles: list[Any] = []

    def __enter__(self) -> ResidualCatcher:
        for i, mod in enumerate(self.modules):

            def _make(idx: int) -> Any:
                def _hook(_m: Any, _inp: Any, out: Any) -> None:
                    self.captures[idx] = capture_final_token(self.modules[idx], out)

                return _hook

            self._handles.append(mod.register_forward_hook(_make(i)))
        return self

    def __exit__(self, *exc: object) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def stacked(self) -> np.ndarray:
        if any(c is None for c in self.captures):
            raise RuntimeError("missing layer capture; did the forward run?")
        return np.stack([c for c in self.captures if c is not None], axis=0)
