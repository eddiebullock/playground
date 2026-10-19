"""Removable steering / ablation hooks.

add:    h <- h + alpha * ||h_bar|| * v_hat
ablate: h <- h - (h · v_hat) v_hat

Positions: last_token | post_media | all.
``alpha=0`` in add mode must be bit-identical to the unhooked forward.
"""

from __future__ import annotations

from types import TracebackType
from typing import Any, Literal

import numpy as np

from esvlm.types import SteerMode, SteerPosition

Tensor = Any


def unit_vector(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        raise ValueError("direction has near-zero norm")
    return arr / norm


class SteeringHook:
    """Context-manager hook. Remove on exit even if the forward fails."""

    def __init__(
        self,
        module: Any,
        direction: np.ndarray,
        *,
        alpha: float = 1.0,
        mode: SteerMode = "add",
        position: SteerPosition = "last_token",
        media_end: int | None = None,
        scale: Literal["mean_norm", "unit"] = "mean_norm",
    ) -> None:
        self.module = module
        self.direction = unit_vector(direction)
        self.alpha = float(alpha)
        self.mode = mode
        self.position = position
        self.media_end = media_end
        self.scale = scale
        self._handle: Any = None
        self.hook_calls = 0
        self._dir_t: Tensor | None = None

    def __enter__(self) -> SteeringHook:
        self.attach()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.remove()

    def attach(self) -> None:
        if self._handle is not None:
            return
        import torch

        hook = self

        def _hook(_mod: Any, _inp: Any, out: Any) -> Any:
            hook.hook_calls += 1
            hidden = out[0] if isinstance(out, tuple) else out
            if not hasattr(hidden, "shape") or hidden.ndim < 2:
                return out
            patched = hook._apply(hidden, torch)
            if isinstance(out, tuple):
                return (patched,) + out[1:]
            return patched

        self._handle = self.module.register_forward_hook(_hook)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
            self._dir_t = None

    def _direction_tensor(self, hidden: Tensor, torch: Any) -> Tensor:
        if self._dir_t is None or self._dir_t.device != hidden.device or self._dir_t.dtype != hidden.dtype:
            self._dir_t = torch.as_tensor(self.direction, device=hidden.device, dtype=hidden.dtype)
        return self._dir_t

    def _apply(self, hidden: Tensor, torch: Any) -> Tensor:
        direction = self._direction_tensor(hidden, torch)
        if direction.shape[-1] != hidden.shape[-1]:
            raise ValueError(
                f"direction dim {direction.shape[-1]} != hidden {hidden.shape[-1]}"
            )
        if self.mode == "add" and self.alpha == 0.0:
            return hidden
        h = hidden
        if self.mode == "add":
            scale = self._scale(h, torch)
            delta = (self.alpha * scale) * direction
            return self._write(h, h + delta) if self.position == "all" else self._scatter(h, delta)
        # ablate: remove the projection everywhere we write
        proj = (h * direction).sum(dim=-1, keepdim=True) * direction
        if self.position == "all":
            return h - proj
        return self._mask_sub(h, proj)

    def _scale(self, hidden: Tensor, torch: Any) -> Tensor:
        if self.scale == "unit":
            return hidden.new_tensor(1.0)
        # ||h_bar|| : mean token-norm over the sequence, keep graph-free
        norms = torch.linalg.vector_norm(hidden.detach(), dim=-1).mean()
        return norms.to(hidden.dtype)

    def _scatter(self, hidden: Tensor, delta: Tensor) -> Tensor:
        h = hidden.clone()
        if self.position == "last_token":
            h[..., -1, :] = h[..., -1, :] + delta
            return h
        start = 0 if self.media_end is None else int(self.media_end)
        h[..., start:, :] = h[..., start:, :] + delta
        return h

    def _mask_sub(self, hidden: Tensor, proj: Tensor) -> Tensor:
        h = hidden.clone()
        if self.position == "last_token":
            h[..., -1, :] = h[..., -1, :] - proj[..., -1, :]
            return h
        start = 0 if self.media_end is None else int(self.media_end)
        h[..., start:, :] = h[..., start:, :] - proj[..., start:, :]
        return h

    def _write(self, hidden: Tensor, replaced: Tensor) -> Tensor:
        return replaced
