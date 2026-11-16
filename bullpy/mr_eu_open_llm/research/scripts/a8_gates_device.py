#!/usr/bin/env python
"""A8, proposed dated deviation (NEEDS_INPUT item 14): the frozen gate script on a model split over GPUs.

The frozen `scripts/vision_gates.py` hooks every decoder layer (`LayerLab`). The hooks index each
layer's hidden state with the image positions, which are held on the device of the inputs, the first
GPU. When `device_map="auto"` splits a model over two or more GPUs, the hooks of the layers on another
GPU fail: `indices should be either on cpu or on the same device as the indexed tensor`. This module
runs the frozen script with one change, in those hooks only:

- before a frozen hook runs, the positions it indexes with are moved to the device of the tensor the
  hook receives, and they are restored after it;
- after it runs, what it captured is moved to the positions' own device, where a one-GPU run holds
  it, so every later comparison runs on the device it would run on with one GPU.

The frozen hook bodies run unchanged, and a move between devices copies values exactly, so only where
tensors are held changes, never a value. The frozen file is imported, not edited, and stays
byte-identical to its pin. Models with per-layer inputs (Gemma) are refused: the change is for
Qwen3-VL-32B only.

    python -m research.scripts.a8_gates_device --model qwen3vl --gates g2,g3,g4 ...   (the frozen arguments)
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
# As the frozen jobs, which run `python -m ...` from the project root.
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from scripts import vision_gates as vg  # noqa: E402

FROZEN_LAYER_LAB = vg.LayerLab


class DeviceLayerLab(FROZEN_LAYER_LAB):
    """The frozen LayerLab, with each frozen hook run where the tensor it indexes is held."""

    def __init__(self, adapter: Any, positions: torch.Tensor, **kwargs: Any) -> None:
        if adapter.has_per_layer:
            raise SystemExit("a8_gates_device covers models without per-layer inputs only (Qwen3-VL).")
        super().__init__(adapter, positions, **kwargs)
        self._home_positions = positions
        self._home = positions.device

    def _run_on(self, device: torch.device, frozen: Callable[[], Any]) -> Any:
        self.positions = self._home_positions.to(device)
        try:
            return frozen()
        finally:
            self.positions = self._home_positions

    def _pre(self, index: int) -> Callable[..., Any]:
        frozen = super()._pre(index)

        def hook(module: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
            hidden = args[0] if args else kwargs["hidden_states"]
            out = self._run_on(hidden.device, lambda: frozen(module, args, kwargs))
            if index in self.inputs:
                self.inputs[index] = self.inputs[index].to(self._home)
            if torch.is_tensor(self.masks.get(index)):
                self.masks[index] = self.masks[index].to(self._home)
            return out

        return hook

    def _post(self, index: int) -> Callable[..., None]:
        frozen = super()._post(index)

        def hook(module: Any, args: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            self._run_on(hidden.device, lambda: frozen(module, args, output))
            self.outputs[index] = self.outputs[index].to(self._home)
            self.last[index] = self.last[index].to(self._home)

        return hook


@contextlib.contextmanager
def device_hooks() -> Iterator[None]:
    """The frozen script's functions look LayerLab up in its module at call time; swap it, then restore."""
    saved = vg.LayerLab
    vg.LayerLab = DeviceLayerLab
    try:
        yield
    finally:
        vg.LayerLab = saved


def run(argv: Sequence[str]) -> None:
    """The frozen main() with the device-aware hooks, as `python -m scripts.vision_gates <argv>` runs it."""
    saved = sys.argv
    sys.argv = [vg.__file__, *argv]
    try:
        with device_hooks():
            vg.main()
    finally:
        sys.argv = saved


if __name__ == "__main__":
    run(sys.argv[1:])
