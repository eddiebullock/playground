"""
Hold operators and the exit cut for amendment A7, on top of the frozen gate code.

Everything here goes through scripts/vision_gates.py unchanged: the exit cut is its
`_cut_fn` with every route, holds use `vision_intervention` and `LayerLab`, and token holds
use `swap_tokens`. Nothing is removed from the model state except by the exit cut, which is
used for validity only (C2 and C3). See research/PREREG_A7_DRAFT.md, "Three kinds of cut".

Blocks, from the design block:
- qwen3vl: A is the main route, B the three deepstack routes, all at the vision exit.
- gemma4: A is the hidden state at image positions entering language model layer 0, B the
  per-layer input at image positions in every layer. Both are the language model's entry.

Guard. Any site in the language model past the entry is refused, and Gemma image-position
sites at layer 24 or later are refused by name, so an incomplete cut cannot be run by
mistake (the Qwen layer 4 pilot and Gemma's guaranteed false null at L24).

The Session class runs one forward per readout with any of these active. It takes an
`encode` and a `readout` callable, so the same code serves the cluster (A2's blind readout)
and the CPU toys of the Level 1 recovery.
"""

from __future__ import annotations

import re
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.vision_gates import (  # noqa: E402
    Adapter,
    LayerLab,
    LayoutError,
    _cut_fn,
    image_positions,
    strip_pixels,
    swap_tokens,
    vision_intervention,
)

GEMMA_FIRST_KV_SHARED = 24  # STATE.md: image-position cuts here leave the output unchanged


class SiteRefused(ValueError):
    """A site that is not a complete cut, or a block of one, at the language model entry."""


ENTRY_SITES = {
    "qwen3vl": ("exit", "main", "deepstack", "tokens"),
    "gemma4": ("exit", "hidden_L0", "per_layer_all", "tokens"),
}
_LAYER = re.compile(r"L(\d+)")


def check_site(model_key: str, site: str) -> str:
    """The site if it is at the vision exit or the language model entry; otherwise refuses."""
    allowed = ENTRY_SITES.get(model_key)
    if allowed is None:
        raise SiteRefused(f"A7 has no sites for {model_key}.")
    layers = [int(x) for x in _LAYER.findall(site)]
    if model_key == "gemma4" and any(k >= GEMMA_FIRST_KV_SHARED for k in layers):
        raise SiteRefused(f"{site}: Gemma layers 24 and later reuse stored keys and values; the cut is a guaranteed false null.")
    if any(k > 0 for k in layers):
        raise SiteRefused(f"{site}: past the language model entry, text has already read the image; no such cut is complete.")
    if site not in allowed:
        raise SiteRefused(f"{site} is not an A7 site on {model_key}; allowed: {allowed}.")
    return site


def block_sites(model_key: str, routes: Mapping[str, Any]) -> Dict[str, Tuple[str, ...]]:
    """What each block holds. On Qwen the routes are named in the design block."""
    if model_key == "qwen3vl":
        block_b = tuple(routes["B"]) if isinstance(routes["B"], (list, tuple)) else (routes["B"],)
        return {"A": (routes["A"],), "B": block_b}
    if model_key == "gemma4":
        return {"A": ("hidden_L0",), "B": ("per_layer_all",)}
    raise SiteRefused(f"A7 has no blocks for {model_key}.")


@dataclass
class State:
    """What a forward put at the site, for one image: exit routes and the entry streams."""

    routes: Dict[str, torch.Tensor] = field(default_factory=dict)
    hidden0: Optional[torch.Tensor] = None
    per_layer: Dict[int, torch.Tensor] = field(default_factory=dict)


@contextmanager
def capturing(adapter: Adapter, positions: torch.Tensor) -> Iterator[State]:
    """Capture exit routes and the entry streams during one or more forwards.

    Every forward of a blind readout sends the same image, and the captured tensors are
    position-local functions of it, so later forwards overwrite them with equal values.
    """
    state = State()
    lab = LayerLab(adapter, positions, capture=True, input_layers=[0], keep_outputs=False)
    with vision_intervention(adapter, None, capture=state.routes), lab:
        yield state
    state.hidden0 = lab.inputs.get(0)
    if adapter.has_per_layer:
        state.per_layer = {k: v[positions].clone() for k, v in lab.per_layer.items()}


def _route_hold(names: Sequence[str], source: State) -> Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    def fn(routes: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = dict(routes)
        for name in names:
            if name not in routes or name not in source.routes:
                raise LayoutError(f"Route {name} missing: {sorted(routes)} versus {sorted(source.routes)}")
            if routes[name].shape != source.routes[name].shape:
                raise LayoutError(f"{name}: {routes[name].shape} versus {source.routes[name].shape}")
            out[name] = source.routes[name]
        return out

    return fn


@contextmanager
def holding(
    adapter: Adapter,
    model_key: str,
    positions: torch.Tensor,
    blocks: Iterable[str],
    source: State,
    route_blocks: Mapping[str, Any],
) -> Iterator[None]:
    """Hold the named blocks at the source image's values, whatever pixels are sent."""
    chosen = sorted(set(blocks))
    if not chosen or any(b not in ("A", "B") for b in chosen):
        raise SiteRefused(f"Blocks must be A, B or both, not {chosen}.")
    sites = block_sites(model_key, route_blocks)
    for block in chosen:
        for site in sites[block]:
            check_site(model_key, "deepstack" if site.startswith("deepstack") else site)
    with ExitStack() as stack:
        if model_key == "qwen3vl":
            names = [name for block in chosen for name in sites[block]]
            stack.enter_context(vision_intervention(adapter, _route_hold(names, source)))
        else:
            hidden: Dict[int, torch.Tensor] = {}
            per_layer: Dict[int, torch.Tensor] = {}
            if "A" in chosen:
                if source.hidden0 is None:
                    raise LayoutError("No captured layer 0 state to hold.")
                hidden = {0: source.hidden0}
            if "B" in chosen:
                n_layers = len(adapter.lm_layers)
                if sorted(source.per_layer) != list(range(n_layers)):
                    raise LayoutError(f"Per-layer input captured for {len(source.per_layer)} of {n_layers} layers.")
                per_layer = dict(source.per_layer)
            stack.enter_context(LayerLab(adapter, positions, replace_hidden=hidden, replace_per_layer=per_layer))
        yield


@contextmanager
def holding_tokens(adapter: Adapter, indices: Sequence[int], source: State) -> Iterator[None]:
    """Hold the chosen exit tokens, in every route, at the source image's values."""
    idx = np.asarray(sorted(set(int(i) for i in indices)), dtype=np.int64)

    def fn(routes: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return swap_tokens(routes, source.routes, idx)

    with vision_intervention(adapter, fn):
        yield


def exit_cut(adapter: Adapter, placeholder: torch.Tensor):
    """Every route at the exit replaced by what the image-free forward has in its place (G3)."""
    return vision_intervention(adapter, _cut_fn(adapter, placeholder))


Encode = Callable[..., Dict[str, Any]]
Readout = Callable[[Dict[str, Any]], np.ndarray]


class Session:
    """Readouts of one model under the A7 operations. One instance per model and job.

    encode(clip, condition) returns model inputs; readout(inputs) returns the 106 unrounded
    word logprobs as float64. vision_only(inputs), if given, runs only the vision stack.
    """

    def __init__(
        self,
        adapter: Adapter,
        model_key: str,
        encode: Encode,
        readout: Readout,
        route_blocks: Mapping[str, Any],
        vision_only: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> None:
        if model_key not in ENTRY_SITES:
            raise SiteRefused(f"A7 has no sites for {model_key}.")
        self.adapter = adapter
        self.model_key = model_key
        self.encode = encode
        self.readout_fn = readout
        self.route_blocks = route_blocks
        self.vision_only = vision_only

    def positions(self, inputs: Dict[str, Any]) -> torch.Tensor:
        return image_positions(self.adapter, inputs)

    def _read(self, inputs: Dict[str, Any]) -> np.ndarray:
        values = np.asarray(self.readout_fn(inputs), dtype=np.float64)
        return values

    def native(self, inputs: Dict[str, Any]) -> Tuple[np.ndarray, State]:
        with capturing(self.adapter, self.positions(inputs)) as state:
            values = self._read(inputs)
        return values, state

    def plain(self, inputs: Dict[str, Any]) -> np.ndarray:
        return self._read(inputs)

    def image_free(self, inputs: Dict[str, Any]) -> Tuple[np.ndarray, torch.Tensor]:
        """A2's image-free row, and the placeholder the exit cut writes (G3's definition)."""
        positions = self.positions(inputs)
        free = strip_pixels(inputs, self.adapter.pixel_keys)
        lab = LayerLab(self.adapter, positions, capture=True, input_layers=[0], keep_outputs=False)
        with lab:
            values = self._read(free)
        return values, lab.inputs[0]

    def exit_cut(self, inputs: Dict[str, Any], placeholder: torch.Tensor) -> np.ndarray:
        check_site(self.model_key, "exit")
        with exit_cut(self.adapter, placeholder):
            return self._read(inputs)

    def hold(self, inputs: Dict[str, Any], blocks: Iterable[str], source: State) -> np.ndarray:
        with holding(self.adapter, self.model_key, self.positions(inputs), blocks, source, self.route_blocks):
            return self._read(inputs)

    def token_hold(self, inputs: Dict[str, Any], indices: Sequence[int], source: State) -> np.ndarray:
        check_site(self.model_key, "tokens")
        with holding_tokens(self.adapter, indices, source):
            return self._read(inputs)

    def exit_routes(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Exit routes for one image, by the vision stack alone when that is available."""
        routes: Dict[str, torch.Tensor] = {}
        with torch.no_grad(), vision_intervention(self.adapter, None, capture=routes):
            if self.vision_only is not None:
                self.vision_only(inputs)
            else:
                self._read(inputs)
        return routes


def relative_change(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> float:
    """Relative L2 of b from a over the given tensors: the rider "block change under the edit"."""
    num = sum(float((b[k].float() - a[k].float()).norm() ** 2) for k in a)
    den = sum(float(a[k].float().norm() ** 2) for k in a)
    return float(np.sqrt(num / den)) if den > 0 else float("nan")


def block_tensors(model_key: str, state: State, route_blocks: Mapping[str, Any]) -> Dict[str, Dict[str, torch.Tensor]]:
    """The tensors that make up each block, for the relative change rider."""
    if model_key == "qwen3vl":
        sites = block_sites(model_key, route_blocks)
        return {b: {n: state.routes[n] for n in names} for b, names in sites.items()}
    return {
        "A": {"hidden_L0": state.hidden0},
        "B": {f"per_layer_{k}": v for k, v in state.per_layer.items()},
    }


def token_change(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> np.ndarray:
    """Per exit token, the L2 norm of its change, summed over routes (the token map measure)."""
    total = None
    for name in a:
        diff = (b[name].float() - a[name].float()).reshape(a[name].shape[0], -1).norm(dim=1)
        total = diff if total is None else total + diff
    return total.cpu().numpy().astype(np.float64)
