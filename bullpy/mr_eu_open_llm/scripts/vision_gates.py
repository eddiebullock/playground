"""
Validity gates G2 to G4 for causal work on vision paths.

G2 image-token invariance: same clip, different option lists, compare activations at
   image positions per layer.
G3 hook validity: replaying a captured value changes logits by zero, and cutting every
   vision route gives the image-free forward. Single-route cuts show each route is live.
G4 positive control: swap image tokens with a different-label clip at 10, 25, 50, 100
   percent, and measure the shift in answer logprobs.

All gate prompts rotate the options so the target is never first (amendment A1). The
rules in GATE_RULES were fixed before any model was run; do not change them after
seeing results.

Definitions used throughout:
- "route": a tensor leaving the vision stack for the LM. Qwen3-VL has main plus three
  DeepStack routes, Gemma 4 and Molmo2 have main only. Gemma's per-layer signal is
  derived from main inside the LM, so it is patched at the decoder layers.
- "image-free forward": same input_ids, attention mask and layout tensors with pixels
  removed, so placeholders and position ids stay. A prompt with no image tokens has a
  different token sequence and cannot match exactly; it is reported for information.
- "answer": each option is scored by joint logprob of its tokens after the prefill
  "EMOTION:", the format the prompt requests.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from paper_v2.option_score import ABSTAIN_LABEL, rotate_options

SEED = 42
N_CLIPS = 40
DOSES = (0.10, 0.25, 0.50, 1.00)
PREFILL = "EMOTION:"
MODEL_ORDER = ("qwen3vl", "gemma4", "molmo2")

GATE_RULES: Dict[str, Any] = {
    "fixed_before_any_model_run": True,
    "G2": {
        "pass": "max relative L2 at image positions, over layers and clips, is within tolerance",
        "relative_l2_tolerance": {"float32": 1e-4, "bfloat16": 1e-2},
        "applies_when": "ids up to the last image token match and the mask blocks image-to-text",
        "exact_zero_is_reported_separately": True,
    },
    "G3": {
        "pass": "max abs last-position logit difference within tolerance for every self-patch, "
        "for cut versus image-free, and for cut(clip A) versus cut(clip B)",
        "logit_tolerance": {"float32": 1e-3, "bfloat16": "2 ulp at the largest logit magnitude"},
    },
    "G4": {
        "primary_metric": "RMS change in mean-centred option joint logprob, in nats, per clip",
        "monotonic": "mean strictly increases over 10, 25, 50, 100 percent and the actor "
        "bootstrap 95 percent CI of (100 percent minus 10 percent) excludes 0",
        "large": "mean primary metric at 100 percent >= 1.0 nat AND argmax flip rate at "
        "100 percent >= 0.25",
        "hook_check": "the 100 percent swap equals a native forward on the donor clip",
        "threshold_note": "the brief gave no number for 'large'; these were set by the "
        "implementer before running and must not be tuned",
    },
}


class LayoutError(RuntimeError):
    """Own clip and donor clip do not share a token layout, so a token swap is undefined."""


# ---------------------------------------------------------------------------
# Pure helpers: sample, prompts, doses, metrics, verdicts
# ---------------------------------------------------------------------------


def stable_int(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def gate_options(options: Sequence[str], rotation: int, target: str) -> List[str]:
    """Rotate named options; refuses any order that lists the target first."""
    rotated = rotate_options(options, rotation)
    if rotated[0] == target:
        raise ValueError(f"Target-first order is not allowed in gate prompts: {rotated}")
    if rotated[-1] != ABSTAIN_LABEL:
        raise ValueError(f"Abstain option must stay last: {rotated}")
    return rotated


def rotation_for_position(position: int) -> int:
    """Cycle rotations 1 to 4. Rotation 0 is the target-first order."""
    return 1 + int(position) % 4


def _partner_pool(item: Dict[str, Any], items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        c
        for c in items
        if c["trial_id"] != item["trial_id"]
        and c["label"] != item["label"]
        and c["actor"] != item["actor"]
        and c["signature"] == item["signature"]
    ]


def select_sample(
    items: Sequence[Dict[str, Any]], *, n: int = N_CLIPS, seed: int = SEED
) -> List[Dict[str, Any]]:
    """Fix a clip sample, balanced by actor, using no model output.

    Each item needs trial_id, label, actor, signature and options. Quotas differ by at
    most one across actors. Clips are drawn round-robin, so any prefix is actor-diverse.
    Each clip gets a rotation and a donor with a different label, a different actor and
    the same frame count and size. Donors whose label is a foil in this clip's options
    are preferred, which makes the direction of the swap effect predictable.
    """
    eligible = [i for i in items if _partner_pool(i, items)]
    by_actor: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in eligible:
        by_actor[item["actor"]].append(item)
    actors = sorted(by_actor, key=lambda a: stable_int(seed, "actor", a))
    if sum(len(v) for v in by_actor.values()) < n:
        raise ValueError(f"Only {len(eligible)} eligible clips for a sample of {n}")

    quota = {a: n // len(actors) + (1 if i < n % len(actors) else 0) for i, a in enumerate(actors)}
    chosen: List[Dict[str, Any]] = []
    label_counts: Counter = Counter()
    used = set()
    while len(chosen) < n:
        progressed = False
        for actor in actors:
            if len(chosen) >= n:
                break
            free = [i for i in by_actor[actor] if i["trial_id"] not in used]
            if quota[actor] <= 0 or not free:
                continue
            pick = min(
                free,
                key=lambda i: (label_counts[i["label"]], stable_int(seed, "clip", i["trial_id"])),
            )
            used.add(pick["trial_id"])
            label_counts[pick["label"]] += 1
            quota[actor] -= 1
            chosen.append(pick)
            progressed = True
        if not progressed:
            raise ValueError("Actor quotas cannot be met; check the eligible pool")

    donor_use: Counter = Counter()
    out: List[Dict[str, Any]] = []
    for position, item in enumerate(chosen):
        pool = _partner_pool(item, items)
        named = set(item["options"]) - {ABSTAIN_LABEL, item["label"]}
        donor = min(
            pool,
            key=lambda c: (
                c["label"] not in named,
                donor_use[c["trial_id"]],
                stable_int(seed, "donor", item["trial_id"], c["trial_id"]),
            ),
        )
        donor_use[donor["trial_id"]] += 1
        rotation = rotation_for_position(position)
        out.append(
            {
                **item,
                "rotation": rotation,
                "gate_options": gate_options(item["options"], rotation, item["label"]),
                "donor_id": donor["trial_id"],
                "donor_label": donor["label"],
                "donor_actor": donor["actor"],
                "donor_label_in_options": donor["label"] in named,
                "foreign_options": gate_options(donor["options"], rotation, donor["label"]),
                "second_rotation_options": gate_options(
                    item["options"], rotation_for_position(position + 1), item["label"]
                ),
            }
        )
    return out


def nested_dose_indices(n_tokens: int, doses: Sequence[float], key: Any) -> Dict[float, np.ndarray]:
    """Token subsets for each dose. Subsets are nested, so a curve is one path per clip."""
    perm = np.random.default_rng(stable_int(SEED, "dose", key) % 2**32).permutation(n_tokens)
    out: Dict[float, np.ndarray] = {}
    for dose in doses:
        count = n_tokens if dose >= 1.0 else max(1, int(round(n_tokens * dose)))
        out[float(dose)] = np.sort(perm[:count])
    return out


def centred(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr - arr.mean()


def softmax(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    exps = np.exp(arr - arr.max())
    return exps / exps.sum()


def answer_shift(base: Sequence[float], other: Sequence[float]) -> Dict[str, float]:
    """Shift between two option logprob vectors; means are removed as only gaps matter."""
    delta = centred(other) - centred(base)
    return {
        "rms_nats": float(np.sqrt(np.mean(delta**2))),
        "total_variation": float(0.5 * np.abs(softmax(other) - softmax(base)).sum()),
        "flipped": float(int(np.argmax(other)) != int(np.argmax(base))),
    }


def actor_bootstrap_ci(
    values: Sequence[float], actors: Sequence[str], *, draws: int = 10000, seed: int = SEED
) -> Tuple[float, float]:
    """95 percent CI of the mean, resampling actors, which are the independent units."""
    arr = np.asarray(values, dtype=np.float64)
    groups = defaultdict(list)
    for value, actor in zip(arr, actors):
        groups[actor].append(value)
    sums = np.array([np.sum(v) for v in groups.values()])
    counts = np.array([len(v) for v in groups.values()], dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(draws, len(sums)))
    means = sums[idx].sum(1) / counts[idx].sum(1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def logit_tolerance(dtype_name: str, max_abs_logit: float) -> float:
    if dtype_name == "float32":
        return float(GATE_RULES["G3"]["logit_tolerance"]["float32"])
    exponent = math.floor(math.log2(max(float(max_abs_logit), 1.0)))
    return 2.0 * 2.0 ** (exponent - 7)


def tensor_distance(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    delta = (a.float() - b.float()).reshape(-1)
    return {
        "max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
        "l2": float(delta.norm()),
        "rel_l2": float(delta.norm() / b.float().norm().clamp_min(1e-12)),
    }


def g2_verdict(summary: Dict[str, Any], dtype_name: str) -> Dict[str, Any]:
    tolerance = GATE_RULES["G2"]["relative_l2_tolerance"][dtype_name]
    worst = max(row["max_rel_l2"] for row in summary["per_layer"].values())
    return {
        "dtype": dtype_name,
        "tolerance": tolerance,
        "worst_rel_l2": worst,
        "exact_zero": all(row["max_abs"] == 0.0 for row in summary["per_layer"].values()),
        "mask_permits_invariance": summary["mask_permits_invariance"],
        "pass": bool(worst <= tolerance and summary["mask_permits_invariance"]),
    }


def g3_verdict(clips: Sequence[Dict[str, Any]], dtype_name: str) -> Dict[str, Any]:
    checks: Dict[str, Dict[str, Any]] = {}
    names = sorted({name for clip in clips for name in clip["required"]})
    for name in names:
        rows = [clip["required"][name] for clip in clips if name in clip["required"]]
        worst = max(row["max_abs"] for row in rows)
        top = max(clip["max_abs_logit"] for clip in clips)
        tolerance = logit_tolerance(dtype_name, top)
        checks[name] = {
            "worst_max_abs": worst,
            "tolerance": tolerance,
            "exact_zero": worst == 0.0,
            "pass": bool(worst <= tolerance),
        }
    return {"dtype": dtype_name, "checks": checks, "pass": all(c["pass"] for c in checks.values())}


def g4_verdict(clips: Sequence[Dict[str, Any]], *, draws: int = 10000) -> Dict[str, Any]:
    actors = [clip["actor"] for clip in clips]
    curve: Dict[str, Dict[str, Any]] = {}
    for dose in DOSES:
        key = f"{dose:.2f}"
        row: Dict[str, Any] = {}
        for metric in ("rms_nats", "total_variation", "flipped"):
            values = [clip["shift"][key][metric] for clip in clips]
            row[metric] = float(np.mean(values))
            row[f"{metric}_ci95"] = actor_bootstrap_ci(values, actors, draws=draws)
        curve[key] = row
    means = [curve[f"{d:.2f}"]["rms_nats"] for d in DOSES]
    span = [
        clip["shift"]["1.00"]["rms_nats"] - clip["shift"]["0.10"]["rms_nats"] for clip in clips
    ]
    span_ci = actor_bootstrap_ci(span, actors, draws=draws)
    strictly_up = all(b > a for a, b in zip(means, means[1:]))
    monotonic = bool(strictly_up and span_ci[0] > 0)
    top = curve["1.00"]
    large = bool(top["rms_nats"] >= 1.0 and top["flipped"] >= 0.25)
    native = max(clip["native_vs_swap_max_abs"] for clip in clips)
    native_ok = bool(native <= 1e-3)

    directed = [c["target_vs_donor_logodds_shift"] for c in clips if c["donor_label_in_options"]]
    secondary = {
        "n_clips_donor_label_in_options": len(directed),
        "mean_shift_toward_donor_label_nats": float(np.mean(directed)) if directed else None,
        "accuracy_own_clip": float(np.mean([c["correct_base"] for c in clips])),
        "accuracy_full_swap": float(np.mean([c["correct_full_swap"] for c in clips])),
        "mean_image_vs_free_rms_nats": float(np.mean([c["image_free_rms_nats"] for c in clips])),
    }
    return {
        "n_clips": len(clips),
        "n_actors": len(set(actors)),
        "curve": curve,
        "span_100_minus_10_ci95": span_ci,
        "monotonic": monotonic,
        "large": large,
        "native_vs_swap_worst_max_abs": native,
        "native_equivalence_ok": native_ok,
        "secondary": secondary,
        "pass": bool(monotonic and large and native_ok),
    }


# ---------------------------------------------------------------------------
# Model adapters
# ---------------------------------------------------------------------------


@dataclass
class Adapter:
    """Where one architecture keeps its vision routes and its LM layers."""

    key: str
    vision_module: torch.nn.Module
    lm_layers: Any
    image_token_id: int
    additive: bool
    pixel_keys: Tuple[str, ...]
    read_routes: Callable[[Any], Dict[str, torch.Tensor]]
    write_routes: Callable[[Any, Dict[str, torch.Tensor]], Any]
    has_per_layer: bool = False


def _tensor_routes(output: Any) -> Dict[str, torch.Tensor]:
    return {"main": output}


def _tensor_write(output: Any, routes: Dict[str, torch.Tensor]) -> Any:
    return routes["main"].to(output.device, output.dtype).reshape(output.shape)


def _qwen_routes(output: Any) -> Dict[str, torch.Tensor]:
    routes = {"main": output.pooler_output}
    for i, tensor in enumerate(output.deepstack_features):
        routes[f"deepstack_{i}"] = tensor
    return routes


def _qwen_write(output: Any, routes: Dict[str, torch.Tensor]) -> Any:
    output.pooler_output = routes["main"].to(output.pooler_output.device, output.pooler_output.dtype)
    output.deepstack_features = [
        routes[f"deepstack_{i}"].to(t.device, t.dtype)
        for i, t in enumerate(output.deepstack_features)
    ]
    return output


def adapter_for(model_key: str, model: Any) -> Adapter:
    inner = model.model
    if model_key == "qwen3vl":
        return Adapter(
            key=model_key,
            vision_module=inner.visual,
            lm_layers=inner.language_model.layers,
            image_token_id=int(model.config.image_token_id),
            additive=False,
            pixel_keys=("pixel_values",),
            read_routes=_qwen_routes,
            write_routes=_qwen_write,
        )
    if model_key == "gemma4":
        return Adapter(
            key=model_key,
            vision_module=inner.embed_vision,
            lm_layers=inner.language_model.layers,
            image_token_id=int(model.config.image_token_id),
            additive=False,
            pixel_keys=("pixel_values", "image_position_ids"),
            read_routes=_tensor_routes,
            write_routes=_tensor_write,
            has_per_layer=True,
        )
    if model_key == "molmo2":
        return Adapter(
            key=model_key,
            vision_module=inner.vision_backbone,
            lm_layers=inner.transformer.blocks,
            image_token_id=int(model.config.image_patch_id),
            additive=True,
            pixel_keys=("pixel_values", "image_token_pooling", "image_grids", "image_num_crops"),
            read_routes=_tensor_routes,
            write_routes=_tensor_write,
        )
    raise ValueError(f"No vision adapter for {model_key}")


def image_positions(adapter: Adapter, inputs: Dict[str, Any]) -> torch.Tensor:
    positions = (inputs["input_ids"][0] == adapter.image_token_id).nonzero().squeeze(-1)
    if positions.numel() == 0:
        raise RuntimeError(f"No image tokens found for {adapter.key}")
    return positions


def strip_pixels(inputs: Dict[str, Any], pixel_keys: Iterable[str]) -> Dict[str, Any]:
    drop = set(pixel_keys)
    return {k: v for k, v in inputs.items() if k not in drop}


def append_tokens(inputs: Dict[str, Any], ids: Sequence[int]) -> Dict[str, Any]:
    """Add assistant text after the generation prompt, extending sequence tensors."""
    if not ids:
        return dict(inputs)
    out = dict(inputs)
    input_ids = inputs["input_ids"]
    extra = torch.tensor([list(ids)], device=input_ids.device, dtype=input_ids.dtype)
    out["input_ids"] = torch.cat([input_ids, extra], dim=1)
    for key, fill in (("attention_mask", 1), ("mm_token_type_ids", 0), ("token_type_ids", 0)):
        value = inputs.get(key)
        if value is not None:
            pad = torch.full_like(extra, fill, dtype=value.dtype, device=value.device)
            out[key] = torch.cat([value, pad], dim=1)
    return out


def same_layout(a: Dict[str, Any], b: Dict[str, Any], *, ignore: Iterable[str] = ()) -> bool:
    """True when both inputs hold the same tensors apart from pixels."""
    skip = {"pixel_values", "pixel_values_videos", *ignore}
    if set(a) != set(b):
        return False
    for key, value in a.items():
        if key in skip or not torch.is_tensor(value):
            continue
        if value.shape != b[key].shape or not torch.equal(value, b[key]):
            return False
    return True


# ---------------------------------------------------------------------------
# Interventions
# ---------------------------------------------------------------------------

RouteFn = Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]


@contextmanager
def vision_intervention(
    adapter: Adapter, fn: Optional[RouteFn] = None, capture: Optional[Dict[str, torch.Tensor]] = None
):
    """Capture or rewrite vision routes as they leave the vision stack."""

    def hook(_module: Any, _args: Any, output: Any) -> Any:
        routes = adapter.read_routes(output)
        if capture is not None:
            capture.update({k: v.detach().clone() for k, v in routes.items()})
        if fn is None:
            return None
        return adapter.write_routes(output, fn(routes))

    handle = adapter.vision_module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


class LayerLab:
    """Capture or rewrite LM decoder state at image positions, per layer.

    inputs[k] is the state entering layer k and outputs[k] the state leaving it, both at
    image positions. per_layer[k] is Gemma's per-layer input over the whole sequence.
    replace_hidden and replace_per_layer map a layer index to a tensor for image
    positions, written after capture.
    """

    def __init__(
        self,
        adapter: Adapter,
        positions: torch.Tensor,
        *,
        capture: bool = False,
        input_layers: Optional[Iterable[int]] = None,
        keep_outputs: bool = True,
        replace_hidden: Optional[Dict[int, torch.Tensor]] = None,
        replace_per_layer: Optional[Dict[int, torch.Tensor]] = None,
    ) -> None:
        self.adapter = adapter
        self.positions = positions
        self.capture = capture
        self.input_layers = None if input_layers is None else set(input_layers)
        self.keep_outputs = keep_outputs
        self.replace_hidden = replace_hidden or {}
        self.replace_per_layer = replace_per_layer or {}
        self.inputs: Dict[int, torch.Tensor] = {}
        self.outputs: Dict[int, torch.Tensor] = {}
        self.last: Dict[int, torch.Tensor] = {}
        self.per_layer: Dict[int, torch.Tensor] = {}
        self.masks: Dict[int, Any] = {}
        self._handles: List[Any] = []

    def _pre(self, index: int) -> Callable[..., Any]:
        def hook(_module: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
            new_args = list(args)
            hidden = new_args[0] if new_args else kwargs["hidden_states"]
            if self.capture:
                if self.input_layers is None or index in self.input_layers:
                    self.inputs[index] = hidden[0, self.positions].detach().clone()
                self.masks[index] = kwargs.get("attention_mask")
            replacement = self.replace_hidden.get(index)
            if replacement is not None:
                hidden = hidden.clone()
                hidden[0, self.positions] = replacement.to(hidden.device, hidden.dtype)
                if new_args:
                    new_args[0] = hidden
                else:
                    kwargs["hidden_states"] = hidden
            if self.adapter.has_per_layer:
                in_args = len(new_args) > 1
                per_layer = new_args[1] if in_args else kwargs.get("per_layer_input")
                if self.capture:
                    self.per_layer[index] = per_layer[0].detach().clone()
                patch = self.replace_per_layer.get(index)
                if patch is not None:
                    per_layer = per_layer.clone()
                    per_layer[0, self.positions] = patch.to(per_layer.device, per_layer.dtype)
                    if in_args:
                        new_args[1] = per_layer
                    else:
                        kwargs["per_layer_input"] = per_layer
            return tuple(new_args), kwargs

        return hook

    def _post(self, index: int) -> Callable[..., None]:
        def hook(_module: Any, _args: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            self.outputs[index] = hidden[0, self.positions].detach().clone()
            self.last[index] = hidden[0, -1].detach().clone()

        return hook

    def __enter__(self) -> "LayerLab":
        for index, layer in enumerate(self.adapter.lm_layers):
            self._handles.append(
                layer.register_forward_pre_hook(self._pre(index), with_kwargs=True)
            )
            if self.capture and self.keep_outputs:
                self._handles.append(layer.register_forward_hook(self._post(index)))
        return self

    def __exit__(self, *_exc: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def forward_logits(model: Any, inputs: Dict[str, Any]) -> torch.Tensor:
    with torch.no_grad():
        return model(**inputs, use_cache=False).logits


def last_logits(model: Any, inputs: Dict[str, Any]) -> torch.Tensor:
    return forward_logits(model, inputs)[0, -1].float().clone()


def swap_tokens(
    own: Dict[str, torch.Tensor], donor: Dict[str, torch.Tensor], indices: np.ndarray
) -> Dict[str, torch.Tensor]:
    """Replace the chosen tokens in every route, so no route keeps the own clip's copy."""
    if set(own) != set(donor):
        raise LayoutError(f"Route mismatch: {sorted(own)} versus {sorted(donor)}")
    out: Dict[str, torch.Tensor] = {}
    idx = torch.as_tensor(indices, dtype=torch.long)
    for name, tensor in own.items():
        if tensor.shape != donor[name].shape:
            raise LayoutError(f"{name}: {tensor.shape} versus {donor[name].shape}")
        swapped = tensor.clone()
        target = idx.to(swapped.device)
        swapped[target] = donor[name].to(swapped.device, swapped.dtype)[target]
        out[name] = swapped
    return out


def mask_report(mask: Any, positions: torch.Tensor) -> Dict[str, Any]:
    """Can any image query attend past the image block, according to the mask?"""
    if mask is None:
        return {"kind": "implicit_causal", "image_can_read_later": False}
    if not torch.is_tensor(mask) or mask.dim() != 4:
        return {"kind": type(mask).__name__, "image_can_read_later": None}
    allowed = mask if mask.dtype == torch.bool else (mask == 0)
    after = int(positions.max()) + 1
    block = allowed[0, 0][positions][:, after:]
    return {"kind": f"tensor_{mask.dtype}", "image_can_read_later": bool(block.any())}


def first_changed_layer(a: Dict[int, torch.Tensor], b: Dict[int, torch.Tensor]) -> Optional[int]:
    for index in sorted(a):
        if not torch.equal(a[index], b[index]):
            return index
    return None


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


class GateContext:
    """What a gate needs from a model: inputs for (vision clip, options), token ids."""

    model: Any
    adapter: Adapter

    def encode(
        self, vision_clip: str, options: Sequence[str], extra_ids: Sequence[int] = ()
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def encode_text_only(self, options: Sequence[str], extra_ids: Sequence[int]) -> Optional[Dict[str, Any]]:
        return None

    def prefill_ids(self) -> List[int]:
        raise NotImplementedError

    def label_ids(self, option: str) -> List[int]:
        raise NotImplementedError

    def end_clip(self) -> None:
        return None


def run_g2_clip(ctx: GateContext, clip: Dict[str, Any]) -> Dict[str, Any]:
    prefill = ctx.prefill_ids()

    def capture(options: Sequence[str]) -> Tuple[LayerLab, torch.Tensor]:
        inputs = ctx.encode(clip["trial_id"], options, prefill)
        positions = image_positions(ctx.adapter, inputs)
        # Layer 0 input is the embedding; later inputs repeat earlier outputs. On Qwen a
        # capture of every layer costs tens of GB on the longest clips.
        lab = LayerLab(ctx.adapter, positions, capture=True, input_layers=[0])
        with lab:
            forward_logits(ctx.model, inputs)
        return lab, inputs["input_ids"][0]

    reference, ref_ids = capture(clip["gate_options"])
    positions = reference.positions
    masks = {str(i): mask_report(m, positions) for i, m in list(reference.masks.items())[:2]}
    last_image = int(positions.max())
    comparisons: Dict[str, Any] = {}
    alternatives = (
        ("second_rotation", clip["second_rotation_options"]),
        ("foreign_option_set", clip["foreign_options"]),
    )
    for name, options in alternatives:
        other, other_ids = capture(options)
        same_prefix = bool(
            other_ids.numel() > last_image
            and torch.equal(ref_ids[: last_image + 1], other_ids[: last_image + 1])
        )
        same_positions = bool(torch.equal(positions, other.positions))
        layers: Dict[str, Dict[str, float]] = {}
        if same_positions:
            layers["embed"] = tensor_distance(other.inputs[0], reference.inputs[0])
            for index in sorted(reference.outputs):
                layers[str(index)] = tensor_distance(other.outputs[index], reference.outputs[index])
        last_control = {
            str(i): tensor_distance(other.last[i], reference.last[i])["rel_l2"]
            for i in sorted(reference.last)
        }
        comparisons[name] = {
            "same_prefix_through_image": same_prefix,
            "same_image_positions": same_positions,
            "same_total_length": bool(other_ids.numel() == ref_ids.numel()),
            "layers": layers,
            "last_token_rel_l2_control": last_control,
        }
        del other
    return {
        "trial_id": clip["trial_id"],
        "n_image_tokens": int(positions.numel()),
        "first_image_position": int(positions.min()),
        "n_tokens": int(ref_ids.numel()),
        "masks": masks,
        "comparisons": comparisons,
    }


def summarise_g2(clips: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    per_layer: Dict[str, Dict[str, float]] = {}
    control: Dict[str, float] = {}
    nonzero: List[Dict[str, Any]] = []
    permits = True
    for clip in clips:
        for mask in clip["masks"].values():
            permits &= mask["image_can_read_later"] is False
        for name, comparison in clip["comparisons"].items():
            permits &= comparison["same_prefix_through_image"]
            permits &= comparison["same_image_positions"]
            for layer, row in comparison["layers"].items():
                slot = per_layer.setdefault(layer, {"max_abs": 0.0, "max_rel_l2": 0.0, "n_nonzero": 0})
                slot["max_abs"] = max(slot["max_abs"], row["max_abs"])
                slot["max_rel_l2"] = max(slot["max_rel_l2"], row["rel_l2"])
                slot["n_nonzero"] += int(row["max_abs"] != 0.0)
            for layer, value in comparison["last_token_rel_l2_control"].items():
                control[layer] = min(control.get(layer, float("inf")), value)
            changed = [k for k, row in comparison["layers"].items() if row["max_abs"] != 0.0]
            if changed:
                nonzero.append(
                    {
                        "trial_id": clip["trial_id"],
                        "comparison": name,
                        "same_total_length": comparison["same_total_length"],
                        "first_changed_layer": changed[0],
                        "n_layers_changed": len(changed),
                        "max_rel_l2": max(comparison["layers"][k]["rel_l2"] for k in changed),
                    }
                )
    keys = sorted(control, key=int)
    return {
        "n_clips": len(clips),
        "mask_permits_invariance": bool(permits),
        "per_layer": per_layer,
        "nonzero_comparisons": nonzero,
        "control_min_last_token_rel_l2": {k: control[k] for k in keys},
        "control_note": "last token sits after the option list, so it must differ. A zero here "
        "means the measure is blind.",
    }


def _cut_fn(
    adapter: Adapter, placeholder: torch.Tensor, routes_to_cut: Optional[Iterable[str]] = None
) -> RouteFn:
    """Replace vision routes with what an image-free forward has in their place."""
    chosen = None if routes_to_cut is None else set(routes_to_cut)

    def fn(routes: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = dict(routes)
        for name, tensor in routes.items():
            if chosen is not None and name not in chosen:
                continue
            if name == "main" and not adapter.additive:
                out[name] = placeholder.to(tensor.device, tensor.dtype).reshape(tensor.shape)
            else:
                out[name] = torch.zeros_like(tensor)
        return out

    return fn


def _route_row(logits: torch.Tensor, full: torch.Tensor, free: torch.Tensor) -> Dict[str, float]:
    versus_free = tensor_distance(logits, free)
    versus_full = tensor_distance(logits, full)
    total = float((full - free).norm())
    return {
        "max_abs_vs_image_free": versus_free["max_abs"],
        "max_abs_vs_full": versus_full["max_abs"],
        "fraction_of_image_effect_left": versus_free["l2"] / total if total > 0 else float("nan"),
    }


def _route_tests(
    ctx: GateContext,
    inputs: Dict[str, Any],
    positions: torch.Tensor,
    full: torch.Tensor,
    free: torch.Tensor,
    lab_full: LayerLab,
    lab_free: LayerLab,
    captured: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """Single-route cuts: each architecture claim becomes a measured prediction."""
    adapter, model = ctx.adapter, ctx.model
    placeholder = lab_free.inputs[0]
    layers = sorted(lab_free.inputs)
    tests: Dict[str, Any] = {}

    def from_free(start: int, stop: Optional[int] = None) -> Dict[int, torch.Tensor]:
        return {k: lab_free.inputs[k] for k in layers if k >= start and (stop is None or k < stop)}

    per_layer_free: Dict[int, torch.Tensor] = {}
    if adapter.has_per_layer:
        per_layer_free = {k: lab_free.per_layer[k][positions] for k in layers}

    def late_cut(start: int) -> torch.Tensor:
        with LayerLab(adapter, positions, replace_hidden=from_free(start), replace_per_layer=per_layer_free):
            return last_logits(model, inputs)

    deepstack = sorted(k for k in captured if k.startswith("deepstack_"))
    if deepstack:
        with vision_intervention(adapter, _cut_fn(adapter, placeholder, ["main"])):
            tests["main_only_cut_deepstack_live"] = _route_row(last_logits(model, inputs), full, free)
        with vision_intervention(adapter, _cut_fn(adapter, placeholder, deepstack)):
            tests["deepstack_only_cut_main_live"] = _route_row(last_logits(model, inputs), full, free)
        for name in deepstack:
            lab = LayerLab(
                adapter,
                positions,
                capture=True,
                input_layers=range(len(deepstack) + 2),
                keep_outputs=False,
            )
            with vision_intervention(adapter, _cut_fn(adapter, placeholder, [name])), lab:
                row = _route_row(last_logits(model, inputs), full, free)
            # Injection follows the decoder layer call, so a layer output hook runs before it.
            # The state entering layer k+1 is the first place injection after layer k shows.
            entering = first_changed_layer(lab.inputs, {k: lab_full.inputs[k] for k in lab.inputs})
            row["injected_after_lm_layer"] = None if entering is None else entering - 1
            tests[f"{name}_cut"] = row
        tests["late_cut_image_positions_from_L3"] = _route_row(late_cut(3), full, free)

    if adapter.has_per_layer:
        with LayerLab(adapter, positions, replace_hidden=from_free(0, 1)):
            tests["main_stream_cut_L0_per_layer_live"] = _route_row(last_logits(model, inputs), full, free)
        with LayerLab(adapter, positions, replace_per_layer=per_layer_free):
            tests["per_layer_cut_main_stream_live"] = _route_row(last_logits(model, inputs), full, free)
        tests["main_stream_and_per_layer_cut"] = _route_row(late_cut(0), full, free)
        tests["late_cut_image_positions_from_L1"] = _route_row(late_cut(1), full, free)
        mask = torch.zeros(lab_full.per_layer[0].shape[0], dtype=torch.bool, device=positions.device)
        mask[positions] = True
        sampled = sorted({0, layers[len(layers) // 2], layers[-1]})
        tests["per_layer_input_difference"] = {
            str(k): {
                "image_positions_max_abs": tensor_distance(
                    lab_full.per_layer[k][mask], lab_free.per_layer[k][mask]
                )["max_abs"],
                "text_positions_max_abs": tensor_distance(
                    lab_full.per_layer[k][~mask], lab_free.per_layer[k][~mask]
                )["max_abs"],
            }
            for k in sampled
        }
        shared = [
            k
            for k, layer in enumerate(adapter.lm_layers)
            if getattr(getattr(layer, "self_attn", None), "is_kv_shared_layer", False)
        ]
        if shared:
            first = min(shared)
            patch_late = {k: per_layer_free[k] for k in shared}
            with LayerLab(adapter, positions, replace_hidden=from_free(first), replace_per_layer=patch_late):
                tests["image_positions_from_first_kv_shared_layer"] = _route_row(
                    last_logits(model, inputs), full, free
                )
            start = max(0, first - len(shared))
            window = range(start, first)
            patch_early = {k: per_layer_free[k] for k in window}
            with LayerLab(
                adapter,
                positions,
                replace_hidden=from_free(start, first),
                replace_per_layer=patch_early,
            ):
                tests["image_positions_window_before_kv_sharing"] = _route_row(
                    last_logits(model, inputs), full, free
                )
            tests["kv_shared_layers"] = {"first": first, "count": len(shared)}

    if adapter.additive:
        tests["late_cut_image_positions_from_L1"] = _route_row(late_cut(1), full, free)
    return tests


def run_g3_clip(ctx: GateContext, clip: Dict[str, Any]) -> Dict[str, Any]:
    adapter, model = ctx.adapter, ctx.model
    prefill = ctx.prefill_ids()
    options = clip["gate_options"]
    inputs = ctx.encode(clip["trial_id"], options, prefill)
    donor_inputs = ctx.encode(clip["donor_id"], options, prefill)
    if not same_layout(inputs, donor_inputs):
        raise LayoutError(f"{clip['trial_id']} versus donor {clip['donor_id']}")
    free_inputs = strip_pixels(inputs, adapter.pixel_keys)
    positions = image_positions(adapter, inputs)

    captured: Dict[str, torch.Tensor] = {}
    lab_full = LayerLab(adapter, positions, capture=True, keep_outputs=False)
    with vision_intervention(adapter, capture=captured), lab_full:
        full = last_logits(model, inputs)
    repeat = last_logits(model, inputs)
    lab_free = LayerLab(adapter, positions, capture=True, keep_outputs=False)
    with lab_free:
        free = last_logits(model, free_inputs)

    required: Dict[str, Dict[str, float]] = {"repeat_forward": tensor_distance(repeat, full)}

    with vision_intervention(adapter, lambda _r: {k: v.clone() for k, v in captured.items()}):
        required["self_patch_vision_routes"] = tensor_distance(last_logits(model, inputs), full)
    with LayerLab(adapter, positions, replace_hidden=dict(lab_full.inputs)):
        required["self_patch_all_lm_layers"] = tensor_distance(last_logits(model, inputs), full)
    if adapter.has_per_layer:
        replay = {k: v[positions] for k, v in lab_full.per_layer.items()}
        with LayerLab(adapter, positions, replace_per_layer=replay):
            required["self_patch_per_layer_inputs"] = tensor_distance(last_logits(model, inputs), full)

    cut = _cut_fn(adapter, lab_free.inputs[0])
    with vision_intervention(adapter, cut):
        cut_logits = last_logits(model, inputs)
    with vision_intervention(adapter, cut):
        cut_other = last_logits(model, donor_inputs)
    required["all_vision_cut_vs_image_free"] = tensor_distance(cut_logits, free)
    required["cut_clip_vs_cut_donor"] = tensor_distance(cut_other, cut_logits)

    routes = _route_tests(ctx, inputs, positions, full, free, lab_full, lab_free, captured)

    informational: Dict[str, Any] = {
        "full_vs_image_free": tensor_distance(full, free),
    }
    text_only = ctx.encode_text_only(options, prefill)
    if text_only is not None:
        informational["cut_vs_prompt_with_no_image_tokens"] = tensor_distance(
            last_logits(model, text_only), cut_logits
        )
    return {
        "trial_id": clip["trial_id"],
        "n_image_tokens": int(positions.numel()),
        "max_abs_logit": float(full.abs().max()),
        "required": required,
        "route_tests": routes,
        "informational": informational,
    }


def score_options(
    ctx: GateContext,
    vision_clip: str,
    options: Sequence[str],
    *,
    route_fn: Optional[RouteFn] = None,
    image_free: bool = False,
) -> Dict[str, List[float]]:
    """Joint logprob and raw first-token logit per option, in the given option order."""
    prefill = ctx.prefill_ids()
    joint: List[float] = []
    first: List[float] = []
    for option in options:
        label = ctx.label_ids(option)
        inputs = ctx.encode(vision_clip, options, list(prefill) + list(label))
        if image_free:
            inputs = strip_pixels(inputs, ctx.adapter.pixel_keys)
        with vision_intervention(ctx.adapter, route_fn):
            logits = forward_logits(ctx.model, inputs)[0]
        start = logits.shape[0] - len(label)
        steps = logits[start - 1 : logits.shape[0] - 1].float()
        logprobs = torch.log_softmax(steps, dim=-1)
        ids = torch.tensor(label, device=logprobs.device)
        joint.append(float(logprobs[torch.arange(len(label)), ids].sum()))
        first.append(float(steps[0, ids[0]]))
    return {"joint_logprob": joint, "first_token_logit": first}


def run_g4_clip(ctx: GateContext, clip: Dict[str, Any], doses: Sequence[float] = DOSES) -> Dict[str, Any]:
    options = clip["gate_options"]
    prefill = ctx.prefill_ids()
    own: Dict[str, torch.Tensor] = {}
    donor: Dict[str, torch.Tensor] = {}
    own_inputs = ctx.encode(clip["trial_id"], options, prefill)
    donor_inputs = ctx.encode(clip["donor_id"], options, prefill)
    if not same_layout(own_inputs, donor_inputs):
        raise LayoutError(f"{clip['trial_id']} versus donor {clip['donor_id']}")
    with vision_intervention(ctx.adapter, capture=own):
        forward_logits(ctx.model, own_inputs)
    with vision_intervention(ctx.adapter, capture=donor):
        forward_logits(ctx.model, donor_inputs)

    n_tokens = int(next(iter(own.values())).shape[0])
    indices = nested_dose_indices(n_tokens, doses, clip["trial_id"])
    base = score_options(ctx, clip["trial_id"], options)
    swapped: Dict[str, Dict[str, List[float]]] = {}
    for dose in doses:
        chosen = indices[float(dose)]
        swapped[f"{dose:.2f}"] = score_options(
            ctx, clip["trial_id"], options, route_fn=lambda _r, c=chosen: swap_tokens(own, donor, c)
        )
    native = score_options(ctx, clip["donor_id"], options)
    free = score_options(ctx, clip["trial_id"], options, image_free=True)

    base_joint = base["joint_logprob"]
    full_joint = swapped[f"{max(doses):.2f}"]["joint_logprob"]
    target = options.index(clip["label"])
    donor_slot = options.index(clip["donor_label"]) if clip["donor_label_in_options"] else None
    logodds = None
    if donor_slot is not None:
        before = base_joint[donor_slot] - base_joint[target]
        logodds = float(full_joint[donor_slot] - full_joint[target] - before)
    return {
        "trial_id": clip["trial_id"],
        "actor": clip["actor"],
        "donor_id": clip["donor_id"],
        "donor_label_in_options": clip["donor_label_in_options"],
        "options": list(options),
        "target_position": target,
        "n_tokens": n_tokens,
        "base": base,
        "swapped": swapped,
        "native_donor": native,
        "image_free": free,
        "shift": {k: answer_shift(base_joint, v["joint_logprob"]) for k, v in swapped.items()},
        "shift_first_token": {
            k: answer_shift(base["first_token_logit"], v["first_token_logit"]) for k, v in swapped.items()
        },
        "native_vs_swap_max_abs": float(
            np.max(np.abs(np.asarray(native["joint_logprob"]) - np.asarray(full_joint)))
        ),
        "image_free_rms_nats": answer_shift(base_joint, free["joint_logprob"])["rms_nats"],
        "correct_base": float(int(np.argmax(base_joint)) == target),
        "correct_full_swap": float(int(np.argmax(full_joint)) == target),
        "target_vs_donor_logodds_shift": logodds,
    }


# ---------------------------------------------------------------------------
# Architecture probes
# ---------------------------------------------------------------------------


def probe_architecture(ctx: GateContext, model_key: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Record structure as loaded, and test read points that source reading suggested."""
    model = ctx.model
    report: Dict[str, Any] = {"n_lm_layers": len(ctx.adapter.lm_layers)}
    if model_key == "qwen3vl":
        visual = model.model.visual
        indexes = list(visual.deepstack_visual_indexes)
        blocks: Dict[int, torch.Tensor] = {}
        mergers: Dict[int, torch.Tensor] = {}
        handles = []
        for slot, index in enumerate(indexes):

            def on_block(_m: Any, _a: Any, out: Any, i: int = index) -> None:
                blocks[i] = out.detach().clone()

            def on_merger(_m: Any, args: Any, i: int = index) -> None:
                mergers[i] = args[0].detach().clone()

            handles.append(visual.blocks[index].register_forward_hook(on_block))
            handles.append(visual.deepstack_merger_list[slot].register_forward_pre_hook(on_merger))
        try:
            forward_logits(model, inputs)
        finally:
            for handle in handles:
                handle.remove()
        report.update(
            {
                "n_vit_layers": len(visual.blocks),
                "deepstack_visual_indexes": indexes,
                "deepstack_input_equals_vit_block_output": {
                    str(i): bool(torch.equal(blocks[i], mergers[i])) for i in indexes
                },
            }
        )
    elif model_key == "gemma4":
        cfg = model.config.get_text_config()
        vision = model.model.vision_tower
        layers = getattr(getattr(vision, "encoder", None), "layers", [])
        attention = [getattr(layer, "self_attn", None) for layer in ctx.adapter.lm_layers]
        report.update(
            {
                "n_vit_layers": len(layers),
                "hidden_size_per_layer_input": getattr(cfg, "hidden_size_per_layer_input", None),
                "use_bidirectional_attention": getattr(cfg, "use_bidirectional_attention", None),
                "layer_types": dict(Counter(getattr(cfg, "layer_types", []))),
                "num_kv_shared_layers": getattr(cfg, "num_kv_shared_layers", None),
                "kv_shared_layers": [
                    i for i, a in enumerate(attention) if getattr(a, "is_kv_shared_layer", False)
                ],
                "kv_store_layers": [
                    i for i, a in enumerate(attention) if getattr(a, "store_full_length_kv", False)
                ],
                "final_logit_softcapping": getattr(cfg, "final_logit_softcapping", None),
            }
        )
    elif model_key == "molmo2":
        backbone = model.model.vision_backbone
        resolved = [int(i) for i in backbone.vit_layers]
        captured: List[int] = []
        baseline = last_logits(model, inputs)

        def zeroed(index: int) -> Dict[str, float]:
            def hook(_m: Any, _a: Any, out: Any) -> Any:
                captured.append(len(out))
                out = list(out)
                out[index] = torch.zeros_like(out[index])
                return out

            handle = backbone.image_vit.register_forward_hook(hook)
            try:
                return tensor_distance(last_logits(model, inputs), baseline)
            finally:
                handle.remove()

        control = min(resolved) - 7
        effects = {str(i): zeroed(i)["max_abs"] for i in resolved}
        report.update(
            {
                "vit_layers_resolved": resolved,
                "n_vit_outputs_used": captured[0] if captured else None,
                "logit_change_when_read_point_zeroed": effects,
                "control_layer": control,
                "logit_change_when_control_zeroed": zeroed(control)["max_abs"],
                "token_type_ids_passed": "token_type_ids" in inputs,
            }
        )
    return report


# ---------------------------------------------------------------------------
# Real models
# ---------------------------------------------------------------------------


class RealContext(GateContext):
    def __init__(self, model_key: str, dtype_name: str, trials: Dict[str, Dict[str, Any]]) -> None:
        from transformers import AutoProcessor

        from scripts.evaluate import load_hf_model_for_key, resolve_model_path

        self.model_key = model_key
        self.dtype_name = dtype_name
        self.dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_name]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.trials = trials
        path = resolve_model_path(model_key)
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.model = load_hf_model_for_key(model_key, path, self.device, self.dtype)
        self.model.eval()
        self.adapter = adapter_for(model_key, self.model)
        self._media: Dict[str, Tuple[Any, Any]] = {}
        self._encoded: Dict[Tuple[str, Tuple[str, ...]], Dict[str, Any]] = {}

    def _media_for(self, clip: str) -> Tuple[Any, Any]:
        from config import FRAME_POLICY
        from scripts.activation_forward import prepare_trial_media

        if clip not in self._media:
            images, for_processor, _ = prepare_trial_media(
                Path(self.trials[clip]["stimulus_path"]),
                model_key=self.model_key,
                fps=float(FRAME_POLICY["fps"]),
                max_frames=int(FRAME_POLICY["max_frames"]),
            )
            self._media[clip] = (images, for_processor)
        return self._media[clip]

    def encode(
        self, vision_clip: str, options: Sequence[str], extra_ids: Sequence[int] = ()
    ) -> Dict[str, Any]:
        from scripts.activation_forward import build_forward_inputs
        from scripts.prompts import build_4afc_prompt

        key = (vision_clip, tuple(options))
        if key not in self._encoded:
            images, for_processor = self._media_for(vision_clip)
            self._encoded[key] = build_forward_inputs(
                self.model_key,
                self.model,
                self.processor,
                prompt=build_4afc_prompt(options, condition="video_only"),
                images=images,
                images_for_processor=for_processor,
                device=self.device,
                dtype=self.dtype,
            )
        return append_tokens(self._encoded[key], extra_ids)

    def encode_text_only(
        self, options: Sequence[str], extra_ids: Sequence[int]
    ) -> Optional[Dict[str, Any]]:
        from scripts.activation_forward import resolve_model_device
        from scripts.prompts import build_4afc_prompt

        prompt = build_4afc_prompt(options, condition="video_only")
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(text, return_tensors="pt")
        device = resolve_model_device(self.model)
        inputs = {k: v.to(device) for k, v in encoded.items() if k in {"input_ids", "attention_mask"}}
        return append_tokens(inputs, extra_ids)

    def prefill_ids(self) -> List[int]:
        return list(self.tokenizer(PREFILL, add_special_tokens=False).input_ids)

    def label_ids(self, option: str) -> List[int]:
        return list(self.tokenizer(f" {option}", add_special_tokens=False).input_ids)

    def end_clip(self) -> None:
        self._media.clear()
        self._encoded.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def release(self) -> None:
        del self.model, self.adapter, self.processor, self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def clip_signature(path: Path, fps: float, max_frames: int) -> Optional[Tuple[int, int, int]]:
    """Frame count and size, which fix the token layout for all three processors."""
    import cv2

    from scripts.frame_sampling import frame_indices_for_video, video_duration_seconds

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = video_duration_seconds(capture)
    capture.release()
    if frames <= 0:
        return None
    count = len(frame_indices_for_video(frames, duration, fps=fps, max_frames=max_frames))
    return (count, width, height)


def build_items(manifest: Path, lookup: Path, data_root: Path) -> List[Dict[str, Any]]:
    from config import FRAME_POLICY
    from paper_v2.schema import actor_id_from_eu_filename
    from scripts.evaluate import load_trials_from_manifest

    trials, _ = load_trials_from_manifest(manifest, data_root)
    human = json.loads(lookup.read_text(encoding="utf-8"))["trials"]
    items: List[Dict[str, Any]] = []
    for trial in trials:
        trial_id = str(trial["trial_id"])
        entry = human.get(trial_id)
        if not entry or len(entry.get("human_options", [])) != 6:
            continue
        signature = clip_signature(
            Path(trial["stimulus_path"]), float(FRAME_POLICY["fps"]), int(FRAME_POLICY["max_frames"])
        )
        if signature is None:
            continue
        items.append(
            {
                "trial_id": trial_id,
                "label": entry["human_target_label"],
                "actor": actor_id_from_eu_filename(trial_id),
                "options": list(entry["human_options"]),
                "signature": list(signature),
                "stimulus_path": str(trial["stimulus_path"]),
            }
        )
    return items


def run_stage(
    model_key: str,
    dtype_name: str,
    gates: Sequence[str],
    sample: Sequence[Dict[str, Any]],
    trials: Dict[str, Dict[str, Any]],
    *,
    n_g3: int,
    result: Dict[str, Any],
    save: Callable[[], None],
) -> None:
    started = time.time()
    ctx = RealContext(model_key, dtype_name, trials)
    stage: Dict[str, Any] = {"model_load_seconds": round(time.time() - started, 1)}
    result["stages"][dtype_name] = stage
    first = sample[0]
    inputs = ctx.encode(first["trial_id"], first["gate_options"], ctx.prefill_ids())
    tail = ctx.tokenizer.decode(inputs["input_ids"][0][-90:], skip_special_tokens=False)
    stage["prompt_tail_first_clip"] = tail
    stage["input_keys"] = sorted(inputs)
    stage["architecture"] = probe_architecture(ctx, model_key, inputs)
    if torch.cuda.is_available():
        stage["gpu"] = torch.cuda.get_device_name(0)
    save()

    if "g2" in gates:
        clips = []
        tick = time.time()
        for clip in sample:
            clips.append(run_g2_clip(ctx, clip))
            ctx.end_clip()
        summary = summarise_g2(clips)
        stage["g2"] = {
            "seconds": round(time.time() - tick, 1),
            "summary": summary,
            "verdict": g2_verdict(summary, dtype_name),
            "clip_layout": [
                {k: c[k] for k in ("trial_id", "n_image_tokens", "first_image_position", "n_tokens", "masks")}
                for c in clips
            ],
        }
        save()
    if "g3" in gates:
        clips = []
        tick = time.time()
        for clip in sample[:n_g3]:
            clips.append(run_g3_clip(ctx, clip))
            ctx.end_clip()
        stage["g3"] = {
            "seconds": round(time.time() - tick, 1),
            "verdict": g3_verdict(clips, dtype_name),
            "clips": clips,
        }
        save()
    if "g4" in gates:
        clips = []
        tick = time.time()
        for clip in sample:
            clips.append(run_g4_clip(ctx, clip))
            ctx.end_clip()
            stage["g4"] = {"seconds": round(time.time() - tick, 1), "clips": clips, "partial": True}
            save()
        stage["g4"] = {
            "seconds": round(time.time() - tick, 1),
            "verdict": g4_verdict(clips),
            "clips": clips,
        }
        save()
    stage["seconds"] = round(time.time() - started, 1)
    save()
    ctx.release()


def overall_verdict(result: Dict[str, Any]) -> Dict[str, Any]:
    stages = result["stages"]
    out: Dict[str, Any] = {}
    for gate in ("g2", "g3"):
        rows = {k: s[gate]["verdict"]["pass"] for k, s in stages.items() if gate in s}
        out[gate.upper()] = {"by_dtype": rows, "pass": bool(rows) and all(rows.values())}
    g4 = [s["g4"]["verdict"] for s in stages.values() if "verdict" in s.get("g4", {})]
    out["G4"] = {"pass": g4[0]["pass"] if g4 else None}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, choices=MODEL_ORDER)
    parser.add_argument("--manifest", type=Path, default=Path("data/eu_emotions_full_manifest.json"))
    parser.add_argument("--lookup", type=Path, default=Path("data/eu_emotions_human_entropy.json"))
    parser.add_argument("--data_root", type=Path, default=Path("data/eu_emotions"))
    parser.add_argument("--max_clips", type=int, default=None, help="Smoke only: first N of the 40.")
    parser.add_argument("--n_g3", type=int, default=12)
    parser.add_argument("--gates", default="g2,g3,g4")
    parser.add_argument("--dtypes", default="bfloat16,float32")
    parser.add_argument(
        "--float32_max_frames",
        type=int,
        default=None,
        help="Limit the float32 check to clips with at most this many frames.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    items = build_items(args.manifest, args.lookup, args.data_root)
    sample = select_sample(items, n=N_CLIPS, seed=SEED)
    trials = {i["trial_id"]: i for i in items}
    output = args.output or Path(f"research/reproduction/vision_gates_{args.model}.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    import cv2
    import transformers

    result: Dict[str, Any] = {
        "model": args.model,
        "gate_rules": GATE_RULES,
        "seed": SEED,
        "smoke": args.max_clips is not None,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {
            "cv2": cv2.__version__,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "sample": {
            "n_eligible_items": len(items),
            "actor_counts": dict(Counter(c["actor"] for c in sample)),
            "label_counts": dict(Counter(c["label"] for c in sample)),
            "rotation_counts": dict(Counter(c["rotation"] for c in sample)),
            "signature_counts": {
                str(k): v for k, v in Counter(tuple(c["signature"]) for c in sample).items()
            },
            "clips": [
                {
                    k: c[k]
                    for k in (
                        "trial_id",
                        "label",
                        "actor",
                        "signature",
                        "rotation",
                        "gate_options",
                        "donor_id",
                        "donor_label",
                        "donor_actor",
                        "donor_label_in_options",
                    )
                }
                for c in sample
            ],
        },
        "stages": {},
    }

    def save() -> None:
        output.write_text(json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8")

    save()
    gates = [g.strip() for g in args.gates.split(",") if g.strip()]
    for dtype_name in [d.strip() for d in args.dtypes.split(",") if d.strip()]:
        # G4 runs at the evaluation dtype only. float32 is a numerical check on G2 and G3.
        stage_gates = [g for g in gates if g != "g4" or dtype_name == "bfloat16"]
        stage_sample = sample
        if dtype_name == "float32" and args.float32_max_frames is not None:
            # Qwen SDPA allocates 32 x T^2 weights, about 50 GB at 20,000 float32 tokens.
            stage_sample = [c for c in sample if c["signature"][0] <= args.float32_max_frames]
            result["float32_scope"] = {
                "max_frames": args.float32_max_frames,
                "n_clips": len(stage_sample),
                "reason": "memory; the bfloat16 stage covers the full sample",
            }
        if args.max_clips is not None:
            stage_sample = stage_sample[: args.max_clips]
        try:
            run_stage(
                args.model,
                dtype_name,
                stage_gates,
                stage_sample,
                trials,
                n_g3=args.n_g3,
                result=result,
                save=save,
            )
        except torch.cuda.OutOfMemoryError as error:
            if dtype_name != "float32":
                raise
            # The float32 stage is a numerical check; keep what the bfloat16 stage found.
            result["float32_failed"] = str(error)[:300]
            gc.collect()
            torch.cuda.empty_cache()
    result["verdict"] = overall_verdict(result)
    save()
    print(json.dumps(result["verdict"], indent=1))


if __name__ == "__main__":
    main()
