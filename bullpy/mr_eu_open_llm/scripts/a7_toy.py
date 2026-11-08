"""
CPU toys for the Level 1 recovery of amendment A7, and tiny random-weight real architectures.

Two kinds of model, both driven through scripts/a7_cuts.py, which goes through the frozen
`vision_intervention`, `LayerLab`, `swap_tokens` and `_cut_fn` of scripts/vision_gates.py:

1. ToyVLM: a small torch module with the attribute paths of the real models, so that
   `adapter_for` builds its adapter. Two routes (Gemma style: the entry hidden state and a
   per-layer stream; Qwen style: a main route and three deepstack routes) and a hidden third
   route of settable strength in nats, which no hook can see. Options make one block dead,
   make the forward non-deterministic, or mix a share of the edited patches into a far set of
   tokens (uniform attention from the border ring of the token grid to every patch), which
   sets the region test's leak.

2. tiny_gemma4 / tiny_qwen3vl: the real `transformers` classes (local 5.9.0) built from tiny
   configs with random weights. No weights are downloaded. These test the hold wiring on the
   real module code: Gemma's L0 hidden state plus every per-layer input, and Qwen's main and
   deepstack routes. torchvision in this environment does not import, so it is switched off
   for transformers before the import; nothing here uses it.

Nothing here reads a clip, a label or a model output.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.vision_gates import append_tokens  # noqa: E402

N_WORDS = 106
IMAGE_TOKEN = 7
VOCAB = 512
WORD_IDS = list(range(200, 200 + N_WORDS))
SIDE = 16
PATCH = 4
IMAGE_PX = SIDE * PATCH
DIM = 32
N_LAYERS = 4
TEXT_BEFORE = [2, 3, 4]
TEXT_AFTER = [11, 12, 13, 14]
SUFFIX = 15


# ---------------------------------------------------------------------------
# ToyVLM
# ---------------------------------------------------------------------------


class Inner(nn.Module):
    pass


def border_tokens(side: int = SIDE) -> np.ndarray:
    grid = np.zeros((side, side), dtype=bool)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = True
    return grid.reshape(-1)


class ToyEmbedVision(nn.Module):
    """Patches to exit tokens. Optionally adds mix times the patch mean to the far tokens."""

    def __init__(self, mix: float) -> None:
        super().__init__()
        self.embed = nn.Linear(PATCH * PATCH, DIM)
        self.mix = mix
        self.register_buffer("far", torch.tensor(border_tokens(), dtype=torch.float32)[:, None])

    def local(self, patches: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.embed(patches))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        tokens = self.local(patches)
        if self.mix:
            tokens = tokens + self.mix * self.far * tokens.mean(0, keepdim=True)
        return tokens


class ToyQwenVisual(nn.Module):
    """Main route plus three deepstack routes, in the output format Qwen's adapter reads."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Linear(PATCH * PATCH, DIM)
        self.blocks = nn.ModuleList([nn.Linear(DIM, DIM) for _ in range(3)])
        self.mergers = nn.ModuleList([nn.Linear(DIM, DIM) for _ in range(3)])
        self.merger = nn.Linear(DIM, DIM)

    def forward(self, patches: torch.Tensor) -> SimpleNamespace:
        hidden = torch.tanh(self.embed(patches))
        deep = []
        for block, merger in zip(self.blocks, self.mergers):
            hidden = torch.tanh(block(hidden))
            deep.append(0.5 * merger(hidden))
        return SimpleNamespace(pooler_output=self.merger(hidden), deepstack_features=deep)


class ToyLayer(nn.Module):
    """Text reads the mean of the image positions; image positions change only by per-layer input."""

    def __init__(self, per_layer: bool, gate_scale: float) -> None:
        super().__init__()
        self.read = nn.Linear(DIM, DIM, bias=False)
        self.mlp = nn.Linear(DIM, DIM)
        self.gate = nn.Linear(DIM, DIM, bias=False) if per_layer else None
        self.gate_scale = gate_scale

    def forward(self, hidden_states: torch.Tensor, per_layer_input: Optional[torch.Tensor] = None,
                image_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden = hidden_states
        text = ~image_mask
        update = torch.zeros_like(hidden)
        update[0, text] = self.read(hidden[0, image_mask].mean(0)) + 0.3 * torch.tanh(self.mlp(hidden[0, text]))
        hidden = hidden + update
        if self.gate is not None and per_layer_input is not None:
            hidden = hidden + self.gate_scale * torch.tanh(self.gate(hidden)) * per_layer_input
        return hidden


class ToyVLM(nn.Module):
    """style: gemma4 or qwen3vl. leak_nats: third route. dead: block B has no effect.

    noise: added to the logits on every forward (non-deterministic). mix: far-token mixing.
    per_layer_positional: pass Gemma's per-layer input as the second positional argument, as
    the real model does, rather than by keyword.
    """

    def __init__(self, style: str, *, leak_nats: float = 0.0, dead: bool = False, noise: float = 0.0,
                 mix: float = 0.0, per_layer_positional: bool = True, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.style = style
        self.leak_nats = leak_nats
        self.noise = noise
        self.per_layer_positional = per_layer_positional
        self.model = Inner()
        self.model.embed = nn.Embedding(VOCAB, DIM)
        text = Inner()
        gemma = style == "gemma4"
        text.layers = nn.ModuleList([ToyLayer(gemma, 0.0 if dead else 3.0) for _ in range(N_LAYERS)])
        self.model.language_model = text
        if gemma:
            self.model.embed_vision = ToyEmbedVision(mix)
            self.per_layer_proj = nn.Linear(DIM, DIM, bias=False)
        else:
            self.model.visual = ToyQwenVisual()
            self.deep_scale = 0.0 if dead else 1.0
        self.head = nn.Linear(DIM, VOCAB, bias=False)
        bias = torch.zeros(VOCAB)
        bias[WORD_IDS] = -6.0  # word mass is small, so a shift of c on word logits moves their logprobs by about c
        self.register_buffer("bias", bias)
        self.config = SimpleNamespace(image_token_id=IMAGE_TOKEN, get_text_config=lambda: SimpleNamespace())
        self.word_mask = torch.zeros(VOCAB)
        self.word_mask[WORD_IDS] = 1.0

    def forward(self, input_ids: torch.Tensor, pixel_values: Optional[torch.Tensor] = None,
                attention_mask: Any = None, mm_token_type_ids: Any = None, use_cache: Any = None) -> SimpleNamespace:
        inner = self.model
        x = inner.embed(input_ids)
        mask = input_ids == IMAGE_TOKEN
        deep = None
        if pixel_values is not None:
            if self.style == "gemma4":
                feats = inner.embed_vision(pixel_values[0])
                x = x.masked_scatter(mask[..., None], feats)
            else:
                out = inner.visual(pixel_values[0])
                x = x.masked_scatter(mask[..., None], out.pooler_output)
                deep = out.deepstack_features
        per_layer = self.per_layer_proj(x) if self.style == "gemma4" else None
        hidden = x
        for index, layer in enumerate(inner.language_model.layers):
            if self.per_layer_positional:
                hidden = layer(hidden, per_layer, image_mask=mask[0])
            else:
                hidden = layer(hidden, per_layer_input=per_layer, image_mask=mask[0])
            if deep is not None and index < len(deep):
                hidden = hidden.clone()
                hidden[mask] = hidden[mask] + self.deep_scale * deep[index]
        logits = self.head(hidden) + self.bias
        if self.leak_nats and pixel_values is not None:
            # Third route: a pixel-dependent switch, near 0 or 1, times leak_nats on every word logit.
            switch = torch.sigmoid(20.0 * pixel_values.mean())
            logits = logits + self.leak_nats * switch * self.word_mask
        if self.noise:
            logits = logits + self.noise * torch.randn_like(logits)
        return SimpleNamespace(logits=logits)


# ---------------------------------------------------------------------------
# Toy clips, edits and the readout
# ---------------------------------------------------------------------------


def toy_images(n: int = 24, seed: int = 7042) -> List[np.ndarray]:
    """Random 64 by 64 images whose means alternate in sign, so the third route's switch flips."""
    rng = np.random.default_rng(seed)
    return [rng.normal(1.0 if i % 2 == 0 else -1.0, 1.0, (IMAGE_PX, IMAGE_PX)).astype(np.float32) for i in range(n)]


def to_patches(image: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(image))
    p = t.reshape(SIDE, PATCH, SIDE, PATCH).permute(0, 2, 1, 3).reshape(SIDE * SIDE, PATCH * PATCH)
    return p[None]


def toy_inputs(image: np.ndarray) -> Dict[str, torch.Tensor]:
    ids = torch.tensor([TEXT_BEFORE + [IMAGE_TOKEN] * (SIDE * SIDE) + TEXT_AFTER])
    return {"input_ids": ids, "pixel_values": to_patches(image), "attention_mask": torch.ones_like(ids),
            "mm_token_type_ids": (ids == IMAGE_TOKEN).long()}


def region_rect(index: int) -> Tuple[int, int]:
    """Top-left token of a 6 by 6 token region in the centre, moving a little by clip."""
    return 4 + index % 3, 4 + (index // 3) % 3


def region_tokens(index: int) -> List[int]:
    r0, c0 = region_rect(index)
    return [r * SIDE + c for r in range(r0, r0 + 6) for c in range(c0, c0 + 6)]


EDIT_FILL = 4.0


def blur_region(image: np.ndarray, index: int) -> np.ndarray:
    """The toy edit: the region's pixels set to one fill value, so the edit moves the readout
    well above the region test's floor. A toy stand-in for the blur fill, not a blur."""
    r0, c0 = region_rect(index)
    out = image.copy()
    out[r0 * PATCH:(r0 + 6) * PATCH, c0 * PATCH:(c0 + 6) * PATCH] = EDIT_FILL
    return out


def paint_square(image: np.ndarray, centre: Sequence[float], square: float) -> np.ndarray:
    out = image.copy()
    half = int(round(square / 2))
    cx, cy = int(centre[0]), int(centre[1])
    out[max(0, cy - half):cy + half, max(0, cx - half):cx + half] = 0.0
    return out


def mix_for_leak(rho: float) -> float:
    """Far-token mixing whose share of the edit's effect is rho, to first order."""
    if rho <= 0:
        return 0.0
    kappa = rho / (1.0 - rho)
    return kappa * SIDE * SIDE / float(border_tokens().sum())


def readout_for(model: nn.Module, word_ids: Sequence[int] = WORD_IDS):
    """The 106 word logprobs from two forwards, the second with one more token, so that every
    hold must act on more than one forward, as A2's blind readout does with its prefixes."""
    half = len(word_ids) // 2

    def readout(inputs: Dict[str, Any]) -> np.ndarray:
        with torch.no_grad():
            first = torch.log_softmax(model(**inputs, use_cache=False).logits[0, -1].double(), -1)
            second = torch.log_softmax(model(**append_tokens(inputs, [SUFFIX]), use_cache=False).logits[0, -1].double(), -1)
        return np.concatenate([first[list(word_ids[:half])].numpy(), second[list(word_ids[half:])].numpy()])

    return readout


# ---------------------------------------------------------------------------
# Tiny real architectures, random weights, CPU
# ---------------------------------------------------------------------------


def _no_torchvision() -> None:
    import transformers.utils as tu
    import transformers.utils.import_utils as iu

    iu.is_torchvision_available = lambda: False
    tu.is_torchvision_available = lambda: False


TINY_GEMMA_SIDE = 12  # patches per side; pooled 3 by 3 into 4 by 4 = 16 soft tokens
TINY_IMAGE_TOKEN = 290


def tiny_gemma4(seed: int = 0) -> nn.Module:
    """Gemma4ForConditionalGeneration with 6 layers, 2 of them sharing keys and values."""
    _no_torchvision()
    from transformers.models.gemma4 import configuration_gemma4 as c
    from transformers.models.gemma4 import modeling_gemma4 as m

    torch.manual_seed(seed)
    text = dict(vocab_size=300, hidden_size=32, intermediate_size=64, num_hidden_layers=6, num_attention_heads=2,
                num_key_value_heads=1, head_dim=16, global_head_dim=16, vocab_size_per_layer_input=300,
                hidden_size_per_layer_input=8, num_kv_shared_layers=2, sliding_window=8, pad_token_id=0,
                layer_types=["sliding_attention", "sliding_attention", "full_attention",
                             "sliding_attention", "sliding_attention", "full_attention"],
                use_bidirectional_attention="vision")
    vision = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                  num_key_value_heads=2, head_dim=16, patch_size=4, pooling_kernel_size=3, position_embedding_size=64)
    cfg = c.Gemma4Config(text_config=text, vision_config=vision, image_token_id=TINY_IMAGE_TOKEN, boi_token_id=291,
                         eoi_token_id=292, video_token_id=293, audio_token_id=294, boa_token_id=295, eoa_token_index=296)
    return m.Gemma4ForConditionalGeneration(cfg).eval()


def tiny_gemma4_inputs(seed: int) -> Dict[str, torch.Tensor]:
    g = TINY_GEMMA_SIDE
    xs, ys = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="xy")
    positions = torch.stack([xs.flatten(), ys.flatten()], -1)[None]
    gen = torch.Generator().manual_seed(seed)
    pixels = torch.randn(1, g * g, 48, generator=gen)
    ids = torch.tensor([[2, 5, 6, 291] + [TINY_IMAGE_TOKEN] * 16 + [292, 7, 8, 9, 10]])
    return {"input_ids": ids, "pixel_values": pixels, "image_position_ids": positions,
            "attention_mask": torch.ones_like(ids), "mm_token_type_ids": (ids == TINY_IMAGE_TOKEN).long()}


def tiny_qwen3vl(seed: int = 0) -> nn.Module:
    """Qwen3VLForConditionalGeneration with 4 layers and deepstack from ViT blocks 0, 1 and 2."""
    _no_torchvision()
    from transformers.models.qwen3_vl import configuration_qwen3_vl as c
    from transformers.models.qwen3_vl import modeling_qwen3_vl as m

    torch.manual_seed(seed)
    text = dict(vocab_size=300, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=2,
                num_key_value_heads=1, head_dim=16,
                rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [2, 3, 3],
                                 "mrope_interleaved": True})
    vision = dict(depth=4, hidden_size=32, intermediate_size=64, num_heads=2, in_channels=3, patch_size=4,
                  spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=32, num_position_embeddings=64,
                  deepstack_visual_indexes=[0, 1, 2])
    cfg = c.Qwen3VLConfig(text_config=text, vision_config=vision, image_token_id=TINY_IMAGE_TOKEN, video_token_id=291,
                          vision_start_token_id=292, vision_end_token_id=293)
    return m.Qwen3VLForConditionalGeneration(cfg).eval()


def tiny_qwen3vl_inputs(seed: int) -> Dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    pixels = torch.randn(64, 3 * 2 * 4 * 4, generator=gen)
    ids = torch.tensor([[2, 5, 292] + [TINY_IMAGE_TOKEN] * 16 + [293, 7, 8, 9]])
    return {"input_ids": ids, "pixel_values": pixels, "image_grid_thw": torch.tensor([[1, 8, 8]]),
            "attention_mask": torch.ones_like(ids), "mm_token_type_ids": (ids == TINY_IMAGE_TOKEN).long()}


TINY_WORD_IDS = list(range(150, 150 + N_WORDS))
