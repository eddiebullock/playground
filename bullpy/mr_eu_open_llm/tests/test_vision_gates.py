"""Vision gates G2 to G4 on tiny fake models, CPU only.

The fake models keep the attribute paths of the real ones, so the adapters are exercised
rather than bypassed. Broken variants check that each gate can fail.
"""

import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from paper_v2.option_score import ABSTAIN_LABEL  # noqa: E402
from scripts import vision_gates as vg  # noqa: E402

DIM = 24
N_TOKENS = 12
N_LAYERS = 4
IMAGE_TOKEN = 7
OPTIONS = ["Target", "A", "B", "C", "D", ABSTAIN_LABEL]


class TinyAttention(nn.Module):
    def __init__(self, shared: bool = False, store: bool = False) -> None:
        super().__init__()
        self.q = nn.Linear(DIM, DIM, bias=False)
        self.k = nn.Linear(DIM, DIM, bias=False)
        self.v = nn.Linear(DIM, DIM, bias=False)
        self.is_kv_shared_layer = shared
        self.store_full_length_kv = store

    def forward(self, hidden, causal, store):
        query = self.q(hidden)
        if self.is_kv_shared_layer:
            key, value = store["kv"]
        else:
            key, value = self.k(hidden), self.v(hidden)
        if self.store_full_length_kv:
            store["kv"] = (key, value)
        scores = query @ key.transpose(-1, -2) / math.sqrt(DIM)
        if causal:
            n = hidden.shape[1]
            future = torch.triu(torch.ones(n, n, dtype=torch.bool), 1)
            scores = scores.masked_fill(future, float("-inf"))
        return torch.softmax(scores, dim=-1) @ value


class TinyLayer(nn.Module):
    def __init__(self, tuple_output=False, per_layer=False, causal=True, shared=False, store=False):
        super().__init__()
        self.self_attn = TinyAttention(shared, store)
        self.mlp = nn.Sequential(nn.Linear(DIM, DIM), nn.Tanh())
        self.gate = nn.Linear(DIM, DIM, bias=False) if per_layer else None
        self.tuple_output = tuple_output
        self.causal = causal

    def forward(self, hidden_states, per_layer_input=None, attention_mask=None, store=None):
        hidden = hidden_states + self.self_attn(hidden_states, self.causal, store)
        hidden = hidden + self.mlp(hidden)
        if self.gate is not None:
            hidden = hidden + torch.tanh(self.gate(hidden)) * per_layer_input
        return (hidden,) if self.tuple_output else hidden


class QwenVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(DIM, DIM) for _ in range(5)])
        self.deepstack_visual_indexes = [1, 2, 3]
        self.deepstack_merger_list = nn.ModuleList([nn.Linear(DIM, DIM) for _ in range(3)])
        self.merger = nn.Linear(DIM, DIM)

    def forward(self, pixels):
        hidden = pixels
        deep = []
        for index, block in enumerate(self.blocks):
            hidden = block(hidden)
            if index in self.deepstack_visual_indexes:
                slot = self.deepstack_visual_indexes.index(index)
                deep.append(self.deepstack_merger_list[slot](hidden))
        return SimpleNamespace(pooler_output=self.merger(torch.tanh(hidden)), deepstack_features=deep)


class MolmoVit(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(DIM, DIM) for _ in range(10)])

    def forward(self, pixels):
        out, hidden = [], pixels
        for layer in self.layers:
            hidden = torch.tanh(layer(hidden))
            out.append(hidden)
        return out


class MolmoBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vit_layers = [9, 8]
        self.image_vit = MolmoVit()
        self.project = nn.Linear(2 * DIM, DIM)

    def forward(self, pixels):
        out = self.image_vit(pixels)
        return self.project(torch.cat([out[i] for i in self.vit_layers], dim=-1))


class Inner(nn.Module):
    pass


class TinyVLM(nn.Module):
    """style is one of qwen, gemma, molmo. leak adds a route no hook can see."""

    def __init__(self, style, leak=False, causal=True, kv_shared=0):
        super().__init__()
        torch.manual_seed(0)
        self.style = style
        self.leak = leak
        per_layer = style == "gemma"
        first_shared = N_LAYERS - kv_shared
        layers = nn.ModuleList(
            [
                TinyLayer(
                    tuple_output=style == "molmo",
                    per_layer=per_layer,
                    causal=causal,
                    shared=kv_shared > 0 and i >= first_shared,
                    store=kv_shared > 0 and i == first_shared - 1,
                )
                for i in range(N_LAYERS)
            ]
        )
        self.model = Inner()
        self.model.embed = nn.Embedding(100, DIM)
        text = Inner()
        text.layers = layers
        text.config_per_layer = nn.Linear(DIM, DIM, bias=False)
        if style == "molmo":
            self.model.transformer = Inner()
            self.model.transformer.blocks = layers
            self.model.vision_backbone = MolmoBackbone()
            self.model.language_model = text
        else:
            self.model.language_model = text
        if style == "qwen":
            self.model.visual = QwenVisual()
        if style == "gemma":
            self.model.vision_tower = nn.Linear(DIM, DIM)
            self.model.embed_vision = nn.Linear(DIM, DIM)
        self.head = nn.Linear(DIM, 100, bias=False)
        self.config = SimpleNamespace(
            image_token_id=IMAGE_TOKEN,
            image_patch_id=IMAGE_TOKEN,
            get_text_config=lambda: SimpleNamespace(),
        )
        self.leak_head = nn.Linear(DIM, 100, bias=False)

    def forward(self, input_ids, pixel_values=None, attention_mask=None, use_cache=None):
        inner = self.model
        x = inner.embed(input_ids)
        mask = input_ids == IMAGE_TOKEN
        deep = None
        if pixel_values is not None:
            if self.style == "qwen":
                out = inner.visual(pixel_values[0])
                x = x.masked_scatter(mask[..., None], out.pooler_output)
                deep = out.deepstack_features
            elif self.style == "gemma":
                feats = inner.embed_vision(torch.tanh(inner.vision_tower(pixel_values[0])))
                x = x.masked_scatter(mask[..., None], feats)
            else:
                x = x.clone()
                x[mask] += inner.vision_backbone(pixel_values[0])
        per_layer = inner.language_model.config_per_layer(x) if self.style == "gemma" else None
        hidden = x
        store = {}
        for index, layer in enumerate(inner.language_model.layers):
            out = layer(hidden, per_layer, attention_mask=None, store=store)
            hidden = out[0] if isinstance(out, tuple) else out
            if deep is not None and index < len(deep):
                hidden = hidden.clone()
                hidden[mask] = hidden[mask] + deep[index]
        logits = self.head(hidden)
        if self.leak and pixel_values is not None:
            logits = logits + self.leak_head(pixel_values.mean(1, keepdim=True))
        return SimpleNamespace(logits=logits)


class FakeContext(vg.GateContext):
    def __init__(self, model_key, model, image_last=False):
        self.model = model.eval()
        self.adapter = vg.adapter_for(model_key, self.model)
        self.image_last = image_last
        generator = torch.Generator().manual_seed(1)
        self.pixels = {
            name: torch.randn(1, N_TOKENS, DIM, generator=generator)
            for name in ("clip", "donor")
        }
        self.encoded = 0

    def encode(self, vision_clip, options, extra_ids=()):
        self.encoded += 1
        text = [20 + sum(ord(c) for c in option) % 60 for option in options]
        image = [IMAGE_TOKEN] * N_TOKENS
        body = text + image if self.image_last else image + text
        ids = torch.tensor([[1, 2, *body, 3]])
        inputs = {"input_ids": ids, "pixel_values": self.pixels[vision_clip]}
        return vg.append_tokens(inputs, extra_ids)

    def prefill_ids(self):
        return [4, 5]

    def label_ids(self, option):
        return [10 + sum(ord(c) for c in option) % 40, 6]


def make_clip():
    return {
        "trial_id": "clip",
        "donor_id": "donor",
        "actor": "A",
        "label": "Target",
        "donor_label": "B",
        "donor_label_in_options": True,
        "gate_options": vg.gate_options(OPTIONS, 2, "Target"),
        "second_rotation_options": vg.gate_options(OPTIONS, 3, "Target"),
        "foreign_options": vg.gate_options(["Z", "W", "X", "Y", "V", ABSTAIN_LABEL], 2, "Z"),
    }


def make_items(n_actors=17, per_actor=12, labels=21):
    items = []
    for actor in range(n_actors):
        for clip in range(per_actor):
            label = f"L{(actor * 3 + clip) % labels}"
            options = [label] + [f"L{(actor + clip + k) % labels}x" for k in range(1, 5)]
            items.append(
                {
                    "trial_id": f"{actor}/{clip}",
                    "label": label,
                    "actor": chr(65 + actor),
                    "options": options + [ABSTAIN_LABEL],
                    "signature": [2 + clip % 3, 640, 360],
                }
            )
    return items


# Prompts and sample ---------------------------------------------------------


def test_gate_options_refuses_target_first():
    with pytest.raises(ValueError):
        vg.gate_options(OPTIONS, 0, "Target")
    for rotation in (1, 2, 3, 4):
        rotated = vg.gate_options(OPTIONS, rotation, "Target")
        assert rotated[0] != "Target"
        assert rotated[-1] == ABSTAIN_LABEL


def test_sample_is_deterministic_balanced_and_never_target_first():
    items = make_items()
    first = vg.select_sample(items, n=40, seed=42)
    assert [c["trial_id"] for c in first] == [
        c["trial_id"] for c in vg.select_sample(items, n=40, seed=42)
    ]
    assert len({c["trial_id"] for c in first}) == 40
    counts = Counter(c["actor"] for c in first)
    assert len(counts) == 17
    assert max(counts.values()) - min(counts.values()) <= 1
    assert set(Counter(c["rotation"] for c in first).values()) == {10}
    for clip in first:
        for key in ("gate_options", "second_rotation_options"):
            assert clip[key][0] != clip["label"]
        assert clip["gate_options"] != clip["second_rotation_options"]


def test_sample_prefix_covers_actors():
    prefix = vg.select_sample(make_items(), n=40, seed=42)[:12]
    assert len({c["actor"] for c in prefix}) == 12


def test_donor_differs_in_label_and_actor_and_matches_layout():
    items = make_items()
    by_id = {i["trial_id"]: i for i in items}
    for clip in vg.select_sample(items, n=40, seed=42):
        donor = by_id[clip["donor_id"]]
        assert donor["label"] != clip["label"]
        assert donor["actor"] != clip["actor"]
        assert donor["signature"] == clip["signature"]


def test_sample_uses_no_clip_without_a_donor():
    items = make_items(n_actors=3, per_actor=6)
    items[0]["signature"] = [99, 1, 1]
    chosen = vg.select_sample(items, n=6, seed=42)
    assert items[0]["trial_id"] not in {c["trial_id"] for c in chosen}


def test_dose_subsets_are_nested_and_sized():
    subsets = vg.nested_dose_indices(200, vg.DOSES, "clip")
    sizes = [len(subsets[d]) for d in vg.DOSES]
    assert sizes == [20, 50, 100, 200]
    for low, high in zip(vg.DOSES, vg.DOSES[1:]):
        assert set(subsets[low]) <= set(subsets[high])


def test_dose_subsets_depend_on_key_only():
    a = vg.nested_dose_indices(100, vg.DOSES, "x")[0.1]
    b = vg.nested_dose_indices(100, vg.DOSES, "x")[0.1]
    c = vg.nested_dose_indices(100, vg.DOSES, "y")[0.1]
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# Metrics and verdicts --------------------------------------------------------


def test_answer_shift_ignores_a_constant():
    shift = vg.answer_shift([1.0, 2.0, 3.0], [11.0, 12.0, 13.0])
    assert shift["rms_nats"] == pytest.approx(0.0)
    assert shift["total_variation"] == pytest.approx(0.0)
    assert shift["flipped"] == 0.0


def test_answer_shift_detects_a_flip():
    shift = vg.answer_shift([0.0, 5.0], [5.0, 0.0])
    assert shift["flipped"] == 1.0
    assert shift["rms_nats"] == pytest.approx(5.0)


def test_actor_bootstrap_resamples_actors_not_clips():
    low, high = vg.actor_bootstrap_ci([1.0] * 39 + [1.0], ["a"] * 20 + ["b"] * 20, draws=200)
    assert low == high == 1.0
    wide = vg.actor_bootstrap_ci([0.0] * 20 + [1.0] * 20, ["a"] * 20 + ["b"] * 20, draws=500)
    assert wide == (0.0, 1.0)


def test_logit_tolerance_is_fixed_for_float32_and_ulp_for_bfloat16():
    assert vg.logit_tolerance("float32", 100.0) == 1e-3
    assert vg.logit_tolerance("bfloat16", 20.0) == pytest.approx(2 * 0.125)


def _g4_clip(values, flips, native=0.0):
    keys = [f"{d:.2f}" for d in vg.DOSES]
    return {
        "actor": "A",
        "shift": {
            k: {"rms_nats": v, "total_variation": 0.1, "flipped": f}
            for k, v, f in zip(keys, values, flips)
        },
        "native_vs_swap_max_abs": native,
        "donor_label_in_options": False,
        "target_vs_donor_logodds_shift": None,
        "correct_base": 1.0,
        "correct_full_swap": 0.0,
        "image_free_rms_nats": 1.0,
    }


def _g4(clips):
    for i, clip in enumerate(clips):
        clip["actor"] = "ABCD"[i % 4]
    return vg.g4_verdict(clips, draws=200)


def test_g4_passes_only_when_large_and_monotonic():
    good = [_g4_clip([0.2, 0.5, 1.0, 2.0 + i / 10], [0, 0, 1, 1]) for i in range(8)]
    assert _g4(good)["pass"]
    small = [_g4_clip([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1]) for _ in range(8)]
    assert not _g4(small)["large"]
    bent = [_g4_clip([0.5, 2.0, 1.0, 3.0], [0, 0, 1, 1]) for _ in range(8)]
    assert not _g4(bent)["monotonic"]
    no_flip = [_g4_clip([0.2, 0.5, 1.0, 2.0], [0, 0, 0, 0]) for _ in range(8)]
    assert not _g4(no_flip)["large"]


def test_g4_fails_when_full_swap_differs_from_native_forward():
    clips = [_g4_clip([0.2, 0.5, 1.0, 2.0], [0, 0, 1, 1], native=0.5) for _ in range(8)]
    result = _g4(clips)
    assert result["monotonic"] and result["large"]
    assert not result["pass"]


def test_append_tokens_extends_sequence_tensors_only():
    inputs = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
        "mm_token_type_ids": torch.ones(1, 2, dtype=torch.long),
        "pixel_values": torch.zeros(1, 2),
    }
    out = vg.append_tokens(inputs, [9])
    assert out["input_ids"].tolist() == [[1, 2, 9]]
    assert out["attention_mask"].tolist() == [[1, 1, 1]]
    assert out["mm_token_type_ids"].tolist() == [[1, 1, 0]]
    assert out["pixel_values"].shape == (1, 2)


def test_mask_report_reads_bool_float_and_missing_masks():
    positions = torch.tensor([1, 2])
    blocked = torch.tril(torch.ones(1, 1, 4, 4, dtype=torch.bool))
    open_mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    assert vg.mask_report(None, positions)["image_can_read_later"] is False
    assert vg.mask_report(blocked, positions)["image_can_read_later"] is False
    assert vg.mask_report(open_mask, positions)["image_can_read_later"] is True
    floats = torch.zeros(1, 1, 4, 4).masked_fill(~blocked, float("-inf"))
    assert vg.mask_report(floats, positions)["image_can_read_later"] is False


# G2 ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,style", [("qwen3vl", "qwen"), ("gemma4", "gemma"), ("molmo2", "molmo")]
)
def test_g2_image_tokens_are_invariant_under_causal_attention(key, style):
    ctx = FakeContext(key, TinyVLM(style))
    summary = vg.summarise_g2([vg.run_g2_clip(ctx, make_clip())])
    assert summary["mask_permits_invariance"]
    assert all(row["max_abs"] == 0.0 for row in summary["per_layer"].values())
    assert min(summary["control_min_last_token_rel_l2"].values()) > 0
    verdict = vg.g2_verdict(summary, "float32")
    assert verdict["pass"] and verdict["exact_zero"]


def test_g2_finds_dependence_under_bidirectional_attention():
    ctx = FakeContext("qwen3vl", TinyVLM("qwen", causal=False))
    summary = vg.summarise_g2([vg.run_g2_clip(ctx, make_clip())])
    assert summary["per_layer"]["0"]["max_abs"] > 0
    assert not vg.g2_verdict(summary, "float32")["pass"]


def test_g2_records_which_comparison_changed():
    ctx = FakeContext("qwen3vl", TinyVLM("qwen", causal=False))
    summary = vg.summarise_g2([vg.run_g2_clip(ctx, make_clip())])
    names = {row["comparison"] for row in summary["nonzero_comparisons"]}
    assert names == {"second_rotation", "foreign_option_set"}
    assert summary["nonzero_comparisons"][0]["first_changed_layer"] == "0"


def test_layer_lab_can_keep_only_chosen_inputs():
    model = TinyVLM("qwen")
    ctx = FakeContext("qwen3vl", model)
    inputs = ctx.encode("clip", OPTIONS, [])
    lab = vg.LayerLab(ctx.adapter, vg.image_positions(ctx.adapter, inputs), capture=True, input_layers=[0])
    with lab:
        vg.forward_logits(model, inputs)
    assert list(lab.inputs) == [0]
    assert len(lab.outputs) == N_LAYERS


def test_layer_lab_can_skip_outputs():
    model = TinyVLM("qwen")
    ctx = FakeContext("qwen3vl", model)
    inputs = ctx.encode("clip", OPTIONS, [])
    lab = vg.LayerLab(ctx.adapter, vg.image_positions(ctx.adapter, inputs), capture=True, keep_outputs=False)
    with lab:
        vg.forward_logits(model, inputs)
    assert len(lab.inputs) == N_LAYERS
    assert not lab.outputs and not lab.last


def test_g2_flags_text_before_image_as_not_applicable():
    ctx = FakeContext("qwen3vl", TinyVLM("qwen"), image_last=True)
    summary = vg.summarise_g2([vg.run_g2_clip(ctx, make_clip())])
    assert not summary["mask_permits_invariance"]
    assert not vg.g2_verdict(summary, "float32")["pass"]


# G3 ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,style", [("qwen3vl", "qwen"), ("gemma4", "gemma"), ("molmo2", "molmo")]
)
def test_g3_passes_on_clean_models(key, style):
    ctx = FakeContext(key, TinyVLM(style))
    row = vg.run_g3_clip(ctx, make_clip())
    for name, value in row["required"].items():
        assert value["max_abs"] == 0.0, name
    assert row["informational"]["full_vs_image_free"]["max_abs"] > 0
    assert vg.g3_verdict([row], "float32")["pass"]


def test_g3_fails_when_a_route_is_not_cut():
    ctx = FakeContext("qwen3vl", TinyVLM("qwen", leak=True))
    row = vg.run_g3_clip(ctx, make_clip())
    assert row["required"]["all_vision_cut_vs_image_free"]["max_abs"] > 1e-3
    assert row["required"]["cut_clip_vs_cut_donor"]["max_abs"] > 1e-3
    assert not vg.g3_verdict([row], "float32")["pass"]


def test_g3_shows_deepstack_survives_a_main_only_cut():
    ctx = FakeContext("qwen3vl", TinyVLM("qwen"))
    tests = vg.run_g3_clip(ctx, make_clip())["route_tests"]
    assert tests["main_only_cut_deepstack_live"]["max_abs_vs_image_free"] > 1e-3
    assert tests["deepstack_only_cut_main_live"]["max_abs_vs_image_free"] > 1e-3


def test_g3_locates_each_deepstack_injection_layer():
    ctx = FakeContext("qwen3vl", TinyVLM("qwen"))
    tests = vg.run_g3_clip(ctx, make_clip())["route_tests"]
    for slot in range(3):
        assert tests[f"deepstack_{slot}_cut"]["injected_after_lm_layer"] == slot


def test_g3_late_cut_is_incomplete_because_text_already_read_the_image():
    ctx = FakeContext("qwen3vl", TinyVLM("qwen"))
    tests = vg.run_g3_clip(ctx, make_clip())["route_tests"]
    assert tests["late_cut_image_positions_from_L3"]["max_abs_vs_image_free"] > 1e-3


def test_g3_molmo_late_cut_is_incomplete():
    ctx = FakeContext("molmo2", TinyVLM("molmo"))
    tests = vg.run_g3_clip(ctx, make_clip())["route_tests"]
    assert tests["late_cut_image_positions_from_L1"]["max_abs_vs_image_free"] > 1e-3


def test_g3_per_layer_route_survives_a_stream_cut():
    ctx = FakeContext("gemma4", TinyVLM("gemma"))
    tests = vg.run_g3_clip(ctx, make_clip())["route_tests"]
    assert tests["main_stream_cut_L0_per_layer_live"]["max_abs_vs_image_free"] > 1e-3
    assert tests["per_layer_cut_main_stream_live"]["max_abs_vs_full"] > 1e-3
    assert tests["main_stream_and_per_layer_cut"]["max_abs_vs_image_free"] == 0.0
    for row in tests["per_layer_input_difference"].values():
        assert row["image_positions_max_abs"] > 0
        assert row["text_positions_max_abs"] == 0.0


def test_g3_patch_after_kv_sharing_cannot_reach_the_answer():
    ctx = FakeContext("gemma4", TinyVLM("gemma", kv_shared=2))
    tests = vg.run_g3_clip(ctx, make_clip())["route_tests"]
    assert tests["kv_shared_layers"] == {"first": 2, "count": 2}
    assert tests["image_positions_from_first_kv_shared_layer"]["max_abs_vs_full"] == 0.0
    assert tests["image_positions_window_before_kv_sharing"]["max_abs_vs_full"] > 1e-3


# G4 ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,style", [("qwen3vl", "qwen"), ("gemma4", "gemma"), ("molmo2", "molmo")]
)
def test_g4_full_swap_matches_native_donor_forward(key, style):
    ctx = FakeContext(key, TinyVLM(style))
    row = vg.run_g4_clip(ctx, make_clip())
    assert row["native_vs_swap_max_abs"] == 0.0


def test_g4_swap_replaces_every_route():
    own = {"main": torch.zeros(4, 2), "deepstack_0": torch.zeros(4, 2)}
    donor = {"main": torch.ones(4, 2), "deepstack_0": torch.ones(4, 2)}
    out = vg.swap_tokens(own, donor, np.array([0, 2]))
    for tensor in out.values():
        assert tensor[:, 0].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert own["main"].sum() == 0


def test_g4_swap_refuses_mismatched_layouts():
    with pytest.raises(vg.LayoutError):
        vg.swap_tokens({"main": torch.zeros(4, 2)}, {"main": torch.zeros(3, 2)}, np.array([0]))


def test_g4_scores_every_option_in_the_rotated_order():
    ctx = FakeContext("qwen3vl", TinyVLM("qwen"))
    clip = make_clip()
    row = vg.run_g4_clip(ctx, clip)
    assert row["options"] == clip["gate_options"]
    assert row["target_position"] != 0
    assert len(row["base"]["joint_logprob"]) == 6
    assert set(row["swapped"]) == {"0.10", "0.25", "0.50", "1.00"}
    assert row["image_free_rms_nats"] > 0


def test_joint_logprob_reads_the_label_positions():
    ctx = FakeContext("molmo2", TinyVLM("molmo"))
    options = make_clip()["gate_options"]
    scored = vg.score_options(ctx, "clip", options)
    inputs = ctx.encode("clip", options, ctx.prefill_ids() + ctx.label_ids(options[0]))
    logits = vg.forward_logits(ctx.model, inputs)[0]
    label = ctx.label_ids(options[0])
    logprobs = torch.log_softmax(logits, dim=-1)
    n = logits.shape[0]
    expected = logprobs[n - 3, label[0]] + logprobs[n - 2, label[1]]
    assert scored["joint_logprob"][0] == pytest.approx(float(expected))
    assert scored["first_token_logit"][0] == pytest.approx(float(logits[n - 3, label[0]]))


def test_hooks_are_removed_after_use():
    model = TinyVLM("qwen")
    ctx = FakeContext("qwen3vl", model)
    clip = make_clip()
    inputs = ctx.encode("clip", clip["gate_options"], [])
    before = vg.last_logits(model, inputs)
    vg.run_g3_clip(ctx, clip)
    assert torch.equal(vg.last_logits(model, inputs), before)
    assert not model.model.visual._forward_hooks
    for layer in ctx.adapter.lm_layers:
        assert not layer._forward_pre_hooks and not layer._forward_hooks


# Architecture probes --------------------------------------------------------


def test_probe_reads_qwen_deepstack_inputs():
    ctx = FakeContext("qwen3vl", TinyVLM("qwen"))
    inputs = ctx.encode("clip", OPTIONS, [])
    report = vg.probe_architecture(ctx, "qwen3vl", inputs)
    assert report["deepstack_visual_indexes"] == [1, 2, 3]
    assert all(report["deepstack_input_equals_vit_block_output"].values())


def test_probe_separates_molmo_read_points_from_a_control_layer():
    ctx = FakeContext("molmo2", TinyVLM("molmo"))
    inputs = ctx.encode("clip", OPTIONS, [])
    report = vg.probe_architecture(ctx, "molmo2", inputs)
    assert report["vit_layers_resolved"] == [9, 8]
    assert all(v > 0 for v in report["logit_change_when_read_point_zeroed"].values())
    assert report["logit_change_when_control_zeroed"] == 0.0


def test_probe_lists_gemma_kv_shared_layers():
    ctx = FakeContext("gemma4", TinyVLM("gemma", kv_shared=2))
    inputs = ctx.encode("clip", OPTIONS, [])
    report = vg.probe_architecture(ctx, "gemma4", inputs)
    assert report["kv_shared_layers"] == [2, 3]
    assert report["kv_store_layers"] == [1]
