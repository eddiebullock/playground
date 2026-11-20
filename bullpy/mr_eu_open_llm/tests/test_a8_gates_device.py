"""The device-aware gate hooks (research/scripts/a8_gates_device.py) on the tiny fake Qwen model.

`device_map="auto"` splits a large model over GPUs: the inputs and the first layers on one device,
later layers on another, each layer moving its inputs to its own device inside its forward, so a hook
sees a tensor where the previous layer left it. The tiny model of tests/test_vision_gates.py is split
the same way here, over Apple's MPS (which holds the inputs, as the first GPU does) and the CPU. On that
split the frozen hooks fail with the cluster's error, and the device-aware hooks give the frozen hooks'
values. Tests that need a second device are skipped where there is none.
"""

import json
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT, ROOT / "tests", ROOT / "research" / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import a8_gates_device as gd  # noqa: E402
import test_vision_gates as tvg  # noqa: E402
from scripts import vision_gates as vg  # noqa: E402

HAS_MPS = torch.backends.mps.is_available()
CPU = torch.device("cpu")
HOME = torch.device("mps") if HAS_MPS else CPU
needs_two = pytest.mark.skipif(not HAS_MPS, reason="needs a second device (MPS)")
SPLIT = 3  # Qwen's deepstack enters after layers 0 to 2, which stay with the inputs; layer 3 goes to the CPU.


class DeviceAttention(tvg.TinyAttention):
    """tvg.TinyAttention with its causal mask built where the scores are: same weights, same arithmetic."""

    def forward(self, hidden, causal, store):
        query, key, value = self.q(hidden), self.k(hidden), self.v(hidden)
        scores = query @ key.transpose(-1, -2) / math.sqrt(tvg.DIM)
        if causal:
            n = hidden.shape[1]
            future = torch.triu(torch.ones(n, n, dtype=torch.bool, device=hidden.device), 1)
            scores = scores.masked_fill(future, float("-inf"))
        return torch.softmax(scores, dim=-1) @ value


def _align(module, device):
    """Put a module on a device and move its tensor arguments there inside each call, as accelerate does."""
    module.to(device)
    forward = module.forward

    def moved(*args, **kwargs):
        args = tuple(a.to(device) if torch.is_tensor(a) else a for a in args)
        kwargs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in kwargs.items()}
        return forward(*args, **kwargs)

    module.forward = moved
    return module


def qwen(home, split=None):
    """The tiny Qwen model on `home`, with its decoder layers from `split` on the CPU."""
    model = tvg.TinyVLM("qwen")
    for layer in model.model.language_model.layers:
        layer.self_attn.__class__ = DeviceAttention
    _align(model, home)
    if split is not None:
        for layer in model.model.language_model.layers[split:]:
            _align(layer, CPU)
        _align(model.head, home)
    return model.eval()


class Context(tvg.FakeContext):
    """tvg.FakeContext with its inputs on a chosen device; the real context puts them on the first GPU."""

    def __init__(self, model, device):
        super().__init__("qwen3vl", model)
        self.device = device

    def encode(self, vision_clip, options, extra_ids=()):
        inputs = super().encode(vision_clip, options, extra_ids)
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}


def gates(ctx):
    clip = tvg.make_clip()
    return {"g2": vg.run_g2_clip(ctx, clip), "g3": vg.run_g3_clip(ctx, clip), "g4": vg.run_g4_clip(ctx, clip)}


@needs_two
def test_the_frozen_hooks_fail_on_a_split_model_with_the_clusters_error():
    ctx = Context(qwen(HOME, SPLIT), HOME)
    with pytest.raises(RuntimeError, match="indices should be either on cpu or on the same device as the indexed tensor"):
        vg.run_g2_clip(ctx, tvg.make_clip())


@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=needs_two)])
def test_on_one_device_the_device_hooks_change_nothing(device):
    device = torch.device(device)
    frozen = gates(Context(qwen(device), device))
    with gd.device_hooks():
        patched = gates(Context(qwen(device), device))
    assert json.dumps(patched, sort_keys=True) == json.dumps(frozen, sort_keys=True)


@needs_two
def test_on_a_split_model_the_captures_are_the_frozen_hooks_values_held_where_they_hold_them():
    model = qwen(HOME, SPLIT)
    ctx = Context(model, HOME)
    inputs = ctx.encode("clip", tvg.OPTIONS, [])
    positions = vg.image_positions(ctx.adapter, inputs)
    assert positions.device.type == HOME.type
    # Reference: the frozen hooks given the positions on the CPU, an index every device accepts.
    ref = vg.LayerLab(ctx.adapter, positions.cpu(), capture=True)
    with ref:
        ref_logits = vg.forward_logits(model, inputs)
    with gd.device_hooks():
        lab = vg.LayerLab(ctx.adapter, positions, capture=True)
    with lab:
        logits = vg.forward_logits(model, inputs)
    assert torch.equal(logits.cpu(), ref_logits.cpu())
    assert lab.outputs[SPLIT].device.type == "cpu" and lab.outputs[0].device.type == HOME.type
    for name in ("inputs", "outputs", "last"):
        mine, theirs = getattr(lab, name), getattr(ref, name)
        assert sorted(mine) == sorted(theirs) == list(range(tvg.N_LAYERS))
        for k in mine:
            assert mine[k].device == theirs[k].device, (name, k)
            assert torch.equal(mine[k].cpu(), theirs[k].cpu()), (name, k)
    assert lab.positions is positions


@needs_two
def test_on_a_split_model_a_replacement_gives_the_frozen_hooks_logits():
    model = qwen(HOME, SPLIT)
    ctx = Context(model, HOME)
    inputs = ctx.encode("clip", tvg.OPTIONS, [])
    positions = vg.image_positions(ctx.adapter, inputs)
    with gd.device_hooks():
        free = vg.LayerLab(ctx.adapter, positions, capture=True, keep_outputs=False)
    with free:
        vg.forward_logits(model, vg.strip_pixels(inputs, ctx.adapter.pixel_keys))
    patch = dict(free.inputs)
    ref = vg.LayerLab(ctx.adapter, positions.cpu(), replace_hidden=patch)
    with ref:
        want = vg.last_logits(model, inputs)
    with gd.device_hooks():
        lab = vg.LayerLab(ctx.adapter, positions, replace_hidden=patch)
    with lab:
        got = vg.last_logits(model, inputs)
    assert torch.equal(got.cpu(), want.cpu())
    assert not torch.equal(got.cpu(), vg.last_logits(model, inputs).cpu())


@needs_two
def test_on_a_split_model_the_gates_give_the_frozen_hooks_results():
    model = qwen(HOME, SPLIT)
    # Frozen hooks, with the inputs (so the positions) on the CPU; the model moves them to the first device.
    # Every value is equal, not just close: the same tensors, on the same devices, give the same values.
    want = gates(Context(model, CPU))
    with gd.device_hooks():
        got = gates(Context(model, HOME))
    assert json.dumps(got, sort_keys=True) == json.dumps(want, sort_keys=True)
    summary = vg.summarise_g2([got["g2"]])
    assert all(row["max_abs"] == 0.0 for row in summary["per_layer"].values())
    assert vg.g2_verdict(summary, "float32")["pass"]
    assert vg.g3_verdict([got["g3"]], "float32")["pass"]
    assert got["g4"]["native_vs_swap_max_abs"] == 0.0


def test_models_with_per_layer_inputs_are_refused():
    ctx = tvg.FakeContext("gemma4", tvg.TinyVLM("gemma"))
    inputs = ctx.encode("clip", tvg.OPTIONS, [])
    with gd.device_hooks(), pytest.raises(SystemExit, match="per-layer"):
        vg.LayerLab(ctx.adapter, vg.image_positions(ctx.adapter, inputs))


def test_run_calls_the_frozen_main_with_the_device_hooks_then_restores(monkeypatch):
    seen = {}

    def fake_main():
        seen["cls"], seen["argv"] = vg.LayerLab, list(sys.argv)
        raise RuntimeError("stop")

    monkeypatch.setattr(vg, "main", fake_main)
    argv = list(sys.argv)
    with pytest.raises(RuntimeError, match="stop"):
        gd.run(["--model", "qwen3vl", "--gates", "g2"])
    assert seen["cls"] is gd.DeviceLayerLab
    assert seen["argv"] == [vg.__file__, "--model", "qwen3vl", "--gates", "g2"]
    assert vg.LayerLab is gd.FROZEN_LAYER_LAB
    assert sys.argv == argv


# The 8B check's rule (research/scripts/a8_gates_device_check.py), on gate files of the real format's shape.

import copy  # noqa: E402

import a8_gates_device_check as check  # noqa: E402


def gate_file(seconds=10.0):
    return {
        "model": "qwen3vl", "smoke": False, "seed": 7, "script_sha256": "ab" * 32,
        "versions": {"torch": "2.4.1"},
        "stages": {"bfloat16": {
            "model_load_seconds": seconds, "gpu": "NVIDIA A100-SXM4-80GB", "input_keys": ["input_ids", "pixel_values"],
            "g2": {"seconds": seconds, "summary": {"per_layer": {"0": {"max_abs": 0.0}}}, "verdict": {"pass": True}},
            "g3": {"seconds": seconds, "clips": [{"required": {"repeat_forward": {"max_abs": 0.0}}, "drop": 0.31}]},
            "g4": {"seconds": seconds, "clips": [{"native_vs_swap_max_abs": 0.0, "shift": [0.5, 1.25]}]},
        }},
        "verdict": {"G2": {"pass": True}, "G3": {"pass": True}, "G4": {"pass": True}},
    }


def test_the_check_passes_when_only_the_timings_differ():
    report = check.compare(gate_file(10.0), gate_file(11.0), gate_file(12.5))
    assert report["passed"]
    assert report["n_compared"] > 0
    assert "/stages/bfloat16/g2/seconds" in report["timings_not_compared"]
    assert "/stages/bfloat16/model_load_seconds" in report["timings_not_compared"]


def test_the_check_fails_when_the_patched_file_differs_anywhere_else():
    patched = gate_file()
    patched["stages"]["bfloat16"]["g4"]["clips"][0]["shift"][1] = 1.2500001
    report = check.compare(gate_file(), gate_file(), patched)
    assert not report["passed"]
    assert report["patched_differs_at"] == ["/stages/bfloat16/g4/clips[0]/shift[1]"]


def test_the_check_fails_when_the_two_frozen_runs_differ():
    other = gate_file()
    other["stages"]["bfloat16"]["g3"]["clips"][0]["drop"] = 0.32
    report = check.compare(gate_file(), other, gate_file())
    assert not report["passed"]
    assert report["frozen_runs_differ_at"] == ["/stages/bfloat16/g3/clips[0]/drop"]


def test_the_check_fails_on_a_missing_or_extra_leaf_and_on_an_incomplete_run():
    extra = gate_file()
    extra["float32_failed"] = "CUDA out of memory"
    assert check.compare(gate_file(), gate_file(), extra)["leaves_not_in_all_three"]["patched"] == ["/float32_failed"]
    assert not check.compare(gate_file(), gate_file(), extra)["passed"]
    shorter = gate_file()
    shorter["stages"]["bfloat16"]["g4"]["clips"][0]["shift"].pop()
    assert not check.compare(gate_file(), gate_file(), shorter)["passed"]
    smoke, unfinished = gate_file(), gate_file()
    smoke["smoke"] = True
    del unfinished["verdict"]
    for bad in (smoke, unfinished):
        report = check.compare(gate_file(), gate_file(), bad)
        assert not report["complete"]["patched"] and not report["passed"]


def test_the_check_tells_types_apart_and_nan_from_numbers():
    as_int = gate_file()
    as_int["seed"] = 7.0
    assert check.compare(gate_file(), gate_file(), as_int)["patched_differs_at"] == ["/seed"]
    nan = gate_file()
    nan["stages"]["bfloat16"]["g3"]["clips"][0]["drop"] = float("nan")
    assert check.compare(nan, copy.deepcopy(nan), copy.deepcopy(nan))["passed"]


def test_the_check_writes_its_report_once_and_exits_non_zero_on_a_failure(tmp_path):
    paths = []
    for name, obj in (("a", gate_file(1.0)), ("b", gate_file(2.0)), ("p", gate_file(3.0))):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(obj))
        paths.append(str(path))
    report = tmp_path / "report.json"
    out = check.main(["--frozen", paths[0], paths[1], "--patched", paths[2], "--report", str(report)])
    assert out["passed"] and json.loads(report.read_text())["passed"]
    with pytest.raises(SystemExit, match="runs once"):
        check.main(["--frozen", paths[0], paths[1], "--patched", paths[2], "--report", str(report)])
    bad = gate_file()
    bad["seed"] = 8
    (tmp_path / "p.json").write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="did not pass"):
        check.main(["--frozen", paths[0], paths[1], "--patched", paths[2], "--report", str(tmp_path / "r2.json")])
    assert json.loads((tmp_path / "r2.json").read_text())["patched_differs_at"] == ["/seed"]


# The launcher's use of the device hooks (research/scripts/a8_launch.py, DEVICE_HOOKS).

import a8_launch  # noqa: E402


def test_the_launcher_uses_the_device_hooks_for_qwen3vl_32b_gates_only():
    assert a8_launch.DEVICE_HOOKS == {"qwen3vl_32b": "DECISIONS 2026-11-17, row 102"}
    source = (ROOT / "research/scripts/a8_launch.py").read_text(encoding="utf-8")
    assert 'if args.step == "gates" and args.a8_id in DEVICE_HOOKS:' in source
    # The record says so before the attempt starts, so a killed attempt is on record with it.
    assert source.index('record["gates_device_hooks"]') < source.index('record["status"] = "started"')


def test_the_device_hooks_run_the_frozen_gate_arguments_and_a_clean_exit_finishes(monkeypatch):
    seen = []

    def fake_run(argv, code=None):
        seen.append(list(argv))
        raise SystemExit(code)

    design = a8_launch.a8_design.load_design()
    spec = a8_launch.command(design, "gates", {"a8_id": "qwen3vl_32b", "key": "qwen3vl"})
    for code in (None, 0):
        monkeypatch.setattr(gd, "run", lambda argv, code=code: fake_run(argv, code))
        a8_launch.run_device_hooks(spec)
    assert seen == [spec["argv"], spec["argv"]]
    assert spec["argv"][:2] == ["--model", "qwen3vl"] and spec["argv"][-2:] == ["--float32_max_frames", "6"]
    monkeypatch.setattr(gd, "run", lambda argv: fake_run(argv, 2))
    with pytest.raises(SystemExit) as info:
        a8_launch.run_device_hooks(spec)
    assert info.value.code == 2
    with pytest.raises(SystemExit, match="frozen gate script only"):
        a8_launch.run_device_hooks(dict(spec, module="scripts.face_input_validity"))
