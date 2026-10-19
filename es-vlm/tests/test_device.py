from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from esvlm.utils.device import detect_device, torch_device, torch_dtype  # noqa: E402


def test_detect_device_returns_known_kind() -> None:
    spec = detect_device()
    assert spec.kind in {"cuda", "mps", "cpu"}
    assert spec.dtype_name in {"bfloat16", "float16", "float32"}
    if spec.kind == "cpu":
        assert spec.dtype_name == "float32"
    if spec.kind == "cuda":
        assert spec.dtype_name == "bfloat16"
    device = torch_device(spec)
    dtype = torch_dtype(spec)
    x = torch.zeros(2, device=device, dtype=dtype)
    assert x.device.type == spec.kind
    assert x.dtype == dtype


def test_prefer_cpu() -> None:
    spec = detect_device(prefer="cpu")
    assert spec.kind == "cpu"
    assert spec.dtype_name == "float32"
