"""Single place for device / dtype decisions.

CUDA uses bf16. MPS uses bf16 when the build reports it, otherwise fp16,
and always sets ``PYTORCH_ENABLE_MPS_FALLBACK=1``. CPU uses fp32 for
tiny tests.
"""

from __future__ import annotations

import os
from typing import Any

from esvlm.types import DeviceSpec


def detect_device(*, prefer: str | None = None) -> DeviceSpec:
    """Return the runtime device without importing torch at module load.

    ``prefer`` may be ``cuda``, ``mps``, ``cpu``, or None (auto).
    """

    torch = _torch()
    requested = (prefer or os.environ.get("ESVLM_DEVICE") or "auto").lower()
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        return DeviceSpec(kind="cuda", dtype_name="bfloat16", device_str="cuda")
    if requested == "mps" or (
        requested == "auto" and getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        dtype_name = "bfloat16" if _mps_bf16_ok(torch) else "float16"
        return DeviceSpec(kind="mps", dtype_name=dtype_name, device_str="mps")
    return DeviceSpec(kind="cpu", dtype_name="float32", device_str="cpu")


def torch_dtype(spec: DeviceSpec) -> Any:
    torch = _torch()
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[spec.dtype_name]


def torch_device(spec: DeviceSpec) -> Any:
    torch = _torch()
    return torch.device(spec.device_str)


def _mps_bf16_ok(torch: Any) -> bool:
    try:
        return bool(torch.backends.mps.is_macos_or_newer(14, 0))
    except Exception:
        return False


def _torch() -> Any:
    import torch

    return torch
