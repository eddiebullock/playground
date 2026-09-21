"""Laptop-side advisory record. Mirrors thinker/schemas.py JSON fields."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass
class AdvisoryView:
    hazard: str = "UNKNOWN"
    action: str = "SLOW"
    speed_cap_mps: Optional[float] = 3.0
    raw_text: str = ""
    model_ts: float = 0.0
    inference_ms: float = 0.0
    model_id: str = "none"
    parse_ok: bool = False
    frame_index: int = -1
    rtt_ms: float = 0.0
    is_stale: bool = True
    age_ms: float = 0.0
    status: str = "default"  # default | ok | timeout | error | unreachable

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def from_server_json(body: dict[str, Any], rtt_ms: float) -> AdvisoryView:
    return AdvisoryView(
        hazard=str(body.get("hazard", "UNKNOWN")),
        action=str(body.get("action", "SLOW")),
        speed_cap_mps=body.get("speed_cap_mps"),
        raw_text=str(body.get("raw_text", "")),
        model_ts=float(body.get("model_ts") or 0.0),
        inference_ms=float(body.get("inference_ms") or 0.0),
        model_id=str(body.get("model_id", "")),
        parse_ok=bool(body.get("parse_ok", False)),
        frame_index=int(body.get("frame_index", -1)),
        rtt_ms=rtt_ms,
        is_stale=False,
        age_ms=0.0,
        status="ok",
    )


def safe_default() -> AdvisoryView:
    return AdvisoryView(
        hazard="UNKNOWN",
        action="SLOW",
        speed_cap_mps=3.0,
        raw_text="HAZARD=UNKNOWN; ACTION=SLOW; SPEED_CAP_MPS=3.0",
        status="default",
        is_stale=True,
    )
