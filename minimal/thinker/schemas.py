"""Structured advisory contract shared by the Thinker server.

Keep this file inside thinker/ so the HPC copy is self-contained.
Laptop-side code duplicates the parse/format helpers in integration/schemas.py
on purpose: swapping to an in-process Thinker later means importing this module
instead of posting HTTP, not rewriting the combiner.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Optional

VALID_HAZARDS = (
    "NONE",
    "PEDESTRIAN",
    "CYCLIST",
    "VEHICLE",
    "ROAD_BLOCK",
    "UNKNOWN",
)
VALID_ACTIONS = (
    "KEEP",
    "SLOW",
    "STOP",
    "NUDGE_LEFT",
    "NUDGE_RIGHT",
)

PROMPT_V1 = (
    "You are a driving safety advisor for an autonomous delivery vehicle. "
    "Look at this front camera frame. Reply with EXACTLY one line and nothing else, "
    "using this format:\n"
    "HAZARD=<NONE|PEDESTRIAN|CYCLIST|VEHICLE|ROAD_BLOCK|UNKNOWN>; "
    "ACTION=<KEEP|SLOW|STOP|NUDGE_LEFT|NUDGE_RIGHT>; "
    "SPEED_CAP_MPS=<number or NONE>\n"
    "Rules: ACTION=STOP only if collision is imminent. ACTION=SLOW if a road user "
    "is close. ACTION=NUDGE_LEFT/RIGHT only for a small lateral offset. "
    "SPEED_CAP_MPS is a hard cap in meters per second, or NONE if no cap."
)

_LINE_RE = re.compile(
    r"HAZARD\s*=\s*(?P<hazard>[A-Z_]+)\s*;\s*"
    r"ACTION\s*=\s*(?P<action>[A-Z_]+)\s*;\s*"
    r"SPEED_CAP_MPS\s*=\s*(?P<cap>[A-Za-z0-9.+-]+)",
    re.IGNORECASE,
)


@dataclass
class Advisory:
    hazard: str
    action: str
    speed_cap_mps: Optional[float]
    raw_text: str
    model_ts: float
    inference_ms: float
    model_id: str
    parse_ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def format_line(hazard: str, action: str, speed_cap_mps: Optional[float]) -> str:
    cap = "NONE" if speed_cap_mps is None else f"{speed_cap_mps:.1f}"
    return f"HAZARD={hazard}; ACTION={action}; SPEED_CAP_MPS={cap}"


def parse_advisory_text(text: str) -> tuple[str, str, Optional[float], bool]:
    """Return (hazard, action, speed_cap_mps, parse_ok)."""
    cleaned = (text or "").strip().replace("\n", " ")
    match = _LINE_RE.search(cleaned)
    if not match:
        return "UNKNOWN", "SLOW", 3.0, False
    hazard = match.group("hazard").upper()
    action = match.group("action").upper()
    cap_raw = match.group("cap").upper()
    if hazard not in VALID_HAZARDS:
        hazard = "UNKNOWN"
    if action not in VALID_ACTIONS:
        action = "SLOW"
    if cap_raw in ("NONE", "NULL", "NA"):
        cap: Optional[float] = None
    else:
        try:
            cap = float(cap_raw)
        except ValueError:
            cap = 3.0
    return hazard, action, cap, True


def safe_default(model_id: str, model_ts: float) -> Advisory:
    return Advisory(
        hazard="UNKNOWN",
        action="SLOW",
        speed_cap_mps=3.0,
        raw_text=format_line("UNKNOWN", "SLOW", 3.0),
        model_ts=model_ts,
        inference_ms=0.0,
        model_id=model_id,
        parse_ok=False,
    )
