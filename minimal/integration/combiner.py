"""Thinker advises; it does not replace the Driver trajectory.

Rules:
- KEEP: pass Driver waypoints and speed through.
- SLOW / STOP: hard-cap target speed (and scale waypoint spacing).
- NUDGE_LEFT / NUDGE_RIGHT: add a lateral offset to every waypoint (advisory).
- Explicit SPEED_CAP_MPS from the Thinker, if present, wins over the action default.
- Missing/failed Thinker: last-known advisory, or the config safe default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from driver.interface import DriverOutput
from .advisory import AdvisoryView


@dataclass
class CombinedDecision:
    waypoints: np.ndarray
    target_speed_mps: float
    speed_capped: bool
    lateral_nudge_m: float
    reason: str

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "waypoints": self.waypoints.tolist(),
            "target_speed_mps": self.target_speed_mps,
            "speed_capped": self.speed_capped,
            "lateral_nudge_m": self.lateral_nudge_m,
            "reason": self.reason,
        }


def combine(
    driver_out: DriverOutput,
    advisory: AdvisoryView,
    cfg: dict[str, Any],
) -> CombinedDecision:
    caps = cfg.get("action_speed_caps_mps") or {}
    nudge_y = float(cfg.get("nudge_y_m", 0.4))
    action = advisory.action if advisory.action else cfg.get("default_action", "SLOW")

    cap: Optional[float] = advisory.speed_cap_mps
    if cap is None and action in caps:
        cap = caps[action]

    wps = driver_out.waypoints.copy()
    speed = float(driver_out.target_speed_mps)
    speed_capped = False
    if cap is not None and speed > float(cap):
        scale = float(cap) / max(speed, 1e-3)
        wps[:, 0:2] = wps[:, 0:2] * scale
        speed = float(cap)
        speed_capped = True

    lateral = 0.0
    if action == "NUDGE_LEFT":
        lateral = nudge_y
    elif action == "NUDGE_RIGHT":
        lateral = -nudge_y
    if lateral != 0.0 and wps.shape[1] >= 2:
        wps[:, 1] = wps[:, 1] + lateral

    reason = (
        f"action={action} hazard={advisory.hazard} "
        f"cap={cap} stale={advisory.is_stale} status={advisory.status}"
    )
    return CombinedDecision(
        waypoints=wps,
        target_speed_mps=speed,
        speed_capped=speed_capped,
        lateral_nudge_m=lateral,
        reason=reason,
    )
