"""CPU-only public-practice Driver.

Produces a kinematic waypoint fan-out from current speed. This is NOT
TransFuser++. Use it to wire the async loop today; swap backend=garage
in configs/public.yaml once the official checkpoint is on disk.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .interface import DriverOutput


class DummyDriver:
    backend_name = "dummy"

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.n_waypoints = int(cfg.get("n_waypoints", 8))
        self.dt = 0.5  # seconds between waypoints, typical TF++ horizon step

    def predict(
        self,
        frame_bgr: np.ndarray,
        speed_mps: float,
        command: str = "LANEFOLLOW",
    ) -> DriverOutput:
        t0 = time.time()
        # Tiny image-dependent wobble so logs are not identical every frame.
        mean = float(frame_bgr.mean()) if frame_bgr.size else 128.0
        yaw_bias = (mean - 128.0) / 128.0 * 0.02
        if command == "LEFT":
            yaw_bias += 0.05
        elif command == "RIGHT":
            yaw_bias -= 0.05
        speed = max(0.5, float(speed_mps))
        xs, ys, yaws = [], [], []
        x = y = yaw = 0.0
        for _ in range(self.n_waypoints):
            yaw = yaw + yaw_bias
            x = x + speed * self.dt * np.cos(yaw)
            y = y + speed * self.dt * np.sin(yaw)
            xs.append(x)
            ys.append(y)
            yaws.append(yaw)
        wps = np.stack([xs, ys, yaws], axis=1).astype(np.float32)
        bev = np.zeros((64, 64), dtype=np.uint8)
        bev[32:, :] = 1  # fake road band
        return DriverOutput(
            waypoints=wps,
            target_speed_mps=speed,
            bev_semantic=bev,
            boxes=[],
            inference_ms=(time.time() - t0) * 1000.0,
            backend=self.backend_name,
            extras={"command": command, "note": "dummy_not_transfuser"},
        )
