"""Driver interface. Swap DummyDriver for GarageDriver (or a team class) here only."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Protocol

import numpy as np


COMMANDS = {
    "LEFT": 0,
    "RIGHT": 1,
    "STRAIGHT": 2,
    "LANEFOLLOW": 3,
    "CHANGELANELEFT": 4,
    "CHANGELANERIGHT": 5,
}


@dataclass
class DriverOutput:
    waypoints: np.ndarray          # (N, 3) x, y, yaw in ego frame, metres / radians
    target_speed_mps: float
    bev_semantic: Optional[np.ndarray] = None
    boxes: list[dict[str, Any]] = field(default_factory=list)
    inference_ms: float = 0.0
    backend: str = "dummy"
    extras: dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        d = asdict(self)
        d["waypoints"] = self.waypoints.tolist()
        if self.bev_semantic is not None:
            # Keep logs small: shape + a few stats, not the full map.
            d["bev_semantic"] = {
                "shape": list(self.bev_semantic.shape),
                "nonzero": int(np.count_nonzero(self.bev_semantic)),
            }
        return d


class Driver(Protocol):
    backend_name: str

    def predict(
        self,
        frame_bgr: np.ndarray,
        speed_mps: float,
        command: str = "LANEFOLLOW",
    ) -> DriverOutput:
        ...


def load_driver(cfg: dict[str, Any]) -> Driver:
    backend = cfg.get("backend", "dummy")
    if backend == "dummy":
        from .dummy_driver import DummyDriver

        return DummyDriver(cfg)
    if backend == "garage":
        from .garage_driver import GarageDriver

        return GarageDriver(cfg)
    raise ValueError(f"unknown driver backend: {backend}")
