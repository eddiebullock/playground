#!/usr/bin/env python3
"""Run the Driver alone on one sample frame. No Thinker involved."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from driver.interface import load_driver


def main() -> int:
    cfg_path = ROOT / "configs" / "public.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    frames_dir = ROOT / cfg["replay"]["frames_dir"]
    frames = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    if not frames:
        print(f"no frames in {frames_dir}; run python -m replay.fetch_footage")
        return 1
    driver = load_driver(cfg["driver"])
    frame = cv2.imread(str(frames[0]))
    out = driver.predict(frame, speed_mps=8.0, command="LANEFOLLOW")
    print(f"backend={out.backend}  inference_ms={out.inference_ms:.1f}")
    print(f"target_speed_mps={out.target_speed_mps:.2f}")
    print("waypoints (x, y, yaw):")
    for i, row in enumerate(out.waypoints):
        print(f"  {i}: {row.tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
