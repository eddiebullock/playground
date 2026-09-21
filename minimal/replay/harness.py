#!/usr/bin/env python3
"""Frame-by-frame replay of Driver + async Thinker + combiner.

Example (laptop, mock Thinker already running on :8000):

    python -m replay.harness --config configs/public.yaml

In-process mock (no HTTP server):

    python -m replay.harness --config configs/public.yaml --thinker local-mock

HPC Thinker (after SSH tunnel):

    python -m replay.harness --config configs/public.yaml --thinker-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from driver.interface import load_driver
from integration.advisory import AdvisoryView
from integration.combiner import combine
from integration.thinker_client import AsyncThinker, HttpThinkerBackend, LocalMockBackend


def list_frames(frames_dir: Path) -> list[Path]:
    files = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    # glob twice can duplicate if we are not careful; unique while preserving order
    seen = set()
    out = []
    for f in files:
        if f.name in seen:
            continue
        seen.add(f.name)
        out.append(f)
    return out


def stamp_stale(adv: AdvisoryView, frame_index: int) -> AdvisoryView:
    adv.is_stale = (adv.status != "ok") or (adv.frame_index != frame_index)
    return adv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "public.yaml"))
    parser.add_argument(
        "--thinker",
        default="http",
        choices=["http", "local-mock"],
        help="http = network Thinker (HPC or local mock server). local-mock = in-process.",
    )
    parser.add_argument("--thinker-url", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frames-dir", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    frames_dir = Path(args.frames_dir or cfg["replay"]["frames_dir"])
    if not frames_dir.is_absolute():
        frames_dir = ROOT / frames_dir
    frames = list_frames(frames_dir)
    if not frames:
        print(f"[harness] no frames in {frames_dir}. Run: python -m replay.fetch_footage")
        return 1
    limit = args.max_frames or int(cfg["replay"].get("max_frames") or len(frames))
    frames = frames[:limit]

    driver = load_driver(cfg["driver"])
    print(f"[harness] driver backend={driver.backend_name}  frames={len(frames)}")

    if args.thinker == "local-mock":
        backend = LocalMockBackend()
        print("[harness] Thinker = in-process mock (not the HPC model)")
    else:
        url = args.thinker_url or cfg["thinker"]["url"]
        timeout_s = float(cfg["thinker"].get("timeout_s", 8.0))
        backend = HttpThinkerBackend(url=url, timeout_s=timeout_s)
        healthy = backend.health()
        print(f"[harness] Thinker HTTP {url} health={healthy}")
        if not healthy:
            print("[harness] server down — loop will keep last-known / safe default")

    thinker = AsyncThinker(backend)
    log_dir = ROOT / cfg["replay"]["log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"replay_{int(time.time())}.jsonl"

    speed = float(cfg["driver"].get("default_speed_mps", 8.0))
    command = str(cfg["driver"].get("default_command", "LANEFOLLOW"))

    print(f"[harness] logging to {log_path}")
    header = (
        "frame  driver_ms  think_rtt  think_stale  action      "
        "speed  cap?  loop_ms"
    )
    print(header)

    try:
        with open(log_path, "w", encoding="utf-8") as logf:
            for i, path in enumerate(frames):
                loop_t0 = time.time()
                frame = cv2.imread(str(path))
                if frame is None:
                    print(f"[harness] skip unreadable {path}")
                    continue

                thinker.submit(frame, frame_index=i)
                driver_out = driver.predict(frame, speed_mps=speed, command=command)
                adv = stamp_stale(thinker.latest(), frame_index=i)
                combined = combine(driver_out, adv, cfg["combiner"])
                loop_ms = (time.time() - loop_t0) * 1000.0

                rec = {
                    "frame_index": i,
                    "frame_path": str(path.relative_to(ROOT)),
                    "wall_ts": time.time(),
                    "driver": driver_out.to_jsonable(),
                    "thinker": adv.to_dict(),
                    "combined": combined.to_jsonable(),
                    "timings_ms": {
                        "driver": driver_out.inference_ms,
                        "thinker_inference": adv.inference_ms,
                        "thinker_rtt": adv.rtt_ms,
                        "thinker_age": adv.age_ms,
                        "loop": loop_ms,
                    },
                    "source": cfg.get("source", "public_practice"),
                }
                logf.write(json.dumps(rec) + "\n")
                logf.flush()

                stale = "stale" if adv.is_stale else "fresh"
                cap = "yes" if combined.speed_capped else "no"
                print(
                    f"{i:5d}  {driver_out.inference_ms:8.1f}  {adv.rtt_ms:8.1f}  "
                    f"{stale:11s}  {adv.action:10s}  {combined.target_speed_mps:5.1f}  "
                    f"{cap:3s}  {loop_ms:7.1f}"
                )
    finally:
        thinker.close()

    print(f"[harness] done. {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
