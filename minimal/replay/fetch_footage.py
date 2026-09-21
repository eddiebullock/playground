#!/usr/bin/env python3
"""Download a short public driving clip and explode it into data/frames.

Tries, in order:
1. comma2k19 example HEVC already on GitHub (ego dashcam, ~1 min)
2. Intel OpenVINO sample car-detection.mp4 (intersection CCTV, easy mp4)
3. comma2k19 preview.png duplicated into a short placeholder sequence

None of these are the team's proprietary footage. License notes are printed.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
FRAMES = ROOT / "data" / "frames"

COMMA_VIDEO = (
    "https://github.com/commaai/comma2k19/raw/master/"
    "Example_1/b0c9d2329ad1606b%7C2018-08-02--08-34-47/40/video.hevc"
)
COMMA_PREVIEW = (
    "https://github.com/commaai/comma2k19/raw/master/"
    "Example_1/b0c9d2329ad1606b%7C2018-08-02--08-34-47/40/preview.png"
)
INTEL_MP4 = (
    "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4"
)


def download(url: str, dest: Path, timeout: int = 60) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] GET {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "driver-thinker-practice"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if len(data) < 1000:
            print(f"[fetch] too small ({len(data)} bytes) — likely a git-lfs pointer, skip")
            return False
        dest.write_bytes(data)
        print(f"[fetch] wrote {dest} ({len(data)} bytes)")
        return True
    except Exception as exc:
        print(f"[fetch] failed: {exc}")
        return False


def ffmpeg_extract(video: Path, out_dir: Path, fps: float = 5.0, max_frames: int = 120) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"fps={fps}",
        "-frames:v",
        str(max_frames),
        str(out_dir / "%06d.jpg"),
    ]
    print("[fetch]", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr[-2000:])
        return 0
    return len(list(out_dir.glob("*.jpg")))


def placeholder_from_png(png: Path, out_dir: Path, n: int = 30) -> int:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(png))
    if img is None:
        return 0
    for i in range(n):
        # Slight brightness drift so DummyDriver waypoints are not identical.
        scale = 0.92 + 0.16 * (i / max(n - 1, 1))
        frame = cv2.convertScaleAbs(img, alpha=scale, beta=0)
        cv2.imwrite(str(out_dir / f"{i:06d}.jpg"), frame)
    return n


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    FRAMES.mkdir(parents=True, exist_ok=True)

    hevc = RAW / "comma2k19_example.hevc"
    if download(COMMA_VIDEO, hevc):
        n = ffmpeg_extract(hevc, FRAMES)
        if n:
            print(f"[fetch] comma2k19 -> {n} frames  (research dataset, see commaai/comma2k19)")
            return 0

    mp4 = RAW / "car-detection.mp4"
    if download(INTEL_MP4, mp4):
        n = ffmpeg_extract(mp4, FRAMES)
        if n:
            print(
                f"[fetch] intel sample-videos car-detection.mp4 -> {n} frames  "
                "(CCTV, not ego-vehicle; fine for plumbing)"
            )
            return 0

    png = RAW / "comma2k19_preview.png"
    if download(COMMA_PREVIEW, png):
        n = placeholder_from_png(png, FRAMES)
        print(f"[fetch] placeholder sequence from preview.png -> {n} frames")
        print("[fetch] drop a real clip into data/raw/ and re-run this script when you have one")
        return 0 if n else 1

    print("[fetch] nothing downloaded. Put any mp4/hevc in data/raw/ and run ffmpeg yourself.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
