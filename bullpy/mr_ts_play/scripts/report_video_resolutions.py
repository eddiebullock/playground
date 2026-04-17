#!/usr/bin/env python3
"""
Report decoded video frame resolutions for dataset roots.

This is meant to support writing accurate Methods text like:
  "Frames were extracted at native video resolution (X×Y pixels ...)"

It samples up to N videos under each root (recursive), opens with OpenCV, and
counts observed (width, height) pairs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import subprocess
from typing import Optional


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".flv", ".wmv"}


def iter_video_files(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            yield p


def get_resolution_ffprobe(video_path: Path) -> Optional[tuple[int, int]]:
    """
    Return (w, h) using ffprobe, or None if unavailable/fails.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0:s=x",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    out = (result.stdout or "").strip()
    if not out or "x" not in out:
        return None
    try:
        w_s, h_s = out.split("x", 1)
        w = int(w_s)
        h = int(h_s)
    except ValueError:
        return None
    if w <= 0 or h <= 0:
        return None
    return (w, h)


def sample_resolutions(root: Path, max_videos: int, stride: int) -> tuple[Counter, int, int]:
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError:
        cv2 = None

    counts: Counter[tuple[int, int]] = Counter()
    seen = 0
    opened = 0
    examples: list[tuple[Path, int, int]] = []

    for i, video_path in enumerate(iter_video_files(root)):
        if stride > 1 and (i % stride) != 0:
            continue
        if seen >= max_videos:
            break

        seen += 1
        w = h = 0
        if cv2 is not None:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        else:
            res = get_resolution_ffprobe(video_path)
            if res is None:
                continue
            w, h = res

        if w > 0 and h > 0:
            counts[(w, h)] += 1
            opened += 1
            if len(examples) < 5:
                examples.append((video_path, w, h))

    # stash examples on the function so main() can print them without changing the return type
    sample_resolutions._examples = examples  # type: ignore[attr-defined]
    return counts, seen, opened


def fmt_counts(counts: Counter, top_k: int) -> str:
    if not counts:
        return "  (no decodable videos found)\n"
    lines = []
    for (w, h), n in counts.most_common(top_k):
        lines.append(f"  - {w}×{h}: {n}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mindreading-root", type=str, default="", help="MindReading dataset root directory")
    ap.add_argument("--eu-emotion-root", type=str, default="", help="EU-Emotion dataset root directory")
    ap.add_argument("--max-videos", type=int, default=200, help="Max videos to sample per root")
    ap.add_argument("--stride", type=int, default=1, help="Only consider every Nth video (speed)")
    ap.add_argument("--top-k", type=int, default=10, help="Show top-K resolutions")
    args = ap.parse_args()

    roots: list[tuple[str, str]] = []
    if args.mindreading_root:
        roots.append(("MindReading", args.mindreading_root))
    if args.eu_emotion_root:
        roots.append(("EU-Emotion", args.eu_emotion_root))

    if not roots:
        raise SystemExit("Provide at least one of --mindreading-root or --eu-emotion-root")

    for label, root_s in roots:
        root = Path(root_s).expanduser()
        print("=" * 72)
        print(f"{label} root: {root}")
        if not root.exists():
            print("  (root does not exist on this machine)")
            continue
        counts, seen, opened = sample_resolutions(root, max_videos=args.max_videos, stride=args.stride)
        print(f"  sampled_files_considered: {seen}")
        print(f"  successfully_opened:      {opened}")
        examples = getattr(sample_resolutions, "_examples", [])
        if examples:
            print("  example_files:")
            for p, w, h in examples:
                print(f"  - {w}×{h}  {p}")
        print("  top_resolutions:")
        print(fmt_counts(counts, args.top_k), end="")


if __name__ == "__main__":
    main()

