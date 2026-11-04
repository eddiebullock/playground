#!/usr/bin/env python
"""Stop a job before it runs on a decoder that corrupts interlaced clips.

cv2 4.13 returns junk frames on EU-Emotion clips flagged interlaced. This reads a
known interlaced clip through the study loader, repeatedly, and exits non-zero
unless the decoder is pinned and the frames are stable and not blank.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REQUIRED_PREFIX = "4.11."
DEFAULT_CLIP = "HF26_cut.mp4"
N_READS = 3
MIN_MEAN_LEVEL = 5.0


def frame_digest(frames) -> str:
    h = hashlib.sha256()
    for frame in frames:
        h.update(np.asarray(frame).tobytes())
    return h.hexdigest()


def check_reads(reads, *, min_mean=MIN_MEAN_LEVEL):
    """Return problems found across repeated reads of one clip."""
    problems = []
    if len({frame_digest(r) for r in reads}) != 1:
        problems.append("frames differ between reads")
    for r in reads:
        if not r or min(float(np.asarray(f).mean()) for f in r) < min_mean:
            problems.append("blank frame")
            break
        if len(r) > 1 and len({frame_digest([f]) for f in r}) == 1:
            problems.append("all sampled frames identical")
            break
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", type=Path, default=ROOT / "data/eu_emotions")
    ap.add_argument("--clip", default=DEFAULT_CLIP)
    args = ap.parse_args()
    import cv2

    from scripts.frame_sampling import load_frames_from_video

    if not cv2.__version__.startswith(REQUIRED_PREFIX):
        print(f"FAIL: cv2 {cv2.__version__}; need {REQUIRED_PREFIX}x", file=sys.stderr)
        return 1
    matches = sorted(args.data_root.rglob(args.clip))
    if not matches:
        print(f"FAIL: {args.clip} not found under {args.data_root}", file=sys.stderr)
        return 1
    reads = [load_frames_from_video(str(matches[0]), fps=1.0, max_frames=16)[0] for _ in range(N_READS)]
    problems = check_reads(reads)
    if problems:
        print(f"FAIL: {'; '.join(problems)}", file=sys.stderr)
        return 1
    print(f"decoder ok: cv2 {cv2.__version__}, {len(reads[0])} frames, stable over {N_READS} reads")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
