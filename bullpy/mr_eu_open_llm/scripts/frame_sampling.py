"""
Protocol v2 frame sampling: 1 fps, cap at max_frames, uniform in time.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from config import FRAME_POLICY, SEED


def video_duration_seconds(cap: cv2.VideoCapture) -> float:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0:
        fps = 25.0
    if frame_count <= 0:
        return 0.0
    return frame_count / fps


def frame_indices_for_video(
    frame_count: int,
    duration_seconds: Optional[float] = None,
    fps: float = 1.0,
    max_frames: int = 16,
) -> List[int]:
    """
    Sample n = min(max_frames, max(1, floor(duration * fps))) indices uniformly from 0..frame_count-1.
    """
    if frame_count <= 0:
        return [0]
    if duration_seconds is None:
        duration_seconds = float(frame_count)
    n = min(max_frames, max(1, int(math.floor(duration_seconds * fps))))
    if n <= 1:
        return [0]
    indices = [int(round(i * (frame_count - 1) / (n - 1))) for i in range(n)]
    return sorted(set(indices))


def video_readable(video_path: Path) -> bool:
    """Quick check that OpenCV can open the file and read at least one frame index."""
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return frame_count > 0


def load_frames_from_video(
    video_path: Path,
    fps: float = FRAME_POLICY["fps"],
    max_frames: int = FRAME_POLICY["max_frames"],
) -> Tuple[List[Image.Image], List[int]]:
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Video has no frames: {video_path}")

    duration = video_duration_seconds(cap)
    indices = frame_indices_for_video(
        frame_count,
        duration_seconds=duration,
        fps=fps,
        max_frames=max_frames,
    )

    frames: List[Image.Image] = []
    for idx in indices:
        idx = max(0, min(frame_count - 1, idx))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()
    if not frames:
        raise RuntimeError(f"Failed to extract frames from: {video_path}")
    return frames, indices


def load_stimulus_as_images(
    stimulus_path: Path,
    fps: float = FRAME_POLICY["fps"],
    max_frames: int = FRAME_POLICY["max_frames"],
) -> Tuple[List[Image.Image], List[int]]:
    ext = stimulus_path.suffix.lower()
    if ext in {".jpg", ".jpeg", ".png", ".webp"}:
        img = Image.open(stimulus_path).convert("RGB")
        return [img], [0]
    if ext in {".mp4", ".mov", ".avi", ".mkv"}:
        return load_frames_from_video(stimulus_path, fps=fps, max_frames=max_frames)
    raise ValueError(f"Unsupported stimulus type: {stimulus_path}")


def ablation_trial_ids(manifest_trials: list, n: int = 30, seed: int = SEED) -> List[str]:
    """Deterministic subset of trial_ids for frame ablation."""
    ids = [t.get("trial_id", str(i)) for i, t in enumerate(manifest_trials)]
    rng = np.random.default_rng(seed)
    if len(ids) <= n:
        return ids
    chosen = rng.choice(np.array(ids, dtype=object), size=n, replace=False)
    return [str(x) for x in chosen.tolist()]


def frame_policy_tag(
    fps: float,
    max_frames: int,
    frame_mode: Optional[str] = None,
) -> str:
    base = f"fps{int(fps)}_cap{max_frames}"
    default_mode = FRAME_POLICY.get("default_mode", "composite_grid")
    if frame_mode and frame_mode != default_mode:
        return f"{base}_{frame_mode}"
    return base
