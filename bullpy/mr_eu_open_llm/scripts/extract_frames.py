import argparse
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np

from config import SEED


def compute_frame_indices(
    num_frames: int,
    strategy: str = "4frame_default",
) -> Sequence[float]:
    """
    Return normalized frame positions in [0, 1] for a given strategy.

    Strategies:
      - '4frame_default': [0.0, 0.25, 0.5, 0.75]
      - '8frame_ablation': [0.0, 0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 1.0]
    """
    if strategy == "4frame_default":
        return [0.0, 0.25, 0.5, 0.75]
    if strategy == "8frame_ablation":
        return [0.0, 0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 1.0]
    raise ValueError(f"Unknown strategy: {strategy}")


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    strategy: str = "4frame_default",
) -> List[Path]:
    """
    Extract frames from a single video according to the given sampling strategy.

    Returns a list of output frame paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # Placeholder implementation; fill with actual OpenCV frame extraction.
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from Mindreading/EU-Emotions videos.")
    parser.add_argument("--input_root", type=Path, required=True, help="Root directory containing video files.")
    parser.add_argument("--output_root", type=Path, required=True, help="Output directory for extracted frames.")
    parser.add_argument(
        "--strategy",
        type=str,
        default="4frame_default",
        choices=["4frame_default", "8frame_ablation"],
        help="Frame sampling strategy.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers to use.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for any stochastic components.",
    )

    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    # TODO: walk the input directory, detect video files, and call extract_frames_from_video.


if __name__ == "__main__":
    main()

