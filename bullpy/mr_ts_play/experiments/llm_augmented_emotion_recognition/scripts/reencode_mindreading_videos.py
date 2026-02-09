#!/usr/bin/env python3
"""
Re-encode MindReading .mov videos that fail to decode to H.264 MP4.

Identifies videos that OpenCV cannot read, re-encodes them with ffmpeg
(H.264 video, AAC audio, .mp4 container), preserves folder structure,
and validates re-encoded files. Produces updated trial definitions pointing
to .mp4 paths for use with --data-root pointing at the output directory.

Usage:
  python reencode_mindreading_videos.py \\
    --trial-definitions data/trial_definitions/mindreading_emotions_test.json \\
    --data-root /Volumes/MindReading/Emotions \\
    [--output-root results/mindreading_reencoded]  # optional; default is results/mindreading_reencoded \\
    [--dry-run]
"""

import argparse
import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Add project root for optional cv2 use
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Random seed for reproducibility (documented in REPRODUCIBILITY.md)
SEED = 42


def video_can_decode(video_path: Path, num_frames: int = 4) -> bool:
    """
    Test if a video file can be opened and frames read (same logic as inference).
    Returns True if the video decodes successfully.
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available; skipping decode check. Install with: pip install opencv-python")
        return True

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return False

    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    read_ok = True
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, _ = cap.read()
        if not ret:
            read_ok = False
            break
    cap.release()
    return read_ok


def identify_failing_videos(
    trials: List[Dict],
    data_root: Path,
) -> Tuple[List[Dict], List[Dict]]:
    """
    For each trial, resolve video path and test decode. Return (success_trials, fail_trials).
    """
    success_trials = []
    fail_trials = []
    for trial in trials:
        stimulus_path = trial.get("stimulus_path", "")
        if not stimulus_path:
            fail_trials.append(trial)
            continue
        video_path = data_root / stimulus_path
        if not video_path.exists():
            logger.warning("Video not found: %s", video_path)
            fail_trials.append(trial)
            continue
        if video_can_decode(video_path):
            success_trials.append(trial)
        else:
            fail_trials.append(trial)
    return success_trials, fail_trials


def check_ffmpeg_available() -> bool:
    """Return True if ffmpeg is on PATH."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def reencode_video_ffmpeg(src: Path, dst: Path) -> bool:
    """
    Re-encode video to H.264 MP4 with AAC audio. Returns True on success.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    # H.264 baseline + yuv420p for OpenCV/QuickTime compatibility on macOS
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-profile:v",
        "baseline",
        "-level",
        "3.0",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            logger.error("ffmpeg failed for %s: %s", src, result.stderr[:500] if result.stderr else "")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg timeout for %s", src)
        return False
    except FileNotFoundError:
        logger.error("ffmpeg not found. Install ffmpeg (e.g. brew install ffmpeg).")
        return False
    except Exception as e:
        logger.error("ffmpeg error for %s: %s", src, e)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Re-encode failing MindReading .mov videos to H.264 MP4"
    )
    parser.add_argument(
        "--trial-definitions",
        type=str,
        required=True,
        help="Path to trial definitions JSON (e.g. mindreading_emotions_test.json)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory for original videos (e.g. /Volumes/MindReading/Emotions)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root directory for re-encoded .mp4 files (default: results/mindreading_reencoded in current dir)",
    )
    parser.add_argument(
        "--output-trials",
        type=str,
        default=None,
        help="Path to write updated trial definitions (stimulus_path .mov -> .mp4 for re-encoded). Default: <output_root>/mindreading_emotions_test_reencoded.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only identify failing videos; do not re-encode or write files",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run decode check and report counts; no re-encode, no write",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root) if args.output_root else Path("results/mindreading_reencoded").resolve()

    with open(args.trial_definitions, "r") as f:
        data = json.load(f)
    trials = data.get("trials", data) if isinstance(data, dict) else data

    logger.info("Loaded %d trials from %s", len(trials), args.trial_definitions)
    logger.info("Identifying videos that fail to decode (same logic as inference)...")

    success_trials, fail_trials = identify_failing_videos(trials, data_root)
    logger.info("Decode check: %d success, %d fail", len(success_trials), len(fail_trials))

    if args.validate_only:
        if fail_trials:
            out = output_root / "failing_videos_list.json"
            output_root.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(
                    [
                        {
                            "trial_id": t.get("trial_id"),
                            "stimulus_path": t.get("stimulus_path"),
                            "emotion": t.get("correct_label") or t.get("emotion"),
                        }
                        for t in fail_trials
                    ],
                    f,
                    indent=2,
                )
            logger.info("Wrote list of failing videos to %s", out)
        return

    if not fail_trials:
        logger.info("No failing videos to re-encode.")
        return

    if args.dry_run:
        logger.info("Dry run: would re-encode %d videos to %s", len(fail_trials), output_root)
        for t in fail_trials[:5]:
            logger.info("  %s", t.get("stimulus_path"))
        if len(fail_trials) > 5:
            logger.info("  ... and %d more", len(fail_trials) - 5)
        return

    if not check_ffmpeg_available():
        logger.error(
            "ffmpeg is not installed or not on PATH. Re-encoding cannot run.\n"
            "  Install ffmpeg first, e.g.:  brew install ffmpeg\n"
            "  Then re-run this script. If the failing videos cannot be opened manually,\n"
            "  they may be corrupted; ffmpeg may still be able to read some. If re-encoding\n"
            "  still fails for all, use the selection-bias analysis and report results on\n"
            "  the %d valid trials only (see analyze_mindreading_selection_bias.py).",
            len(success_trials),
        )
        sys.exit(1)

    output_root.mkdir(parents=True, exist_ok=True)
    reencoded_paths = {}  # stimulus_path (mov) -> new path (mp4)
    failed_to_encode = []

    for i, trial in enumerate(fail_trials):
        stimulus_path = trial.get("stimulus_path", "")
        src = data_root / stimulus_path
        # Preserve structure: same relative path under output_root, but .mp4
        rel_mp4 = Path(stimulus_path).with_suffix(".mp4")
        dst = output_root / rel_mp4
        reencoded_paths[stimulus_path] = str(rel_mp4)
        logger.info("Re-encoding %d/%d: %s -> %s", i + 1, len(fail_trials), src.name, dst.name)
        if reencode_video_ffmpeg(src, dst):
            if not video_can_decode(dst):
                logger.warning("Re-encoded file failed validation: %s", dst)
                failed_to_encode.append(stimulus_path)
        else:
            failed_to_encode.append(stimulus_path)

    logger.info("Re-encoding complete. Failed to encode: %d", len(failed_to_encode))
    for p in failed_to_encode[:10]:
        logger.warning("  %s", p)
    if len(failed_to_encode) > 10:
        logger.warning("  ... and %d more", len(failed_to_encode) - 10)

    # Copy working videos to output_root so one data root has full set (working .mov + re-encoded .mp4)
    import shutil
    logger.info("Copying %d working videos to output root (same structure)...", len(success_trials))
    for i, trial in enumerate(success_trials):
        stimulus_path = trial.get("stimulus_path", "")
        src = data_root / stimulus_path
        dst = output_root / stimulus_path
        if not dst.exists() or src.stat().st_size != dst.stat().st_size:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        if (i + 1) % 100 == 0:
            logger.info("  Copied %d/%d", i + 1, len(success_trials))

    # Updated trial list: re-encoded use .mp4 path; success keep .mov (now under output_root)
    updated_trials = []
    for t in trials:
        sp = t.get("stimulus_path", "")
        if sp in reencoded_paths and sp not in failed_to_encode:
            updated = dict(t)
            updated["stimulus_path"] = reencoded_paths[sp]
            updated_trials.append(updated)
        else:
            updated_trials.append(t)

    out_trials_path = args.output_trials or str(output_root / "mindreading_emotions_test_reencoded.json")
    out_meta = {
        "trials": updated_trials,
        "metadata": {
            "source_trial_definitions": args.trial_definitions,
            "data_root_original": str(data_root),
            "data_root_reencoded": str(output_root),
            "usage": "Run inference with --data-root set to data_root_reencoded and this file as --trial-definitions to run the full test set.",
            "reencoded_count": len(reencoded_paths) - len(failed_to_encode),
            "reencode_failed_count": len(failed_to_encode),
            "seed": SEED,
        },
    }
    with open(out_trials_path, "w") as f:
        json.dump(out_meta, f, indent=2)
    logger.info("Wrote updated trial definitions to %s", out_trials_path)
    logger.info("To run full test set: use --data-root %s and --trial-definitions %s", output_root, out_trials_path)


if __name__ == "__main__":
    main()
