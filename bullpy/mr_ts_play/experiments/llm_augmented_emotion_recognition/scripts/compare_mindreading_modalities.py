#!/usr/bin/env python3
"""
Compare multimodal vs video-only accuracy for MindReading experiment.

Loads two summary.json files (multimodal and video-only runs) and prints
a comparison table and delta (audio contribution). Optionally writes
a comparison table to CSV/JSON.

Usage:
  python compare_mindreading_modalities.py \\
    --multimodal-summary results/mindreading_multimodal/summary.json \\
    --video-only-summary results/mindreading_video_only/summary.json \\
    --output-dir results/mindreading_comparison
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Compare multimodal vs video-only accuracy"
    )
    parser.add_argument(
        "--multimodal-summary",
        type=str,
        required=True,
        help="Path to summary.json from run with --use-audio",
    )
    parser.add_argument(
        "--video-only-summary",
        type=str,
        required=True,
        help="Path to summary.json from run with --video-only",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/mindreading_comparison",
        help="Directory to write comparison table (CSV/JSON)",
    )
    args = parser.parse_args()

    with open(args.multimodal_summary, "r") as f:
        multi = json.load(f)
    with open(args.video_only_summary, "r") as f:
        video = json.load(f)

    acc_multi = multi.get("accuracy", 0.0)
    acc_video = video.get("accuracy", 0.0)
    valid_multi = multi.get("valid_predictions", 0)
    valid_video = video.get("valid_predictions", 0)
    correct_multi = multi.get("correct", 0)
    correct_video = video.get("correct", 0)

    delta = acc_multi - acc_video
    delta_pct = delta * 100.0

    # Print table
    print("=" * 60)
    print("MindReading: Multimodal vs Video-Only")
    print("=" * 60)
    print(f"{'Condition':<20} {'Valid N':<10} {'Correct':<10} {'Accuracy':<12}")
    print("-" * 60)
    print(f"{'Multimodal (V+A)':<20} {valid_multi:<10} {correct_multi:<10} {acc_multi:.2%}")
    print(f"{'Video-only':<20} {valid_video:<10} {correct_video:<10} {acc_video:.2%}")
    print("-" * 60)
    print(f"Delta (audio contribution): {delta_pct:+.2f} pp  (multimodal - video-only)")
    print("=" * 60)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = {
        "multimodal": {
            "valid_predictions": valid_multi,
            "correct": correct_multi,
            "accuracy": acc_multi,
        },
        "video_only": {
            "valid_predictions": valid_video,
            "correct": correct_video,
            "accuracy": acc_video,
        },
        "delta_accuracy_pp": round(delta_pct, 2),
        "note": "Audio consisted of single-word utterances of the emotion label.",
    }
    out_json = out_dir / "modality_comparison.json"
    with open(out_json, "w") as f:
        json.dump(table, f, indent=2)
    logger.info("Wrote %s", out_json)

    out_csv = out_dir / "modality_comparison.csv"
    with open(out_csv, "w") as f:
        f.write("condition,valid_predictions,correct,accuracy\n")
        f.write("multimodal,%d,%d,%.4f\n" % (valid_multi, correct_multi, acc_multi))
        f.write("video_only,%d,%d,%.4f\n" % (valid_video, correct_video, acc_video))
        f.write("delta_pp,,,%.2f\n" % delta_pct)
    logger.info("Wrote %s", out_csv)


if __name__ == "__main__":
    main()
