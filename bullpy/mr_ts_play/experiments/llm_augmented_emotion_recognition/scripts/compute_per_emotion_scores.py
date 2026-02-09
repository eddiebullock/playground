#!/usr/bin/env python3
"""
Compute per-emotion accuracy from multimodal experiment predictions.json.

Usage:
  python compute_per_emotion_scores.py results/eu_emotion_gemini3_pro/predictions.json
  python compute_per_emotion_scores.py results/eu_emotion_gpt5/predictions.json --output results/eu_emotion_gpt5/per_emotion.json

Reads predictions.json, groups by correct_label (emotion), and computes:
  - count (valid trials for that emotion)
  - correct
  - accuracy
Writes per_emotion.json (and optionally per_emotion.csv) in the same dir as predictions
unless --output is given.
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Compute per-emotion accuracy from predictions.json")
    parser.add_argument("predictions_file", type=str, help="Path to predictions.json")
    parser.add_argument("--output", type=str, default=None, help="Output path for per_emotion.json (default: same dir as predictions)")
    parser.add_argument("--csv", action="store_true", help="Also write per_emotion.csv")
    args = parser.parse_args()

    pred_path = Path(args.predictions_file)
    if not pred_path.exists():
        print(f"Error: {pred_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(pred_path) as f:
        predictions = json.load(f)

    # Group by correct_label; only count trials with valid prediction (is_correct is not None)
    by_emotion = defaultdict(lambda: {"count": 0, "correct": 0})
    for p in predictions:
        correct_label = p.get("correct_label") or ""
        if not correct_label:
            continue
        is_correct = p.get("is_correct")
        if is_correct is None:
            continue
        by_emotion[correct_label]["count"] += 1
        if is_correct is True:
            by_emotion[correct_label]["correct"] += 1

    # Build per-emotion stats
    per_emotion = {}
    for emotion in sorted(by_emotion.keys()):
        d = by_emotion[emotion]
        n = d["count"]
        c = d["correct"]
        acc = c / n if n > 0 else 0.0
        per_emotion[emotion] = {"count": n, "correct": c, "accuracy": round(acc, 4)}

    out_dir = pred_path.parent
    out_json = Path(args.output) if args.output else out_dir / "per_emotion.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(per_emotion, f, indent=2)
    print(f"Wrote {out_json}")

    if args.csv:
        out_csv = out_json.with_suffix(".csv")
        with open(out_csv, "w") as f:
            f.write("emotion,count,correct,accuracy\n")
            for emotion in sorted(per_emotion.keys()):
                d = per_emotion[emotion]
                f.write(f"{emotion},{d['count']},{d['correct']},{d['accuracy']:.4f}\n")
        print(f"Wrote {out_csv}")

    # Print summary table
    print("\nPer-emotion accuracy:")
    print("-" * 50)
    for emotion in sorted(per_emotion.keys()):
        d = per_emotion[emotion]
        print(f"  {emotion}: {d['correct']}/{d['count']} = {d['accuracy']:.1%}")
    total_valid = sum(d["count"] for d in per_emotion.values())
    total_correct = sum(d["correct"] for d in per_emotion.values())
    print("-" * 50)
    print(f"  TOTAL: {total_correct}/{total_valid} = {total_correct/total_valid:.1%}")


if __name__ == "__main__":
    main()
