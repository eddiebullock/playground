#!/usr/bin/env python3
"""
Partition MindReading correct_label set into:
- included: at least one test trial with decodable video (ffprobe)
- excluded: appears in trial JSON but every test trial for that label fails ffprobe

Requires ffprobe on PATH. Does not modify data.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Optional


def ffprobe_ok(video_path: Path) -> bool:
    try:
        r = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    out = (r.stdout or "").strip()
    if not out:
        return False
    # Typical: "320,240," or "320,240"
    parts = [p for p in out.replace("\n", ",").split(",") if p.strip()]
    if len(parts) < 2:
        return False
    try:
        w, h = int(parts[0]), int(parts[1])
        return w > 0 and h > 0
    except ValueError:
        return False


def canonical_label(t: dict) -> Optional[str]:
    lab = t.get("correct_label") or t.get("emotion")
    if not lab or not isinstance(lab, str):
        return None
    return lab.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--trials",
        type=Path,
        default=Path("data/trial_definitions/mindreading_emotions_test.json"),
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="MindReading Emotions root (contains 01/, 02/, ... scenario folders)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/mindreading_decode_split"),
        help="Writes included.txt, excluded.txt, summary.json",
    )
    args = ap.parse_args()

    with open(args.trials) as f:
        data = json.load(f)
    trials = data.get("trials", data)

    by_label: dict[str, list[Path]] = defaultdict(list)
    for t in trials:
        lab = canonical_label(t)
        if not lab:
            continue
        rel = t.get("stimulus_path")
        if not rel:
            continue
        p = args.data_root / str(rel)
        by_label[lab].append(p)

    included: set[str] = set()
    excluded: set[str] = set()
    per_label_decode: dict[str, dict] = {}

    for lab in sorted(by_label.keys(), key=str.lower):
        paths = by_label[lab]
        any_ok = False
        n_ok = n_fail = n_missing = 0
        for p in paths:
            if not p.exists():
                n_missing += 1
                continue
            if ffprobe_ok(p):
                any_ok = True
                n_ok += 1
            else:
                n_fail += 1
        per_label_decode[lab] = {
            "trials": len(paths),
            "decodable": n_ok,
            "ffprobe_fail": n_fail,
            "missing_file": n_missing,
        }
        if any_ok:
            included.add(lab)
        else:
            excluded.add(lab)

    inc_sorted = sorted(included, key=str.lower)
    exc_sorted = sorted(excluded, key=str.lower)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "included.txt").write_text("\n".join(inc_sorted) + "\n")
    (args.out_dir / "excluded.txt").write_text("\n".join(exc_sorted) + "\n")
    summary = {
        "trials_file": str(args.trials),
        "data_root": str(args.data_root),
        "unique_labels": len(by_label),
        "included_count": len(inc_sorted),
        "excluded_count": len(exc_sorted),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    if len(exc_sorted) <= 60:
        print("\nExcluded:")
        for x in exc_sorted:
            print(f"  {x}")


if __name__ == "__main__":
    main()
