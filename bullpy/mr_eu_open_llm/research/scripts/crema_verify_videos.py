#!/usr/bin/env python3
"""Check fetched CREMA-D clips: real video, matching hash, sensible length, decodable.

Run on CSD3 with the project env (needs cv2 and ffprobe), passing paths as arguments:

    python -I crema_verify_videos.py --video-dir DIR --ffprobe PATH --out check.csv

The cv2 pass reproduces how scripts/frame_sampling.py seeks: open, read the frame count,
seek to evenly spaced indices and read one frame at each.
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter

import cv2
import numpy as np

N_FRAMES = 8
MIN_SECONDS, MAX_SECONDS = 1.0, 6.0  # spoken sentences of a few seconds


def probe(ffprobe, path):
    out = subprocess.check_output(
        [ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", path]
    )
    info = json.loads(out.decode("utf-8"))
    video = [s for s in info["streams"] if s["codec_type"] == "video"]
    audio = [s for s in info["streams"] if s["codec_type"] == "audio"]
    return {
        "container": info["format"]["format_name"],
        "duration_s": round(float(info["format"]["duration"]), 3),
        "codec": video[0]["codec_name"] if video else "",
        "width": video[0].get("width", 0) if video else 0,
        "height": video[0].get("height", 0) if video else 0,
        "has_audio": bool(audio),
    }


def seek_check(path):
    """Frames read, mean luminance spread, and the mean difference between sampled frames."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"cv2_frames": 0, "cv2_fps": 0.0, "frames_read": 0, "frame_sd": 0.0, "frame_change": 0.0}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    for idx in np.linspace(0, max(count - 1, 0), N_FRAMES).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float))
    cap.release()
    change = float(np.mean([np.abs(a - b).mean() for a, b in zip(frames, frames[1:])])) if len(frames) > 1 else 0.0
    return {
        "cv2_frames": count,
        "cv2_fps": round(fps, 3),
        "frames_read": len(frames),
        "frame_sd": round(float(np.mean([f.std() for f in frames])), 2) if frames else 0.0,
        "frame_change": round(change, 3),
    }


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--ffprobe", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(os.path.join(args.video_dir, "crema_d_lfs_manifest.csv"), newline="", encoding="utf-8") as fh:
        expected = list(csv.DictReader(fh))
    rows = []
    problems = []
    for entry in expected:
        clip_id = entry["clip_id"]
        path = os.path.join(args.video_dir, clip_id + ".flv")
        row = {"clip_id": clip_id}
        issues = []
        if not os.path.isfile(path):
            problems.append((clip_id, "missing"))
            continue
        with open(path, "rb") as fh:
            head = fh.read(64)
        if head.startswith(b"version https://git-lfs"):
            issues.append("lfs pointer")
        if not head.startswith(b"FLV"):
            issues.append("no FLV magic")
        digest = sha256(path)
        if digest != entry["sha256"] or os.path.getsize(path) != int(entry["size"]):
            issues.append("hash or size mismatch")
        row["size"] = os.path.getsize(path)
        row["sha256"] = digest
        row.update(probe(args.ffprobe, path))
        row.update(seek_check(path))
        if not MIN_SECONDS <= row["duration_s"] <= MAX_SECONDS:
            issues.append("duration {}".format(row["duration_s"]))
        if row["frames_read"] < N_FRAMES:
            issues.append("read {} of {} seeks".format(row["frames_read"], N_FRAMES))
        if row["frame_sd"] < 5:
            issues.append("blank frames")
        rows.append(row)
        problems.extend((clip_id, issue) for issue in issues)

    fields = ["clip_id", "size", "sha256", "container", "duration_s", "codec", "width", "height", "has_audio",
              "cv2_frames", "cv2_fps", "frames_read", "frame_sd", "frame_change"]  # fmt: skip
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    durations = np.array([r["duration_s"] for r in rows])
    implied = np.array([r["cv2_frames"] / r["cv2_fps"] for r in rows if r["cv2_fps"]])
    print("checked", len(rows), "of", len(expected))
    print("total bytes", sum(r["size"] for r in rows), "unique hashes", len({r["sha256"] for r in rows}))
    print("duration s: min {:.2f} median {:.2f} max {:.2f}".format(durations.min(), np.median(durations), durations.max()))
    print("cv2 frames/fps s: min {:.2f} median {:.2f} max {:.2f}".format(implied.min(), np.median(implied), implied.max()))
    print("container", Counter(r["container"] for r in rows), "codec", Counter(r["codec"] for r in rows))
    print("size", Counter((r["width"], r["height"]) for r in rows), "audio", Counter(r["has_audio"] for r in rows))
    print("fps", Counter(r["cv2_fps"] for r in rows).most_common(5))
    print("problems", len(problems), problems[:40])
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
