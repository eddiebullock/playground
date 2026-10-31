"""Hand-built stimulus features for the Phase 2 input baseline.

Videos live on CSD3. This script is safe to run there; locally it writes
whatever it can (actor/duration proxies live in probe.py from eval JSON).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _video_stats_cv2(path: Path, *, max_frames: int = 16) -> Optional[Dict[str, float]]:
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (n / fps) if fps > 0 else float("nan")
    frames = []
    step = max(1, n // max_frames) if n > 0 else 1
    idx = 0
    ok, frame = cap.read()
    while ok and len(frames) < max_frames:
        if idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            frames.append(gray)
        ok, frame = cap.read()
        idx += 1
    cap.release()
    if not frames:
        return {"duration_s": duration, "mean_luminance": float("nan"), "motion_energy": float("nan")}
    stack = np.stack(frames, axis=0)
    lum = float(stack.mean() / 255.0)
    if len(frames) >= 2:
        diffs = np.abs(np.diff(stack, axis=0))
        motion = float(diffs.mean() / 255.0)
    else:
        motion = 0.0
    return {"duration_s": float(duration), "mean_luminance": lum, "motion_energy": motion}


def extract_all(
    *,
    manifest_path: Path,
    data_root: Path,
    out_path: Path,
    max_frames: int = 16,
) -> Dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    trials: Dict[str, Any] = {}
    n_ok = 0
    n_miss = 0
    for trial in manifest["trials"]:
        tid = trial["trial_id"]
        rel = trial.get("stimulus_path") or tid
        path = Path(rel)
        if not path.is_file():
            path = data_root / rel
        if not path.is_file():
            # HPC layout used in eval JSON
            alt = data_root / "emotions" / "HD Version - Face, Body, Social" / "Faces - HD Version" / Path(rel).name
            if alt.is_file():
                path = alt
        stats = _video_stats_cv2(path, max_frames=max_frames) if path.is_file() else None
        if stats is None:
            n_miss += 1
            trials[tid] = {"status": "missing", "path": str(path)}
        else:
            n_ok += 1
            trials[tid] = {"status": "ok", "path": str(path), **stats}
    payload = {
        "n_ok": n_ok,
        "n_missing": n_miss,
        "data_root": str(data_root),
        "trials": trials,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/eu_emotions_full_manifest.json")
    ap.add_argument("--data_root", type=Path, default=ROOT / "data/eu_emotions")
    ap.add_argument("--out", type=Path, default=ROOT / "research/reproduction/stimulus_features.json")
    ap.add_argument("--max_frames", type=int, default=16)
    args = ap.parse_args()
    payload = extract_all(
        manifest_path=args.manifest,
        data_root=args.data_root,
        out_path=args.out,
        max_frames=args.max_frames,
    )
    print(f"wrote {args.out} ok={payload['n_ok']} missing={payload['n_missing']}")


if __name__ == "__main__":
    main()
