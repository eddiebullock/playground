"""Turn the readout ladder's calibration geometry into the records list A7 Stage 0 reads.

Unpinned helper. It changes no pinned A7 or A2 code and reads one file only: the geometry JSON
that `scripts/readout_ladder.py` (step_geometry) writes as {"items": {trial_id: record}}, each
record from `run_stimulus_interventions.clip_geometry_record` plus "path".

Each output record is the geometry record unchanged, plus two keys:

- clip_id: the trial_id, which `a7_stage0.main` indexes records by.
- apex_landmarks: record["landmarks"][str(record["apex"])], the FaceMesh points of the apex
  frame in that frame's pixel coordinates. This is the mapping A2 itself uses:
  `_landmark_table(record)` keys landmarks by int frame index and `build_media` passes
  `clip.landmarks.get(clip.apex)` to `apply_condition`.

It refuses, writing nothing, if any item failed (clip_failure, box_failure, no apex, no face on
the apex frame) or lacks a key Stage 0 or face_views reads. The clip ids themselves are checked
by Stage 0 against the pinned calibration list.

    python3 research/scripts/a7_stage0_records.py \
        --geometry results/readout_ladder/geometry.json --output results/a7/stage0_records.json
"""



import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Keys read from a record by a7_stage0.main, readout_ladder.face_views and run.check_decoder.
# frame_sha256 is read with .get there, so a missing one would silently skip the decoder check;
# it is required here so that check always runs.
NEEDED = ("trial_id", "path", "apex", "landmarks", "face_box", "face_width_px", "frame_sha256")
FAILURE_KEYS = ("clip_failure", "box_failure")


class Refused(Exception):
    pass


def convert(geometry: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = geometry.get("items") if isinstance(geometry, dict) else None
    if not isinstance(items, dict) or not items:
        raise Refused("geometry file has no 'items' mapping")
    problems: List[str] = []
    records: List[Dict[str, Any]] = []
    for trial_id, record in items.items():
        if not isinstance(record, dict):
            problems.append(f"{trial_id}: record is not an object")
            continue
        failed = [k for k in FAILURE_KEYS if record.get(k)]
        if failed:
            problems.append(f"{trial_id}: failed ({', '.join(f'{k}={record[k]}' for k in failed)})")
            continue
        missing = [k for k in NEEDED if k not in record]
        if missing:
            problems.append(f"{trial_id}: missing keys {missing}")
            continue
        if record["trial_id"] != trial_id:
            problems.append(f"{trial_id}: record trial_id is {record['trial_id']!r}")
            continue
        apex = record["apex"]
        if apex is None or isinstance(apex, bool) or not isinstance(apex, int):
            problems.append(f"{trial_id}: no apex frame (apex={apex!r})")
            continue
        landmarks = record["landmarks"]
        if not isinstance(landmarks, dict) or str(apex) not in landmarks:
            problems.append(f"{trial_id}: no landmarks entry for apex frame {apex}")
            continue
        points = landmarks[str(apex)]
        if not points or not all(isinstance(p, list) and len(p) == 2 for p in points):
            problems.append(f"{trial_id}: apex frame {apex} has no usable landmarks")
            continue
        if "clip_id" in record or "apex_landmarks" in record:
            problems.append(f"{trial_id}: already has clip_id or apex_landmarks")
            continue
        out = dict(record)
        out["clip_id"] = trial_id
        out["apex_landmarks"] = points
        records.append(out)
    if problems:
        raise Refused("; ".join(problems))
    return records


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        print(f"{args.output} exists; not overwritten.", file=sys.stderr)
        return 1
    try:
        records = convert(json.loads(args.geometry.read_text(encoding="utf-8")))
    except Refused as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records) + "\n", encoding="utf-8")
    print(json.dumps({"n": len(records), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
