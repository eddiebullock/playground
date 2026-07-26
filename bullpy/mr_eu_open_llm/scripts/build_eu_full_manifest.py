#!/usr/bin/env python3
"""Build a full-EU Faces manifest for study3 (not the 118-trial subset).

Scans emotion packs for face videos and writes JSON that evaluate.py can load:

  { "trials": [ {trial_id, stimulus_path, correct_label, label, edition}, ... ] }

Audio is NOT listed here — at eval time resolve_eu_multimodal_audio pairs UK Voices
by normalized emotion label under:
  <data_root>/EU Emotion - UK Voices/Fixed - amplified volume/

Usage (Mac, against OneDrive while HPC sync runs):

  python -m scripts.build_eu_full_manifest \\
    --data-root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_Emotions" \\
    --out data/eu_emotions_full_manifest.json \\
    --edition edited

  # After HPC sync, rebuild or copy JSON to study3 and point --data_root there:
  #   --data-root /home/eb2007/rds/hpc-work/study3/data/eu_emotions

Defaults: EDITED faces only (skip Original .mov). Use --edition both|original to change.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Set

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
Edition = Literal["edited", "original", "both"]


def normalize_pack_key(name: str) -> str:
    """emotions -> emotions_0 style key for stable trial_ids."""
    if name == "emotions":
        return "emotions"
    m = re.fullmatch(r"emotions\s+(\d+)", name)
    if m:
        return f"emotions_{m.group(1)}"
    return re.sub(r"\s+", "_", name)


def iter_emotion_packs(data_root: Path) -> List[Path]:
    packs = [
        p
        for p in data_root.iterdir()
        if p.is_dir() and (p.name == "emotions" or p.name.startswith("emotions "))
    ]

    def sort_key(p: Path) -> tuple:
        if p.name == "emotions":
            return (0, 0)
        m = re.fullmatch(r"emotions\s+(\d+)", p.name)
        return (1, int(m.group(1)) if m else 9999)

    return sorted(packs, key=sort_key)


def edition_from_path(path: Path) -> Optional[str]:
    parts = path.parts
    for i, part in enumerate(parts):
        if part == "Faces - HD Version" and i + 1 < len(parts):
            ed = parts[i + 1]
            if ed in {"EDITED", "Original"}:
                return ed
    return None


def label_from_path(path: Path) -> str:
    # .../Faces - HD Version/{EDITED|Original}/<Label>/<file>
    return path.parent.name


def should_keep(edition: Optional[str], want: Edition) -> bool:
    if edition is None:
        return False
    if want == "both":
        return edition in {"EDITED", "Original"}
    if want == "edited":
        return edition == "EDITED"
    return edition == "Original"


def scan_faces(data_root: Path, *, edition: Edition) -> List[Dict[str, Any]]:
    trials: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()

    for pack in iter_emotion_packs(data_root):
        faces_roots = list(pack.glob("**/Faces - HD Version"))
        for faces_root in faces_roots:
            for path in sorted(faces_root.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
                    continue
                ed = edition_from_path(path)
                if not should_keep(ed, edition):
                    continue
                rel = path.relative_to(data_root).as_posix()
                label = label_from_path(path)
                pack_key = normalize_pack_key(pack.name)
                trial_id = f"{pack_key}/{ed}/{label}/{path.name}"
                if trial_id in seen_ids:
                    continue
                seen_ids.add(trial_id)
                trials.append(
                    {
                        "trial_id": trial_id,
                        "stimulus_path": rel,
                        "correct_label": label,
                        "label": label,
                        "edition": ed,
                        "pack": pack.name,
                    }
                )
    return trials


def summarize(trials: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    labels: Dict[str, int] = {}
    editions: Dict[str, int] = {}
    for t in trials:
        labels[t["correct_label"]] = labels.get(t["correct_label"], 0) + 1
        editions[t["edition"]] = editions.get(t["edition"], 0) + 1
    return {
        "n_labels": len(labels),
        "labels": dict(sorted(labels.items())),
        "editions": editions,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="EU root containing emotions* packs (+ optional UK Voices sibling tree).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/eu_emotions_full_manifest.json"),
        help="Output manifest JSON path.",
    )
    ap.add_argument(
        "--edition",
        choices=("edited", "original", "both"),
        default="edited",
        help="Which Faces trees to include (default: edited only).",
    )
    ap.add_argument(
        "--n-options",
        type=int,
        default=6,
        help="Stage-2 forced-choice size recorded in manifest (study3 default: 6).",
    )
    args = ap.parse_args()

    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise SystemExit(f"data-root not found: {data_root}")

    trials = scan_faces(data_root, edition=args.edition)
    stats = summarize(trials)
    obj = {
        "dataset": "eu_emotions",
        "dataset_root": str(data_root),
        "edition_filter": args.edition,
        "n_options": int(args.n_options),
        "n_trials": len(trials),
        "summary": stats,
        "notes": [
            "Audio paired at eval via UK Voices Fixed-amplified by normalized correct_label.",
            f"{args.n_options}AFC foils generated at runtime (not stored in manifest); pass --n_options {args.n_options} or rely on this field.",
            "stimulus_path is relative to dataset_root / --data_root used at eval.",
            "Study2 used 4AFC on the 118-trial subset; study3 full-EU defaults to 6AFC.",
        ],
        "trials": trials,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out} ({len(trials)} trials)")
    print(f"  editions: {stats['editions']}")
    print(f"  n_labels: {stats['n_labels']}")
    print(f"  sample labels: {list(stats['labels'])[:8]}...")


if __name__ == "__main__":
    main()
