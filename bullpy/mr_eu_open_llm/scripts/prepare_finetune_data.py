"""
Prepare Mindreading JSONL splits for LoRA fine-tuning (Study 1).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import logging

from config import FINETUNE_MODALITY, LOCAL_DATA_DIR, SEED, TRAINING_DEFAULTS
from scripts.mindreading_audio_resolver import (
    extract_audio_from_video,
    ffmpeg_usable,
    resolve_item_folder_audio,
    resolve_mindreading_v_video,
)

logger = logging.getLogger(__name__)

SCENARIO_DIR_RE = re.compile(r"^(?:0[1-9]|1\d|2[0-4])$")
EXCLUDED_TOP_DIRS = {
    "emotions",
    "rewards",
    "definitions",
    "daniel",
    "driveloc",
}
MEDIA_EXT = {".mp4", ".mov", ".avi", ".mkv", ".jpg", ".jpeg", ".png"}
# V6T clips are audio companions (e.g. 1701501V6Ttempting.mov); OpenCV cannot decode them.
V6T_FILENAME_RE = re.compile(r"V\d+T", re.IGNORECASE)


def is_mindreading_face_video_filename(name: str) -> bool:
    if "T" in name and "V" not in name:
        return False
    if "V" not in name:
        return False
    if V6T_FILENAME_RE.search(name):
        return False
    return True


def infer_label_from_filename(name: str) -> Optional[str]:
    """Extract mental-state word from Mindreading V/T filename when present."""
    stem = Path(name).stem
    for marker in ("V", "T"):
        if marker not in stem:
            continue
        tail = stem.split(marker, 1)[1]
        m = re.match(r"^([A-Za-z][A-Za-z]+)", tail)
        if m:
            return m.group(1).lower()
    return None


def infer_mindreading_label(video_path: Path) -> str:
    from_name = infer_label_from_filename(video_path.name)
    if from_name:
        return from_name
    return video_path.parent.name


def iter_mindreading_media_files(root: Path):
    """Scenario item folders (01-24, Stories, scenarios); exclude Emotions/Rewards trees."""
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if name.casefold() in EXCLUDED_TOP_DIRS:
            continue
        if SCENARIO_DIR_RE.match(name):
            yield from child.rglob("*")
    for sub in ("Stories", "scenarios"):
        p = root / sub
        if p.is_dir():
            yield from p.rglob("*")


def resolve_training_audio(video_path: Path, *, modality: str) -> tuple[Optional[str], Optional[str]]:
    if modality not in {"audio_only", "multimodal"}:
        return None, None
    if not ffmpeg_usable():
        return None, "ffmpeg_missing"
    vp = resolve_mindreading_v_video(video_path)
    ap, rule = resolve_item_folder_audio(vp)
    if ap is not None and ap.suffix.lower() in {".mov", ".mp4", ".m4v", ".avi", ".webm"}:
        extracted = extract_audio_from_video(ap)
        if extracted is not None and extracted.exists():
            ap = extracted
            rule = f"{rule}_extracted"
    if ap is None:
        return None, rule
    return str(ap.resolve()), rule


def discover_mindreading_trials(
    root: Path,
    *,
    modality: str = FINETUNE_MODALITY,
) -> List[Dict[str, Any]]:
    """Scan scenario item folders for V-marker face videos (not Emotions/Audio/)."""
    trials: List[Dict[str, Any]] = []
    if not root.exists():
        return trials
    for p in iter_mindreading_media_files(root):
        if not p.is_file() or p.suffix.lower() not in MEDIA_EXT:
            continue
        norm = str(p).replace("\\", "/")
        if "/Emotions/Audio/" in norm or "/Emotions/Rewards/" in norm:
            continue
        name = p.name
        if not is_mindreading_face_video_filename(name):
            continue
        rel = p.relative_to(root).as_posix()
        label = infer_mindreading_label(p)
        video_abs = resolve_mindreading_v_video(p.resolve())
        if p.suffix.lower() in {".mov", ".mp4", ".avi", ".mkv"}:
            from scripts.frame_sampling import video_readable

            if not video_readable(video_abs):
                continue
        audio_path, audio_rule = resolve_training_audio(video_abs, modality=modality)
        trials.append(
            {
                "id": rel,
                "trial_id": rel,
                "stimulus_path": str(video_abs),
                "stimulus_relpath": rel,
                "media_paths": [rel],
                "video_path": str(video_abs),
                "label": label,
                "audio_path": audio_path,
                "audio_rule": audio_rule,
                "modality": modality,
            }
        )
    return trials


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def split_records(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = SEED,
) -> Dict[str, List[Dict[str, Any]]]:
    import numpy as np

    rng = np.random.default_rng(seed)
    idx = np.arange(len(records))
    rng.shuffle(idx)
    n = len(records)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return {
        "train": [records[i] for i in train_idx],
        "val": [records[i] for i in val_idx],
        "test": [records[i] for i in test_idx],
    }


def write_mindreading_manifest(
    records: List[Dict[str, Any]],
    *,
    output_path: Path,
    dataset_root: Path,
) -> None:
    trials = []
    for rec in records:
        rel = rec.get("stimulus_relpath") or rec.get("media_paths", [None])[0]
        trials.append(
            {
                "trial_id": rec.get("trial_id") or rec.get("id"),
                "stimulus_path": rel,
                "label": rec.get("label"),
                "correct_label": rec.get("label"),
            }
        )
    obj = {
        "dataset": "mindreading",
        "dataset_root": str(dataset_root),
        "n_trials": len(trials),
        "trials": trials,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare Mindreading fine-tune JSONL files.")
    ap.add_argument("--root", type=Path, default=LOCAL_DATA_DIR / "mindreading")
    ap.add_argument("--output_dir", type=Path, default=LOCAL_DATA_DIR / "mindreading")
    ap.add_argument("--train_subset", type=int, default=TRAINING_DEFAULTS["mindreading_train_subset"])
    ap.add_argument("--val_subset", type=int, default=TRAINING_DEFAULTS["mindreading_val_subset"])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--modality",
        type=str,
        default=FINETUNE_MODALITY,
        choices=("video_only", "audio_only", "multimodal"),
    )
    ap.add_argument(
        "--write_manifest",
        action="store_true",
        help="Also write data/mindreading_test_manifest.json from test split.",
    )
    args = ap.parse_args()

    if args.modality in {"audio_only", "multimodal"} and not ffmpeg_usable():
        print(
            "WARNING: ffmpeg not usable; audio_path will be empty in JSONL. "
            "On HPC: conda install -c conda-forge ffmpeg (do not use module load ffmpeg)."
        )

    records = discover_mindreading_trials(args.root, modality=args.modality)
    if not records:
        print(f"No trials found under {args.root}; create data/mindreading/ first.")
        return

    with_audio = sum(1 for r in records if r.get("audio_path"))
    print(f"Discovered {len(records)} V-marker trials ({with_audio} with audio for {args.modality}).")

    splits = split_records(records, seed=args.seed)
    out = args.output_dir
    write_jsonl(out / "train_full.jsonl", splits["train"])
    write_jsonl(out / "val_full.jsonl", splits["val"])
    write_jsonl(out / "test_full.jsonl", splits["test"])

    import numpy as np

    rng = np.random.default_rng(args.seed)
    train_pool = splits["train"]
    val_pool = splits["val"]
    if len(train_pool) >= args.train_subset:
        tr_idx = rng.choice(len(train_pool), size=args.train_subset, replace=False)
        write_jsonl(out / "train_subset_100.jsonl", [train_pool[i] for i in tr_idx])
    if len(val_pool) >= args.val_subset:
        va_idx = rng.choice(len(val_pool), size=args.val_subset, replace=False)
        write_jsonl(out / "val_subset_50.jsonl", [val_pool[i] for i in va_idx])

    if args.write_manifest:
        manifest_path = args.root.parent / "mindreading_test_manifest.json"
        write_mindreading_manifest(splits["test"], output_path=manifest_path, dataset_root=args.root)
        print(f"Wrote {manifest_path}")

    print(f"Wrote JSONL files to {out}")


if __name__ == "__main__":
    main()
