from __future__ import annotations

"""EU-Emotions audio resolver: face video + UK Voices pairing by normalized emotion label."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".aif", ".aiff"}
UK_VOICES_DIRNAME = "EU Emotion - UK Voices"
# On CSD3 the same tree is often synced as data/eu_emotions_118/EU/ (without the long name).
UK_VOICES_DIR_ALIASES = (UK_VOICES_DIRNAME, "EU")
UK_VOICES_SUBDIRS = ("Fixed - amplified volume", "Original")
_VOICE_INDEX_CACHE: Dict[str, Dict[str, List[Path]]] = {}


def _looks_like_uk_voices_tree(path: Path) -> bool:
    return any((path / sub).is_dir() for sub in UK_VOICES_SUBDIRS)


def normalize_emotion_label(label: str) -> str:
    s = label.strip().lower().replace("-", " ")
    return re.sub(r"\s+", " ", s)


def find_eu_voices_root(base_data_dir: Path) -> Optional[Path]:
    base_data_dir = base_data_dir.resolve()
    seen: set[Path] = set()
    for root in [base_data_dir, *base_data_dir.parents[:4]]:
        if root in seen:
            continue
        seen.add(root)
        for dirname in UK_VOICES_DIR_ALIASES:
            candidate = root / dirname
            if candidate.is_dir() and _looks_like_uk_voices_tree(candidate):
                return candidate
    return None


def build_uk_voice_index(voices_root: Path) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    for subname in UK_VOICES_SUBDIRS:
        sub = voices_root / subname
        if not sub.is_dir():
            continue
        for folder in sorted(sub.iterdir()):
            if not folder.is_dir():
                continue
            label = normalize_emotion_label(folder.name)
            files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
            if files:
                index.setdefault(label, []).extend(files)
    return index


def get_uk_voice_index(base_data_dir: Path) -> Dict[str, List[Path]]:
    voices_root = find_eu_voices_root(base_data_dir)
    if voices_root is None:
        return {}
    cache_key = str(voices_root.resolve())
    if cache_key not in _VOICE_INDEX_CACHE:
        _VOICE_INDEX_CACHE[cache_key] = build_uk_voice_index(voices_root)
    return _VOICE_INDEX_CACHE[cache_key]


def _pick_deterministic(candidates: Sequence[Path], *, trial_id: str, seed: int) -> Path:
    ordered = sorted(candidates, key=lambda p: str(p))
    digest = hashlib.sha256(f"{trial_id}|{seed}".encode("utf-8")).hexdigest()
    return ordered[int(digest, 16) % len(ordered)]


def emotion_label_for_trial(trial: Mapping[str, Any], video_path: Path) -> str:
    for key in ("correct_label", "emotion", "label"):
        if trial.get(key):
            return str(trial[key])
    return video_path.parent.name


def resolve_audio_next_to_video(video_path: Path) -> Tuple[Optional[Path], str]:
    if not video_path.exists():
        return None, "video_missing"
    folder = video_path.parent
    for ext in sorted(AUDIO_EXTS):
        candidate = folder / f"{video_path.stem}{ext}"
        if candidate.exists():
            return candidate, "same_stem"
    audio_files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    if len(audio_files) == 1:
        return audio_files[0], "single_audio_in_folder"
    return None, "not_found"


def resolve_uk_voice_by_label(
    *, emotion_label: str, base_data_dir: Path, trial_id: str, seed: int
) -> Tuple[Optional[Path], str]:
    norm = normalize_emotion_label(emotion_label)
    index = get_uk_voice_index(base_data_dir)
    if not index:
        return None, "uk_voices_root_missing"
    candidates = index.get(norm)
    if not candidates:
        return None, f"uk_voices_label_missing:{norm}"
    if len(candidates) == 1:
        return candidates[0], "uk_voices_single_match"
    return _pick_deterministic(candidates, trial_id=trial_id, seed=seed), "uk_voices_label_hash_pick"


def resolve_eu_multimodal_audio(
    video_path: Path, *, emotion_label: str, base_data_dir: Path, trial_id: str, seed: int
) -> Tuple[Optional[Path], str]:
    ap, rule = resolve_audio_next_to_video(video_path)
    if ap is not None:
        return ap, rule
    return resolve_uk_voice_by_label(
        emotion_label=emotion_label, base_data_dir=base_data_dir, trial_id=trial_id, seed=seed
    )


def resolve_eu_audio_only(
    *, emotion_label: str, base_data_dir: Path, trial_id: str, seed: int
) -> Tuple[Optional[Path], str]:
    return resolve_uk_voice_by_label(
        emotion_label=emotion_label, base_data_dir=base_data_dir, trial_id=trial_id, seed=seed
    )


def _is_video_file(path: Path) -> bool:
    return path.suffix.lower() in {".mp4", ".m4v", ".avi", ".webm", ".mov"}


def build_audio_mapping_audit(
    trials: Sequence[Dict[str, Any]], *, base_data_dir: Path, condition: str, seed: int = 0,
    max_audit_rows: Optional[int] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, t in enumerate(trials):
        if max_audit_rows is not None and i >= max_audit_rows:
            break
        trial_id = str(t.get("trial_id", f"trial_{i}"))
        stimulus_path = t.get("stimulus_path") or t.get("video_path")
        if not stimulus_path:
            out.append({"trial_id": trial_id, "stimulus_path": None, "resolved_video_path": None,
                        "resolved_audio_path": None, "matching_rule": "missing_stimulus_path"})
            continue
        vp = Path(str(stimulus_path))
        if not vp.is_absolute():
            vp = (base_data_dir / vp).resolve()
        emotion_label = emotion_label_for_trial(t, vp) if _is_video_file(vp) else str(
            t.get("correct_label") or t.get("emotion") or t.get("label") or ""
        )
        ap, rule = None, "not_applicable"
        if condition == "audio_only":
            ap, rule = resolve_eu_audio_only(
                emotion_label=emotion_label, base_data_dir=base_data_dir, trial_id=trial_id, seed=seed
            )
        elif condition == "multimodal" and _is_video_file(vp):
            ap, rule = resolve_eu_multimodal_audio(
                vp, emotion_label=emotion_label, base_data_dir=base_data_dir, trial_id=trial_id, seed=seed
            )
        out.append({"trial_id": trial_id, "stimulus_path": str(stimulus_path),
                    "emotion_label": emotion_label or None,
                    "resolved_video_path": str(vp) if _is_video_file(vp) else None,
                    "resolved_audio_path": str(ap) if ap else None, "matching_rule": rule})
    return out


def save_audio_mapping_audit(audit_rows: Iterable[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(list(audit_rows), indent=2, ensure_ascii=False), encoding="utf-8")
