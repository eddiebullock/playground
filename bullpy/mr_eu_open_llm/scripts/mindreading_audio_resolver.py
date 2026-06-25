from __future__ import annotations

"""Mindreading audio resolver: item-folder T-files; blocks Emotions/Audio leakage."""

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from config import LOCAL_CACHE_DIR

logger = logging.getLogger(__name__)

_FFMPEG_WARNED = False

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".aif", ".aiff"}


def _warn_ffmpeg_once(context: str, exc: Exception) -> None:
    global _FFMPEG_WARNED
    if _FFMPEG_WARNED:
        return
    logger.warning(
        "%s (%s); further per-file warnings suppressed. "
        "On HPC: export CONDA_PKGS_DIRS=~/rds/hpc-work/study2/conda_pkgs && "
        "conda install -c conda-forge ffmpeg",
        context,
        str(exc).splitlines()[0],
    )
    _FFMPEG_WARNED = True
VIDEO_EXTS = {".mov", ".mp4", ".m4v", ".avi", ".webm"}


class LeakageAudioPathError(RuntimeError):
    pass


def ffmpeg_usable() -> bool:
    """True when ffmpeg/ffprobe exist and ffmpeg can execute (system modules may break on compute nodes)."""
    for cmd in ("ffmpeg", "ffprobe"):
        if not shutil.which(cmd):
            return False
    try:
        proc = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _hard_guard_no_leakage(audio_path: Optional[Path]) -> None:
    if audio_path is None:
        return
    if "/Emotions/Audio/" in str(audio_path).replace("\\", "/"):
        raise LeakageAudioPathError(f"Resolved audio path points to leakage directory: {audio_path}")


def resolve_mindreading_v_video(video_path: Path) -> Path:
    if not video_path.exists():
        return video_path
    name = video_path.name
    if "V" in name:
        return video_path
    if "T" in name and "V" not in name:
        prefix = name[:7]
        tail = name.split("T", 1)[1]
        candidates = sorted(video_path.parent.glob(f"{prefix}*V{tail}"))
        if candidates:
            return candidates[0]
    return video_path


def resolve_item_folder_audio(video_path: Path) -> Tuple[Optional[Path], str]:
    if not video_path.exists():
        return None, "video_missing"

    video_path = resolve_mindreading_v_video(video_path)
    item_dir = video_path.parent

    for ext in sorted(AUDIO_EXTS):
        candidate = item_dir / f"{video_path.stem}{ext}"
        if candidate.exists():
            _hard_guard_no_leakage(candidate)
            return candidate, "same_stem"

    try:
        vname = video_path.name
        # Audio-only companion .mov (T marker), e.g. 2100601U5Tincredulous.mov
        if len(vname) >= 10 and "T" in vname and "V" not in vname:
            extracted = extract_audio_from_video(video_path)
            if extracted is not None and extracted.exists():
                _hard_guard_no_leakage(extracted)
                return extracted, "T_file_extracted"
        if len(vname) >= 10 and "V" in vname:
            prefix = vname[:7]
            tail = vname.split("V", 1)[1]
            candidates = sorted(item_dir.glob(f"{prefix}*T{tail}"))
            if not candidates and "." not in tail:
                candidates = sorted(item_dir.glob(f"{prefix}*T{tail}.*"))
            if candidates:
                extracted = extract_audio_from_video(candidates[0])
                if extracted is not None and extracted.exists():
                    _hard_guard_no_leakage(extracted)
                    return extracted, "matched_T_file_extracted"
    except Exception as e:
        _warn_ffmpeg_once("T-file / audio extraction unavailable", e)

    audio_files = sorted(p for p in item_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    if len(audio_files) == 1:
        candidate = audio_files[0]
        _hard_guard_no_leakage(candidate)
        if candidate.suffix.lower() == ".mov":
            extracted = extract_audio_from_video(candidate)
            if extracted is not None and extracted.exists():
                return extracted, "single_audio_in_folder_extracted"
        return candidate, "single_audio_in_folder"

    if video_path.suffix.lower() in VIDEO_EXTS:
        try:
            if has_audio_stream(video_path):
                extracted = extract_audio_from_video(video_path)
                if extracted is not None and extracted.exists():
                    _hard_guard_no_leakage(extracted)
                    return extracted, "extracted_from_video"
        except Exception as e:
            _warn_ffmpeg_once("Audio extraction unavailable", e)

    return None, "not_found"


def has_audio_stream(video_path: Path) -> bool:
    if not video_path.exists():
        return False
    cmd = ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(video_path)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        return False
    try:
        data = json.loads(proc.stdout.decode("utf-8", errors="ignore") or "{}")
        return any(s.get("codec_type") == "audio" for s in (data.get("streams") or []))
    except Exception:
        return False


def extract_audio_from_video(video_path: Path, *, out_root: Optional[Path] = None) -> Optional[Path]:
    if not video_path.exists():
        return None
    out_root = out_root or LOCAL_CACHE_DIR
    out_dir = out_root / "extracted_audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_path.stem}.wav"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(out_path)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode('utf-8', errors='ignore')[:400]}")
    return out_path if out_path.exists() and out_path.stat().st_size > 0 else None


def build_audio_mapping_audit(
    trials: Sequence[Dict[str, Any]], *, base_data_dir: Path, max_audit_rows: Optional[int] = None
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
        vp = Path(stimulus_path)
        if not vp.is_absolute():
            vp = (base_data_dir / vp).resolve()
        vp = resolve_mindreading_v_video(vp)
        audio_path, rule = resolve_item_folder_audio(vp)
        out.append({"trial_id": trial_id, "stimulus_path": str(stimulus_path),
                    "resolved_video_path": str(vp),
                    "resolved_audio_path": str(audio_path) if audio_path else None, "matching_rule": rule})
    return out


def save_audio_mapping_audit(audit_rows: Iterable[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(list(audit_rows), indent=2, ensure_ascii=False), encoding="utf-8")
