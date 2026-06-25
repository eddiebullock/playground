from __future__ import annotations

"""Discover canonical Study 1 EU-Emotions baseline JSONs per model."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import LOCAL_RESULTS_DIR, PROTOCOL_VERSION, STUDY_MODELS

# Preferred condition per model for primary EU baseline reporting.
CANONICAL_CONDITION: Dict[str, str] = {
    "qwen2vl": "video_only",
    "llavanext": "video_only",
    "gemma4": "multimodal",
}

MIN_FULL_TRIALS = 100


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def score_baseline_json(obj: Dict[str, Any], *, preferred_condition: str) -> Tuple[int, int, float]:
    """Higher is better: full trial count, matching condition, newer mtime tie-break outside."""
    n_scored = int(obj.get("n_scored") or 0)
    cond = str(obj.get("condition") or "")
    cond_match = 1 if cond == preferred_condition else 0
    has_entropy = 1 if obj.get("mean_semantic_entropy") is not None else 0
    return (n_scored, cond_match, has_entropy)


def discover_canonical_baseline(
    model_key: str,
    *,
    results_root: Path = LOCAL_RESULTS_DIR / "baseline" / "eu_emotions",
) -> Optional[Path]:
    model_dir = results_root / model_key
    if not model_dir.is_dir():
        return None
    preferred = CANONICAL_CONDITION.get(model_key, "video_only")
    candidates: List[Tuple[Tuple[int, int, int], float, Path]] = []
    for path in sorted(model_dir.glob("eval_v2_*.json")):
        obj = _load_json(path)
        if obj is None or obj.get("protocol_version") != PROTOCOL_VERSION:
            continue
        if obj.get("model") != model_key:
            continue
        rank = score_baseline_json(obj, preferred_condition=preferred)
        if rank[0] < MIN_FULL_TRIALS:
            continue
        candidates.append((rank, path.stat().st_mtime, path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]


def discover_all_canonical(
    *,
    results_root: Path = LOCAL_RESULTS_DIR / "baseline" / "eu_emotions",
) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for model in STUDY_MODELS:
        p = discover_canonical_baseline(model, results_root=results_root)
        if p is not None:
            out[model] = p
    return out
