from __future__ import annotations

"""Load EU-Emotions label definitions for semantic entropy embeddings."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from config import LOCAL_DATA_DIR, PROJECT_ROOT

DEFAULT_DEFINITIONS_PATH = LOCAL_DATA_DIR / "eu_emotion_definitions.json"
LOW_INTENSITY_SUFFIX = " low intensity"


def definitions_path() -> Path:
    if DEFAULT_DEFINITIONS_PATH.is_file():
        return DEFAULT_DEFINITIONS_PATH
    alt = PROJECT_ROOT / "data" / "eu_emotion_definitions.json"
    return alt if alt.is_file() else DEFAULT_DEFINITIONS_PATH


@lru_cache(maxsize=1)
def load_eu_emotion_definitions(path: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    p = Path(path) if path else definitions_path()
    if not p.is_file():
        return {}
    obj = json.loads(p.read_text(encoding="utf-8"))
    base = obj.get("base_emotions") or {}
    out: Dict[str, Dict[str, str]] = {}
    for key, rec in base.items():
        if isinstance(rec, dict):
            out[str(key).casefold()] = {
                "canonical_name": str(rec.get("canonical_name", key)),
                "definition": str(rec.get("definition", "")),
            }
    return out


def base_emotion_key(label: str) -> str:
    s = str(label).strip()
    low = s.casefold()
    if low.endswith(LOW_INTENSITY_SUFFIX):
        return s[: -len(LOW_INTENSITY_SUFFIX)].strip().casefold()
    return low


def definition_for_label(label: str, defs: Optional[Dict[str, Dict[str, str]]] = None) -> Optional[Dict[str, str]]:
    defs = defs if defs is not None else load_eu_emotion_definitions()
    return defs.get(base_emotion_key(label))


def embedding_text_for_label(label: str, *, rich: bool = True) -> str:
    """Text to embed for a fine EU label (with optional intensity variant)."""
    s = str(label).strip()
    defs = load_eu_emotion_definitions()
    rec = definition_for_label(s, defs)
    low = s.casefold()
    is_low = low.endswith(LOW_INTENSITY_SUFFIX)
    base = s[: -len(LOW_INTENSITY_SUFFIX)].strip() if is_low else s

    if rec and rec.get("definition"):
        canon = rec.get("canonical_name", base)
        if is_low:
            return (
                f"Mental state: {canon} ({base}) — low intensity. "
                f"Definition: {rec['definition']} The person expresses this with low emotional intensity."
            )
        return f"Mental state: {canon} ({base}). Definition: {rec['definition']}"

    if rich:
        if is_low:
            return (
                f"Mental state label: {s}. The person appears to express {base} "
                f"with low emotional intensity."
            )
        return f"Mental state label: {s}. The person appears to express {s}."
    return s
