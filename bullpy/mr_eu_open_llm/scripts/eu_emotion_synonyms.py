"""Load auditable EU-Emotions synonym sets for free-response judging."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set

from config import LOCAL_DATA_DIR

_SYNONYMS_PATH = LOCAL_DATA_DIR / "eu_emotion_synonyms.json"
_WORD_BOUNDARY = r"\b{}\b"

# Fallback when data/eu_emotion_synonyms.json not synced to HPC yet (must match JSON on disk).
_BUILTIN_SYNONYM_CONFIG: Dict = {
    "version": "eu_emotions_synonyms_v1",
    "collapse_intensity_to_base": True,
    "synonyms": {
        "afraid": ["afraid", "fear", "fearful", "frightened", "scared", "terrified", "anxious", "apprehensive"],
        "angry": ["angry", "anger", "mad", "furious", "irritated", "annoyed", "hostile", "enraged"],
        "ashamed": ["ashamed", "shame", "embarrassed", "humiliated", "mortified"],
        "bored": ["bored", "boredom", "uninterested", "dull", "apathetic"],
        "disappointed": ["disappointed", "disappointment", "let down", "disheartened"],
        "disgusted": ["disgusted", "disgust", "revolted", "repulsed", "sickened"],
        "excited": ["excited", "excitement", "enthusiastic", "thrilled", "elated"],
        "frustrated": ["frustrated", "frustration", "thwarted", "blocked"],
        "happy": ["happy", "happiness", "joyful", "joy", "pleased", "cheerful", "delighted"],
        "hurt": ["hurt", "wounded", "upset", "pained", "emotional pain"],
        "interested": ["interested", "interest", "curious", "engaged", "attentive"],
        "jealous": ["jealous", "jealousy", "envious", "envy", "covetous"],
        "joking": ["joking", "joke", "playful", "teasing", "humorous", "amused", "funny"],
        "kind": ["kind", "kindness", "friendly", "caring", "compassionate", "warm"],
        "proud": ["proud", "pride", "satisfied", "accomplished"],
        "sad": ["sad", "sadness", "unhappy", "sorrowful", "melancholy", "downcast", "gloomy"],
        "sneaky": ["sneaky", "sly", "devious", "cunning", "deceptive", "secretive"],
        "surprised": ["surprised", "surprise", "shocked", "astonished", "amazed", "startled"],
        "unfriendly": ["unfriendly", "cold", "hostile", "unkind", "disdainful", "aloof"],
        "worried": ["worried", "worry", "concerned", "uneasy", "nervous", "anxious"],
    },
}


def _word_boundary_match(term: str, haystack: str) -> bool:
    return re.search(_WORD_BOUNDARY.format(re.escape(term)), haystack, re.IGNORECASE) is not None


def _normalize_label(label: str) -> str:
    return " ".join(str(label).strip().lower().split())


def collapse_intensity_label(label: str) -> str:
    """Map e.g. 'afraid low intensity' -> 'afraid'."""
    s = _normalize_label(label)
    suffix = " low intensity"
    if s.endswith(suffix):
        return s[: -len(suffix)].strip()
    return s


@lru_cache(maxsize=1)
def load_synonym_config() -> Dict:
    if _SYNONYMS_PATH.is_file():
        return json.loads(_SYNONYMS_PATH.read_text(encoding="utf-8"))
    return dict(_BUILTIN_SYNONYM_CONFIG)


def synonym_config_source() -> str:
    """'file' or 'builtin' — for logging in verification reports."""
    return "file" if _SYNONYMS_PATH.is_file() else "builtin"


def base_emotion_keys() -> List[str]:
    cfg = load_synonym_config()
    return sorted(cfg.get("synonyms", {}).keys())


def accepted_terms_for_label(label: str) -> Set[str]:
    """All accepted strings for a trial label (includes synonyms + label variants)."""
    cfg = load_synonym_config()
    collapse = bool(cfg.get("collapse_intensity_to_base", True))
    raw = _normalize_label(label)
    base = collapse_intensity_label(raw) if collapse else raw
    terms: Set[str] = {raw, base}
    for key in (base, raw):
        for syn in cfg.get("synonyms", {}).get(key, []):
            terms.add(_normalize_label(syn))
    if raw != base:
        terms.add(raw)
    return {t for t in terms if t}


def match_label_in_text(text: str, label: str) -> Optional[str]:
    """
    Return the matched accepted term if *text* expresses *label* (rule-based).
    Prefers longer multi-word synonyms first.
    """
    if not text or not label:
        return None
    haystack = " ".join(str(text).lower().split())
    for term in sorted(accepted_terms_for_label(label), key=lambda x: (-len(x), x)):
        if " " in term:
            if term in haystack:
                return term
        else:
            if _word_boundary_match(term, haystack):
                return term
    return None


def judge_free_response_rule(text: str, gold_label: str) -> Dict:
    """Rule-based free-response judge (no LLM)."""
    matched = match_label_in_text(text, gold_label)
    return {
        "correct": matched is not None,
        "matched_term": matched,
        "gold_label": gold_label,
        "gold_base": collapse_intensity_label(gold_label),
        "method": "rule_synonym",
    }
