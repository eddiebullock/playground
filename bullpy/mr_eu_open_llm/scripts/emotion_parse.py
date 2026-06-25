"""Strict 4AFC emotion parsing (shared by evaluate, tolerant_parse, patching)."""

from __future__ import annotations

import json
import re
from typing import Any, Optional, Sequence, Tuple


def _is_instruction_emotion_placeholder(raw: str) -> bool:
    s = raw.strip().lower()
    if "<" in raw and ">" in raw:
        return True
    if "option labels" in s or "brief justification" in s:
        return True
    return False


def _match_raw_to_option(raw: str, options: Sequence[str]) -> Optional[str]:
    """
    Map raw EMOTION field text to one of the four options (4AFC).
    Prefer longer labels on substring match so e.g. 'Afraid Low Intensity' beats 'Afraid'.
    """
    raw0 = raw.strip().strip(" .\"'`")
    m_paren = re.match(r"^(\d+)\s*[\)\.:]\s*(.+)$", raw0)
    if m_paren:
        idx = int(m_paren.group(1)) - 1
        rest = m_paren.group(2).strip()
        if 0 <= idx < len(options):
            if rest.lower() == options[idx].lower() or rest.lower() in options[idx].lower():
                return options[idx]
            em = _match_raw_to_option(rest, options)
            if em is not None:
                return em
        em = _match_raw_to_option(rest, options)
        if em is not None:
            return em

    mnum = re.match(r"^(\d+)\s*[\)\.]?\s*$", raw0)
    if mnum:
        idx = int(mnum.group(1)) - 1
        if 0 <= idx < len(options):
            return options[idx]

    for opt in options:
        if raw0.lower() == opt.lower():
            return opt

    for opt in sorted(options, key=lambda x: -len(str(x))):
        if opt.lower() in raw0.lower():
            return opt

    return None


def parse_emotion(output_text: Any, options: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (emotion, reasoning). Emotion is forced to one of options if possible.

    If output_text still includes the user prompt (should be avoided: decode only new tokens),
    we skip instruction placeholders and use the last EMOTION / REASONING lines.
    """
    emotion = None
    reasoning = None
    if not isinstance(output_text, str):
        try:
            text = json.dumps(output_text, ensure_ascii=False)
        except Exception:
            text = str(output_text)
    else:
        text = output_text
    text = text.strip()

    for m in reversed(list(re.finditer(r"(?im)^\s*EMOTION\s*[:\-]\s*(.+?)\s*$", text))):
        raw = m.group(1).strip()
        if _is_instruction_emotion_placeholder(raw):
            continue
        emotion = _match_raw_to_option(raw, options)
        if emotion is not None:
            break

    for r in reversed(list(re.finditer(r"(?im)^\s*REASONING\s*:\s*(.+?)\s*$", text))):
        rs = r.group(1).strip()
        if rs.lower() in {"<brief justification>", "brief justification"}:
            continue
        if "<" in rs and "brief" in rs.lower():
            continue
        reasoning = rs
        break

    if emotion is None and text:
        lower = text.lower()
        for opt in sorted(options, key=lambda x: -len(str(x))):
            if str(opt).lower() in lower:
                emotion = str(opt)
                break

    return emotion, reasoning
