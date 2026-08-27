"""Minimal EMOTION:/REASONING: parser for study4 RMET outputs (no study3 import)."""

from __future__ import annotations

import re
from typing import Optional, Sequence, Tuple


def _match_raw_to_option(raw: str, options: Sequence[str]) -> Optional[str]:
    s = raw.strip().strip("\"'").strip()
    # Strip leading "3) " / "3." numbering
    s = re.sub(r"^\d+\)\s*", "", s)
    s = re.sub(r"^\d+\.\s*", "", s)
    s = s.strip()
    by_cf = {o.casefold(): o for o in options}
    if s.casefold() in by_cf:
        return by_cf[s.casefold()]
    for o in sorted(options, key=lambda x: -len(str(x))):
        if str(o).casefold() in s.casefold():
            return str(o)
    return None


def parse_rmet_emotion(output_text: str, options: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
    """Return (emotion_label, reasoning) forced to one of options when possible."""
    if not isinstance(output_text, str):
        text = str(output_text)
    else:
        text = output_text
    text = text.strip()
    emotion = None
    reasoning = None

    for m in reversed(list(re.finditer(r"(?im)^\s*EMOTION\s*[:\-]\s*(.+?)\s*$", text))):
        raw = m.group(1).strip()
        if "<" in raw and "option" in raw.lower():
            continue
        emotion = _match_raw_to_option(raw, options)
        if emotion is not None:
            break

    for r in reversed(list(re.finditer(r"(?im)^\s*REASONING\s*:\s*(.+?)\s*$", text))):
        rs = r.group(1).strip()
        if rs.lower() in {"<brief justification>", "brief justification"}:
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
