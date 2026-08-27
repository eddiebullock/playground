"""
study4 RMET 4AFC prompt — local to study4_rmet.

Wording adapted from study3's build_4afc_prompt for a static eye-region image.
This module does not import or modify study3 scripts/prompts.py.
"""

from __future__ import annotations

from typing import Sequence


def build_rmet_4afc_prompt(options: Sequence[str]) -> str:
    """Forced-choice prompt for one RMET item (eyes-only image; labels in text only)."""
    n = len(options)
    opts = "\n".join([f"{i + 1}) {opt}" for i, opt in enumerate(options)])
    return (
        f"You are performing a {n}-alternative forced-choice mental state recognition task.\n"
        "Analyze the image of the eye region only. Choose exactly one label.\n"
        f"\nOPTIONS:\n{opts}\n\n"
        "Respond with:\n"
        "EMOTION: <one of the option labels exactly>\n"
        "REASONING: <brief justification>\n"
    )
