"""Parse a 40-item completion. Refusals and unparseable answers are kept."""

from __future__ import annotations

import re

from scog.a1.prompts import RESPONSE_LABELS

_LINE = re.compile(r"^\s*(\d+)\s*[\.\)\:\-]\s*(.*?)\s*$")
_REFUSAL = (
    "i can't",
    "i cannot",
    "i won't",
    "i will not",
    "i am unable",
    "i'm unable",
    "as an ai",
    "i must decline",
    "i'm sorry",
)


def _norm(text: str) -> str:
    folded = text.strip().lower().rstrip(".")
    return " ".join(folded.split())


def parse_completion(text: str, *, n_items: int = 40) -> dict:
    if text is None or not str(text).strip():
        raise ValueError("completion is empty")
    answers: list[str | None] = [None] * n_items
    for raw in str(text).splitlines():
        match = _LINE.match(raw)
        if not match:
            continue
        number = int(match.group(1))
        if number < 1 or number > n_items:
            continue
        label = _norm(match.group(2))
        if label in RESPONSE_LABELS:
            answers[number - 1] = label
    n_parsed = sum(answer is not None for answer in answers)
    folded = _norm(text)
    refused = n_parsed == 0 and any(phrase in folded for phrase in _REFUSAL)
    if n_parsed == n_items:
        status = "complete"
    elif refused:
        status = "refusal"
    else:
        status = "unparseable"
    return {
        "status": status,
        "answers": answers,
        "n_parsed": n_parsed,
        "n_items": n_items,
    }
