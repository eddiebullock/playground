"""
Generate three Substack Notes from an approved article (manual copy-paste queue).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from src.generation.llm import generate_text

logger = logging.getLogger(__name__)


NOTES_PROMPT = """From the article below, write 3 Substack Notes for "Grey Matters".

Return ONLY valid JSON (no markdown), shape:
{{
  "hook_note": "...",
  "personal_note": "...",
  "question_note": "..."
}}

Rules:
- Each note: 150-220 words max, punchy, standalone.
- hook_note: the most surprising finding; close by nudging readers to the full post (today's edition / latest send).
- personal_note: Eddie's personal reaction, no direct plug for the article.
- question_note: genuine question to spark comments.

Article topic: {topic}

Article markdown:
---
{article}
---
"""


@dataclass(frozen=True)
class NotesBundle:
    """Three notes plus suggested schedule dates."""

    note_1: str
    note_2: str
    note_3: str
    scheduled_day_1: dt.date
    scheduled_day_2: dt.date
    scheduled_day_3: dt.date


def _parse_notes_json(raw: str) -> Optional[tuple[str, str, str]]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
        a, b, c = data.get("hook_note"), data.get("personal_note"), data.get("question_note")
        if not all(isinstance(x, str) and x.strip() for x in (a, b, c)):
            return None
        return a.strip(), b.strip(), c.strip()
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("Notes JSON parse failed: %s", e)
        return None


def generate_notes(
    article_markdown: str,
    topic: str,
    publish_anchor_date: Optional[dt.date] = None,
) -> Optional[NotesBundle]:
    """
    Generate three notes and schedule days publish, publish+3, publish+7.

    Args:
        article_markdown: Full article body.
        topic: Topic label.
        publish_anchor_date: Day-of-publish; defaults to today (UTC date).

    Returns:
        NotesBundle or None if generation fails.
    """
    anchor = publish_anchor_date or dt.datetime.now(dt.timezone.utc).date()
    prompt = NOTES_PROMPT.format(topic=topic.strip(), article=article_markdown.strip()[:12000])
    raw = generate_text(
        prompt,
        system_instruction="You output only compact JSON.",
        max_output_tokens=2048,
        temperature=0.8,
    )
    if not raw:
        return None
    parsed = _parse_notes_json(raw)
    if not parsed:
        return None
    n1, n2, n3 = parsed
    return NotesBundle(
        note_1=n1,
        note_2=n2,
        note_3=n3,
        scheduled_day_1=anchor,
        scheduled_day_2=anchor + dt.timedelta(days=3),
        scheduled_day_3=anchor + dt.timedelta(days=7),
    )
