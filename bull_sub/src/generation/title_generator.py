"""
Generate and score title variants for drafts using Gemini plus historical open-rate patterns.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from sqlalchemy.orm import Session

from src.analytics.performance_tracker import TITLE_PATTERNS, get_top_performing_titles
from src.generation.llm import generate_text

logger = logging.getLogger(__name__)


TITLE_PROMPT_TEMPLATE = """You create newsletter subject lines for "Grey Matters" (neuroscience, AI, mental health).
Topic: {topic}

Produce exactly 5 titles, one per pattern below. Return ONLY valid JSON (no markdown), shape:
{{"variants":[
  {{"pattern":"question","title":"..."}},
  {{"pattern":"number","title":"..."}},
  {{"pattern":"contrarian","title":"..."}},
  {{"pattern":"personal","title":"..."}},
  {{"pattern":"curiosity_gap","title":"..."}}
]}}

Patterns (meaning):
1. question — e.g. "Why Does X Actually Happen?"
2. number — e.g. "3 Things Neuroscience Says About X"
3. contrarian — e.g. "X Is Not What You Think"
4. personal — e.g. "What X Taught Me About Y"
5. curiosity_gap — e.g. "The Surprising Link Between X and Y"

Rules: max ~90 characters each, no clickbait lies, no ALL CAPS.
"""


def generate_title_variants_json(topic: str) -> Optional[list[dict[str, str]]]:
    """
    Ask Gemini for five titled variants as structured JSON.

    Args:
        topic: Article topic.

    Returns:
        List of dicts with keys pattern, title; or None on failure.
    """
    raw = generate_text(
        TITLE_PROMPT_TEMPLATE.format(topic=topic.strip()),
        system_instruction="You output only compact JSON. No code fences.",
        max_output_tokens=1024,
        temperature=0.85,
    )
    if not raw:
        return None
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
        variants = data.get("variants")
        if not isinstance(variants, list) or len(variants) < 5:
            logger.warning("Title JSON missing variants: %s", cleaned[:200])
            return None
        out: list[dict[str, str]] = []
        for v in variants[:5]:
            if isinstance(v, dict) and v.get("title"):
                out.append({"pattern": str(v.get("pattern", "")), "title": str(v["title"]).strip()})
        return out if len(out) >= 5 else None
    except json.JSONDecodeError as e:
        logger.warning("Could not parse title JSON: %s — %s", e, cleaned[:300])
        return None


def score_titles(
    variants: list[dict[str, Any]],
    session: Session,
) -> list[dict[str, Any]]:
    """
    Score each title using historical open-rate pattern lifts from past posts.

    Args:
        variants: Items with at least ``title`` (and optional ``pattern``).
        session: DB session for analytics.

    Returns:
        Same variants enriched with numeric ``score`` and ``recommended`` flag (one winner).
    """
    lifts = get_top_performing_titles(session, min_posts=5)
    baseline = 50.0
    enriched: list[dict[str, Any]] = []
    for v in variants:
        title = str(v.get("title", "")).strip()
        score = baseline
        hits: list[str] = []
        for name, rx in TITLE_PATTERNS.items():
            if rx.search(title):
                hits.append(name)
                score += float(lifts.get(name, 0.0))
        row = {**v, "title": title, "pattern_hits": hits, "score": round(score, 2)}
        enriched.append(row)

    if not enriched:
        return []

    best_idx = max(range(len(enriched)), key=lambda i: enriched[i]["score"])
    for i, row in enumerate(enriched):
        row["recommended"] = i == best_idx
    return enriched


def generate_and_score_titles(topic: str, session: Session) -> list[dict[str, Any]]:
    """
    Generate titles via Gemini, then score them against historical patterns.

    Args:
        topic: Topic string.
        session: DB session.

    Returns:
        Scored variant dicts; empty list if generation fails.
    """
    gen = generate_title_variants_json(topic)
    if not gen:
        return []
    return score_titles(gen, session)


def pick_recommended_title(scored: list[dict[str, Any]]) -> str:
    """
    Return the recommended title string, or the first title if none flagged.

    Args:
        scored: Output of ``score_titles``.

    Returns:
        Title string.
    """
    for row in scored:
        if row.get("recommended"):
            return str(row["title"])
    return str(scored[0]["title"]) if scored else "Untitled"
