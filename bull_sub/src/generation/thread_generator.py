"""
Generate Bluesky and Twitter/X thread variants from an article.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from src.generation.llm import generate_text

logger = logging.getLogger(__name__)

MAX_LEN_BLUESKY = 280
MAX_LEN_TWITTER = 280


THREAD_PROMPT = """Create two social threads from this newsletter article.

Topic: {topic}
Substack article URL (use verbatim in final CTA post): {url}

Return ONLY valid JSON (no markdown):
{{
  "bluesky": ["post1", "post2", ...],
  "twitter": ["post1", "post2", ...]
}}

Bluesky thread rules:
- 8-12 posts, each max {bsky_len} characters (hard limit).
- Post 1: hook — most interesting line.
- Posts 2-7: one key insight per post, conversational.
- Post 8: personal angle / Eddie's take.
- Post 9 (or last): CTA with the Substack URL above.
- If you need fewer than 12 to stay clear, use 8-10.

Twitter rules:
- Same structure and count, slightly punchier / more terse tone.
- Each max {tw_len} characters.

Article (markdown):
---
{article}
---
"""


@dataclass(frozen=True)
class ThreadBundle:
    """Parallel threads for Bluesky and Twitter."""

    bluesky: list[str]
    twitter: list[str]


def _clip(s: str, n: int) -> str:
    """Clip string to n characters at word boundary when possible."""
    s = s.strip()
    if len(s) <= n:
        return s
    cut = s[: n - 1]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + "…"


def _parse_thread_json(raw: str) -> Optional[tuple[list[str], list[str]]]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
        b, t = data.get("bluesky"), data.get("twitter")
        if not isinstance(b, list) or not isinstance(t, list):
            return None
        bs = [str(x).strip() for x in b if str(x).strip()]
        ts = [str(x).strip() for x in t if str(x).strip()]
        if len(bs) < 3 or len(ts) < 3:
            return None
        return bs, ts
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("Thread JSON parse failed: %s", e)
        return None


def generate_threads(
    article_markdown: str,
    topic: str,
    substack_article_url: str,
) -> Optional[ThreadBundle]:
    """
    Generate Bluesky and Twitter thread post lists.

    Args:
        article_markdown: Article body.
        topic: Topic label.
        substack_article_url: Link for the CTA post.

    Returns:
        ThreadBundle or None on failure.
    """
    prompt = THREAD_PROMPT.format(
        topic=topic.strip(),
        url=substack_article_url.strip(),
        bsky_len=MAX_LEN_BLUESKY,
        tw_len=MAX_LEN_TWITTER,
        article=article_markdown.strip()[:12000],
    )
    raw = generate_text(
        prompt,
        system_instruction="You output only compact JSON. Respect character limits.",
        max_output_tokens=4096,
        temperature=0.75,
    )
    if not raw:
        return None
    parsed = _parse_thread_json(raw)
    if not parsed:
        return None
    bs, ts = parsed
    bs = [_clip(p, MAX_LEN_BLUESKY) for p in bs]
    ts = [_clip(p, MAX_LEN_TWITTER) for p in ts]
    return ThreadBundle(bluesky=bs, twitter=ts)
