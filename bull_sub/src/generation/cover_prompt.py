"""
Generate a text-only prompt for cover/thumbnail images (paste into DALL-E, Midjourney, etc.).
"""

from __future__ import annotations

import logging
from typing import Optional

from src.generation.llm import generate_text

logger = logging.getLogger(__name__)


def generate_cover_image_prompt(
    topic: str,
    title: str,
    article_excerpt: str,
) -> Optional[str]:
    """
    Produce one concise image-generation prompt aligned with the article.

    No image bytes are created; use the result in external tools (DALL-E, Midjourney, Canva, etc.).

    Args:
        topic: Draft topic.
        title: Working title.
        article_excerpt: Start of the article body (truncated by caller if needed).

    Returns:
        A single prompt paragraph, or None if generation fails.
    """
    topic = topic.strip()
    title = title.strip()
    excerpt = (article_excerpt or "").strip()[:4000]

    system = """You write a single image-generation prompt for a newsletter cover image.
Rules:
- Output plain text only: one or two short paragraphs, no bullet lists, no quotes around the whole thing.
- Style: editorial / magazine, tasteful, could be photographic or illustrated; specify mood and lighting.
- The image must contain NO text, NO letters, NO logos (Substack will add typography).
- Avoid clichéd brain clipart; prefer subtle, human, scientific-but-warm vibes fitting a PhD science writer.
- UK/European tone is fine; no watermarks."""

    user = f"""Topic: {topic}
Working title: {title}

Article excerpt (for tone only):
---
{excerpt}
---

Write the image prompt."""

    out = generate_text(
        user,
        system_instruction=system,
        max_output_tokens=512,
        temperature=0.85,
    )
    if not out:
        return None
    return out.strip()
