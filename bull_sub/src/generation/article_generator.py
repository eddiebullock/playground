"""
Generate full Substack article drafts in Eddie's voice using Gemini.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from config import ARTICLE_TARGET_WORDS_MAX, ARTICLE_TARGET_WORDS_MIN, VOICE
from src.generation.llm import generate_text

logger = logging.getLogger(__name__)


def _word_count(text: str) -> int:
    """Approximate English word count."""
    return len(re.findall(r"\b[\w'-]+\b", text))


def build_voice_system_instruction() -> str:
    """
    System-style instruction encoding Grey Matters voice and structure.

    Returns:
        Instruction string passed to Gemini.
    """
    v = VOICE
    return f"""You are the writer of "{v.publication_name}", a Substack about {v.niche}.
Author angle: {v.author_angle}.
Tone: {v.tone}.

Writing rules:
- Opens with a personal hook or surprising fact.
- Bridge lived experience (rugby, CrossFit, PhD life) with the science.
- Explain complex ideas accessibly without dumbing down.
- Use "I" frequently; personal, not academic.
- Evidence-based: mention uncertainty where it exists; avoid hype.
- End with a question or reflection, then a soft CTA to subscribe or share.
- Structure with clear sections (use markdown ## headings):
  ## Hook
  ## Personal angle
  ## The science
  ## Practical insight
  ## Reflection
- Length: {ARTICLE_TARGET_WORDS_MIN}-{ARTICLE_TARGET_WORDS_MAX} words (aim for the middle).
- Do not include a title line at the top; body only.
"""


def generate_article(
    topic: str,
    trend_context: str,
    historical_context: str,
) -> Optional[str]:
    """
    Generate a draft article for a topic with trend and audience context.

    Args:
        topic: Core topic to write about.
        trend_context: What is trending / timely about this topic (from Trends).
        historical_context: What this audience has responded to before (from analytics).

    Returns:
        Markdown article body, or None if generation fails.
    """
    topic = topic.strip()
    if not topic:
        logger.warning("generate_article called with empty topic")
        return None

    prompt = f"""Write this week's newsletter piece.

Topic (focus): {topic}

Trend / timeliness context:
{trend_context.strip() or "(no extra trend context)"}

Historical audience context (what has performed well before):
{historical_context.strip() or "(no historical data yet — write for a curious general audience)"}
"""

    body = generate_text(
        prompt,
        system_instruction=build_voice_system_instruction(),
        max_output_tokens=8192,
        temperature=0.75,
    )
    if not body:
        return None

    wc = _word_count(body)
    if wc < ARTICLE_TARGET_WORDS_MIN * 0.85:
        logger.warning("Article shorter than target (%d words): %s", wc, topic)
    elif wc > ARTICLE_TARGET_WORDS_MAX * 1.15:
        logger.warning("Article longer than target (%d words): %s", wc, topic)

    return body
