"""
Generate full Substack article drafts in Eddie's voice using Gemini.

**Where this fits:** ``generate_draft_for_topic`` in ``src/pipeline.py`` calls
``generate_article()`` with trend + historical strings, then ``title_generator`` and
``cover_prompt`` run. To change how pieces read, edit the system prompt and user prompt
below (and optionally ``config.VOICE`` / word targets).
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from config import (
    ARTICLE_GENERATION_TEMPERATURE,
    ARTICLE_TARGET_WORDS_MAX,
    ARTICLE_TARGET_WORDS_MIN,
    VOICE,
)
from src.generation.llm import generate_text

logger = logging.getLogger(__name__)


def _word_count(text: str) -> int:
    """Approximate English word count."""
    return len(re.findall(r"\b[\w'-]+\b", text))


def build_voice_system_instruction() -> str:
    """
    System-style instruction encoding Grey Matters voice, arc, and section discipline.

    Returns:
        Instruction string passed to Gemini.
    """
    v = VOICE
    lo, hi = ARTICLE_TARGET_WORDS_MIN, ARTICLE_TARGET_WORDS_MAX
    mid = (lo + hi) // 2
    return f"""You are the writer of "{v.publication_name}", a Substack about {v.niche}.
Author angle: {v.author_angle}.
Tone: {v.tone}.

## Output
- Markdown body only (no YAML front matter, no title line at the very top).
- Target length: **{mid} words** (stay within **{lo}–{hi}** words total).
- Use **exactly five** level-2 headings: `## ...` — each heading must be a **specific, human phrase**
  (e.g. "## Why your brain treats rejection like pain"), **not** the literal words
  "Hook", "Personal angle", "The science", "Practical insight", or "Reflection".

## Story arc (this order)
1. **Opening block** (~100–130 words under first `##`): A concrete hook — scene, surprise, or honest question.
   Pull the reader in fast; avoid throat-clearing ("In today's newsletter…").
2. **Personal bridge** (~140–180 words): Connect your life (rugby, CrossFit, PhD) to the topic — one vivid moment beats three generic claims.
3. **Science / evidence** (~400–480 words): Explain mechanisms or findings clearly. Use plain language; define jargon once.
   Where evidence is mixed or early, say so. Short paragraphs (3–5 sentences max).
4. **What to do with it** (~180–240 words): Actionable, realistic takeaways — habits, reframes, or how to think about a decision. Not a lecture.
5. **Land the plane** (~100–130 words): Honest reflection or question to the reader, then a **soft** CTA (subscribe / share / reply) — warm, not salesy.

## Style rules
- First person ("I") often; sound like a thoughtful friend who reads papers, not a journal abstract.
- Vary sentence length; avoid repeating the same openers ("Importantly…", "Moreover…").
- No fabricated citations or statistics — if you reference a study generally, keep claims defensible.
- Do not use emojis unless essential (prefer none).
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

    prompt = f"""Write this week's newsletter edition.

**Topic to centre the piece on:** {topic}

**Timeliness / search interest (use lightly — do not turn into a trends essay):**
{trend_context.strip() or "(no extra trend context — lean on timeless explanation + your angle)"}

**What we know about this readership from past sends (honour if relevant; ignore if thin):**
{historical_context.strip() or "(no historical analytics yet — write for a smart non-specialist who likes depth)"}

Before you write: pick a **single spine** for the piece (one thesis line you could say in a sentence) and thread it through all five sections."""

    body = generate_text(
        prompt,
        system_instruction=build_voice_system_instruction(),
        max_output_tokens=8192,
        temperature=ARTICLE_GENERATION_TEMPERATURE,
    )
    if not body:
        return None

    wc = _word_count(body)
    if wc < ARTICLE_TARGET_WORDS_MIN * 0.85:
        logger.warning("Article shorter than target (%d words): %s", wc, topic)
    elif wc > ARTICLE_TARGET_WORDS_MAX * 1.15:
        logger.warning("Article longer than target (%d words): %s", wc, topic)

    return body
