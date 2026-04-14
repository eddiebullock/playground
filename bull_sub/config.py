"""
Project configuration for the Grey Matters Substack content engine.

This module keeps constants and style configuration separate from business logic
so the pipeline remains easy to test and modify.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
EXPORTS_DIR = DATA_DIR / "exports"
DB_PATH = DATA_DIR / "bull_sub.db"
GOOGLE_TRENDS_CACHE_PATH = DATA_DIR / "google_trends_cache.json"
GOOGLE_TRENDS_CACHE_TTL_HOURS = 24


SUBSTACK_EXPORT_REQUIRED_COLUMNS = {
    "title",
    "published_at",
    "open_rate",
    "click_rate",
}

# Note: these CSVs vary; we treat these as optional when present.
SUBSTACK_EXPORT_OPTIONAL_COLUMNS = {
    "emails_sent",
    "new_subscribers",
    "paid_conversions",
    "topics",
    "word_count",
}


GOOGLE_TRENDS_KEYWORD_GROUPS: list[list[str]] = [
    ["neuroscience", "brain science", "cognitive science"],
    ["artificial intelligence", "AI mental health", "AI brain"],
    ["mental health", "anxiety", "depression research"],
    ["sports science", "athlete psychology", "performance"],
]


TOPIC_SCORING_TREND_WEIGHT = 0.40
TOPIC_SCORING_HISTORICAL_WEIGHT = 0.60
RECENTLY_COVERED_DAYS = 30


ENGAGEMENT_SPIKE_STD_MULTIPLIER = 1.5


DEFAULT_SCHEDULE_CRON = {"hour": 9, "minute": 0}


BLUESKY_POST_DELAY_SECONDS = 2


@dataclass(frozen=True)
class VoiceConfig:
    """High-level voice spec for article generation."""

    publication_name: str = "Grey Matters"
    niche: str = "Neuroscience, AI, mental health, science communication"
    author_angle: str = (
        "Cambridge PhD student, ex-pro rugby player, CrossFit coach"
    )
    tone: str = (
        "Accessible, intellectually curious, first-person, evidence-based, occasionally personal"
    )


VOICE = VoiceConfig()


ARTICLE_TARGET_WORDS_MIN = 900
ARTICLE_TARGET_WORDS_MAX = 1100

# Gemini model id (google-generativeai). ``gemini-2.0-flash`` is deprecated for new API keys; use 2.5 Flash.
# Override with GEMINI_MODEL_NAME in .env if needed.
GEMINI_MODEL_NAME_DEFAULT = "gemini-2.5-flash"

# If the primary model returns 404 (not enabled), try these in order.
GEMINI_MODEL_FALLBACKS: list[str] = [
    "gemini-2.5-flash-lite",
    "gemini-1.5-flash-8b",
    "gemini-1.5-flash",
]
# Runtime resolution happens in ``src.generation.llm`` after ``load_dotenv`` (use ``GEMINI_MODEL_NAME`` env).

