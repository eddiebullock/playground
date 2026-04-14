"""
Orchestration helpers: CSV import, topic selection, and draft generation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from sqlalchemy.orm import Session

from src.alerts.cta_detector import build_historical_context_for_topic, run_cta_detection
from src.analytics.csv_parser import ImportResult, import_exports_folder
from src.db.models import Draft, Note, Thread
from src.generation.article_generator import generate_article
from src.generation.cover_prompt import generate_cover_image_prompt
from src.generation.notes_generator import generate_notes
from src.generation.thread_generator import generate_threads
from src.generation.title_generator import generate_and_score_titles, pick_recommended_title
from src.trends.google_trends import TrendTopic, get_trending_topics
from src.trends.topic_scorer import (
    filter_recently_covered,
    persist_topic_scores,
    rank_topic_opportunities,
)

if TYPE_CHECKING:
    from src.trends.topic_scorer import RankedTopic

logger = logging.getLogger(__name__)


def import_csvs_and_alerts(session: Session) -> ImportResult:
    """
    Import Substack CSVs from ``data/exports/`` and run CTA detection.

    Args:
        session: Active SQLAlchemy session.

    Returns:
        Import statistics.
    """
    imp = import_exports_folder(session)
    run_cta_detection(session)
    return imp


def pick_next_topic(
    session: Session,
    ranked_topics: list["RankedTopic"],
) -> tuple[Optional[str], Optional[float]]:
    """
    Choose the best topic not recently covered; fall back to top ranked.

    Args:
        session: DB session.
        ranked_topics: List of ``RankedTopic`` from ``rank_topic_opportunities``.

    Returns:
        (topic, combined_score) or (None, None).
    """
    if not ranked_topics:
        return None, None
    names = [r.topic for r in ranked_topics]
    fresh = set(filter_recently_covered(session, names))
    for r in ranked_topics:
        if r.topic in fresh:
            return r.topic, r.combined_score
    r0 = ranked_topics[0]
    return r0.topic, r0.combined_score


def build_trend_context(trend_topics: list[TrendTopic], limit: int = 14) -> str:
    """Format trend topics into a short string for the article prompt."""
    parts = [f"{t.topic}={t.trend_score:.0f}" for t in trend_topics[:limit]]
    return "Trending keywords (Google Trends relative interest 0-100): " + ", ".join(parts)


def generate_draft_for_topic(
    session: Session,
    topic: str,
    combined_score: Optional[float] = None,
    trend_topics: Optional[list[TrendTopic]] = None,
) -> Optional[Draft]:
    """
    Generate article body + scored titles and persist a ``pending`` draft.

    Args:
        session: DB session.
        topic: Focus topic.
        combined_score: Optional score from topic ranking.
        trend_topics: Optional pre-fetched trends; if None, loads from cache/network.

    Returns:
        New ``Draft`` instance (added to session), or None if generation fails.
    """
    topic = topic.strip()
    if not topic:
        return None

    if trend_topics is None:
        trend_topics, _ = get_trending_topics(force_refresh=False)

    trend_ctx = build_trend_context(trend_topics)
    hist = build_historical_context_for_topic(session, topic)
    body = generate_article(topic, trend_ctx, hist)
    if not body:
        logger.warning("Article generation failed for topic=%s", topic)
        return None

    scored_titles = generate_and_score_titles(topic, session)
    if not scored_titles:
        title = f"Notes on {topic}"
        scored_titles = [
            {
                "title": title,
                "score": 50.0,
                "recommended": True,
                "pattern_hits": [],
                "pattern": "fallback",
            }
        ]
    else:
        title = pick_recommended_title(scored_titles)

    cover_prompt = generate_cover_image_prompt(topic, title[:500], body)
    if not cover_prompt:
        logger.warning("Cover image prompt generation failed; draft saved without it")

    draft = Draft(
        title=title[:500],
        content=body,
        topic=topic[:200],
        title_variants=scored_titles,
        cover_image_prompt=cover_prompt,
        combined_score=combined_score,
        status="pending",
    )
    session.add(draft)
    session.flush()
    logger.info("Created pending draft id=%s topic=%s", draft.id, topic)
    return draft


def refresh_trends_and_rank(
    session: Session,
    force_refresh: bool = False,
) -> list["RankedTopic"]:
    """
    Fetch trends, rank, and persist ``topic_scores`` rows.

    Returns:
        Ranked list of ``RankedTopic``.
    """
    topics, _ = get_trending_topics(force_refresh=force_refresh)
    ranked = rank_topic_opportunities(session, topics)
    persist_topic_scores(session, ranked)
    return ranked


def generate_notes_and_threads_for_draft(
    session: Session,
    draft: Draft,
    full_article_url: str,
) -> tuple[Optional[Note], Optional[Thread], Optional[Thread]]:
    """
    After approval: generate Notes + Bluesky/Twitter threads and store rows.

    Args:
        session: DB session.
        draft: Approved draft.
        full_article_url: Public URL of the published post (for thread CTA).

    Returns:
        (note_row, bluesky_thread, twitter_thread) — any may be None on failure.
    """
    article_url = full_article_url.strip()
    bundle = generate_notes(draft.content, draft.topic)
    if not bundle:
        return None, None, None

    note_row = Note(
        draft_id=draft.id,
        note_1=bundle.note_1,
        note_2=bundle.note_2,
        note_3=bundle.note_3,
        scheduled_day_1=bundle.scheduled_day_1,
        scheduled_day_2=bundle.scheduled_day_2,
        scheduled_day_3=bundle.scheduled_day_3,
        status="pending",
    )
    session.add(note_row)

    threads = generate_threads(draft.content, draft.topic, article_url)
    bsky_row: Optional[Thread] = None
    tw_row: Optional[Thread] = None
    if threads:
        bsky_row = Thread(
            draft_id=draft.id,
            platform="bluesky",
            thread_content=threads.bluesky,
            status="pending",
        )
        tw_row = Thread(
            draft_id=draft.id,
            platform="twitter",
            thread_content=threads.twitter,
            status="pending",
        )
        session.add(bsky_row)
        session.add(tw_row)

    return note_row, bsky_row, tw_row


def run_full_pipeline() -> str:
    """
    Daily pipeline: import CSVs, refresh trends, pick topic, create draft.

    Returns:
        Human-readable summary line.
    """
    from dotenv import load_dotenv

    from config import PROJECT_ROOT
    from src.db.session import init_db, session_scope

    load_dotenv(PROJECT_ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    engine = init_db()

    with session_scope(engine) as s:
        imp = import_csvs_and_alerts(s)
        logger.info(
            "CSV import: files=%d rows=%d upserted=%d skipped=%d",
            imp.files_seen,
            imp.rows_seen,
            imp.upserted_posts,
            imp.skipped_rows,
        )

        ranked = refresh_trends_and_rank(s, force_refresh=False)
        topic, combined = pick_next_topic(s, ranked)
        if not topic:
            return "Pipeline complete. No trending topics available (check cache / pytrends)."

        draft = generate_draft_for_topic(s, topic, combined_score=combined)
        if not draft:
            return f"Pipeline complete. Draft generation failed for topic: {topic}"

    return f"Pipeline complete. New draft ready: {topic}"
