"""
Combine Google Trends signals with historical newsletter performance to rank topics.

Weights: 40% trend, 60% historical (from config). Drops topics that appear in
recent posts (topics field) or recent drafts within ``RECENTLY_COVERED_DAYS``.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from config import (
    RECENTLY_COVERED_DAYS,
    TOPIC_SCORING_HISTORICAL_WEIGHT,
    TOPIC_SCORING_TREND_WEIGHT,
)
from src.analytics.performance_tracker import calculate_topic_performance
from src.db.models import Draft, Post, TopicScore
from src.trends.google_trends import TrendTopic

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RankedTopic:
    """A topic with trend, historical, and combined scores."""

    topic: str
    trend_score: float
    historical_performance_score: float
    combined_score: float
    source: str


def combine_scores(trend_score: float, historical_performance_score: float) -> float:
    """
    Weighted average of trend (40%) and historical (60%) scores.

    Args:
        trend_score: Google Trends relative score 0-100.
        historical_performance_score: Normalized historical score 0-100.

    Returns:
        Combined score 0-100.
    """
    t = float(trend_score)
    h = float(historical_performance_score)
    return (
        TOPIC_SCORING_TREND_WEIGHT * t
        + TOPIC_SCORING_HISTORICAL_WEIGHT * h
    )


def open_rate_avg_to_historical_score(avg_open_rate_percent: Optional[float]) -> float:
    """
    Map average open rate (percent) to a 0-100 historical score.

    If there is no data, returns a neutral 50. Otherwise maps roughly
    25%%-55%% open rate to 0-100 (clamped).

    Args:
        avg_open_rate_percent: Mean open rate as percent, or None.

    Returns:
        Score in [0, 100].
    """
    if avg_open_rate_percent is None:
        return 50.0
    lo, hi = 25.0, 55.0
    x = float(avg_open_rate_percent)
    if hi <= lo:
        return 50.0
    scaled = (x - lo) / (hi - lo) * 100.0
    return max(0.0, min(100.0, scaled))


def filter_recently_covered(
    session: Session,
    topics: list[str],
    days: int = RECENTLY_COVERED_DAYS,
) -> list[str]:
    """
    Remove topics that appear in recent posts (by topics/tags) or recent drafts.

    Args:
        session: Database session.
        topics: Candidate topic strings.
        days: Lookback window in days.

    Returns:
        Topics that are not recently covered.
    """
    if not topics:
        return []
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=int(days))

    out: list[str] = []
    for t in topics:
        key = t.strip()
        if not key:
            continue
        if _is_topic_recently_covered(session, key, cutoff):
            continue
        out.append(key)
    return out


def _is_topic_recently_covered(session: Session, topic: str, cutoff: dt.datetime) -> bool:
    """Return True if this topic string matches a recent draft or post topics field."""
    draft_hit = session.scalar(
        select(Draft.id)
        .where(Draft.created_at >= cutoff, Draft.topic.ilike(f"%{topic}%"))
        .limit(1)
    )
    if draft_hit is not None:
        return True

    post_hit = session.scalar(
        select(Post.id)
        .where(
            Post.published_at >= cutoff,
            Post.topics.is_not(None),
            Post.topics.ilike(f"%{topic}%"),
        )
        .limit(1)
    )
    return post_hit is not None


def rank_topic_opportunities(
    session: Session,
    trend_topics: list[TrendTopic],
) -> list[RankedTopic]:
    """
    Score and rank trend topics using historical open rates and config weights.

    Args:
        session: Database session (for historical lookups).
        trend_topics: Rows from ``get_trending_topics``.

    Returns:
        Ranked topics, highest ``combined_score`` first.
    """
    ranked: list[RankedTopic] = []
    for tt in trend_topics:
        avg = calculate_topic_performance(session, tt.topic)
        hist = open_rate_avg_to_historical_score(avg)
        combined = combine_scores(tt.trend_score, hist)
        ranked.append(
            RankedTopic(
                topic=tt.topic,
                trend_score=tt.trend_score,
                historical_performance_score=hist,
                combined_score=combined,
                source=tt.source,
            )
        )
    ranked.sort(key=lambda r: r.combined_score, reverse=True)
    return ranked


def persist_topic_scores(session: Session, ranked: list[RankedTopic]) -> int:
    """
    Insert ranked topic scores into ``topic_scores`` (one row per topic).

    Args:
        session: Database session.
        ranked: RankedTopic rows to persist.

    Returns:
        Number of rows inserted.
    """
    n = 0
    for r in ranked:
        session.add(
            TopicScore(
                topic=r.topic,
                trend_score=r.trend_score,
                historical_performance_score=r.historical_performance_score,
                combined_score=r.combined_score,
            )
        )
        n += 1
    logger.info("Persisted %d topic_scores rows", n)
    return n
