"""
Performance tracking utilities computed from the `posts` table.

These functions are used for:
- topic performance scoring (historical context)
- detecting engagement spikes
- learning title patterns correlated with high open rates
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Optional

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from config import ENGAGEMENT_SPIKE_STD_MULTIPLIER
from src.db.models import Post


TITLE_PATTERNS = {
    "question_mark": re.compile(r"\?"),
    "starts_how": re.compile(r"^\s*how\b", re.IGNORECASE),
    "starts_why": re.compile(r"^\s*why\b", re.IGNORECASE),
    "has_number": re.compile(r"\b\d+\b"),
    "first_person": re.compile(r"\b(i|my|me)\b", re.IGNORECASE),
}


@dataclass(frozen=True)
class EngagementSpike:
    """A post that exceeds the spike threshold."""

    post_id: int
    title: str
    published_at: object
    open_rate: float
    threshold: float


def _split_topics(topics: Optional[str]) -> list[str]:
    if topics is None:
        return []
    raw = topics.strip()
    if raw == "":
        return []
    # Accept either comma-separated or semicolon-separated tags.
    parts = re.split(r"[;,]", raw)
    out: list[str] = []
    for p in parts:
        t = p.strip()
        if t:
            out.append(t)
    return out


def calculate_topic_performance(session: Session, topic: str) -> Optional[float]:
    """
    Calculate average open rate for posts tagged with a topic.

    Args:
        session: SQLAlchemy session.
        topic: Topic string to match (substring match within `topics`).

    Returns:
        Average open rate (percent) or None if insufficient data.
    """
    topic = topic.strip()
    if not topic:
        return None

    # We keep topics as a raw string; best-effort substring matching.
    stmt = (
        select(func.avg(Post.open_rate))
        .where(and_(Post.open_rate.is_not(None), Post.topics.is_not(None)))
        .where(Post.topics.ilike(f"%{topic}%"))
    )
    v = session.scalar(stmt)
    return float(v) if v is not None else None


def detect_engagement_spike(
    session: Session,
    std_multiplier: float = ENGAGEMENT_SPIKE_STD_MULTIPLIER,
) -> list[EngagementSpike]:
    """
    Detect engagement spikes based on open_rate distribution.

    Spike condition:
        open_rate > mean + std_multiplier * std_dev

    Args:
        session: SQLAlchemy session.
        std_multiplier: Standard deviation multiplier.

    Returns:
        List of EngagementSpike objects.
    """
    # Pull open rates into Python to compute std dev robustly for small N.
    rows = session.execute(select(Post.id, Post.title, Post.published_at, Post.open_rate).where(Post.open_rate.is_not(None))).all()
    vals = [float(r.open_rate) for r in rows if r.open_rate is not None]
    if len(vals) < 5:
        return []

    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / max(1, (len(vals) - 1))
    std = math.sqrt(var)
    threshold = mean + float(std_multiplier) * std

    spikes: list[EngagementSpike] = []
    for r in rows:
        if r.open_rate is None:
            continue
        if float(r.open_rate) > threshold:
            spikes.append(
                EngagementSpike(
                    post_id=int(r.id),
                    title=str(r.title),
                    published_at=r.published_at,
                    open_rate=float(r.open_rate),
                    threshold=float(threshold),
                )
            )
    return spikes


def get_top_performing_titles(
    session: Session,
    min_posts: int = 10,
) -> dict[str, float]:
    """
    Estimate which title patterns correlate with higher open rates.

    This is intentionally lightweight: for each binary pattern feature, compute
    the average open rate when present vs overall average.

    Args:
        session: SQLAlchemy session.
        min_posts: Minimum number of posts required for analysis.

    Returns:
        Mapping of pattern_name -> lift (pattern_avg - overall_avg).
    """
    posts = session.execute(select(Post.title, Post.open_rate).where(Post.open_rate.is_not(None))).all()
    if len(posts) < min_posts:
        return {}

    overall = sum(float(p.open_rate) for p in posts if p.open_rate is not None) / len(posts)
    lifts: dict[str, float] = {}

    for name, rx in TITLE_PATTERNS.items():
        subset = [float(p.open_rate) for p in posts if p.open_rate is not None and rx.search(str(p.title or ""))]
        if len(subset) < max(3, min_posts // 4):
            continue
        lifts[name] = (sum(subset) / len(subset)) - overall

    return dict(sorted(lifts.items(), key=lambda kv: kv[1], reverse=True))


def track_paid_conversion_by_topic(session: Session) -> list[tuple[str, int]]:
    """
    Track total paid conversions by topic tag.

    Since topics are stored as free-form strings, we split on commas/semicolons.

    Returns:
        List of (topic, total_paid_conversions) sorted descending.
    """
    rows = session.execute(
        select(Post.topics, Post.paid_conversions).where(Post.paid_conversions.is_not(None))
    ).all()
    totals: dict[str, int] = {}
    for r in rows:
        paid = int(r.paid_conversions or 0)
        for t in _split_topics(r.topics):
            totals[t] = totals.get(t, 0) + paid
    return sorted(totals.items(), key=lambda kv: kv[1], reverse=True)

