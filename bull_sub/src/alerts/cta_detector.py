"""
Detect engagement spikes and high-converting topics; create actionable alerts.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from config import ENGAGEMENT_SPIKE_STD_MULTIPLIER
from src.analytics.performance_tracker import (
    detect_engagement_spike,
    track_paid_conversion_by_topic,
)
from src.db.models import Alert

logger = logging.getLogger(__name__)


def _spike_cta_email(title: str, open_rate: float) -> str:
    """Draft a soft paid-tier CTA email referencing a high-performing post."""
    return (
        f"If you liked \"{title}\" and want more of that depth, paid subscribers get extra essays, "
        f"behind-the-scenes from the lab, and threads I do not put in the free edition. "
        f"(This one landed around {open_rate:.1f}% opens — thank you for reading.) "
        f"If you have been on the fence, this is a good week to upgrade."
    )


def _alert_exists_for_post(session: Session, post_id: int, alert_type: str) -> bool:
    """Return True if an un-actioned alert already exists for this post and type."""
    q = select(Alert.id).where(
        and_(
            Alert.post_id == post_id,
            Alert.alert_type == alert_type,
            Alert.actioned.is_(False),
        )
    ).limit(1)
    return session.scalar(q) is not None


def create_engagement_spike_alerts(session: Session) -> int:
    """
    For posts above the open-rate spike threshold, insert ``engagement_spike`` alerts.

    Args:
        session: DB session.

    Returns:
        Number of new alerts created.
    """
    spikes = detect_engagement_spike(session, std_multiplier=ENGAGEMENT_SPIKE_STD_MULTIPLIER)
    n = 0
    for s in spikes:
        if _alert_exists_for_post(session, s.post_id, "engagement_spike"):
            continue
        session.add(
            Alert(
                post_id=s.post_id,
                alert_type="engagement_spike",
                cta_draft=_spike_cta_email(s.title, s.open_rate),
            )
        )
        n += 1
    if n:
        logger.info("Created %d engagement spike alerts", n)
    return n


def create_paid_topic_recommendation_alerts(session: Session, min_total_conversions: int = 2) -> int:
    """
    Flag topics that accumulate paid conversions (write more on what converts).

    Args:
        session: DB session.
        min_total_conversions: Minimum summed conversions to trigger.

    Returns:
        Number of new alerts created.
    """
    ranked = track_paid_conversion_by_topic(session)
    n = 0
    for topic, total in ranked[:5]:
        if total < min_total_conversions:
            continue
        snippet = (
            f"Topic signal: \"{topic}\" has driven {total} paid conversion(s) in imported data. "
            f"Consider another deep dive while interest is warm."
        )
        if session.scalar(
            select(Alert.id)
            .where(
                and_(
                    Alert.alert_type == "paid_cta",
                    Alert.cta_draft.is_not(None),
                    Alert.cta_draft.like(f"%{topic}%"),
                    Alert.actioned.is_(False),
                )
            )
            .limit(1)
        ):
            continue
        session.add(Alert(post_id=None, alert_type="paid_cta", cta_draft=snippet))
        n += 1
    if n:
        logger.info("Created %d paid-topic recommendation alerts", n)
    return n


def run_cta_detection(session: Session) -> dict[str, int]:
    """
    Run all CTA / spike detectors after a CSV import or pipeline step.

    Args:
        session: DB session.

    Returns:
        Counts per detector.
    """
    return {
        "engagement_spike": create_engagement_spike_alerts(session),
        "paid_topic": create_paid_topic_recommendation_alerts(session),
    }


def build_historical_context_for_topic(session: Session, topic: str) -> str:
    """
    Summarize historical performance hints for prompt context.

    Args:
        session: DB session.
        topic: Topic being written.

    Returns:
        Multi-line string for the article generator.
    """
    from src.analytics.performance_tracker import calculate_topic_performance, get_top_performing_titles

    lines: list[str] = []
    avg = calculate_topic_performance(session, topic)
    if avg is not None:
        lines.append(f"Average open rate for posts tagged like this topic: ~{avg:.1f}% (from your imports).")
    lifts = get_top_performing_titles(session, min_posts=5)
    if lifts:
        top = ", ".join(f"{k} (+{v:.1f} pts vs avg)" for k, v in list(lifts.items())[:3])
        lines.append(f"Title patterns that have correlated with stronger opens: {top}.")
    if not lines:
        return "No historical import data yet; optimize for clarity and curiosity."
    return "\n".join(lines)
