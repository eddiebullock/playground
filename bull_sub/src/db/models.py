"""
ORM models for the Grey Matters content engine.

Schema mirrors the user's spec and stays deliberately simple (SQLite-first).
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base


def utcnow() -> dt.datetime:
    """Return timezone-aware UTC now."""
    return dt.datetime.now(dt.timezone.utc)


class Post(Base):
    """Imported Substack post performance record."""

    __tablename__ = "posts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    published_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    open_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    click_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    new_subscribers: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    paid_conversions: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    topics: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # comma-separated or raw string
    word_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    best_title_variant: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    __table_args__ = (
        UniqueConstraint("title", "published_at", name="uq_posts_title_published_at"),
        Index("idx_posts_published_at", "published_at"),
        Index("idx_posts_title", "title"),
    )


class TopicScore(Base):
    """Topic opportunity score at a point in time."""

    __tablename__ = "topic_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    topic: Mapped[str] = mapped_column(String(200), nullable=False, index=True)

    trend_score: Mapped[float] = mapped_column(Float, nullable=False)
    historical_performance_score: Mapped[float] = mapped_column(Float, nullable=False)
    combined_score: Mapped[float] = mapped_column(Float, nullable=False, index=True)

    fetched_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class Draft(Base):
    """Generated draft (article + title variants) awaiting approval."""

    __tablename__ = "drafts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    title: Mapped[str] = mapped_column(String(500), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    topic: Mapped[str] = mapped_column(String(200), nullable=False, index=True)

    title_variants: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    combined_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending", index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    notes: Mapped[list["Note"]] = relationship(back_populates="draft", cascade="all, delete-orphan")
    threads: Mapped[list["Thread"]] = relationship(back_populates="draft", cascade="all, delete-orphan")

    __table_args__ = (Index("idx_drafts_created_at", "created_at"),)


class Note(Base):
    """Substack Notes generated from an approved draft (manual copy/paste)."""

    __tablename__ = "notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    draft_id: Mapped[int] = mapped_column(ForeignKey("drafts.id", ondelete="CASCADE"), nullable=False, index=True)

    note_1: Mapped[str] = mapped_column(Text, nullable=False)
    note_2: Mapped[str] = mapped_column(Text, nullable=False)
    note_3: Mapped[str] = mapped_column(Text, nullable=False)

    scheduled_day_1: Mapped[Optional[dt.date]] = mapped_column(nullable=True)
    scheduled_day_2: Mapped[Optional[dt.date]] = mapped_column(nullable=True)
    scheduled_day_3: Mapped[Optional[dt.date]] = mapped_column(nullable=True)

    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending", index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    draft: Mapped[Draft] = relationship(back_populates="notes")


class Thread(Base):
    """Social thread (Bluesky/Twitter) generated from a draft."""

    __tablename__ = "threads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    draft_id: Mapped[int] = mapped_column(ForeignKey("drafts.id", ondelete="CASCADE"), nullable=False, index=True)

    platform: Mapped[str] = mapped_column(String(20), nullable=False, index=True)  # bluesky/twitter
    thread_content: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)

    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending", index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    draft: Mapped[Draft] = relationship(back_populates="threads")

    __table_args__ = (Index("idx_threads_platform_status", "platform", "status"),)


class Alert(Base):
    """Actionable alert (engagement spike or paid CTA opportunity)."""

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    post_id: Mapped[Optional[int]] = mapped_column(ForeignKey("posts.id", ondelete="SET NULL"), nullable=True, index=True)

    alert_type: Mapped[str] = mapped_column(String(30), nullable=False, index=True)  # engagement_spike/paid_cta
    cta_draft: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    triggered_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    actioned: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)

