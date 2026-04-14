"""Basic unit tests (no network, no API keys)."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.analytics.csv_parser import _map_row, normalize_col, parse_rate_to_percent
from src.db import models as _models  # noqa: F401
from src.db.base import Base
from src.trends.topic_scorer import combine_scores, filter_recently_covered


def _memory_session() -> Session:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, class_=Session, future=True)()


def test_normalize_col() -> None:
    assert normalize_col(" Open Rate % ") == "open_rate_pct"


def test_parse_rate_to_percent() -> None:
    assert parse_rate_to_percent("48.2%") == 48.2
    assert parse_rate_to_percent("0.48") == 48.0
    assert parse_rate_to_percent(None) is None


def test_combine_scores_weights() -> None:
    assert abs(combine_scores(100.0, 0.0) - 40.0) < 1e-6
    assert abs(combine_scores(0.0, 100.0) - 60.0) < 1e-6


def test_map_row_substack_style_slug_and_post_date() -> None:
    row = {
        "slug": "brain-plasticity-notes",
        "post_date": "2024-06-01 12:00:00",
        "email_open_rate": "42%",
        "ctr": "5%",
    }
    m = _map_row(row)
    assert m is not None
    assert m["title"] == "brain-plasticity-notes"
    assert m["open_rate"] == 42.0
    assert m["click_rate"] == 5.0


def test_filter_recently_covered_empty_db() -> None:
    s = _memory_session()
    try:
        out = filter_recently_covered(s, ["neuroscience", "anxiety"], days=30)
        assert out == ["neuroscience", "anxiety"]
    finally:
        s.close()
