"""
Parse Substack CSV exports dropped into `data/exports/` and upsert into the `posts` table.

Substack export schemas vary across time and publication settings, so this parser is defensive:
- Column names are normalized (case/whitespace/punctuation)
- Rates are parsed from either percent strings ("48.2%") or floats ("0.482" / "48.2")
- Missing optional columns are tolerated

The importer is idempotent via a unique constraint on (title, published_at).
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from dateutil import parser as dtparser
from dateutil import tz
from sqlalchemy import func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from config import EXPORTS_DIR, SUBSTACK_EXPORT_OPTIONAL_COLUMNS, SUBSTACK_EXPORT_REQUIRED_COLUMNS
from src.db.models import Post

logger = logging.getLogger(__name__)


def normalize_col(name: str) -> str:
    """Normalize a CSV column name into a simple snake-ish token."""
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("%", "pct")
    )


def parse_optional_int(v: Any) -> Optional[int]:
    """Parse an optional integer from a CSV field."""
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def parse_optional_float(v: Any) -> Optional[float]:
    """Parse an optional float from a CSV field."""
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def parse_rate_to_percent(v: Any) -> Optional[float]:
    """
    Parse a rate field into a percentage in [0, 100] when possible.

    Accepts:
    - "48.2%" -> 48.2
    - "0.482" -> 48.2
    - "48.2"  -> 48.2
    """
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    try:
        if s.endswith("%"):
            return float(s[:-1].strip())
        x = float(s)
        if 0 <= x <= 1:
            return x * 100.0
        return x
    except Exception:
        return None


def parse_published_at(v: Any) -> Optional[Any]:
    """Parse a datetime string into a timezone-aware datetime (UTC if absent)."""
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    try:
        dt = dtparser.parse(s)
        if dt.tzinfo is None:
            # Treat naive timestamps as UTC.
            return dt.replace(tzinfo=tz.UTC)
        return dt.astimezone(tz.UTC)
    except Exception:
        return None


@dataclass(frozen=True)
class ImportResult:
    """Summary of a CSV import run."""

    files_seen: int
    rows_seen: int
    upserted_posts: int
    skipped_rows: int


def _iter_csv_files(exports_dir: Path) -> list[Path]:
    return sorted([p for p in exports_dir.glob("*.csv") if p.is_file()], key=lambda p: p.stat().st_mtime)


def _read_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def import_exports_folder(
    session: Session,
    exports_dir: Path = EXPORTS_DIR,
) -> ImportResult:
    """
    Import all CSV exports from a folder into the database.

    Args:
        session: SQLAlchemy session.
        exports_dir: Folder containing Substack CSV exports.

    Returns:
        ImportResult summary.
    """
    exports_dir.mkdir(parents=True, exist_ok=True)
    csv_files = _iter_csv_files(exports_dir)

    rows_seen = 0
    upserted = 0
    skipped = 0

    for csv_path in csv_files:
        logger.info("Importing Substack export: %s", csv_path.name)
        for row in _read_rows(csv_path):
            rows_seen += 1
            mapped = _map_row(row)
            if mapped is None:
                skipped += 1
                continue
            _upsert_post(session, mapped)
            upserted += 1

    return ImportResult(files_seen=len(csv_files), rows_seen=rows_seen, upserted_posts=upserted, skipped_rows=skipped)


def _map_row(row: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    Map a raw CSV row to Post fields.

    Returns None if required fields are missing/unparseable.
    """
    norm = {normalize_col(k): v for k, v in row.items() if k is not None}

    # Try a couple of common aliases.
    title = norm.get("title") or norm.get("post_title") or norm.get("subject")
    published_at = norm.get("published_at") or norm.get("publish_date") or norm.get("date")

    open_rate = norm.get("open_rate") or norm.get("open_rate_pct") or norm.get("open_rate_percent")
    click_rate = norm.get("click_rate") or norm.get("click_rate_pct") or norm.get("click_rate_percent")

    if title is None or str(title).strip() == "":
        return None
    dt = parse_published_at(published_at)
    if dt is None:
        return None

    mapped: dict[str, Any] = {
        "title": str(title).strip(),
        "published_at": dt,
        "open_rate": parse_rate_to_percent(open_rate),
        "click_rate": parse_rate_to_percent(click_rate),
    }

    # Optional fields (best-effort).
    if "new_subscribers" in norm:
        mapped["new_subscribers"] = parse_optional_int(norm.get("new_subscribers"))
    if "paid_conversions" in norm:
        mapped["paid_conversions"] = parse_optional_int(norm.get("paid_conversions"))
    if "topics" in norm:
        topics = norm.get("topics")
        mapped["topics"] = str(topics).strip() if topics is not None and str(topics).strip() != "" else None
    if "word_count" in norm:
        mapped["word_count"] = parse_optional_int(norm.get("word_count"))

    # If the export has unexpected column names, we still keep going; required columns were validated above.
    missing_required = [c for c in SUBSTACK_EXPORT_REQUIRED_COLUMNS if c not in {"title", "published_at", "open_rate", "click_rate"}]
    _ = missing_required  # reserved for future strict-mode validation
    _ = SUBSTACK_EXPORT_OPTIONAL_COLUMNS

    return mapped


def _upsert_post(session: Session, values: dict[str, Any]) -> None:
    """
    Upsert a Post using SQLite's ON CONFLICT handling.

    Conflict key: (title, published_at)
    """
    stmt = sqlite_insert(Post).values(**values)
    update_cols = {
        "open_rate": stmt.excluded.open_rate,
        "click_rate": stmt.excluded.click_rate,
        "new_subscribers": stmt.excluded.new_subscribers,
        "paid_conversions": stmt.excluded.paid_conversions,
        "topics": stmt.excluded.topics,
        "word_count": stmt.excluded.word_count,
    }
    stmt = stmt.on_conflict_do_update(
        index_elements=["title", "published_at"],
        set_=update_cols,
    )
    session.execute(stmt)


def count_posts(session: Session) -> int:
    """
    Count total posts in the database.

    Args:
        session: SQLAlchemy session.

    Returns:
        Number of posts.
    """
    return int(session.scalar(select(func.count(Post.id))) or 0)

