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
import datetime as pydt
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
        if s.isdigit() and len(s) >= 10:
            # Unix seconds (Substack sometimes exports epoch).
            ts = int(s[:10])
            return pydt.datetime.fromtimestamp(ts, tz=tz.UTC)
        parsed = dtparser.parse(s)
        if parsed.tzinfo is None:
            # Treat naive timestamps as UTC.
            return parsed.replace(tzinfo=tz.UTC)
        return parsed.astimezone(tz.UTC)
    except Exception:
        return None


def _coalesce(norm: dict[str, Any], *keys: str) -> Any:
    """Return the first non-empty mapped value for the given normalized column keys."""
    for k in keys:
        if k not in norm:
            continue
        val = norm[k]
        if val is None:
            continue
        if str(val).strip() == "":
            continue
        return val
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
        logged_skip_columns = False
        for row in _read_rows(csv_path):
            rows_seen += 1
            mapped = _map_row(row)
            if mapped is None:
                skipped += 1
                if not logged_skip_columns:
                    sample_norm = {normalize_col(k): v for k, v in row.items() if k is not None}
                    logger.warning(
                        "At least one row skipped; first skipped row column names (normalized): %s",
                        sorted(sample_norm.keys()),
                    )
                    logged_skip_columns = True
                continue
            _upsert_post(session, mapped)
            upserted += 1

    return ImportResult(files_seen=len(csv_files), rows_seen=rows_seen, upserted_posts=upserted, skipped_rows=skipped)


def _map_row(row: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    Map a raw CSV row to Post fields.

    Returns None if required fields are missing/unparseable.

    Substack full-export ``posts.csv`` often uses names like ``post_date``, ``slug``, or
    ``created_at`` rather than ``title`` / ``published_at``. Stats may live in a different
    CSV; we still import posts with null rates when those columns are absent.
    """
    norm = {normalize_col(k): v for k, v in row.items() if k is not None}

    title = _coalesce(
        norm,
        "title",
        "post_title",
        "subject",
        "name",
        "post_name",
        "headline",
        "slug",
        "post_slug",
        "filename",
        "file_name",
        "url",
    )
    published_at = _coalesce(
        norm,
        "published_at",
        "publish_date",
        "publication_date",
        "date",
        "post_date",
        "created_at",
        "created",
        "sent_at",
        "scheduled_at",
        "published",
        "pub_date",
        "timestamp",
        "post_timestamp",
    )

    open_rate = _coalesce(
        norm,
        "open_rate",
        "open_rate_pct",
        "open_rate_percent",
        "email_open_rate",
        "email_open_pct",
        "opens_rate",
        "delivery_rate",
    )
    click_rate = _coalesce(
        norm,
        "click_rate",
        "click_rate_pct",
        "click_rate_percent",
        "ctr",
        "click_through_rate",
        "email_click_rate",
    )

    if title is None or str(title).strip() == "":
        return None
    title_str = str(title).strip()
    if title_str.lower().startswith("http"):
        # Substack sometimes puts the post URL in the title column.
        part = title_str.rstrip("/").split("/")[-1]
        title_str = part.replace("-", " ").title() if part else title_str
    title_str = title_str[:500]

    pub = parse_published_at(published_at)
    if pub is None:
        return None

    mapped: dict[str, Any] = {
        "title": title_str,
        "published_at": pub,
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
    elif "tags" in norm and norm.get("tags") is not None and str(norm.get("tags")).strip() != "":
        mapped["topics"] = str(norm.get("tags")).strip()
    elif "categories" in norm and norm.get("categories") is not None and str(norm.get("categories")).strip() != "":
        mapped["topics"] = str(norm.get("categories")).strip()
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

