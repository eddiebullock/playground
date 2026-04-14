"""
Google Trends integration via `pytrends` (no API key).

Fetches relative interest (0-100) for niche keyword groups, caches results for
24 hours to reduce rate limiting, and falls back to stale cache on errors.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from config import (
    GOOGLE_TRENDS_CACHE_PATH,
    GOOGLE_TRENDS_CACHE_TTL_HOURS,
    GOOGLE_TRENDS_KEYWORD_GROUPS,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrendTopic:
    """A single keyword with its Google Trends relative score."""

    topic: str
    trend_score: float
    source: str = "google_trends"


def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


def _cache_is_fresh(fetched_at: datetime, ttl_hours: float) -> bool:
    """Return True if cache timestamp is within TTL."""
    age = _utc_now() - fetched_at
    return age < timedelta(hours=ttl_hours)


def _read_cache(path: Path) -> Optional[dict[str, Any]]:
    """Load cache JSON from disk if present."""
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not read trends cache %s: %s", path, e)
        return None


def _write_cache(path: Path, payload: dict[str, Any]) -> None:
    """Write cache JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp.replace(path)
    except Exception as e:
        logger.warning("Could not write trends cache %s: %s", path, e)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def _parse_cache_payload(raw: dict[str, Any]) -> tuple[datetime, list[TrendTopic]]:
    """Parse cached JSON into (fetched_at, topics)."""
    iso = raw.get("fetched_at_utc")
    if not iso:
        raise ValueError("cache missing fetched_at_utc")
    fetched_at = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)
    items = raw.get("topics") or []
    topics: list[TrendTopic] = []
    for it in items:
        topics.append(
            TrendTopic(
                topic=str(it["topic"]),
                trend_score=float(it["trend_score"]),
                source=str(it.get("source") or "google_trends"),
            )
        )
    return fetched_at, topics


def _fallback_topics_from_config(keyword_groups: list[list[str]]) -> list[TrendTopic]:
    """
    When Trends cannot be fetched, use configured keywords with a neutral score.

    Args:
        keyword_groups: Same structure as ``GOOGLE_TRENDS_KEYWORD_GROUPS``.

    Returns:
        De-duplicated ``TrendTopic`` rows (score 50, source ``config_fallback``).
    """
    seen: set[str] = set()
    out: list[TrendTopic] = []
    for group in keyword_groups:
        for raw in group:
            k = str(raw).strip()
            if not k:
                continue
            key = k.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(TrendTopic(topic=k, trend_score=50.0, source="config_fallback"))
    return out


def _fetch_via_pytrends(keyword_groups: list[list[str]]) -> list[TrendTopic]:
    """
    Query pytrends for each keyword group and return per-keyword scores.

    Google allows up to 5 keywords per comparison batch; each inner list should
    stay within that limit (our config uses groups of 3).

    Args:
        keyword_groups: Lists of keywords to query together.

    Returns:
        TrendTopic rows (one per keyword).

    Raises:
        Exception: Propagates pytrends/network errors to the caller.
    """
    try:
        from pytrends.request import TrendReq
    except ImportError as e:
        raise RuntimeError("pytrends is not installed") from e

    pytrends = TrendReq(hl="en-GB", tz=0)
    out: list[TrendTopic] = []
    timeframe = "today 3-m"

    for group in keyword_groups:
        keywords = [k.strip() for k in group if k and str(k).strip()]
        if not keywords:
            continue
        if len(keywords) > 5:
            logger.warning(
                "Truncating keyword group to 5 terms (pytrends limit): %s",
                keywords,
            )
            keywords = keywords[:5]

        pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo="", gprop="")
        df = pytrends.interest_over_time()
        if df is None or df.empty:
            for kw in keywords:
                out.append(TrendTopic(topic=kw, trend_score=0.0))
            continue

        # Mean of last 7 available rows (relative0-100).
        tail = df[keywords].tail(7)
        for kw in keywords:
            if kw not in tail.columns:
                out.append(TrendTopic(topic=kw, trend_score=0.0))
                continue
            series = tail[kw].astype(float)
            score = float(series.mean()) if len(series) else 0.0
            score = max(0.0, min(100.0, score))
            out.append(TrendTopic(topic=kw, trend_score=score))

    return out


def get_trending_topics(
    keyword_groups: list[list[str]] | None = None,
    cache_path: Path = GOOGLE_TRENDS_CACHE_PATH,
    ttl_hours: float = GOOGLE_TRENDS_CACHE_TTL_HOURS,
    force_refresh: bool = False,
) -> tuple[list[TrendTopic], bool]:
    """
    Return trending topics with scores 0-100, using cache when fresh.

    Args:
        keyword_groups: Keyword batches; defaults to config niche groups.
        cache_path: JSON cache file path.
        ttl_hours: Cache time-to-live in hours.
        force_refresh: If True, skip freshness check and hit the network.

    Returns:
        (topics, used_network) where used_network is True if pytrends ran.
    """
    groups = keyword_groups or GOOGLE_TRENDS_KEYWORD_GROUPS
    raw = _read_cache(cache_path)

    if not force_refresh and raw:
        try:
            fetched_at, topics = _parse_cache_payload(raw)
            if _cache_is_fresh(fetched_at, ttl_hours) and topics:
                logger.info(
                    "Using fresh Google Trends cache from %s (%d topics)",
                    fetched_at.isoformat(),
                    len(topics),
                )
                return topics, False
        except Exception as e:
            logger.warning("Trends cache parse failed, will refresh: %s", e)

    try:
        topics = _fetch_via_pytrends(groups)
        payload = {
            "fetched_at_utc": _utc_now().isoformat(),
            "topics": [
                {"topic": t.topic, "trend_score": t.trend_score, "source": t.source}
                for t in topics
            ],
        }
        _write_cache(cache_path, payload)
        logger.info("Fetched Google Trends for %d keywords", len(topics))
        return topics, True
    except Exception as e:
        logger.warning(
            "Google Trends fetch failed (%s). Using cached data if available.",
            e,
        )
        if raw:
            try:
                _, topics = _parse_cache_payload(raw)
                if topics:
                    logger.info("Using stale Google Trends cache (%d topics)", len(topics))
                    return topics, False
            except Exception as e2:
                logger.warning("Stale cache unusable: %s", e2)
        fb = _fallback_topics_from_config(groups)
        logger.warning(
            "Using config keyword fallback (%d topics); trend scores are neutral (50).",
            len(fb),
        )
        return fb, False
