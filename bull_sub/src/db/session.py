"""
Engine/session helpers for the local SQLite database.

The app is intentionally migration-free to keep the stack lean. We use
`Base.metadata.create_all()` at startup.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from config import DB_PATH
from src.db.base import Base


@dataclass(frozen=True)
class DbConfig:
    """Configuration for creating the DB engine."""

    path: Path = DB_PATH


def get_engine(cfg: DbConfig | None = None) -> Engine:
    """
    Create a SQLAlchemy engine for SQLite.

    Args:
        cfg: Optional database configuration (path override).

    Returns:
        SQLAlchemy Engine instance.
    """
    cfg = cfg or DbConfig()
    cfg.path.parent.mkdir(parents=True, exist_ok=True)
    url = f"sqlite:///{cfg.path.as_posix()}"
    # `check_same_thread=False` allows Streamlit (multi-threaded) access.
    return create_engine(url, future=True, connect_args={"check_same_thread": False})


def init_db(engine: Engine | None = None) -> Engine:
    """
    Initialize the SQLite database by creating missing tables.

    Args:
        engine: Optional engine. If omitted, a new engine is created.

    Returns:
        The engine used for initialization.
    """
    engine = engine or get_engine()
    # Import models so they register with Base.metadata before create_all().
    import src.db.models as _models  # noqa: F401

    Base.metadata.create_all(engine)
    return engine


def get_session(engine: Engine | None = None) -> Session:
    """
    Create a new SQLAlchemy Session.

    Args:
        engine: Optional engine. If omitted, a new engine is created.

    Returns:
        A SQLAlchemy Session.
    """
    engine = engine or get_engine()
    maker = sessionmaker(bind=engine, class_=Session, autoflush=False, autocommit=False, future=True)
    return maker()


@contextmanager
def session_scope(engine: Engine | None = None) -> Iterator[Session]:
    """
    Context manager-style generator for a transactional session.

    Usage:
        with session_scope() as s:
            ...
    """
    s = get_session(engine=engine)
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()

