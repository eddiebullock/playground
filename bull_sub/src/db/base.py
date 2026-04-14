"""
SQLAlchemy base + common helpers.

We keep base definitions in a separate module to avoid import cycles.
"""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""

