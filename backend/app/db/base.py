"""Declarative Base for ORM models. See docs/architecture.md §3.3.

PLACEHOLDER — not wired in the skeleton (MOCK_MODE jobs live in memory).
Implement when persistence is added; Alembic autogenerate keys off
``Base.metadata``.
"""

from sqlalchemy.orm import declarative_base

Base = declarative_base()
