"""Async SQLAlchemy engine + ``get_db_session`` dependency.

docs/architecture.md §3.3 names Alembic autogenerate as the target. For the
hackathon we deliberately use ``Base.metadata.create_all`` on startup
instead (``init_models``): the schema is a single ``jobs`` table, so there
is nothing to migrate, and ``create_all`` is idempotent and has zero
migration-state failure surface in a one-command ``docker compose up`` that
the user brings up themselves. This deviation is documented in
architecture.md §3.3/§3.10 so the doc does not lie about the code.

Background ML processing runs outside the request scope, so it opens its own
session via :func:`session_scope` (the request-scoped ``get_db_session``
dependency would already be closed by the time the task runs).
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings
from app.core.logger import logger
from app.db.base import Base

# `pool_pre_ping` makes the pool resilient to Postgres restarts (the db
# container may come up a beat after the backend even with depends_on).
engine = create_async_engine(
    settings.DATABASE_URL, pool_pre_ping=True, future=True
)
AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: one session per request, always closed."""
    async with AsyncSessionLocal() as session:
        yield session


@contextlib.asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Standalone session for out-of-request work (the background ML task).

    Commits on success, rolls back on error, always closes.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_models() -> None:
    """Create tables if absent (see module docstring re: create_all vs Alembic).

    Importing ``app.db.models`` registers every table on ``Base.metadata``.
    """
    import app.db.models  # noqa: F401  (populate Base.metadata)

    from sqlalchemy import text

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # create_all only creates *missing tables*, never new columns on an
        # existing one. We deliberately use create_all over Alembic (one
        # table — see module docstring); additive columns are applied here
        # with idempotent, Postgres-native `ADD COLUMN IF NOT EXISTS`. Keep
        # this list append-only; each entry is safe to run every boot.
        for ddl in (
            "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS "
            "rotation VARCHAR(8) NOT NULL DEFAULT 'none'",
            "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS "
            "content_hash VARCHAR(64)",
            "CREATE INDEX IF NOT EXISTS ix_jobs_content_hash "
            "ON jobs (content_hash)",
            "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS phase VARCHAR(16)",
            "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS "
            "mode VARCHAR(8) NOT NULL DEFAULT 'full'",
            "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS "
            "detections_json TEXT",
        ):
            await conn.execute(text(ddl))
    logger.info("DB schema ready (create_all + additive DDL on %s)", settings.DB_HOST)


async def ping() -> None:
    """Raise if Postgres is unreachable — called once at startup."""
    from sqlalchemy import text

    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
