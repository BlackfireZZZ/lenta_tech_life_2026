"""Async engine + ``get_db_session`` dependency. See docs/architecture.md §3.3.

PLACEHOLDER — the skeleton runs MOCK_MODE without a database. The real
async-SQLAlchemy engine/sessionmaker goes here exactly as in the
architecture doc; routes then ``Depends(get_db_session)``.
"""

# from sqlalchemy.ext.asyncio import (
#     AsyncSession, async_sessionmaker, create_async_engine,
# )
# from app.core.config import settings
#
# engine = create_async_engine(settings.DATABASE_URL, pool_pre_ping=True)
# AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession,
#                                        expire_on_commit=False)
#
# async def get_db_session() -> AsyncSession:
#     async with AsyncSessionLocal() as session:
#         yield session
