"""Liveness probe. Real impl also checks Postgres + Redis + ML reachability."""

from fastapi import APIRouter

from app.core.config import settings

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("")
async def health() -> dict:
    return {
        "status": "ok",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "mock_mode": settings.MOCK_MODE,
    }
