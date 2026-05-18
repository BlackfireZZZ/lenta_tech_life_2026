"""FastAPI application assembly for the price-tag gateway.

This is the single public entry point. The frontend talks ONLY to this
service; this service is the only one that talks to the ML service
(``app/ml/client.py``). See ``docs/architecture.md`` §3.

Status: MOCKED skeleton. Endpoints return fake data so the structure and
contracts are reviewable end-to-end before any real logic is written.
DB / cache / auth wiring is documented and stubbed (see the placeholder
modules under ``app/db``, ``app/cache``, ``app/core``).
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.routes import health, jobs
from app.core.config import settings
from app.core.logger import logger


async def _init_db_with_retry(attempts: int = 10, delay: float = 2.0) -> None:
    """Bring the schema up, tolerating a Postgres that is a beat behind.

    docker-compose `depends_on: db: service_healthy` usually means the DB is
    ready, but pool warm-up can still race on a cold `up`; retry a few times
    before giving up loudly.
    """
    from app.db.session import init_models, ping

    last: Exception | None = None
    for i in range(1, attempts + 1):
        try:
            await ping()
            await init_models()
            return
        except Exception as exc:  # noqa: BLE001 - want the loud final raise
            last = exc
            logger.warning("DB not ready (attempt %d/%d): %s", i, attempts, exc)
            await asyncio.sleep(delay)
    raise RuntimeError(f"Postgres unreachable after {attempts} attempts") from last


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "%s v%s starting (MOCK_MODE=%s)",
        settings.APP_NAME, settings.APP_VERSION, settings.MOCK_MODE,
    )
    # MOCK_MODE keeps the gateway standalone (no Postgres). The real product
    # path needs the schema before the first request.
    if not settings.MOCK_MODE:
        await _init_db_with_retry()
    yield
    logger.info("shutting down")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API is always versioned: /api/v1/...
# Auth router (documented in docs/architecture.md §3.8) will mount here
# WITHOUT the /api/v1 prefix and without the CSRF dependency. Stubbed for now.
app.include_router(health.router, prefix="/api/v1")
app.include_router(jobs.router, prefix="/api/v1")
