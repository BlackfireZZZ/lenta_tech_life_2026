"""FastAPI application assembly for the price-tag gateway.

This is the single public entry point. The frontend talks ONLY to this
service; this service is the only one that talks to the ML service
(``app/ml/client.py``). See ``docs/architecture.md`` §3.

Status: MOCKED skeleton. Endpoints return fake data so the structure and
contracts are reviewable end-to-end before any real logic is written.
DB / cache / auth wiring is documented and stubbed (see the placeholder
modules under ``app/db``, ``app/cache``, ``app/core``).
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.routes import health, jobs
from app.core.config import settings
from app.core.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Real impl: ping Postgres + Redis here (see docs/architecture.md §3.3).
    logger.info("%s v%s starting (MOCK mode=%s)", settings.APP_NAME, settings.APP_VERSION, settings.MOCK_MODE)
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
