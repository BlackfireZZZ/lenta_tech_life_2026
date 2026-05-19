"""Redis cache wrapper — ``get_or_set_json(key, fetch, ttl)``.

docs/architecture.md §3.9. Two hard safety rules here, because a cache is
the easiest place to introduce a silent correctness bug:

1. **Fail-open.** Any Redis error (down, timeout, bad payload) ⇒ call
   ``fetch`` and return its value as if the cache were absent. Redis being
   unavailable must never fail a request — it is a latency optimisation,
   not a source of truth. ``CACHE_ENABLED=false`` ⇒ transparent passthrough.
2. **Only the caller decides what is cacheable.** This module never caches
   on its own. The single caller (``GET /jobs/{id}``) caches **only
   terminal** jobs (succeeded/failed) — a *running* job's status/progress
   changes on every poll, so caching it would freeze the progress bar. A
   terminal job is immutable, so a short TTL needs no invalidation.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from app.core.config import settings
from app.core.logger import logger

try:  # redis is a hard dep; guard import so a packaging slip can't 500 the app
    import redis.asyncio as aioredis
except Exception:  # pragma: no cover
    aioredis = None  # type: ignore[assignment]


class RedisCache:
    def __init__(self) -> None:
        self._client = None
        if settings.CACHE_ENABLED and aioredis is not None:
            try:
                self._client = aioredis.from_url(
                    settings.REDIS_URL,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                )
            except Exception as exc:  # pragma: no cover - fail-open
                logger.warning("Redis init failed; cache disabled: %s", exc)
                self._client = None

    async def get_json(self, key: str) -> Any | None:
        """Cached value for ``key``, or ``None`` on miss/any error."""
        if self._client is None:
            return None
        try:
            hit = await self._client.get(key)
            return None if hit is None else json.loads(hit)
        except Exception as exc:  # pragma: no cover - fail-open
            logger.debug("cache get failed (%s): %s", key, exc)
            return None

    async def set_json(self, key: str, value: Any, ttl: int) -> None:
        """Store ``value`` (JSON-serialisable) under ``key``. Never raises."""
        if self._client is None:
            return
        try:
            await self._client.set(key, json.dumps(value), ex=ttl)
        except Exception as exc:  # pragma: no cover - fail-open
            logger.debug("cache set failed (%s): %s", key, exc)

    async def get_or_set_json(
        self, key: str, fetch: Callable[[], Awaitable[Any]], ttl: int
    ) -> Any:
        """Documented convenience (architecture.md §3.9): cached value, else
        compute via ``fetch`` and store. Fail-open throughout. Callers that
        must gate *what* is cacheable (e.g. terminal-only job reads) use
        :meth:`get_json` / :meth:`set_json` directly instead.
        """
        cached = await self.get_json(key)
        if cached is not None:
            return cached
        value = await fetch()
        await self.set_json(key, value, ttl)
        return value


_cache: RedisCache | None = None


def get_redis_client() -> RedisCache:
    """Process-wide cache singleton."""
    global _cache
    if _cache is None:
        _cache = RedisCache()
    return _cache
