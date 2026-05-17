"""Redis cache wrapper — ``get_or_set_json(key, fetch, ttl)``.

PLACEHOLDER (docs/architecture.md §3.9). Pattern: cache only GET-list /
status reads, invalidate on any write. Transparent passthrough when
``CACHE_ENABLED=false``. Not wired in the MOCK_MODE skeleton.
"""

# import json
# import redis.asyncio as redis
# from app.core.config import settings
#
# class RedisCache:
#     async def get_or_set_json(self, key, fetch, ttl): ...
#
# def get_redis_client() -> "RedisCache": ...
