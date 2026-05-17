#!/usr/bin/env sh
# Container entrypoint: run DB migrations, then start the API.
# In MOCK_MODE there are no migrations yet — this is a no-op placeholder
# kept so the contract (migrate -> serve) is visible. See architecture.md §3.10.
set -e

# When persistence lands, uncomment:
# uv run alembic upgrade head

exec uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
