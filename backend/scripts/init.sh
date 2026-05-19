#!/usr/bin/env sh
# Container entrypoint: start the API. The schema is brought up in the app
# lifespan via Base.metadata.create_all (single `jobs` table — no migration
# state to manage; idempotent on every boot). This deliberately replaces the
# Alembic migrate→serve step for hackathon reliability; the deviation is
# documented in docs/architecture.md §3.3/§3.10.
set -e

# Run with the system interpreter. Deps are already installed image-wide via
# `uv pip install --system` (see backend/Dockerfile). `uv run` would instead
# create a fresh project .venv and re-resolve every dependency from PyPI on
# every container start — slow and needs network. `python -m uvicorn` uses
# the baked-in packages: instant start, no network.
exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
