# Gateway API image. Small and CPU-only — heavy ML lives in ../ml.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN pip install --no-cache-dir uv

# Dependency layer first for cache.
COPY pyproject.toml ./
RUN uv pip install --system --no-cache \
        fastapi "uvicorn[standard]" pydantic pydantic-settings \
        python-multipart httpx

COPY app ./app
COPY scripts ./scripts
RUN chmod +x scripts/init.sh

EXPOSE 8000
ENTRYPOINT ["scripts/init.sh"]
