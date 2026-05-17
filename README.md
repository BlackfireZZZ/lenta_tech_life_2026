# lenta_tech_life_2026

End-to-end price-tag recognition for the **Lenta Tech Life 2026** hackathon:
a robot drives along Russian supermarket shelves; from its video we detect
price tags and extract structured fields per tag, emitting one CSV row per
unique tag behind an upload-video → download-CSV UI.

## Documentation

All knowledge lives in **[`docs/`](./docs/index.md)** — start at
**[`docs/index.md`](./docs/index.md)**. AI agents: see
[`AGENTS.md`](./AGENTS.md).

| | |
|---|---|
| Official task, CSV schema, metric | [docs/hackathon/task.md](./docs/hackathon/task.md) |
| Organizer-chat intel & gotchas | [docs/hackathon/briefing.md](./docs/hackathon/briefing.md) |
| App/service architecture (backend·frontend·ML) | [docs/architecture.md](./docs/architecture.md) |
| Model & pipeline strategy | [docs/strategy.md](./docs/strategy.md) |
| CLI / profiles / backends / output | [docs/pipeline-reference.md](./docs/pipeline-reference.md) |
| Runbooks (local + Colab) | [docs/runbooks/](./docs/runbooks/local.md) |
| Data layout & external datasets | [docs/data/layout.md](./docs/data/layout.md) · [docs/data/datasets.md](./docs/data/datasets.md) |
| Branch map | [docs/branches.md](./docs/branches.md) |
| Pre-rewrite analysis (historical) | [docs/analysis.md](./docs/analysis.md) |

## Layout

Monorepo: the **model** (training/experiments/research) lives in
`projects/price_tag_pipeline/`; the **product** around it is
`backend/` (API gateway) · `frontend/` (SPA) · `ml/` (deployable service
wrapping the pipeline) + `docker-compose.yaml`. The service stack is
currently a **reviewable skeleton** — structure and contracts in place,
business logic intentionally not written yet. See
[`docs/architecture.md`](./docs/architecture.md).

## Quick start

```bash
# Model — tests need no data, model, or GPU (use a uv-managed .venv).
pip install -r projects/price_tag_pipeline/requirements.txt
pytest projects/price_tag_pipeline/tests -v

# Whole product skeleton (mocked: no GPU/DB needed).
docker compose up --build
# frontend :5173 · backend :8000 (/docs) · ml :8002
```

Model setup, training, inference, CSV export:
[`docs/pipeline-reference.md`](./docs/pipeline-reference.md) and the
[runbooks](./docs/runbooks/local.md). Service architecture & build order:
[`docs/architecture.md`](./docs/architecture.md). `main` is the canonical
branch.
