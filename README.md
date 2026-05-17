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
| Model & pipeline strategy | [docs/strategy.md](./docs/strategy.md) |
| CLI / profiles / backends / output | [docs/pipeline-reference.md](./docs/pipeline-reference.md) |
| Runbooks (local + Colab) | [docs/runbooks/](./docs/runbooks/local.md) |
| Data layout & external datasets | [docs/data/layout.md](./docs/data/layout.md) · [docs/data/datasets.md](./docs/data/datasets.md) |
| Branch map | [docs/branches.md](./docs/branches.md) |
| Pre-rewrite analysis (historical) | [docs/analysis.md](./docs/analysis.md) |

## Quick start

```bash
# Install (Python 3.11/3.12; use a uv-managed .venv, not global pip).
pip install -r projects/price_tag_pipeline/requirements.txt

# Tests need no data, model, or GPU.
pytest projects/price_tag_pipeline/tests -v
```

Full setup, data staging, training, inference, CSV export and the UI are in
[`docs/pipeline-reference.md`](./docs/pipeline-reference.md) and the
[runbooks](./docs/runbooks/local.md). `main` is the canonical branch.
