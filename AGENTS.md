# AGENTS.md — orientation for AI agents

Canonical entry point for any AI agent (Claude Code, other tools) working in
this repo. Tool-agnostic. `CLAUDE.md` is a thin pointer to this file.

## What this project is

End-to-end **price-tag recognition** for the *Lenta Tech Life 2026* hackathon.
A robot drives along Russian supermarket shelves; from its video we detect
price tags and extract structured fields per tag, emit **one CSV row per
unique tag**, behind an upload-video → download-CSV UI.

## Start here

**All knowledge lives in [`docs/`](./docs/index.md). Read
[`docs/index.md`](./docs/index.md) first.** Do not duplicate doc content into
this file, the root `README.md`, or nested READMEs — they are pointers.

Priority reads:
1. [`docs/hackathon/task.md`](./docs/hackathon/task.md) — authoritative task,
   29-column CSV schema, metric, constraints, deadline.
2. [`docs/hackathon/briefing.md`](./docs/hackathon/briefing.md) — metric
   deep-dive, scoring pitfalls, manual-labeling rule, gotchas.
3. [`docs/hackathon/price-tag-guide.md`](./docs/hackathon/price-tag-guide.md) —
   **OCR field-layout bible**: where each CSV field physically sits per tag
   type/mechanic. **Mandatory before any OCR/parser work.**
4. [`docs/strategy.md`](./docs/strategy.md) — model & pipeline strategy;
   §10 reconciles old assumptions with ground truth.
5. [`docs/pipeline-reference.md`](./docs/pipeline-reference.md) — current CLI,
   profiles, backends, output schema.
6. [`docs/architecture.md`](./docs/architecture.md) — the product around the
   model: monorepo layout, backend/frontend/ML contracts, build order.

Raw organizer decks are heavy binaries in `real_data/materials/` (gitignored,
local-only) — **never open them**; they are fully distilled into the docs
above. See [`docs/hackathon/source-materials.md`](./docs/hackathon/source-materials.md).

## Repo layout

```
docs/                          single source of truth (start at index.md)
backend/                       FastAPI gateway — only public service (mocked)
frontend/                      React + Vite SPA (placeholder)
ml/                            deployable ML service wrapping the pipeline (mocked)
docker-compose.yaml            whole product, one command
projects/price_tag_pipeline/   THE MODEL: training, experiments, research
  src/price_tag_pipeline/      detector→rectifier→ocr→qr→parser→aggregator
  scripts/                     CLI entry points (prepare/train/infer/export/UI)
  configs/                     runtime profiles (fast/balanced/hq/zeroshot/…)
  tests/                       parser, aggregator, metrics, data, smoke
data/                          raw/processed/splits/checkpoints (mostly gitignored)
real_data/                     untouched local source of truth — never committed
```

`backend`/`frontend`/`ml` are a **reviewable skeleton**: structure and
contracts are in place; business logic is deliberately not written yet
(backend `MOCK_MODE`, ml fake CSV, frontend placeholder). Architecture and
build order: [`docs/architecture.md`](./docs/architecture.md).

Branch: **`main` is canonical** and supersedes `feature/full-autonomous-demo`.
See [`docs/branches.md`](./docs/branches.md).

## Hard rules (do not violate)

- **No cloud APIs / external online services at inference.** The shipped
  pipeline must run fully local. Heavy training off-box is allowed; document it.
- **No global pip.** Use a `uv`-managed `.venv`. See project memory.
- **`frame_timestamp` is milliseconds** from video start, never a frame index.
- **Non-ASCII paths must work.** Route still-image I/O through
  `price_tag_pipeline.cv_io`; never call `cv2.imread/imwrite` directly.
- **`"нет"` ≠ empty.** Absent on tag → `"нет"`; present but unread → empty.
- Fix root causes and explain them; never silently paper over a bug.
- Disclose every labeled / external / synthetic dataset in the README — the
  organizers make this a hard scoring requirement.

## What the metric actually rewards

A tag scores only if it is matched to GT (**barcode** is the primary key, else
noisy `frame_timestamp+bbox`) **and** ≥80% of its *substantive* fields are
correct. Therefore, in priority order: **barcode recognition → dedup (one row
per tag) → substantive fields → QR**. Skipping a low-confidence tag beats
emitting noise. Full reasoning: [`docs/index.md`](./docs/index.md) "five facts".

## Killer feature (stretch goal, documented)

A planned **shelf-analytics layer** (per-product visible *facings* + gap/OOS
flags) targets the non-metric finals criteria, not the score. Attempt **only
if the core graded pipeline is solid**; keep its output separate from the
graded CSV. Spec: [`docs/strategy.md`](./docs/strategy.md) §12. Dataset
backing: [`docs/data/datasets-research.md`](./docs/data/datasets-research.md).

## Running things

CLI reference and copy-paste commands: [`docs/pipeline-reference.md`](./docs/pipeline-reference.md)
and [`docs/runbooks/`](./docs/runbooks/local.md). Tests don't need data or GPU:

```bash
pytest projects/price_tag_pipeline/tests -v
```

## When you change things

- Editing knowledge → edit the file under `docs/`, not a pointer. If you add a
  doc, link it from [`docs/index.md`](./docs/index.md).
- Editing behaviour → keep [`docs/pipeline-reference.md`](./docs/pipeline-reference.md)
  in sync; it is what humans and agents trust for current state.
- A claim that contradicts the code is a bug in the doc — fix the doc.
