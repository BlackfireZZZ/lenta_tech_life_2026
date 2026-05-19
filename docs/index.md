# Documentation index — Lenta Tech Life 2026

End-to-end price-tag recognition from robot-captured supermarket video:
detect every price tag on the shelf, read its fields, and emit **one CSV
row per unique tag** behind an upload-video → download-CSV web app.

This `docs/` tree is the **single source of truth** for the project. The
root [`README.md`](../README.md) (Russian, the main entry point),
`AGENTS.md` and `CLAUDE.md` are thin pointers here — knowledge is not
duplicated into them.

> **Status.** The whole product is built and real: a React SPA → FastAPI
> gateway (Postgres + Redis) → a GPU ML service that runs the real
> `PriceTagPipeline` (Qwen3-VL recognition), all behind one
> `docker compose up`. A guarded mock mode (`ML_MOCK=1` /
> `MOCK_MODE=true`) lets the stack boot without a GPU for UI work. See
> [architecture.md](./architecture.md).

> **Monorepo.** The model (training / experiments / research) lives in
> `projects/price_tag_pipeline/`; the product around it —
> `backend/` (API gateway), `frontend/` (SPA), `ml/` (deployable service
> wrapping the pipeline) + `docker-compose.yaml` — is described in
> [architecture.md](./architecture.md).

## Read in this order

1. **[hackathon/task.md](./hackathon/task.md)** — the official task: the
   29-column CSV schema, the metric, constraints, deliverables, timeline.
   **Authoritative.**
2. **[hackathon/price-tag-guide.md](./hackathon/price-tag-guide.md)** —
   **the field-layout reference.** Where every CSV field physically sits on
   each tag type; the `discount_amount` <100₽→% / ≥100₽→₽ rule;
   barcode-as-text vs graphical; which formats have no QR/barcode. **Read
   before touching OCR/parsing.**
3. **[architecture.md](./architecture.md)** — the product around the model:
   monorepo layout, backend/frontend/ML contracts, the 29-column CSV that
   ties all three together, build order.
4. **[strategy.md](./strategy.md)** — model & pipeline strategy: detector
   choice, recognition chain, tracking, dedup.
5. **[pipeline-reference.md](./pipeline-reference.md)** — current code: CLI
   entry points, runtime profiles, OCR/detector backends, output schema.
6. **[recognition-pipeline.md](./recognition-pipeline.md)** — the
   crop→fields chain (QR → barcode → smart OCR) and the `CropDecoder` seam.

## By topic

| Need | Doc |
|---|---|
| What exactly to build & how it's scored | [hackathon/task.md](./hackathon/task.md) |
| Where each field sits on the tag (OCR) | [hackathon/price-tag-guide.md](./hackathon/price-tag-guide.md) |
| **App / service architecture (backend + frontend + ML)** | [architecture.md](./architecture.md) |
| **Full product bring-up (docker compose, GPU)** | [runbooks/docker-compose.md](./runbooks/docker-compose.md) |
| Which model / why / training plan | [strategy.md](./strategy.md) |
| Run inference / CSV / UI / CLI flags | [pipeline-reference.md](./pipeline-reference.md) |
| Crop→fields chain + QR/barcode seam | [recognition-pipeline.md](./recognition-pipeline.md) |
| Repair barcode/name from the master catalog | [catalog-reconciliation.md](./catalog-reconciliation.md) |
| **Killer feature: out-of-stock + missing-tag + product cards** | [shelf-audit.md](./shelf-audit.md) |
| Optional product/facing shelf-state layer | [shelf-analytics.md](./shelf-analytics.md) |
| Local annotated-video + CSV runbook | [runbooks/local.md](./runbooks/local.md) |
| New env / `import cv2` fails | [runbooks/venv-setup.md](./runbooks/venv-setup.md) |
| Colab — zero-shot GPU baseline | [runbooks/colab-baseline.md](./runbooks/colab-baseline.md) |
| Colab — full pipeline on real videos | [runbooks/colab-real-data.md](./runbooks/colab-real-data.md) |
| Where data goes when it arrives | [data/layout.md](./data/layout.md) |
| External datasets — curated shortlist | [data/datasets.md](./data/datasets.md) |
| Frontend design system | [DESIGN.md](./DESIGN.md) |
| Team working notes (research, history) | [internal/](./internal/README.md) |

## The five facts that change decisions

Extracted from the hackathon docs because they are easy to get wrong and
they reshape priorities:

1. **Metric = "≥80% of *substantive* fields correct per GT-matched tag"**,
   then the share of such tags over all GT tags. Technical fields
   (`filename`, `frame_timestamp`, bbox) are *not scored* — only used for
   matching.
2. **Barcode is the primary matching key.** A row with no/incorrect barcode
   falls back to noisy `frame_timestamp+bbox` matching → may not match at
   all. Barcode recognition is **P0**, not just one field.
3. **Duplicates hurt.** One physical tag emitted N times → one row matches
   GT, the rest are unmatched noise. Tracking + cross-track dedup is
   **critical** and is explicitly the participant's responsibility.
4. **Cloud APIs are banned at inference.** The shipped pipeline must run
   fully local; lightweight / `rknn int8` is rewarded. Heavy training
   off-box is fine.
5. **`"нет"` vs empty are different.** Field absent on the tag → write
   `"нет"`. Field present but not recognized → leave empty. Confusing them
   loses points.

## Killer feature (stretch goal)

A **shelf-analytics layer** — per-product visible **facings** (how many
units are on the shelf) + optional empty-slot/gap flags — turns the tag CSV
into a shelf-state report. It targets the *non-metric* finals criteria
(maturity, scalability, business applicability), **not** the technical
score, and its output stays physically separate from the graded 29-column
CSV so it cannot corrupt the metric. Spec: [shelf-audit.md](./shelf-audit.md),
[shelf-analytics.md](./shelf-analytics.md).

## Conventions (apply to all work here)

- **No global pip.** Install into a `uv`-managed `.venv` (Python 3.11/3.12),
  never globally — see [runbooks/venv-setup.md](./runbooks/venv-setup.md).
- **`frame_timestamp` is milliseconds** from video start — never a frame
  index.
- **Non-ASCII paths must keep working** (the build machine may sit under a
  Cyrillic Windows username). Route still-image I/O through
  `price_tag_pipeline.cv_io`; never call `cv2.imread/imwrite` directly.
- **Fix root causes, explain them** — don't paper over bugs.
- **Dataset & disclosure.** The labeled set is small (5 videos with the
  provided ground-truth CSV, ~274 tag rows). The main quality lever is
  **external open datasets + open-source-model auto-labeling**, followed by
  fine-tuning on **strong, camera-matched augmentations**. Every external /
  auto-labeled / synthetic dataset is disclosed in the
  [`README.md`](../README.md) (a hard scoring requirement); see
  [data/datasets.md](./data/datasets.md).
