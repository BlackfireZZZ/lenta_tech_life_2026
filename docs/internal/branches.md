# Branch map

This repository is intentionally split into hackathon branches so the team can
switch between safe baseline and richer experimental pipeline.

> **Status (verified 2026-05-19).** `main` is no longer a thin shell — it now
> carries the **full consolidated pipeline** and is *ahead of*
> `feature/full-autonomous-demo` (main HEAD `8c07b601` builds on that branch's
> tip). Treat `main` as the canonical, most-up-to-date branch. The `feature/*`
> branches below are kept for history and for isolated baselines; their
> descriptions are accurate for what each *introduced*, not for "what is most
> complete today".

## `main`

**Canonical branch — full pipeline + real Lenta data layer.**

- Everything from `feature/full-autonomous-demo` plus the real Lenta dataset
  structure, ms/CSV-timestamp fixes, and Unicode-safe I/O
  (`feat(data): real Lenta dataset structure + ms/CSV fixes`).
- 30 modules under `projects/price_tag_pipeline/src/`; full hackathon CSV
  export, training scripts, metrics, tests.
- Dockerfile, pyproject, pre-commit, docs.
- Use this branch for all new work and final runs.

## `feature/price-tag-pipeline-starter`

First production-oriented scaffold:

- Core module boundaries: detector -> rectifier -> OCR -> parser -> aggregator.
- Basic configs, tests, and data layout.
- Useful for reading the architecture history, not recommended for final runs.

## `feature/ml-baseline-colab`

GPU/Colab-ready baseline branch:

- Batch inference scripts.
- Zero-label config based on YOLO-World open-vocabulary detection.
- Hackathon CSV exporter.
- Colab runbook.

Use this branch when you want the simplest known-good no-label baseline.

## `feature/full-autonomous-demo`

The branch that introduced the full autonomous pipeline. **Superseded by
`main`** (main now contains every commit here plus the real-data layer). Kept
for history; do not branch new work off it — use `main`.

What it introduced:

- Full hackathon CSV schema support through `extra_fields`.
- QR-first extraction from crops with OpenCV QR decoder and optional `pyzbar`.
- OCR/QR audit trail for visual debugging.
- Annotated MP4 renderer with detection boxes, recognized fields, and OCR/QR moments.
- Gradio UI: upload video -> annotated MP4 + CSV.
- Updated runbooks for local and Google Colab runs.

Historically the "final pipeline + demo" branch; that role now belongs to
`main`.
