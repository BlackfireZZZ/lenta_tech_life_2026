# Branch map

This repository is intentionally split into hackathon branches so the team can
switch between safe baseline and richer experimental pipeline.

## `main`

Minimal reproducible project shell:

- Dockerfile, pyproject, base README.
- No full ML pipeline guarantees.
- Use only as a stable public landing branch.

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

Current top-score attempt branch:

- Full hackathon CSV schema support through `extra_fields`.
- QR-first extraction from crops with OpenCV QR decoder and optional `pyzbar`.
- OCR/QR audit trail for visual debugging.
- Annotated MP4 renderer with detection boxes, recognized fields, and OCR/QR moments.
- Gradio UI: upload video -> annotated MP4 + CSV.
- Updated runbooks for local and Google Colab runs.

Use this branch for the final hackathon pipeline and demo.
