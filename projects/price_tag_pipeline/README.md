# Price Tag Video Pipeline (YOLO + OCR + Tracking)

Production-grade starter for automatic price tag recognition from robot video.

## What is included

- Modular pipeline: detector -> rectifier -> OCR -> parser -> aggregator.
- Runtime profiles: `fast`, `balanced`, `hq`.
- Built-in support for YOLO tracking mode (ByteTrack/BOTSORT via Ultralytics config).
- OCR gating by quality and frame interval.
- Track-level voting with confidence aggregation.
- JSONL output ready for downstream ingestion.
- Unit tests for critical logic (price parsing and vote aggregation).

## Directory layout

```text
projects/price_tag_pipeline/
  configs/
    fast.yaml
    balanced.yaml
    hq.yaml
  scripts/
    run_price_tag_pipeline.py
  src/price_tag_pipeline/
    aggregator.py
    cli.py
    config.py
    detector.py
    ocr.py
    parser.py
    pipeline.py
    quality.py
    rectifier.py
    types.py
  tests/
    test_aggregator.py
    test_price_parser.py
  examples/
    sample_output.jsonl
```

## Quick start

From repository root:

```bash
python projects/price_tag_pipeline/scripts/run_price_tag_pipeline.py \
  --video /absolute/path/to/video.mp4 \
  --config projects/price_tag_pipeline/configs/balanced.yaml \
  --output /absolute/path/to/out.jsonl
```

## Configuration strategy

- `fast`: lower latency, fewer OCR calls.
- `balanced`: practical default.
- `hq`: more OCR attempts and stricter voting for accuracy.

All settings are YAML-driven and can be changed without code edits.

## Integration notes

- Replace `model_path` with your trained detector checkpoint (price tag class).
- Plug PaddleOCR in `ocr.backend: paddle` after installing `paddleocr`.
- Keep `track_id` enabled for stable multi-frame voting.

## Suggested next steps

1. Train YOLO on your own `price_tag` dataset.
2. Enable PaddleOCR and benchmark on hard video segments.
3. Add business rules for separating promo price, old price, and unit price.
4. Export detector to ONNX/OpenVINO/TensorRT and compare latency profiles.
