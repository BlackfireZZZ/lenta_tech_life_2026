# lenta_tech_life_2026

Production-ready starter for automatic price tag recognition from robot video.

## Included now

- Modular price-tag pipeline in `projects/price_tag_pipeline`
- YOLO-based detection/tracking integration point
- OCR stage abstraction (`noop`, `tesseract`, `paddle`)
- Price parser with currency/unit normalization
- Track-level voting and confidence aggregation
- Runtime profiles: `fast`, `balanced`, `hq`
- Unit tests for parser and aggregation core

## Quick start

```bash
python projects/price_tag_pipeline/scripts/run_price_tag_pipeline.py \
  --video /absolute/path/to/video.mp4 \
  --config projects/price_tag_pipeline/configs/balanced.yaml \
  --output /absolute/path/to/out.jsonl
```

## Project structure

```text
projects/price_tag_pipeline/
  configs/
  scripts/
  src/price_tag_pipeline/
  tests/
```

## Next milestones

1. Train detector on real shelf footage.
2. Enable PaddleOCR backend for production runs.
3. Add CI checks and benchmark suite (FPS/latency/quality).
4. Add fallback server OCR for low-confidence tracks.
