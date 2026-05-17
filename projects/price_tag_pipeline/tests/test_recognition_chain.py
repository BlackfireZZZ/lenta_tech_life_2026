"""Recognition-chain contract tests.

Locks the structural agreement so the separate QR/barcode branch can swap
decoder internals without silently breaking the merge policy:

- chain runs decoders in QR -> barcode -> OCR order and concatenates readings;
- emptiness is decided by the single shared `parsed_is_empty`;
- `build_recognition_chain` honours the `recognition:` enable flags and
  defaults to the full chain when the block is absent.

No GPU / real models / real data: a `noop` OCR backend and fake decoders.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from price_tag_pipeline.config import load_config  # noqa: E402
from price_tag_pipeline.recognition import (  # noqa: E402
    CropDecoder,
    RecognitionChain,
    RecognitionResult,
    build_recognition_chain,
    parsed_is_empty,
)
from price_tag_pipeline.types import ParsedTag  # noqa: E402


class _FakeDecoder(CropDecoder):
    def __init__(self, name: str, results: list[RecognitionResult]) -> None:
        self.name = name
        self._results = results
        self.calls = 0

    def decode(self, crop_bgr: np.ndarray) -> list[RecognitionResult]:
        self.calls += 1
        return self._results


def _result(decoder: str, **fields) -> RecognitionResult:
    parsed = ParsedTag(backend=decoder, **fields)
    return RecognitionResult(
        parsed=parsed,
        decoder=decoder,
        confidence=0.9,
        text="x",
        found=not parsed_is_empty(parsed),
    )


def _img() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


def test_parsed_is_empty_rules() -> None:
    assert parsed_is_empty(ParsedTag())
    assert not parsed_is_empty(ParsedTag(regular_price=99.0))
    assert not parsed_is_empty(ParsedTag(extra_fields={"barcode": "460"}))


def test_chain_preserves_order_and_concatenates() -> None:
    qr = _FakeDecoder("qr", [_result("qr", extra_fields={"barcode": "4601"})])
    bc = _FakeDecoder("barcode", [])
    ocr = _FakeDecoder("ocr", [_result("ocr", product_name="Молоко")])
    chain = RecognitionChain([qr, bc, ocr])

    out = chain.decode(_img())

    assert [r.decoder for r in out] == ["qr", "ocr"]  # barcode returned nothing
    assert qr.calls == bc.calls == ocr.calls == 1  # QR is NOT a short-circuit
    assert chain.decoder_names == ["qr", "barcode", "ocr"]


def test_chain_runs_ocr_even_when_qr_found() -> None:
    """The agreed policy is 'QR-first, OCR fills gaps' — OCR always runs."""
    qr = _FakeDecoder("qr", [_result("qr", regular_price=99.0)])
    ocr = _FakeDecoder("ocr", [_result("ocr", product_name="Сыр")])
    chain = RecognitionChain([qr, ocr])

    out = chain.decode(_img())

    assert ocr.calls == 1
    assert {r.decoder for r in out} == {"qr", "ocr"}


def _write_cfg(tmp_path: Path, recognition_block: str) -> Path:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
runtime: {profile_name: t, output_path: null, log_every_n_frames: 50}
detector: {backend: yolo, model_path: x.pt, conf: 0.25, iou: 0.5, device: null,
           classes: null, tracker_yaml: botsort.yaml}
rectifier: {padding_ratio: 0.06, clahe_clip_limit: 2.5, clahe_tile_grid_size: 8,
            rotate_vertical_tags: true}
ocr: {backend: noop, min_frames_between_ocr_per_track: 5, min_sharpness: 35.0,
      min_crop_area_px: 1400, min_detection_confidence: 0.3}
parser: {min_price: 0.01, max_price: 999999.99}
aggregation: {min_observations_per_track: 2, min_final_confidence: 0.5,
              track_ttl_frames: 60}
"""
        + recognition_block,
        encoding="utf-8",
    )
    return cfg


def test_build_chain_defaults_to_full_chain(tmp_path: Path) -> None:
    cfg = load_config(_write_cfg(tmp_path, ""))  # no recognition: block
    assert cfg.recognition.enable_qr
    assert cfg.recognition.merge_policy == "qr_first_fill_gaps"
    chain = build_recognition_chain(cfg)
    assert chain.decoder_names == ["qr", "barcode", "ocr"]


def test_build_chain_honours_enable_flags(tmp_path: Path) -> None:
    block = "recognition: {enable_qr: true, enable_barcode: false, enable_ocr: true}\n"
    cfg = load_config(_write_cfg(tmp_path, block))
    chain = build_recognition_chain(cfg)
    assert chain.decoder_names == ["qr", "ocr"]
