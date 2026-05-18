"""Detector model-path resolution tests.

No model download is performed here; we monkeypatch the Hugging Face helper so
the test stays fast and offline.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from price_tag_pipeline.detector import resolve_detector_model_path  # noqa: E402


def test_plain_model_path_is_passed_through():
    assert resolve_detector_model_path("data/checkpoints/detector/best.pt") == (
        "data/checkpoints/detector/best.pt"
    )
    assert resolve_detector_model_path("yolo11n.pt") == "yolo11n.pt"


def test_hf_model_uri_downloads_expected_repo_file(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, str]] = []

    def fake_download(*, repo_id: str, filename: str) -> str:
        calls.append((repo_id, filename))
        return "/cache/openfoodfacts-price-tag-best.pt"

    fake_module = types.ModuleType("huggingface_hub")
    fake_module.hf_hub_download = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    resolved = resolve_detector_model_path(
        "hf://openfoodfacts/price-tag-detection/weights/best.pt"
    )

    assert resolved == "/cache/openfoodfacts-price-tag-best.pt"
    assert calls == [("openfoodfacts/price-tag-detection", "weights/best.pt")]


def test_hf_model_uri_requires_owner_repo_and_file():
    with pytest.raises(ValueError, match="hf://owner/repo/path/to/file.pt"):
        resolve_detector_model_path("hf://openfoodfacts/price-tag-detection")
