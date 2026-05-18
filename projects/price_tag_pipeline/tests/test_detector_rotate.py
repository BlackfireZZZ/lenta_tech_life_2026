"""Detector frame-rotation config.

The Lenta scan-robot camera is mounted 90° CW (clips are stored sideways
landscape). The detector path must rotate frames upright before inference;
`rotate` defaults to "ccw" so a fresh config does the right thing on the live
data, and is overridable to "none"/"cw".
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from price_tag_pipeline.config import DetectorConfig, _as_detector  # noqa: E402


def _node(**extra):
    base = {
        "backend": "yolo",
        "model_path": "data/checkpoints/detector/best.pt",
        "conf": 0.25,
        "iou": 0.6,
        "tracker_yaml": "configs/trackers/botsort.yaml",
    }
    base.update(extra)
    return base


def test_rotate_defaults_to_ccw():
    assert DetectorConfig.rotate == "ccw"
    assert _as_detector(_node()).rotate == "ccw"


def test_rotate_override_parsed_and_normalised():
    assert _as_detector(_node(rotate="NONE")).rotate == "none"
    assert _as_detector(_node(rotate=" cw ")).rotate == "cw"
