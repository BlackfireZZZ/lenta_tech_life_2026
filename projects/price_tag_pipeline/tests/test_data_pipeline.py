"""Data-prep smoke tests using synthetic fixtures.

We do not commit real images. The fixture builder writes a tiny set of
synthetic frames + YOLO labels to a temp directory and runs the pipeline
end-to-end through validation + split.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
import numpy as np

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from price_tag_pipeline.data.loaders import ingest_yolo, write_dataset_yaml  # noqa: E402
from price_tag_pipeline.data.splits import build_video_level_folds  # noqa: E402
from price_tag_pipeline.data.validate import validate_dataset  # noqa: E402


def _make_synthetic_raw(root: Path, video_ids: list[str], frames_per_video: int = 4) -> None:
    annot_root = root / "annotations" / "labels"
    frames_root = root / "frames"
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    (root / "annotations" / "classes.txt").write_text("price_tag\n", encoding="utf-8")
    for vid in video_ids:
        (annot_root / vid).mkdir(parents=True, exist_ok=True)
        (frames_root / vid).mkdir(parents=True, exist_ok=True)
        for i in range(frames_per_video):
            img = np.full((480, 640, 3), 128, dtype=np.uint8)
            cv2.rectangle(img, (200, 200), (300, 260), (255, 255, 255), -1)
            cv2.imwrite(str(frames_root / vid / f"{i:06d}.jpg"), img)
            (annot_root / vid / f"{i:06d}.txt").write_text("0 0.391 0.479 0.156 0.125\n", encoding="utf-8")


def test_ingest_validate_split(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    _make_synthetic_raw(raw, ["vid_a", "vid_b", "vid_c"])

    records, classes = ingest_yolo(raw, processed, extract_videos=False)
    assert len(records) == 12  # 3 videos x 4 frames
    assert classes == ["price_tag"]

    report = validate_dataset(processed, classes, read_images=False)
    assert report.is_ok, report.summary()
    assert report.n_images == 12
    assert report.n_boxes == 12

    folds = build_video_level_folds(["vid_a", "vid_b", "vid_c"], n_splits=3)
    assert len(folds) == 3
    # Every video appears in exactly one val set across folds.
    seen_in_val: set[str] = set()
    for fold in folds:
        seen_in_val.update(fold["val"])
    assert seen_in_val == {"vid_a", "vid_b", "vid_c"}

    yaml_path = write_dataset_yaml(processed, classes, fold=folds[0])
    assert yaml_path.exists()
    assert (processed / "train_images.txt").exists()
    assert (processed / "val_images.txt").exists()
