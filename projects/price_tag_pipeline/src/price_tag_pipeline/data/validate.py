"""Dataset integrity validation.

We treat data errors as failure conditions, not warnings. A single bad label
file can silently corrupt training (loss-spike, NaN). This validator fails
loudly with a structured report.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

try:  # OpenCV is only needed when read_images=True (decodes JPEGs).
    import cv2
except ModuleNotFoundError:  # pragma: no cover - exercised on opencv-less envs
    cv2 = None  # type: ignore[assignment]

# Unicode-safe reader — cv2.imread silently returns None on non-ASCII paths.
from ..cv_io import imread  # noqa: E402

LOGGER = logging.getLogger(__name__)


@dataclass
class IntegrityReport:
    n_images: int = 0
    n_label_files: int = 0
    n_boxes: int = 0
    images_missing_label: list[str] = field(default_factory=list)
    labels_missing_image: list[str] = field(default_factory=list)
    bad_boxes: list[tuple[str, int, str]] = field(default_factory=list)  # (label_path, line_no, reason)
    unreadable_images: list[str] = field(default_factory=list)
    classes_in_use: dict[int, int] = field(default_factory=dict)
    expected_classes: int = 0
    per_video_frames: dict[str, int] = field(default_factory=dict)
    per_video_labeled: dict[str, int] = field(default_factory=dict)

    @property
    def is_ok(self) -> bool:
        return (
            not self.images_missing_label
            and not self.labels_missing_image
            and not self.bad_boxes
            and not self.unreadable_images
        )

    def summary(self) -> str:
        lines = [
            f"  images:                 {self.n_images}",
            f"  label files:            {self.n_label_files}",
            f"  total boxes:            {self.n_boxes}",
            f"  images missing label:   {len(self.images_missing_label)}",
            f"  labels missing image:   {len(self.labels_missing_image)}",
            f"  bad boxes:              {len(self.bad_boxes)}",
            f"  unreadable images:      {len(self.unreadable_images)}",
            f"  classes used (id->count): {dict(sorted(self.classes_in_use.items()))}",
            f"  expected classes:       {self.expected_classes}",
        ]
        if self.per_video_frames:
            lines.append("  per-video frame/label counts (first 10):")
            for i, (vid, n) in enumerate(self.per_video_frames.items()):
                if i >= 10:
                    lines.append(f"    ... and {len(self.per_video_frames) - 10} more")
                    break
                labeled = self.per_video_labeled.get(vid, 0)
                lines.append(f"    {vid}: {n} frames / {labeled} labeled")
        return "\n".join(lines)


def validate_dataset(processed_dir: Path, classes: list[str], read_images: bool = True) -> IntegrityReport:
    """Walk processed_dir and validate every (image, label) pair."""
    if read_images and cv2 is None:
        raise ModuleNotFoundError(
            "validate_dataset(read_images=True) needs OpenCV; install opencv-python "
            "or call with read_images=False to skip JPEG decoding."
        )
    report = IntegrityReport(expected_classes=len(classes))
    frames_root = processed_dir / "frames"
    labels_root = processed_dir / "labels"

    if not frames_root.exists() or not labels_root.exists():
        LOGGER.warning("Either frames or labels root missing — empty dataset.")
        return report

    for video_dir in sorted(frames_root.iterdir()):
        if not video_dir.is_dir():
            continue
        vid = video_dir.name
        n_frames = len(list(video_dir.glob("*.jpg")))
        report.per_video_frames[vid] = n_frames
        label_dir = labels_root / vid
        n_labels = len(list(label_dir.glob("*.txt"))) if label_dir.exists() else 0
        report.per_video_labeled[vid] = n_labels

        for img_path in sorted(video_dir.glob("*.jpg")):
            stem = img_path.stem
            label_path = label_dir / f"{stem}.txt"
            report.n_images += 1

            if not label_path.exists():
                report.images_missing_label.append(str(img_path))
                continue

            if read_images:
                im = imread(img_path)
                if im is None:
                    report.unreadable_images.append(str(img_path))
                    continue

            _validate_label_file(label_path, report, expected_classes=len(classes))

        # Reverse direction: labels with no matching image.
        if label_dir.exists():
            for label_path in label_dir.glob("*.txt"):
                stem = label_path.stem
                if not (video_dir / f"{stem}.jpg").exists():
                    report.labels_missing_image.append(str(label_path))

    return report


def _validate_label_file(label_path: Path, report: IntegrityReport, expected_classes: int) -> None:
    report.n_label_files += 1
    try:
        text = label_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        report.bad_boxes.append((str(label_path), 0, "file not utf-8"))
        return

    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            report.bad_boxes.append((str(label_path), i, f"expected 5 tokens, got {len(parts)}"))
            continue
        try:
            cls = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            report.bad_boxes.append((str(label_path), i, "non-numeric token"))
            continue
        if cls < 0 or cls >= expected_classes:
            report.bad_boxes.append((str(label_path), i, f"class id {cls} out of range"))
            continue
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            report.bad_boxes.append((str(label_path), i, "center out of [0,1]"))
            continue
        if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            report.bad_boxes.append((str(label_path), i, "width/height not in (0,1]"))
            continue
        x_min = cx - w / 2
        x_max = cx + w / 2
        y_min = cy - h / 2
        y_max = cy + h / 2
        if x_min < -1e-3 or x_max > 1 + 1e-3 or y_min < -1e-3 or y_max > 1 + 1e-3:
            report.bad_boxes.append((str(label_path), i, "box extends outside image"))
            continue
        report.classes_in_use[cls] = report.classes_in_use.get(cls, 0) + 1
        report.n_boxes += 1
