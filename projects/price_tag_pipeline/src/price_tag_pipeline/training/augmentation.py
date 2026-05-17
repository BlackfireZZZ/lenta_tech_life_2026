"""Augmentation hyper-parameters tuned for moving-robot retail-shelf footage.

These are surfaced as a dataclass so the same defaults are used by training
scripts and (later) by the dataset class that overrides Ultralytics' built-ins
with Albumentations.

Rationale for non-default values is documented in docs/strategy.md §2.3.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UltralyticsAugConfig:
    # HSV jitter
    hsv_h: float = 0.010
    hsv_s: float = 0.55
    hsv_v: float = 0.35

    # Geometric
    degrees: float = 3.0          # small rotation; price tags are usually upright
    translate: float = 0.07
    scale: float = 0.35
    shear: float = 1.0
    perspective: float = 0.0008
    flipud: float = 0.0           # never flip vertically — would invert text
    fliplr: float = 0.5

    # Mosaic / MixUp — mosaic hurts small-object recall above ~0.3
    mosaic: float = 0.5
    mixup: float = 0.0
    copy_paste: float = 0.0

    # Built-in blur/noise (Ultralytics 8.x+ supports motion blur via Albumentations callback;
    # see training script comments for the override hook).
    erasing: float = 0.4          # random erasing helps with partial occlusions

    def as_kwargs(self) -> dict[str, float]:
        return {
            "hsv_h": self.hsv_h,
            "hsv_s": self.hsv_s,
            "hsv_v": self.hsv_v,
            "degrees": self.degrees,
            "translate": self.translate,
            "scale": self.scale,
            "shear": self.shear,
            "perspective": self.perspective,
            "flipud": self.flipud,
            "fliplr": self.fliplr,
            "mosaic": self.mosaic,
            "mixup": self.mixup,
            "copy_paste": self.copy_paste,
            "erasing": self.erasing,
        }
