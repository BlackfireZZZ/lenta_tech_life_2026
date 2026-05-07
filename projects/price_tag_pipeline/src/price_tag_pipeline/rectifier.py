from __future__ import annotations

import cv2
import numpy as np

from .config import RectifierConfig
from .quality import laplacian_sharpness
from .types import CropCandidate


class TagRectifier:
    def __init__(self, cfg: RectifierConfig):
        self.cfg = cfg
        self._clahe = cv2.createCLAHE(
            clipLimit=self.cfg.clahe_clip_limit,
            tileGridSize=(self.cfg.clahe_tile_grid_size, self.cfg.clahe_tile_grid_size),
        )

    def rectify(self, frame: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> CropCandidate | None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox_xyxy
        pad_x = int((x2 - x1) * self.cfg.padding_ratio)
        pad_y = int((y2 - y1) * self.cfg.padding_ratio)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w - 1, x2 + pad_x)
        y2 = min(h - 1, y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2].copy()
        if crop.size == 0:
            return None

        if self.cfg.rotate_vertical_tags and crop.shape[0] > crop.shape[1] * 1.6:
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        norm = self._clahe.apply(gray)
        enhanced = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
        sharpness = laplacian_sharpness(norm)
        area_px = int(enhanced.shape[0] * enhanced.shape[1])
        return CropCandidate(
            image=enhanced,
            sharpness=sharpness,
            area_px=area_px,
            bbox_xyxy=(x1, y1, x2, y2),
        )

