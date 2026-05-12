"""Super-resolution for tiny crops.

For crops below `cfg.rectifier.sr_min_area_px`, optionally upscale 2x or 4x
with Real-ESRGAN before sending to the OCR/VLM stage. Crops that are too
small are often the difference between a parsed tag and a missed one.

Lazy-loaded; if `realesrgan` isn't installed, we silently fall back to
bicubic upsampling (which still helps a little).
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


class SuperResolution:
    """Lazy Real-ESRGAN wrapper. Falls back to bicubic when the lib is missing."""

    def __init__(self, scale: int = 2, model_name: str = "RealESRGAN_x2plus", device: Optional[str] = None):
        self.scale = scale
        self.model_name = model_name
        self.device = device
        self._upsampler = None
        self._impl: Optional[str] = None

    def _ensure_loaded(self) -> None:
        if self._upsampler is not None:
            return
        try:
            from realesrgan import RealESRGANer  # type: ignore
            from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=self.scale,
            )
            self._upsampler = RealESRGANer(
                scale=self.scale,
                model_path=None,  # let realesrgan fetch the default weights
                model=model,
                tile=0,
                pre_pad=0,
                half=True if (self.device or "").startswith("cuda") else False,
                device=self.device,
            )
            self._impl = "realesrgan"
        except ImportError:
            LOGGER.info(
                "Real-ESRGAN not available; falling back to bicubic upsampling. "
                "Install: pip install realesrgan basicsr"
            )
            self._impl = "bicubic"

    def upscale(self, image: np.ndarray) -> np.ndarray:
        self._ensure_loaded()
        if self._impl == "realesrgan":
            try:
                output, _ = self._upsampler.enhance(image, outscale=self.scale)  # type: ignore[union-attr]
                return output
            except Exception as e:
                LOGGER.warning("Real-ESRGAN failed (%s); falling back to bicubic", e)
        # Bicubic fallback.
        h, w = image.shape[:2]
        return cv2.resize(image, (w * self.scale, h * self.scale), interpolation=cv2.INTER_CUBIC)
