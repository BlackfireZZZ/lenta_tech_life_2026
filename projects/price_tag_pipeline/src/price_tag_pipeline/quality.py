from __future__ import annotations

import cv2
import numpy as np


def laplacian_sharpness(image: np.ndarray) -> float:
    if image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


_TENENGRAD_MAX_SIDE = 256


def tenengrad_sharpness(image: np.ndarray) -> float:
    """Tenengrad (Sobel gradient energy) focus measure.

    Used only by the tracker's per-track best-crop ranking — NOT the pipeline
    sharpness gate (that stays on ``laplacian_sharpness``). Tenengrad is more
    robust to motion blur and far less noise-sensitive than Laplacian
    variance, which matters when picking the sharpest of a moving track's
    many crops. Scale-tolerant: large crops are downscaled so the measure is
    cheap and roughly size-invariant in this hot path.
    """
    if image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape[:2]
    longest = max(h, w)
    if longest > _TENENGRAD_MAX_SIDE:
        s = _TENENGRAD_MAX_SIDE / float(longest)
        gray = cv2.resize(gray, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))

