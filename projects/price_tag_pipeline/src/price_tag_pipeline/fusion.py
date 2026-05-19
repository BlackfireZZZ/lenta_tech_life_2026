"""Multi-frame crop fusion — Level-2 tracker-amplified code reading.

A QR/1D symbol is physically identical and static across every frame of a
track; the tracker certifies "these crops are the same tag". In motion *no
single frame* may decode (all motion-blurred), but blur/sensor noise is
roughly zero-mean and uncorrelated across frames while the symbol structure
is constant — so aligning and median-fusing the track's crops cancels the
noise and can yield ONE image that decodes when none of the inputs do
(classic signal averaging / multi-frame restoration).

This is image-level fusion only; the (owner's) recognition reader still does
symbol localisation + decode on the fused image — no reader-seam change.
Pure, dependency-light (cv2 + numpy); returns ``None`` when fusion is not
worthwhile so callers transparently fall back to per-frame decoding.
"""

from __future__ import annotations

import numpy as np


def _tenengrad(gray: np.ndarray) -> float:
    import cv2

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def fuse_crops(images: list[np.ndarray], max_frames: int = 8) -> np.ndarray | None:
    """Align the sharpest ``max_frames`` crops to the sharpest one, median-fuse.

    Reference = the sharpest crop (best-defined geometry). Each other crop is
    resized to the reference size and ECC-aligned (affine; translation
    fallback); frames that fail to converge are dropped. The per-pixel median
    of the aligned stack is returned (median, not mean, so a few
    blurred/occluded frames don't drag the result), lightly unsharp-masked to
    counter the averaging softening. ``None`` if <2 usable crops.
    """
    import cv2

    imgs = [im for im in images if im is not None and im.size]
    if len(imgs) < 2:
        return None

    grays = [
        cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im
        for im in imgs
    ]
    order = sorted(range(len(imgs)), key=lambda i: _tenengrad(grays[i]), reverse=True)
    order = order[:max_frames]
    ref_i = order[0]
    ref = imgs[ref_i]
    h, w = ref.shape[:2]
    ref_gray = grays[ref_i].astype(np.float32)

    stack: list[np.ndarray] = [ref.astype(np.float32)]
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
    for i in order[1:]:
        mv = imgs[i]
        if mv.shape[:2] != (h, w):
            mv = cv2.resize(mv, (w, h), interpolation=cv2.INTER_CUBIC)
        mv_gray = (
            cv2.cvtColor(mv, cv2.COLOR_BGR2GRAY) if mv.ndim == 3 else mv
        ).astype(np.float32)
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(
                ref_gray, mv_gray, warp, cv2.MOTION_AFFINE, crit, None, 5
            )
            aligned = cv2.warpAffine(
                mv.astype(np.float32), warp, (w, h),
                flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )
        except cv2.error:
            continue  # uncorrelated frame — skip rather than pollute
        stack.append(aligned)

    if len(stack) < 2:
        return None

    fused = np.median(np.stack(stack, axis=0), axis=0).astype(np.uint8)
    blur = cv2.GaussianBlur(fused, (0, 0), 1.0)
    sharp = cv2.addWeighted(fused, 1.5, blur, -0.5, 0)
    return sharp
