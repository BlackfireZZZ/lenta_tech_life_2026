"""Sample video frames every *N* seconds and drop near-duplicates.

Two concerns, kept separate so the numeric core is unit-testable without
OpenCV or a real video:

* **Sampling** — pick one frame every ``every_s`` seconds.  ``step`` is
  derived from the container FPS, so 2 s means 2 s regardless of 25/30/60 fps.
* **De-duplication** — the organizer robot frequently *stops* (shelf scan
  pauses); those runs of frames are visually identical and would bloat the
  annotation set with redundant work.  We compare each sampled frame against
  the last *kept* one via a tiny grayscale signature: if the normalized mean
  absolute difference is below ``min_diff`` the frame is a near-duplicate and
  is skipped (compared against the kept frame, not the previous sampled one,
  so a long static run collapses to a single frame — not every Nth).

The decision functions (:func:`frame_signature`, :func:`signature_distance`,
:func:`is_near_duplicate`) are pure NumPy.  Only :func:`iter_sampled_frames`
touches OpenCV, and that import is lazy.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

# Tiny enough that decode/compare is free, large enough that real shelf
# motion (a pan of a few cm) clears the threshold.
_SIG_SIZE = 32

# Sensible default for the Lenta robot footage: identical "robot parked"
# frames sit around ~0.003–0.01; a real pan/shelf change is well above 0.02.
DEFAULT_MIN_DIFF = 0.02


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only sans cv2
        raise RuntimeError(
            "OpenCV is required to read video frames. Install "
            "projects/price_tag_pipeline/requirements/base.txt in the venv."
        ) from exc
    return cv2


def frame_signature(frame_bgr: np.ndarray) -> np.ndarray:
    """Reduce a BGR frame to a small grayscale signature in ``[0, 1]``.

    Pure NumPy (no cv2): luma over a fixed ``_SIG_SIZE`` grid via strided
    block-mean, so aspect ratio / resolution never affects comparability.
    """
    arr = np.asarray(frame_bgr)
    if arr.ndim == 3:
        # BGR → luma. Constant weights, good enough for a change detector.
        gray = arr[..., :3].astype(np.float32) @ np.array(
            [0.114, 0.587, 0.299], dtype=np.float32
        )
    else:
        gray = arr.astype(np.float32)
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((_SIG_SIZE, _SIG_SIZE), dtype=np.float32)
    ys = (np.arange(_SIG_SIZE) * h // _SIG_SIZE).clip(0, h - 1)
    xs = (np.arange(_SIG_SIZE) * w // _SIG_SIZE).clip(0, w - 1)
    sig = gray[np.ix_(ys, xs)]
    return (sig / 255.0).astype(np.float32)


def signature_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized mean absolute difference of two signatures (0 = identical)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        return 1.0
    return float(np.mean(np.abs(a - b)))


def is_near_duplicate(a: np.ndarray, b: np.ndarray, min_diff: float) -> bool:
    """True when ``b`` is within ``min_diff`` of ``a`` (so ``b`` is redundant).

    ``min_diff <= 0`` disables de-duplication (nothing is a duplicate).
    """
    if min_diff <= 0.0:
        return False
    return signature_distance(a, b) < min_diff


def iter_sampled_frames(
    video_path: str | Path,
    *,
    every_s: float = 2.0,
    min_diff: float = DEFAULT_MIN_DIFF,
    max_frames: int | None = None,
) -> Iterator[tuple[int, "np.ndarray", int, float]]:
    """Yield ``(frame_idx, frame_bgr, total_frames, fps)`` for kept frames.

    Decoding is **sequential** (no per-index seek) — far faster than
    ``cap.set(POS_FRAMES)`` per sample and exact on VFR containers.  A frame
    is yielded when it is both on the ``every_s`` grid *and* not a
    near-duplicate of the last kept frame.  ``total_frames``/``fps`` are
    echoed so callers don't reopen the container.
    """
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(every_s * fps)))

    last_sig: np.ndarray | None = None
    kept = 0
    idx = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    idx += 1
                    continue
                sig = frame_signature(frame)
                if last_sig is None or not is_near_duplicate(
                    last_sig, sig, min_diff
                ):
                    last_sig = sig
                    kept += 1
                    yield idx, frame, total, fps
                    if max_frames is not None and kept >= max_frames:
                        break
            idx += 1
    finally:
        cap.release()
