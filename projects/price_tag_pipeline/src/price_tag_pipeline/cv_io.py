"""Unicode-safe image read/write.

`cv2.imread` / `cv2.imwrite` hand the path to OpenCV's C++ layer as a
locale-encoded byte string, so on Windows they **silently** fail (return
`None` / `False`) when the path contains non-ASCII characters — e.g. a
Cyrillic project folder or username. Frames then never get written and
labels are silently orphaned.

OpenCV *video* I/O (`VideoCapture`/`VideoWriter`) is unaffected — verified
empirically — so only still-image I/O needs this shim. The trick: do the
filesystem part in Python (`np.fromfile` / `ndarray.tofile`, both fully
Unicode-aware) and let OpenCV work on an in-memory buffer
(`imdecode`/`imencode`), which never sees the path.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - exercised on opencv-less envs
    cv2 = None  # type: ignore[assignment]


def imread(path, flags: int | None = None) -> "np.ndarray | None":
    """Drop-in `cv2.imread` that works with non-ASCII paths.

    Returns `None` on a missing/empty/undecodable file, matching cv2 semantics.
    """
    if cv2 is None:
        raise ModuleNotFoundError("imread needs OpenCV; install opencv-python.")
    if flags is None:
        flags = cv2.IMREAD_COLOR
    try:
        buf = np.fromfile(os.fspath(path), dtype=np.uint8)
    except (FileNotFoundError, OSError):
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def imwrite(path, img: "np.ndarray", params: "list[int] | None" = None) -> bool:
    """Drop-in `cv2.imwrite` that works with non-ASCII paths.

    Encoding is chosen from the path suffix (default `.jpg`). Creates parent
    directories. Returns False if encoding fails, matching cv2 semantics.
    """
    if cv2 is None:
        raise ModuleNotFoundError("imwrite needs OpenCV; install opencv-python.")
    p = Path(path)
    ext = p.suffix if p.suffix else ".jpg"
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        return False
    p.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(os.fspath(p))
    return True
