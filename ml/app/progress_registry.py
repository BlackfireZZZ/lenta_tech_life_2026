"""In-process job → progress registry.

The ML ``/process`` contract (architecture.md §5.2) is request/response and
intentionally unchanged. But a robot video takes minutes, and task.md §13
mandates a progress bar; architecture.md §5.3 names the async/polling pattern
as the honest target. This registry is the seam for it: the pipeline records
its :class:`ProgressEvent`s here keyed by ``job_id`` while it runs, and the
**additive, optional** ``GET /progress/{job_id}`` endpoint reads them. The
gateway can poll that endpoint and surface it through its existing
``JobResponse.progress`` field — no change to the locked ML contract or its
``backend/app/ml/schemas.py`` mirror.

In-process and best-effort by design: a single ML container, one dict, a
lock. If ML is scaled out, swap this one module for Redis — nothing else
changes (same reason the registry is isolated here).
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Optional

# Bounded so a long-lived service does not leak memory across many jobs.
# Old entries fall off; a finished job is normally read once and discarded.
_MAX_ENTRIES = 256

_LOCK = threading.Lock()
_STORE: "OrderedDict[str, dict[str, Any]]" = OrderedDict()


def record(job_id: str, progress: dict[str, Any]) -> None:
    """Store the latest progress snapshot for ``job_id`` (last write wins)."""
    with _LOCK:
        _STORE[job_id] = progress
        _STORE.move_to_end(job_id)
        while len(_STORE) > _MAX_ENTRIES:
            _STORE.popitem(last=False)


def get(job_id: str) -> Optional[dict[str, Any]]:
    """Latest snapshot for ``job_id``, or ``None`` if unknown/evicted."""
    with _LOCK:
        return _STORE.get(job_id)
