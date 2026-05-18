"""The single place that knows the ML service's address and protocol.

Routes/background tasks call ``ml_client.process(...)`` /
``ml_client.get_progress(...)`` — never httpx directly. Swapping sync HTTP
for a queue (architecture.md §5.3) changes only this file.

Contract (docs/architecture.md §5.2):

    POST {ML_BASE_URL}/process            -> {csv, rows, meta?}
    GET  {ML_BASE_URL}/progress/{job_id}  -> {fraction, phase, ...}  (ADDITIVE)

``MOCK_MODE`` keeps the gateway runnable standalone (no ML container): it
returns a tiny fake CSV and a synthetic progress. With ``MOCK_MODE=false``
(the docker-compose real path) it makes the actual network calls.
"""

from __future__ import annotations

import httpx
from fastapi import HTTPException

from app.core.config import settings
from app.core.logger import logger
from app.ml.schemas import ProcessRequest, ProcessResponse

# A finished job's progress snapshot for the mock path.
_MOCK_PROGRESS = {"fraction": 1.0, "phase": "done", "message": "mock"}


class MLClient:
    def __init__(self) -> None:
        self._base_url = settings.ML_BASE_URL
        self._timeout = settings.ML_TIMEOUT_SECONDS

    async def process(self, req: ProcessRequest) -> ProcessResponse:
        """Run one video end-to-end. Blocks for the whole (minutes-long) run.

        Called from the out-of-band background task (architecture.md §5.3),
        never inside a request handler, so blocking here is fine — the event
        loop stays free (``await`` yields while the ML side computes).
        """
        if settings.MOCK_MODE:
            logger.info("MOCK MLClient.process(job=%s) -> fake CSV", req.job_id)
            return ProcessResponse(
                csv="filename,frame_timestamp,x1,y1,x2,y2,barcode,...\n",
                rows=0,
                meta={"mock": True},
            )

        # A real GPU run (detect + Qwen-VL OCR over a full clip) routinely
        # exceeds any fixed read budget — a flat `timeout=600` made httpx
        # abandon a still-running job and surface a bogus 502. Processing is
        # out-of-band and its liveness is observable via `/progress`, so the
        # READ is intentionally unbounded; only connect/write keep a fast
        # guard so a genuinely-down ML still fails quickly.
        timeout = httpx.Timeout(
            connect=10.0, read=None, write=60.0, pool=None
        )
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url, timeout=timeout
            ) as c:
                r = await c.post("/process", json=req.model_dump())
                r.raise_for_status()
        except httpx.HTTPError as e:
            logger.error("ML /process failed (job=%s): %s", req.job_id, e)
            raise HTTPException(502, "ML service unavailable") from e
        return ProcessResponse.model_validate(r.json())

    async def get_progress(self, job_id: str) -> dict:
        """Latest progress snapshot for a running job (best-effort).

        Additive/optional side-channel (architecture.md §5.5). Never fatal:
        any failure returns ``{}`` so a flaky progress poll cannot fail the
        job — the job's real outcome is decided by ``process()``.
        """
        if settings.MOCK_MODE:
            return dict(_MOCK_PROGRESS)
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url, timeout=10.0
            ) as c:
                r = await c.get(f"/progress/{job_id}")
                r.raise_for_status()
                return r.json()
        except httpx.HTTPError as e:
            logger.debug("ML /progress poll failed (job=%s): %s", job_id, e)
            return {}


ml_client = MLClient()  # one instance per app
