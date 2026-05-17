"""The single place that knows the ML service's address and protocol.

Routes call ``ml_client.process(...)`` — never httpx directly. Swapping
sync HTTP for a queue (architecture.md §5.3) changes only this file.

Status: MOCKED. Returns a fake CSV without any network call so the gateway
runs standalone. The real body (kept as a comment) is a plain httpx POST.
"""

# import httpx
from app.core.config import settings
from app.core.logger import logger
from app.ml.schemas import ProcessRequest, ProcessResponse


class MLClient:
    def __init__(self) -> None:
        self._base_url = settings.ML_BASE_URL
        self._timeout = settings.ML_TIMEOUT_SECONDS

    async def process(self, req: ProcessRequest) -> ProcessResponse:
        if settings.MOCK_MODE:
            logger.info("MOCK MLClient.process(job=%s) -> fake CSV", req.job_id)
            return ProcessResponse(
                csv="filename,frame_timestamp,x1,y1,x2,y2,barcode,...\n",
                rows=0,
                meta={"mock": True},
            )

        # Real implementation (enable when ML service is live):
        #
        # async with httpx.AsyncClient(base_url=self._base_url,
        #                              timeout=self._timeout) as c:
        #     try:
        #         r = await c.post("/process", json=req.model_dump())
        #         r.raise_for_status()
        #     except httpx.HTTPError as e:
        #         logger.error("ML service call failed: %s", e)
        #         raise HTTPException(502, "ML service unavailable") from e
        # return ProcessResponse.model_validate(r.json())
        raise NotImplementedError("Set MOCK_MODE=false and wire the httpx call above")


ml_client = MLClient()  # one instance per app
