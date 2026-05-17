"""ML inference service — a thin deployable wrapper around the price-tag
pipeline. Internal model/training logic is NOT here; it lives in
``projects/price_tag_pipeline/`` (training, experiments, research). This
service only loads that package and exposes a stable HTTP contract to the
gateway. See docs/architecture.md §5.

Contract (the gateway is the only caller — never the frontend):

    GET  /health   -> {"status": "ok"}
    POST /process  -> ProcessResponse  (one video → the 29-column CSV)

Status: MOCKED. ``/process`` returns a fake CSV without running the model so
the whole monorepo boots without GPU/weights. The real call is wired (and
commented) in ``runner.py``.
"""

from fastapi import FastAPI

from app.contract import ProcessRequest, ProcessResponse
from app.runner import run_pipeline

app = FastAPI(title="Lenta Price-Tag ML Service", version="0.1.0")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "ml", "mock": True}


@app.post("/process", response_model=ProcessResponse)
async def process(req: ProcessRequest) -> ProcessResponse:
    return run_pipeline(req)
