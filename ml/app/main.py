"""ML inference service — a thin deployable wrapper around the price-tag
pipeline. Internal model/training logic is NOT here; it lives in
``projects/price_tag_pipeline/`` (training, experiments, research). This
service only loads that package and exposes a stable HTTP contract to the
gateway. See docs/architecture.md §5.

Contract (the gateway is the only caller — never the frontend):

    GET  /health             -> {"status": "ok", "mode": "real"|"mock"}
    POST /process            -> ProcessResponse  (one video → 29-column CSV)
    GET  /progress/{job_id}  -> latest progress snapshot   (ADDITIVE)

``/process`` runs the real pipeline when it can import it (else a mock CSV so
the monorepo boots without GPU/weights — see runner.py). It is a *sync* def
on purpose: FastAPI runs sync routes in a worker thread, so a minutes-long
job does not block the event loop and ``/progress`` stays answerable while
it runs. ``/progress`` is **additive and optional** — the
``ProcessRequest``/``ProcessResponse`` contract and its
``backend/app/ml/schemas.py`` mirror are unchanged.
"""

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.contract import ProcessRequest, ProcessResponse
from app.runner import current_mode, run_pipeline, warm_state, warmup
from app import progress_registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the heavy model NOW, not on the first upload. Background daemon
    # thread so the service is immediately up and /health answers while the
    # ~8 GB Qwen load runs; the process-wide VLM cache + its lock mean a
    # request that races the warm-up just waits for the single shared load.
    threading.Thread(target=warmup, name="vlm-warmup", daemon=True).start()
    yield


app = FastAPI(
    title="Lenta Price-Tag ML Service", version="0.2.0", lifespan=lifespan
)


@app.get("/health")
async def health() -> dict:
    # `warm`: cold|warming|ready|skipped|error — observable, additive.
    return {
        "status": "ok",
        "service": "ml",
        "mode": current_mode(),
        "warm": warm_state(),
    }


@app.post("/process", response_model=ProcessResponse)
def process(req: ProcessRequest) -> ProcessResponse:
    # sync def → threadpool → event loop free for /progress (see module doc).
    return run_pipeline(req)


@app.get("/progress/{job_id}")
async def progress(job_id: str) -> dict:
    snapshot = progress_registry.get(job_id)
    if snapshot is None:
        return {"job_id": job_id, "status": "unknown"}
    return {"job_id": job_id, "status": "running", **snapshot}
