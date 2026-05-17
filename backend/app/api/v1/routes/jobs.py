"""`jobs` resource — upload a shelf video, poll status, download the CSV.

Reference flow (docs/architecture.md §3.6 + §5):

    POST /api/v1/jobs            multipart video  -> {id, status: queued}
    GET  /api/v1/jobs/{id}       -> {id, status, progress, result_csv_url?}
    GET  /api/v1/jobs/{id}/result.csv  -> the 29-column CSV (when succeeded)

Video processing is minutes-long, so the real implementation persists jobs
in Postgres and runs the ML call out-of-band (architecture.md §5.3 "async
jobs" pattern), with the frontend polling GET /jobs/{id}.

Status: MOCKED. An in-memory store fakes a job that "progresses" on each
poll and then succeeds. No DB, no real ML call, no real CSV. This exists so
the contract and UX can be wired and reviewed before the real pipeline lands.
"""

from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, UploadFile, status
from fastapi.responses import PlainTextResponse

from app.api.v1.schemas.job import JobResponse, JobStatus

router = APIRouter(prefix="/jobs", tags=["Jobs"])

# --- MOCK STORE (replace with app/db/models/job.py + Postgres) -------------
_MOCK_JOBS: dict[UUID, dict] = {}
_MOCK_CSV_HEADER = (
    "filename,frame_timestamp,x1,y1,x2,y2,barcode,name,price,..."  # 29 cols — see docs/hackathon/task.md
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


@router.post("", status_code=status.HTTP_201_CREATED, response_model=JobResponse)
async def create_job(video: UploadFile) -> JobResponse:
    job_id = uuid4()
    _MOCK_JOBS[job_id] = {
        "id": job_id,
        "status": JobStatus.queued,
        "progress": 0.0,
        "filename": video.filename or "upload.mp4",
        "rows": None,
        "error": None,
        "result_csv_url": None,
        "created_at": _now(),
        "updated_at": _now(),
    }
    return JobResponse.model_validate(_MOCK_JOBS[job_id])


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: UUID) -> JobResponse:
    job = _MOCK_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")

    # Fake progress so the polling UI has something to show.
    if job["status"] in (JobStatus.queued, JobStatus.running):
        job["status"] = JobStatus.running
        job["progress"] = round(min(job["progress"] + 0.34, 1.0), 2)
        if job["progress"] >= 1.0:
            job["status"] = JobStatus.succeeded
            job["rows"] = 42
            job["result_csv_url"] = f"/api/v1/jobs/{job_id}/result.csv"
        job["updated_at"] = _now()
    return JobResponse.model_validate(job)


@router.get("/{job_id}/result.csv", response_class=PlainTextResponse)
async def download_result(job_id: UUID) -> str:
    job = _MOCK_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    if job["status"] != JobStatus.succeeded:
        raise HTTPException(status.HTTP_409_CONFLICT, "Result not ready")
    # Real impl streams the CSV produced by the ML service.
    return _MOCK_CSV_HEADER + "\n"
