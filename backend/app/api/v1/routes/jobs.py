"""`jobs` resource — upload a shelf video, poll status, review + download.

Reference flow (docs/architecture.md §3.4 + §5):

    POST /api/v1/jobs                  multipart video -> {id, status: queued}
    GET  /api/v1/jobs/{id}             -> {id, status, progress, *_url?}
    GET  /api/v1/jobs/{id}/video       -> the uploaded clip (range-enabled)
    GET  /api/v1/jobs/{id}/predictions -> non-graded per-tag review JSON
    GET  /api/v1/jobs/{id}/result.csv  -> the graded 29-column CSV

Video processing is minutes-long, so the real implementation persists jobs
in Postgres and runs the ML call out-of-band (architecture.md §5.3 "async
jobs" pattern), with the frontend polling GET /jobs/{id}.

Status: MOCKED. An in-memory store fakes a job that "progresses" on each
poll and then succeeds. The uploaded clip is kept on a temp path so it can
be played back; the tags are fabricated deterministically per job id
(``app/jobs_mock.py``) so the review JSON and the CSV always agree. No DB,
no real ML call. This exists so the contract and the full review UX can be
wired and reviewed before the real pipeline lands.
"""

import shutil
from datetime import datetime, timezone
from pathlib import Path
from tempfile import gettempdir
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, PlainTextResponse

from app.api.v1.schemas.job import JobPredictions, JobResponse, JobStatus
from app.jobs_mock import build_csv, build_predictions, generate_tags

router = APIRouter(prefix="/jobs", tags=["Jobs"])

# --- MOCK STORE (replace with app/db/models/job.py + Postgres) -------------
_MOCK_JOBS: dict[UUID, dict] = {}
# Uploaded clips live here so /video can stream them back. Temp by design —
# the real impl uses the shared `uploads` volume (architecture.md §6).
_UPLOAD_ROOT = Path(gettempdir()) / "lenta_jobs"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _get_job_or_404(job_id: UUID) -> dict:
    job = _MOCK_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    return job


def _tags_for(job: dict) -> list:
    """Lazily materialize (and cache) the deterministic mock tags."""
    if job["tags"] is None:
        job["tags"] = generate_tags(job["id"], job["filename"])
    return job["tags"]


@router.post("", status_code=status.HTTP_201_CREATED, response_model=JobResponse)
async def create_job(video: UploadFile) -> JobResponse:
    job_id = uuid4()
    filename = video.filename or "upload.mp4"

    # Persist the clip so the review screen can play it back. Store under a
    # fixed, ASCII-safe name (the original may be non-ASCII); keep the
    # original name only as metadata. Non-ASCII paths must keep working.
    ext = Path(filename).suffix.lower() or ".mp4"
    job_dir = _UPLOAD_ROOT / str(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path = job_dir / f"source{ext}"
    with video_path.open("wb") as out:
        shutil.copyfileobj(video.file, out)

    _MOCK_JOBS[job_id] = {
        "id": job_id,
        "status": JobStatus.queued,
        "progress": 0.0,
        "filename": filename,
        "video_path": video_path,
        "tags": None,
        "rows": None,
        "error": None,
        "result_csv_url": None,
        "predictions_url": None,
        "video_url": None,
        "created_at": _now(),
        "updated_at": _now(),
    }
    return JobResponse.model_validate(_MOCK_JOBS[job_id])


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: UUID) -> JobResponse:
    job = _get_job_or_404(job_id)

    # Fake progress so the polling UI has something to show. ~3 polls -> done.
    if job["status"] in (JobStatus.queued, JobStatus.running):
        job["status"] = JobStatus.running
        job["progress"] = round(min(job["progress"] + 0.34, 1.0), 2)
        if job["progress"] >= 1.0:
            job["status"] = JobStatus.succeeded
            job["rows"] = len(_tags_for(job))
            base = f"/api/v1/jobs/{job_id}"
            job["result_csv_url"] = f"{base}/result.csv"
            job["predictions_url"] = f"{base}/predictions"
            job["video_url"] = f"{base}/video"
        job["updated_at"] = _now()
    return JobResponse.model_validate(job)


@router.get("/{job_id}/video")
async def get_video(job_id: UUID) -> FileResponse:
    job = _get_job_or_404(job_id)
    video_path: Path = job["video_path"]
    if not video_path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Source video not found")
    # FileResponse honours HTTP Range, so the SPA can seek to any tag's
    # timestamp without downloading the whole clip.
    return FileResponse(video_path, filename=f"source{video_path.suffix}")


@router.get("/{job_id}/predictions", response_model=JobPredictions)
async def get_predictions(job_id: UUID) -> JobPredictions:
    job = _get_job_or_404(job_id)
    if job["status"] != JobStatus.succeeded:
        raise HTTPException(status.HTTP_409_CONFLICT, "Predictions not ready")
    base = f"/api/v1/jobs/{job_id}"
    return build_predictions(
        job_id=job_id,
        filename=job["filename"],
        tags=_tags_for(job),
        video_url=f"{base}/video",
        csv_url=f"{base}/result.csv",
    )


@router.get("/{job_id}/result.csv", response_class=PlainTextResponse)
async def download_result(job_id: UUID) -> PlainTextResponse:
    job = _get_job_or_404(job_id)
    if job["status"] != JobStatus.succeeded:
        raise HTTPException(status.HTTP_409_CONFLICT, "Result not ready")
    csv_text = build_csv(_tags_for(job))
    return PlainTextResponse(
        csv_text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="result.csv"'},
    )
