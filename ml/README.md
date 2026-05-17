# ml/ — ML inference service

Thin **deployable** HTTP service. It is *not* the model: training,
experiments and research live in
[`../projects/price_tag_pipeline/`](../projects/price_tag_pipeline/). This
service imports that package's public entry point and exposes a stable
contract to the gateway. The frontend never calls it directly.

```
GET  /health             {"status":"ok","mode":"real"|"mock"}
POST /process            ProcessRequest  -> ProcessResponse (29-col CSV)
GET  /progress/{job_id}  latest progress snapshot   (additive, optional)
```

**Real or mock.** `/process` runs the real pipeline
(`PriceTagPipeline(cfg).run` → `submission.final_tags_to_csv`) when it can
import the package; if `ML_MOCK=1` or the import fails (no GPU/heavy deps),
it returns a fake CSV so the monorepo still boots. `current_mode()` is
reported by `/health`.

`/progress/{job_id}` is **additive**: the pipeline streams its progress into
an in-process registry keyed by the job id while `/process` runs (sync def →
threadpool, so the bar stays answerable). The `ProcessRequest` /
`ProcessResponse` contract and its `backend/app/ml/schemas.py` mirror are
unchanged — this just gives the future async gateway (architecture.md §5.3)
and its `JobResponse.progress` field a real source.

Env: `ML_MOCK` (force mock), `ML_PIPELINE_CONFIG`
(default `configs/balanced.yaml`).

Contract, sync-vs-async rationale and how this wraps the pipeline:
**[`../docs/architecture.md`](../docs/architecture.md) §5**. The model
itself: [`../docs/pipeline-reference.md`](../docs/pipeline-reference.md).

```bash
cd ml
uv venv && uv pip install fastapi "uvicorn[standard]" pydantic
# mock mode boots with just the above; for real inference also:
#   uv pip install -r ../projects/price_tag_pipeline/requirements/base.txt
#   uv pip install -r ../projects/price_tag_pipeline/requirements/ocr.txt
uv run uvicorn app.main:app --port 8002 --reload
```
