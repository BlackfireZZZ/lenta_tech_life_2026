# ml/ — ML inference service

Thin **deployable** HTTP service. It is *not* the model: training,
experiments and research live in
[`../projects/price_tag_pipeline/`](../projects/price_tag_pipeline/). This
service only imports that package and exposes a stable contract to the
gateway (`/health`, `/process`). The frontend never calls it directly.

**Status: mocked.** `/process` returns a fake CSV (no GPU/weights needed).
The contract, the sync-vs-async decision, and how this wraps the pipeline
are in **[`../docs/architecture.md`](../docs/architecture.md) §5**. The model
itself: [`../docs/pipeline-reference.md`](../docs/pipeline-reference.md).

```bash
cd ml
uv venv && uv pip install fastapi "uvicorn[standard]" pydantic
uv run uvicorn app.main:app --port 8002 --reload   # /health, POST /process
```
