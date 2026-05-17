# backend/ — price-tag gateway API

The **only public service**. Frontend talks to this; this is the only thing
that talks to the ML service. Layered `routes → schemas → models` with a thin
`app/ml/` client.

**Status: mocked skeleton.** `MOCK_MODE=true` → endpoints return fake data;
no Postgres/Redis/ML required. Layer-by-layer contracts, the auth model, and
the implementation order live in **[`../docs/architecture.md`](../docs/architecture.md)**
(§3 backend, §5 ML integration). Don't duplicate knowledge here.

```bash
cd backend
uv venv && uv pip install -e .        # or: uv sync
uv run uvicorn app.main:app --reload  # http://localhost:8000/docs
```
