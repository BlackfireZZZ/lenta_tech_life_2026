# Application architecture — monorepo (backend + frontend + ML)

How the **product** around the price-tag model is structured: a public API
gateway, the built SPA, and the model packaged as an internal service.
This is the single source of truth for the web/service architecture — the
per-service `README.md` files are thin pointers here.

> **Current status: the whole product is wired end to end and real.**
> `frontend` is the real upload→poll→review→CSV SPA. `backend` now runs
> `MOCK_MODE=false` in docker-compose: jobs **persist in Postgres**, and it
> calls the **real** `ml` service **out-of-band** (the request returns
> immediately; an asyncio task runs the pipeline and the gateway mirrors
> ML `/progress` into `JobResponse.progress`), then serves the verbatim
> graded CSV and a review payload reconstructed from it (§5.6). `ml` is
> **real**: it runs `PriceTagPipeline` on GPU and streams live progress,
> with a guarded mock fallback so the monorepo still boots without
> GPU/weights (§5). `MOCK_MODE=true` is still supported as a standalone
> skeleton (in-memory jobs, deterministic mock tags, no DB/Redis/ML) for
> frontend-only work. The unifying contract between all three is the graded
> 29-column CSV + per-tag field semantics (§5.6). Each section states
> *as-built* vs. *target* so the doc never lies about the code.

---

## 0. TL;DR

- **Monorepo.** Web/service stack (`backend/`, `frontend/`, `ml/`,
  `docker-compose.yaml`) lives alongside the existing ML research codebase
  (`projects/price_tag_pipeline/`) and the knowledge base (`docs/`).
- **One business resource: `jobs`.** Upload a shelf video → poll status →
  download the graded 29-column CSV. There is no generic CRUD.
- **Backend** = FastAPI gateway, the only public service. Layered
  `routes → schemas → models`, async SQLAlchemy + Postgres, Redis cache,
  thin `app/ml/` client. Target auth: JWT + refresh + CSRF (documented,
  not yet implemented).
- **Frontend** = React 19 + Vite + TypeScript SPA (Tailwind v4 +
  shadcn-style), axios client, DTO types from the backend's OpenAPI. Built:
  upload → poll → review (video + bbox + crop + fields) → CSV.
- **ML** = a *deployable wrapper* (`ml/`) that imports the
  `price_tag_pipeline` package and exposes `/process` + `/health`. The
  model, training and experiments stay in `projects/price_tag_pipeline/`.
  Frontend never calls ML directly — only `frontend → backend → ml`.
- **Infra (local):** Postgres + Redis, all in one `docker compose up`.

---

## 1. Monorepo layout

```
lenta_tech_life_2026/
├── backend/                  FastAPI gateway — the only public service
│   ├── app/
│   │   ├── api/v1/
│   │   │   ├── routes/       HTTP handlers (one file = one resource)
│   │   │   └── schemas/      Pydantic DTOs (Create/Update/Response)
│   │   ├── core/             config, logger (+ security/deps when auth lands)
│   │   ├── db/               base, session, models/  (implemented — §3.3)
│   │   ├── cache/            redis wrapper            (implemented — §3.9)
│   │   ├── ml/               CLIENT to the ML service (not the model!) — §5
│   │   └── main.py           FastAPI assembly
│   ├── scripts/init.sh       entrypoint: migrate → serve
│   ├── pyproject.toml · .env.example · Dockerfile · README.md
├── frontend/                 React + Vite SPA (built, on the real API)
│   ├── src/
│   │   ├── api/              axios client + one module per resource
│   │   ├── pages/            pages by feature (UploadPage, JobPage)
│   │   ├── App.tsx · main.tsx · index.css
│   ├── package.json · vite.config.ts · openapi-ts.config.ts
│   ├── Dockerfile.dev · .env.example · README.md
├── ml/                       Deployable ML service (wraps the pipeline) — §5
│   ├── app/ (main.py · contract.py · runner.py)
│   ├── pyproject.toml · Dockerfile · README.md
├── projects/
│   └── price_tag_pipeline/   THE MODEL: training, experiments, research
├── data/                     datasets / checkpoints (mostly gitignored)
├── real_data/                local source of truth (never committed)
├── docs/                     knowledge base — single source of truth
├── docker-compose.yaml       whole product, one command
└── Dockerfile                separate CUDA *training* image (not in compose)
```

Principle: **one resource → three files** —
`routes/<res>.py` + `schemas/<res>.py` + `models/<res>.py` on the backend,
`api/<res>.ts` + a page on the frontend. Files stay small and focused
(200–400 lines, ~800 max).

### 1.1 The two halves of "ML" — read this once

| | `projects/price_tag_pipeline/` | `ml/` |
|---|---|---|
| What | the actual model: detector→OCR→parser→aggregator, training scripts, experiments | a thin HTTP service |
| Audience | ML work (offline, GPU, iteration) | the running product |
| Stability | churns constantly | stable contract (§5) |
| Ships in image | only via `ml/` | yes |

`ml/` **imports** the package and calls its public entry point
(`PriceTagPipeline(cfg).run(video)`). This single seam (`ml/app/runner.py`)
means model/training churn never breaks the service contract. Model
internals are out of scope here — see
[`pipeline-reference.md`](./pipeline-reference.md) and
[`strategy.md`](./strategy.md).

---

## 2. Stack (versions are a floor — take current)

**Backend** — Python 3.11+, `uv`; FastAPI + Uvicorn(`[standard]`);
`python-multipart` (video upload); `httpx` (→ ML). *Target, when persistence
& auth land:* SQLAlchemy 2.0 (`[asyncio]`) + `asyncpg`, Alembic, `redis`,
`pyjwt`, `argon2-cffi`, `itsdangerous`. dev: `pytest`, `black`, `isort`.

**Frontend** — React 19 + Vite 7 + TypeScript; React Router 7; axios;
`@hey-api/openapi-ts` (DTOs from OpenAPI). *Target UI:* Tailwind v4 +
shadcn-style components, `react-hook-form` + `zod`, `sonner`.

**ML service** — FastAPI thin wrapper. Heavy deps belong to
`price_tag_pipeline`; the mock image carries none of them.

---

## 3. Backend

### 3.1 Layers & data flow (target)

```
HTTP → routes/<res>.py
         │  Depends(get_current_user)   ← auth   (target — §3.8)
         │  Depends(get_db_session)     ← async DB session (target — §3.3)
         │  Depends(get_redis_client)   ← cache  (target — §3.9)
         ├─ input validation: schemas.<Res> (Pydantic)
         ├─ data access: SQLAlchemy select/insert over models.<Res>
         ├─ cache: get_or_set_json on reads, invalidate on writes
         └─ response: schemas.<Res>Response.model_validate(obj)
```

Simple CRUD logic lives directly in the route (no needless service layer —
correct for a hackathon). Heavy domain logic → its own module.

**As-built:** routes are DB-backed (async SQLAlchemy via `session_scope`)
with a fail-open Redis read-cache; auth is still out by design (§3.8). The
data-flow shape above is real minus the auth dependency.

### 3.2 Config — `app/core/config.py`

Single settings source, all from `.env` via `pydantic-settings`, no secrets
in code. `MOCK_MODE` (default `true`) is the master switch: while true,
routes and the ML client return fakes and never touch Postgres/Redis/ML.
Flip it off layer-by-layer as each is implemented. Computed
`DATABASE_URL` / `DATABASE_URL_SYNC` (Alembic) / `REDIS_URL` are already
defined for when persistence lands.

### 3.3 Persistence — `app/db/` *(implemented)*

`base.py` (declarative `Base`), `session.py` (async `create_async_engine`
+ `async_sessionmaker` + `get_db_session` request dependency +
`session_scope` for the out-of-band task + `init_models`/`ping`),
`models/job.py` (the single `Job` entity — UUID PK, `created_at/updated_at`
via `server_default=func.now()`, imported in `models/__init__.py`).

**Deviation from the Alembic target (intentional, hackathon):** the schema
is one `jobs` table, so the app brings it up with
`Base.metadata.create_all` in the FastAPI lifespan (`init_models`, retried
against a just-started Postgres) instead of Alembic migrations. `create_all`
is idempotent and has zero migration-state failure surface in a
one-command `docker compose up`. Alembic stays the documented path for when
the schema grows (§3.10).

### 3.4 The `jobs` resource (the whole product API)

A *job* = one uploaded shelf video → one 29-column result CSV.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/jobs` | multipart video upload → `{id, status: queued}` |
| `GET`  | `/api/v1/jobs/{id}` | status + progress (+ `result_csv_url` when done) |
| `GET`  | `/api/v1/jobs/{id}/result.csv` | the graded CSV (when `succeeded`) |
| `GET`  | `/api/v1/health` | liveness |

Schema: `schemas/job.py` (`JobResponse`, `JobStatus`
queued→running→succeeded/failed). CSV schema itself:
[`hackathon/task.md`](./hackathon/task.md) (29 columns) — the gateway
serves it **verbatim** and must never reshape graded columns. That
29-column CSV is the contract that ties the gateway, the SPA and the real
pipeline together; its three owners and lock-step are §5.6.

**As-built (`MOCK_MODE=false`, the docker-compose default):**
`routes/jobs.py` persists every job in Postgres. `POST /jobs` writes the
clip to the shared `uploads` volume under an ASCII-safe name, inserts a
`queued` row, spawns an asyncio task and returns immediately. The task
(`_process_job`, its own `session_scope`) marks the job `running`, calls
the real ML service, and a sibling poller mirrors ML `/progress` into
`Job.progress` (capped <1.0, stopped+awaited before the terminal write so
it can't race). On success it stores the verbatim 29-column CSV and the
review payload reconstructed from that CSV (`app/predictions.py`, §5.6);
on any failure the job is `failed` with the message — the server never
crashes. `GET /jobs/{id}` reads the row (terminal responses cached in
Redis, fail-open, never a *running* job — §3.9); `/video` streams the
stored clip (range-enabled); `/predictions` and `/result.csv` serve the
persisted review JSON and the verbatim CSV. `MOCK_MODE=true` keeps the old
in-memory deterministic mock (`app/jobs_mock.py`) for standalone work.

### 3.8 Authentication *(target — documented, not implemented)*

The hackathon flow (anonymous upload→download) may not need auth at all; if
it does, copy the proven model 1:1 rather than inventing one:

- **Passwords:** Argon2id (`argon2-cffi`).
- **Access token:** JWT in an httpOnly cookie `access_token` (~30 min).
- **Refresh token:** `JWT.random_part`; `random_part` Argon2-hashed in a
  `sessions` row. Rotation + replay detection: a stale `random_part`
  revokes all of the user's sessions.
- **CSRF:** double-submit cookie (`itsdangerous`-signed `csrf_token`
  + `X-CSRF-Token` header on unsafe methods); GET/HEAD/OPTIONS skip.
- Auth router mounts **without** `/api/v1` and **without** the CSRF
  dependency (no cookies issued yet there).

Decide auth in/out before building it; if in, it is the most expensive part
— budget for it. Placeholder files are intentionally absent to avoid
implying it's wired.

### 3.9 Cache — `app/cache/` *(implemented)*

`RedisCache` exposes `get_json`/`set_json` (+ the documented
`get_or_set_json` convenience), all **fail-open**: any Redis error or
`CACHE_ENABLED=false` ⇒ transparent passthrough, never a failed request.
Wired in `GET /jobs/{id}` for **terminal jobs only** — a *running* job's
status/progress changes every poll, so caching it would freeze the
progress bar; a terminal job is immutable so its short TTL needs no
invalidation. (Marginal value here since the SPA stops polling once
terminal — kept because it makes the spine real and is provably safe.)

### 3.10 Migrations — Alembic *(deferred; create_all in use — §3.3)*

The shipped path is `Base.metadata.create_all` in the app lifespan (§3.3):
one table, idempotent, no migration state to corrupt on a cold
`docker compose up`. Alembic (autogenerate only —
`uv run alembic revision --autogenerate` → `upgrade head`, never
hand-edit, run from `scripts/init.sh` before Uvicorn) remains the
documented path for when the schema grows beyond `jobs`.

---

## 4. Frontend

### 4.1 Principle

SPA on React + Vite. One `src/api/<res>.ts` module per backend resource.
DTO types are **generated** from the backend's OpenAPI, never hand-written
(they are stubbed only until the backend is real). Pages grouped by feature
under `src/pages/`. Route access gated by layout components.

**As-built:** the real SPA — Tailwind v4 + design tokens + shadcn-style
primitives (`components/ui/`), React Router with lazy pages, `api/client.ts`
+ `api/jobs.ts`. `UploadPage` (drag&drop → `POST /jobs`) and `JobPage`
(poll → review: source video with bbox overlay, canvas crop at the tag's
timestamp, grouped recognized fields, summary, tags table, CSV download).
Anonymous (no auth, §3.8). DTO types are hand-kept mirrors of the backend
schema until `npm run generate:types` is run against a live `/openapi.json`
(§4.4); the auth axios client (§4.2) is intentionally not added.

### 4.2 axios client — `src/api/client.ts` (copy 1:1 when auth lands)

One axios instance, `withCredentials: true`, interceptors:
**request** — `X-CSRF-Token` (from `csrf_token` cookie) on unsafe methods,
`Idempotency-Key` on POST. **response 401** — single-flight
`POST /auth/token/refresh` (shared `refreshPromise` so parallel 401s don't
race), retry once, else clear cookies → `/login?next=`. **response 403
CSRF** — `GET /auth/csrf`, retry once. This file is critical auth UX —
copy verbatim.

### 4.3 API module per resource — `src/api/<res>.ts`

Thin wrapper over `apiClient`, types from `generated/`. `jobs.ts` exposes
`create(video)` / `get(id)`; long video jobs use polling on `get(id)`.

### 4.4 Generated types

`openapi-ts.config.ts` reads `./openapi.json`, emits `src/api/generated/`.
Flow: download backend `/openapi.json` → `npm run generate:types`. Wire
`prebuild` → `generate:types` so the contract is typed and free.

### 4.5 AuthContext / 4.6 routing & layouts (target)

`AuthContext` holds `user/isLoading/authError`, calls `getMe()` on mount,
registers a logout callback with the axios client. Route tree:
`Router → ThemeProvider → AuthProvider → ErrorBoundary → Suspense →
Routes` with `AuthLayout`/`PublicOnlyRoutes` and
`ProtectedRoutes`/`MainLayout`; lazy pages; `<Toaster />` at the root.
(Only relevant if auth is in — see §3.8.)

### 4.7 UI stack (target)

Tailwind v4 (`@tailwindcss/vite`), shadcn-style components in
`src/components/ui/` (Radix + `cva` + `cn()`), `react-hook-form` + `zod`,
`lucide-react`, `@ → src` alias (already set in `vite.config.ts`).

---

## 5. ML integration — the contract

> ML internals are out of scope (model/training =
> `projects/price_tag_pipeline/`, see [`strategy.md`](./strategy.md)).
> Here: only how the product talks to it.

### 5.1 Rules

1. **Frontend never calls ML.** Only `frontend → backend → ml`. One place
   for auth, CORS, validation.
2. **Backend → ML through `backend/app/ml/client.py` only.** Routes call
   `ml_client.process(...)`, never `httpx` directly. Swapping sync HTTP for
   a queue changes only that file.
3. **Contract is two Pydantic mirrors:** `backend/app/ml/schemas.py` and
   `ml/app/contract.py` describe the same JSON; keep them in lock-step.
4. **ML is stateless:** in = video reference, out = CSV. All state (jobs,
   results) lives in the backend's Postgres.

### 5.2 Contract

```
POST {ML_BASE_URL}/process            {video_path, job_id, filename} → 200 {csv, rows, meta?}
GET  {ML_BASE_URL}/health                                  → 200 {"status":"ok","mode":…}
GET  {ML_BASE_URL}/progress/{job_id}                        → 200 {fraction,phase,…}   (ADDITIVE)
```

`csv` is the full 29-column submission text
([`hackathon/task.md`](./hackathon/task.md)); the backend persists and
serves it unchanged. `ProcessRequest` gained **`filename`** (optional,
default `""`) — both mirrors updated in lock-step (§5.1 rule 3). It is
**required for correctness**: the clip is stored on disk as an ASCII-safe
`source.<ext>`, so without it every graded CSV's `filename` cell would be
`source` and break GT matching for barcode-less tags. `meta` carries
non-graded raw-clip geometry (`frame_width/height`, `video_duration_s`)
the gateway uses to rebuild the review overlay. **The graded contract is
`/process` + `/health`** and `ProcessResponse` is **unchanged**.
`/progress/{job_id}` is **additive and optional**: the gateway MAY poll it
to fill its own `JobResponse.progress`, but is not required to, and the
two-mirror lock-step does not cover it.

### 5.3 Sync vs. async

Video processing is **minutes-long**, so the gateway uses the async
pattern: `POST /jobs` returns immediately with a `queued` job, the work
runs out-of-band, the frontend polls `GET /api/v1/jobs/{id}`. The contract
in §5.2 is identical — only *who waits* changed. **As-built:** the ML side
runs the real pipeline as a *sync def* (FastAPI threadpool) so `/progress`
stays answerable during a run; the gateway side is a **single in-process
FIFO worker** (`app/jobs_queue.py`) — uploads are processed **strictly one
at a time** because the deliberately cheap rented box has one small GPU and
a single ML run already saturates its VRAM (two in flight ⇒ both far
slower, or OOM). A second upload stays `queued` until the one ahead
finishes; `JobResponse.queue_position` tells the user how many videos are
ahead (0 = running / next up). The queue is in-process (one ML container —
§5.5) but **durable across a restart**: on boot, `queued`/`running` rows in
Postgres are re-enqueued in `created_at` order (Postgres stays the source
of truth; the queue is just the scheduler). If ever scaled to several GPU
boxes, this one worker is the single piece that becomes a shared broker —
the contract is unchanged.

### 5.4 Files

`backend/app/ml/{schemas,client}.py` — contract + the single httpx seam.
**Real:** `client.process()` does the async `POST /process`
(`ML_TIMEOUT_SECONDS`, 502 on failure); `client.get_progress()` polls
`/progress` best-effort (any failure → `{}`, never fails the job).
`MOCK_MODE=true` still returns the fake CSV so the gateway runs standalone.
`ml/app/{contract,runner,main}.py` — contract mirror + the **real**
pipeline bridge: `runner.py` calls
`PriceTagPipeline(cfg).run(video, progress=…)` →
`submission.final_tags_to_csv`, with a guarded mock fallback
(`ML_MOCK=1`, or pipeline import fails → fake CSV so the monorepo still
boots without GPU/weights). `progress_registry.py` is the in-process
`job_id → progress` store behind `GET /progress/{job_id}`. Env:
`ML_MOCK`, `ML_PIPELINE_CONFIG` (default `configs/balanced.yaml`).

### 5.5 Progress side-channel (additive)

The pipeline's single progress signal
([`pipeline-reference.md`](./pipeline-reference.md) "Progress reporting")
is recorded per `job_id` and served by `GET /progress/{job_id}`
(`{fraction 0..1, phase, frames_done/total, tags_finalized, message}`).
This is the concrete source the async target (§5.3) and the backend's
existing `JobResponse.progress` (§3.4) will consume — built **without**
touching the locked `/process` contract. In-process by design (one ML
container); if ML is scaled out, swap `progress_registry.py` for Redis and
nothing else changes.

### 5.6 The unifying contract — the 29-column CSV (read this)

What makes frontend + backend + inference **one product** is not the HTTP
call (the backend doesn't even call ML yet) — it is the graded
**29-column CSV + per-tag field semantics**. It has *three owners* that
must stay in lock-step, exactly like the §5.1-rule-3 Pydantic mirrors
(backend and `ml/` are separate deployables — only `ml/` may import the
pipeline package — so the schema is necessarily mirrored, not shared):

| Owner | Where | Role |
|---|---|---|
| **Producer** | `price_tag_pipeline.submission` — `HACK_CSV_COLUMNS`, `final_tags_to_csv`, `hack_row_from_tag_dict` | the real CSV, used by the ML service & Gradio |
| **Gateway** | `backend/app/api/v1/schemas/job.py:CSV_COLUMNS` (+ `SUBSTANTIVE_FIELDS`/`TECHNICAL_FIELDS`) + `app/jobs_mock.py:build_csv` | served *verbatim* as the graded artifact; today a deterministic mock |
| **Consumer** | frontend `JobPredictions`/`TagPrediction` (`columns` + `fields`) | renders the review screen |

All three agree on: the 29 column names **and order**; the byte format
(UTF-8, `,` separator, `.` decimal, `\n` line terminator,
`QUOTE_MINIMAL`); and the three field states — a value, `"нет"` (absent
on the tag), or `""` (present but unrecognized) — which are scored
(task.md §3.3/§5.3). The backend now does this for real: `app/predictions.py`
parses the producer's verbatim CSV straight into `TagPrediction.fields`,
derives the normalized bbox from the pixel columns (÷ the raw-frame size
from `meta`) and `t_frac` from `frame_timestamp` ÷ clip duration, and the
review UI is unchanged — the seam working
(`backend/tests/test_predictions_reconstruction.py` pins the round-trip).
The lock-step
is enforced on the producer side by `tests/test_submission.py` (column
count/order + `\n`/`QUOTE_MINIMAL` parity with `build_csv`); change one
owner ⇒ change all three.

---

## 6. docker-compose

`docker compose up --build` runs the whole product: `backend:8000`
(`/docs` = Swagger), `frontend:5173`, `ml:8002`, `db:5432`, `redis:6379`.
A named `uploads` volume is shared between `backend` and `ml` for the video
handoff (§5); `ml` also bind-mounts `./real_data/db_hack.csv` read-only for
GT-safe catalog reconciliation (optional, skipped if absent —
[`catalog-reconciliation.md`](./catalog-reconciliation.md)). The root
`./Dockerfile` is the **separate CUDA training image** and is intentionally
not part of compose. The backend runs
`MOCK_MODE=false` (Postgres + real ML call); `backend/.env` is optional
(`env_file: required: false` — every needed value is in the compose
`environment:`). Full bring-up + verification:
[`runbooks/docker-compose.md`](./runbooks/docker-compose.md).

---

## 7. Code conventions

Small focused files (200–400 lines, ~800 max), one resource per file.
Explicit `HTTPException` with status codes, no internals in `detail`. No
secrets in code — `.env` → `settings`, commit only `.env.example`.
Parameterized SQL only (SQLAlchemy). Validation only at the Pydantic
boundary. Python: Black + isort (profile `black`). TS: ESLint, components
`PascalCase`, DTO types only from `generated/`. Branches `feature/<name>`.
These also obey the repo-wide hard rules in
[`../AGENTS.md`](../AGENTS.md) (no cloud APIs at inference, `uv` venvs,
non-ASCII paths, `"нет"` ≠ empty).

## 8. Deliberately dropped (hackathon weight-cut)

No CI/CD, no OpenTelemetry/observability stack, no MCP/AI-assistant, no
RabbitMQ/Celery (the cheap single GPU forces serialization, but a
**single in-process FIFO worker** — §5.3 — covers it; a broker would be
pure overhead for one box), no S3/MinIO (a shared volume covers the video
handoff), no history/audit/rate-limit, no Storybook. What stays
is the clean spine: layered backend + async SQLAlchemy + `create_all`
(Alembic deferred, §3.10) + fail-open Redis + (optional) auth; typed
frontend with a generated contract; ML as a separate service behind a thin
client.

## 9. Build order — status

1. **Auth:** **out** (§3.8) — anonymous upload→download, as designed.
2. **Backend persistence: done** — `db/` + `Job` model, `routes/jobs.py`
   on Postgres via `session_scope`, schema via `create_all` (Alembic
   deferred, §3.3/§3.10).
3. **ML service: done** — `ml/app/runner.py` runs the real
   `PriceTagPipeline` on the GPU image with a guarded mock fallback +
   `/progress` (§5.4/§5.5); `meta` now also carries non-graded raw-clip
   geometry for the review overlay.
4. **backend→ML wiring: done** — real async `client.process()` driven by a
   single in-process FIFO worker (one video at a time on the cheap GPU,
   `app/jobs_queue.py`) + `/progress` poll into `JobResponse.progress`
   (§3.4/§5.3). `ProcessRequest.filename` added (both mirrors) to keep the
   graded `filename` cell correct.
5. **Cache: done** — fail-open `RedisCache`, terminal-only (§3.9).
6. **Frontend:** the built SPA (§4) consumes the real endpoints unchanged —
   the CSV→predictions seam (§5.6) makes the review screen work against the
   real pipeline with no UI change.
7. **End-to-end:** `docker compose up` → upload a real video → poll →
   download the real graded CSV. Runbook + per-service verification:
   [`runbooks/docker-compose.md`](./runbooks/docker-compose.md).

---

**Summary:** the whole product is real — DB-backed gateway, fail-open
Redis, the model kept in `projects/price_tag_pipeline/` behind a thin ML
service called out-of-band, the typed SPA unchanged thanks to the verbatim
29-column CSV contract (§5.6). `MOCK_MODE=true` stays as the standalone
skeleton. Keep this doc honest as the code evolves.
