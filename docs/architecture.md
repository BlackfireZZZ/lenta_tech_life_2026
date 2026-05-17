# Application architecture — monorepo (backend + frontend + ML)

How the **product** around the price-tag model is structured: a public API
gateway, a placeholder SPA, and the model packaged as an internal service.
This is the single source of truth for the web/service architecture — the
per-service `README.md` files are thin pointers here.

> **Current status: scaffolded skeleton.** The structure, the service
> boundaries and every contract are in place and reviewable. Business logic
> is deliberately **not written yet** — `backend` runs in `MOCK_MODE`, `ml`
> returns a fake CSV, `frontend` renders one placeholder page. Each section
> below states what is *as-built* vs. the *target* so the doc never lies
> about the code. Build the real layers against this doc.

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
- **Frontend** = React 19 + Vite + TypeScript SPA, axios client, DTO types
  generated from the backend's OpenAPI. Currently a placeholder.
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
│   │   ├── db/               base, session, models/  (placeholder — §3.3)
│   │   ├── cache/            redis wrapper            (placeholder — §3.9)
│   │   ├── ml/               CLIENT to the ML service (not the model!) — §5
│   │   └── main.py           FastAPI assembly
│   ├── scripts/init.sh       entrypoint: migrate → serve
│   ├── pyproject.toml · .env.example · Dockerfile · README.md
├── frontend/                 React + Vite SPA (placeholder)
│   ├── src/
│   │   ├── api/              axios client + one module per resource
│   │   ├── pages/            pages by feature (Home = placeholder)
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

**As-built:** routes return mock data from an in-memory store; no DB/cache/
auth dependencies are attached yet. The shape above is the target the
placeholders are wired toward.

### 3.2 Config — `app/core/config.py`

Single settings source, all from `.env` via `pydantic-settings`, no secrets
in code. `MOCK_MODE` (default `true`) is the master switch: while true,
routes and the ML client return fakes and never touch Postgres/Redis/ML.
Flip it off layer-by-layer as each is implemented. Computed
`DATABASE_URL` / `DATABASE_URL_SYNC` (Alembic) / `REDIS_URL` are already
defined for when persistence lands.

### 3.3 Persistence — `app/db/` *(placeholder)*

Target: `base.py` (declarative `Base`), `session.py` (async engine +
`get_db_session`), `models/<res>.py` (one entity per file, UUID PKs,
`created_at/updated_at` via `server_default=func.now()`, every model
imported in `models/__init__.py` for Alembic autogenerate). Migrations:
**autogenerate only**, applied by `scripts/init.sh` before Uvicorn.
Files exist as commented placeholders pointing here.

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
serves it **verbatim** and must never reshape graded columns.

**As-built:** `routes/jobs.py` fakes progress on each poll from an in-memory
dict and returns a stub CSV header. Target: persist jobs in Postgres, run
the ML call out-of-band (§5.3), frontend polls `GET /jobs/{id}`.

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

### 3.9 Cache — `app/cache/` *(placeholder)*

Target: `RedisCache.get_or_set_json(key, fetch, ttl)`; cache only
GET-list/status reads, invalidate on any write; transparent passthrough
when `CACHE_ENABLED=false`. Commented placeholder points here.

### 3.10 Migrations — Alembic *(target)*

Autogenerate only (`uv run alembic revision --autogenerate -m "..."` →
`upgrade head`); never hand-edit. `scripts/init.sh` runs `upgrade head`
before serving (the migrate→serve contract is visible there as a no-op
until persistence lands).

---

## 4. Frontend

### 4.1 Principle

SPA on React + Vite. One `src/api/<res>.ts` module per backend resource.
DTO types are **generated** from the backend's OpenAPI, never hand-written
(they are stubbed only until the backend is real). Pages grouped by feature
under `src/pages/`. Route access gated by layout components.

**As-built:** one placeholder page (`pages/Home.tsx`) + a sketched `api/`
seam (`client.ts`, `jobs.ts`). Everything below is the target.

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
POST {ML_BASE_URL}/process   {video_path, job_id}  → 200 {csv, rows, meta?}
GET  {ML_BASE_URL}/health                          → 200 {"status":"ok"}
```

`csv` is the full 29-column submission text
([`hackathon/task.md`](./hackathon/task.md)); the backend persists and
serves it unchanged.

### 5.3 Sync vs. async

Video processing is **minutes-long**, so the honest target is the async
pattern: backend creates a job, runs the ML call out-of-band, frontend
polls `GET /api/v1/jobs/{id}`. The contract in §5.2 stays identical — only
*who waits* changes. The skeleton mocks it synchronously (instant fake CSV)
so the flow is reviewable; introduce a worker/queue only when the real
pipeline forces it.

### 5.4 Files

`backend/app/ml/{schemas,client}.py` — contract + the single httpx seam
(mocked: returns a fake CSV; real body present as a comment).
`ml/app/{contract,runner,main}.py` — contract mirror + the
pipeline bridge (`runner.py`, mocked; real `PriceTagPipeline().run(...)`
call present as a comment) + the FastAPI app.

---

## 6. docker-compose

`docker compose up --build` runs the whole product: `backend:8000`
(`/docs` = Swagger), `frontend:5173`, `ml:8002`, `db:5432`, `redis:6379`.
A named `uploads` volume is shared between `backend` and `ml` for the video
handoff (§5). The root `./Dockerfile` is the **separate CUDA training
image** and is intentionally not part of compose. In the skeleton the
backend runs `MOCK_MODE=true`, so db/redis come up but aren't yet used.

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
RabbitMQ (unless §5.3 forces a queue), no S3/MinIO (a shared volume covers
the video handoff), no history/audit/rate-limit, no Storybook. What stays
is the clean spine: layered backend + async SQLAlchemy + Alembic + Redis +
(optional) auth; typed frontend with a generated contract; ML as a
separate service behind a thin client.

## 9. Build order (from this skeleton)

1. **Decide auth in/out** (§3.8) — it sizes everything else.
2. Backend persistence: implement `db/`, the `Job` model, swap the
   in-memory store in `routes/jobs.py` for Postgres; Alembic init + first
   migration; flip `MOCK_MODE` for the DB path.
3. ML service: enable the real `PriceTagPipeline` call in `ml/app/runner.py`,
   add the pipeline path-dependency, switch the `ml` image to the GPU base.
4. Wire the real `backend/app/ml/client.py` httpx call; choose §5.3
   sync/async; add a worker if async.
5. Cache: implement `app/cache/`, cache the status/list reads.
6. Frontend: real axios client (§4.2), `generate:types`, build the
   upload→poll→download page, then layouts/auth if §1 said auth is in.
7. End-to-end: upload a real video → poll → download a real CSV; validate
   against [`pipeline-reference.md`](./pipeline-reference.md).

---

**Summary:** the skeleton is the proven shape — layered gateway, typed SPA,
ML as a separate service behind a thin client, the model kept in
`projects/price_tag_pipeline/`. Nothing here is implemented yet by design;
fill it in along §9, keeping this doc honest as you go.
