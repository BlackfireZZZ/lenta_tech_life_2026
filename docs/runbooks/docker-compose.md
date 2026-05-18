# Runbook — full product bring-up with docker-compose

One command brings up the whole product for manual end-to-end testing:
frontend + backend gateway + the **real GPU ML** service + Postgres + Redis.
This is the companion to [`architecture.md`](../architecture.md) §6 (the
*why*); this file is the *how*.

## What comes up

| Service  | URL / port            | Notes |
|----------|-----------------------|-------|
| frontend | http://localhost:5173 | React SPA (Vite dev server, HMR) |
| backend  | http://localhost:8000 — `/docs` = Swagger | gateway, `MOCK_MODE=true` |
| ml       | http://localhost:8002 — `/health` | **real** CUDA + Qwen3-VL pipeline |
| db       | localhost:5432        | Postgres (up, not used yet) |
| redis    | localhost:6379        | Redis (up, not used yet) |

## Prerequisites (Windows)

- Docker Desktop with the **WSL2** backend (not Hyper-V).
- An NVIDIA GPU + recent driver. Nothing else to install — Docker Desktop
  passes the GPU through WSL2; the NVIDIA Container Toolkit is **not**
  required on Windows. (Verified target: RTX 4070 Ti / driver 591.86 /
  Docker 29.4.2 — Qwen3-VL-4B needs ~8.5 GB VRAM, fits 12 GB.)
- `backend/.env` must exist (compose reads it via `env_file`). It is
  gitignored; copy it from the template if missing:
  `cp backend/.env.example backend/.env`.

Optional GPU sanity check before a full build:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## Bring it up

```bash
docker compose up --build
```

- **First build is slow & large.** The `ml` image is a CUDA base + torch +
  transformers + the pipeline stack (~10 GB). That is *dependencies*, not
  weights — unavoidable for GPU inference, built once and layer-cached.
- **Weights are NOT in the image.** On the first real `/process` call the
  service downloads the Qwen3-VL-4B (~8 GB) and the OpenFoodFacts detector
  into the `hf_cache` volume (`HF_HOME=/models/hf`). This is a one-time
  download; it persists across container restarts **and image rebuilds**,
  and never enters an image layer. Subsequent runs are instant.

## Verify each service

```bash
curl http://localhost:8002/health      # {"status":"ok","mode":"real"}  ← GPU pipeline live
curl http://localhost:8000/api/v1/health
# open http://localhost:5173 and http://localhost:8000/docs
```

`"mode":"real"` confirms the pipeline imported. `"mode":"mock"` means a
heavy import failed — check `docker compose logs ml`.

## Mock escape hatch (no GPU / fast boot)

Set `ML_MOCK=1` on the `ml` service (compose `environment:`) to serve the
deterministic mock CSV without loading any model. The service still boots
and the contract is identical — useful when iterating on something else.

## Scope note — what "end-to-end" means today

`backend` runs `MOCK_MODE=true`: it serves a deterministic mock job/CSV and
**does not call `ml` yet** (the backend↔ML httpx wiring lands after the ML
work is pulled into `main`). So today:

- **Frontend e2e is fully exercisable now** — upload → poll → review
  (video + bbox + crop + fields) → CSV download — against the backend mock.
  `ml` being real does not change this path.
- **True `frontend → backend → ml → real CSV`** becomes available once the
  backend↔ML call is enabled (flip `MOCK_MODE=false`). The `ml` side and
  all Docker plumbing are already real and ready for that switch.
