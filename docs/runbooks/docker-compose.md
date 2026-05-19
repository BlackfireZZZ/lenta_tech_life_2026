# Runbook — full product bring-up with docker-compose

One command brings up the whole product for manual end-to-end testing:
frontend + backend gateway + the **real GPU ML** service + Postgres + Redis.
This is the companion to [`architecture.md`](../architecture.md) §6 (the
*why*); this file is the *how*.

## What comes up

| Service  | URL / port            | Notes |
|----------|-----------------------|-------|
| frontend | http://localhost:5173 | React SPA (Vite dev server, HMR) |
| backend  | http://localhost:8000 — `/docs` = Swagger | gateway, `MOCK_MODE=false` (Postgres + real ML call) |
| ml       | http://localhost:8002 — `/health` | **real** CUDA + Qwen3-VL pipeline |
| db       | localhost:5432        | Postgres (jobs/results persistence) |
| redis    | localhost:6379        | Redis (fail-open status-read cache) |

## Prerequisites (Windows)

- Docker Desktop with the **WSL2** backend (not Hyper-V).
- An NVIDIA GPU + recent driver. Nothing else to install — Docker Desktop
  passes the GPU through WSL2; the NVIDIA Container Toolkit is **not**
  required on Windows. (Verified target: RTX 4070 Ti / driver 591.86 /
  Docker 29.4.2 — Qwen3-VL-4B needs ~8.5 GB VRAM, fits 12 GB.)
- `backend/.env` is **optional** — `env_file` is `required: false` and the
  compose `environment:` block already sets every value the real path
  needs. Copy the template only to override defaults:
  `cp backend/.env.example backend/.env`.
- **Lenta catalog (optional but recommended).** Put the master catalog at
  `<this dir>/real_data/db_hack.csv` — compose mounts it read-only and the
  ML runner applies GT-safe `barcode`/`product_name` reconciliation as a
  post-step (`docs/catalog-reconciliation.md`). It is gitignored and lives
  only in the primary checkout, so copy it in:
  `cp /e/Hackatons/lenta_tech_life_2026/real_data/db_hack.csv real_data/`.
  If absent the runner logs and skips it — the graded run still works.

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

Then the real end-to-end run:

```bash
# 1. upload a real shelf clip
curl -sF "video=@/path/to/clip.mp4" http://localhost:8000/api/v1/jobs
#   -> {"id":"<job>","status":"queued",...}

# 2. poll until succeeded (progress mirrors the GPU pipeline live)
curl -s http://localhost:8000/api/v1/jobs/<job>

# 3. download the verbatim graded 29-column CSV
curl -s http://localhost:8000/api/v1/jobs/<job>/result.csv -o result.csv
```

The first real `/process` triggers the one-time weight download into
`hf_cache` (Qwen3-VL-4B ~8 GB) — the first job is slow, later ones fast.
Or just drive it from the SPA at http://localhost:5173 (upload → live
progress → review with bbox/crop/fields → **Скачать CSV**).

## Mock escape hatch (no GPU / fast boot)

Set `ML_MOCK=1` on the `ml` service (compose `environment:`) to serve the
deterministic mock CSV without loading any model. The service still boots
and the contract is identical — useful when iterating on something else.

## Scope note — what "end-to-end" means now

`backend` runs `MOCK_MODE=false`: jobs persist in Postgres and it calls the
**real** `ml` service out-of-band. The full `frontend → backend → ml →
real graded CSV` path is live — uploading a clip runs the real GPU
pipeline and the CSV you download is exactly what `price_tag_pipeline`
produced (the gateway serves it verbatim).

- The job is processed asynchronously: `POST /jobs` returns immediately;
  the gateway mirrors the pipeline's live progress into the job status
  while it runs (minutes for a real clip).
- If `real_data/db_hack.csv` is mounted, the CSV is catalog-reconciled
  (GT-safe fill-only) before it is stored — `docker compose logs ml` shows
  `catalog=applied:<changed>/<rows>` (or `skipped` when not mounted).
- Need to iterate on the frontend/gateway without the GPU? Either set
  `ML_MOCK=1` on `ml` (real contract, instant fake CSV) or
  `MOCK_MODE=true` on `backend` (fully standalone, no DB/ML).
- A `failed` job carries the error message; `docker compose logs ml` /
  `... backend` shows the detail.
