# frontend/ — React + Vite SPA

Upload a shelf video → poll job status → **review** (source video with
bounding-box overlay, the price-tag crop at its predicted timestamp, the
recognized fields grouped and labelled) → download the 29-column CSV.

**Status: built, on the real gateway.** Tailwind v4 + the project design
tokens + shadcn-style primitives (`src/components/ui/`), React Router with
lazy pages, `src/api/` axios seam. Anonymous — no auth. In `docker compose`
it consumes the real backend end to end (upload → live progress → review →
CSV). The full target architecture (axios auth client, OpenAPI-generated
DTOs, layouts) is the single source of truth in
**[`../docs/architecture.md`](../docs/architecture.md) §4** — read it there,
don't duplicate it here.

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173  (proxies /api → :8000)
```

The dev server proxies `/api` to the backend gateway (`VITE_API_PROXY`,
default `http://localhost:8000`) so the video stream and the canvas crop
stay same-origin. Start the gateway first (`backend/README.md`). For a
deployed build, set `VITE_API_BASE_URL` to the gateway origin.

DTO types in `src/api/jobs.ts` mirror the backend schema by hand; once a
gateway is reachable, `npm run generate:types` regenerates them from
`/openapi.json` (`openapi-ts.config.ts`).
