# frontend/ — React + Vite SPA

Single-page app: upload a shelf video → poll status → download the CSV.

**Status: placeholder.** Renders one stub page (`src/pages/Home.tsx`); the
`api/` seam is sketched but not wired. The real structure — axios client
(interceptors), `AuthContext`, layout routes, OpenAPI-generated DTO types,
one API module per resource — is fully specified in
**[`../docs/architecture.md`](../docs/architecture.md) §4**. Build it from
there; don't duplicate that knowledge here.

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```
