# Documentation index — Lenta Tech Life 2026

End-to-end price-tag recognition from robot-captured supermarket video:
detect price tags on shelves and extract structured fields per tag, output one
CSV row per unique tag, behind an upload-video → download-CSV UI.

This is the **single source of truth**. The root `README.md`, `AGENTS.md`,
`CLAUDE.md`, and the nested `data/README.md` / `projects/.../README.md` are
thin pointers here — do not duplicate knowledge into them.

> **Branch note.** `main` is the canonical, most-complete branch (it now
> supersedes `feature/full-autonomous-demo`). Older runbooks that say
> `git switch feature/full-autonomous-demo` still work, but `main` is the
> recommended target. See [`branches.md`](./branches.md).

> **Monorepo note.** The model (training/experiments) lives in
> `projects/price_tag_pipeline/`; the product around it —
> `backend/` (API gateway), `frontend/` (SPA), `ml/` (deployable service
> wrapping the pipeline) + `docker-compose.yaml` — is described in
> **[architecture.md](./architecture.md)**. `backend`/`ml` are a reviewable
> skeleton (mocked, real logic not written yet) by design; the `frontend`
> is the built upload→poll→review→CSV SPA wired against that mock.

## Read in this order

1. **[hackathon/task.md](./hackathon/task.md)** — the official task: the 29-column
   CSV schema, the metric, constraints, deliverables, timeline. **Authoritative.**
2. **[hackathon/briefing.md](./hackathon/briefing.md)** — organizer-chat intel:
   metric deep-dive, scoring pitfalls, "best frame" rules, manual-labeling rule
   change, competitor map, the agent checklist. Read for *why* and *gotchas*.
3. **[hackathon/price-tag-guide.md](./hackathon/price-tag-guide.md)** — **the
   OCR field-layout bible.** Where every CSV field physically sits on each tag
   type, mechanic-by-mechanic; the `discount_amount` <100₽→% / ≥100₽→₽ rule;
   barcode-as-text (`ШК:`) vs graphical; which formats have no QR/barcode;
   shelf-talker = same logical tag. **Read before touching OCR/parsing.**
4. **[strategy.md](./strategy.md)** — model & pipeline strategy
   (RF-DETR / YOLO26 / PaddleOCR-VL / BoT-SORT). §10 maps every old assumption
   to the now-known ground truth.
5. **[pipeline-reference.md](./pipeline-reference.md)** — current code: CLI
   entry points, runtime profiles, OCR/detector backends, output schema.
5b. **[recognition-pipeline.md](./recognition-pipeline.md)** — the crop→fields
   chain (QR→barcode→smart OCR), the `CropDecoder` seam, and the contract the
   separate QR/barcode branch plugs into. Read before touching recognition.
6. **[analysis.md](./analysis.md)** — *historical* cold critique of the
   pre-rewrite scaffold; failure-mode catalogue and rationale.

## By topic

| Need | Doc |
|---|---|
| What exactly to build & how it's scored | [hackathon/task.md](./hackathon/task.md), [hackathon/briefing.md](./hackathon/briefing.md) |
| **App/service architecture (backend + frontend + ML)** | [architecture.md](./architecture.md) |
| **Where each field sits on the tag (OCR)** | [hackathon/price-tag-guide.md](./hackathon/price-tag-guide.md) |
| Raw organizer decks (heavy — don't open) | [hackathon/source-materials.md](./hackathon/source-materials.md) |
| Which model / why / training plan | [strategy.md](./strategy.md) |
| **Concrete ML quality improvement plan** | [ml-quality-improvement-plan.md](./ml-quality-improvement-plan.md) |
| Run inference / CSV / UI / CLI flags | [pipeline-reference.md](./pipeline-reference.md) |
| **Crop→fields chain + QR/barcode seam contract** | [recognition-pipeline.md](./recognition-pipeline.md) |
| Optional product/facing shelf-state layer | [shelf-analytics.md](./shelf-analytics.md) |
| Detector experiments (fine-tune OFF base) | [`../projects/price_tag_pipeline/experiments/`](../projects/price_tag_pipeline/experiments/README.md) |
| **`import cv2` fails / new worktree env setup** | [runbooks/venv-setup.md](./runbooks/venv-setup.md) |
| Local annotated-video + CSV runbook | [runbooks/local.md](./runbooks/local.md) |
<<<<<<< HEAD
| Current inference issues + fix plan | [runbooks/inference-current-issues.md](./runbooks/inference-current-issues.md) |
| **Hand-label / extend the dataset (CVAT)** | [runbooks/annotation-cvat.md](./runbooks/annotation-cvat.md) |
| Colab — zero-shot GPU baseline | [runbooks/colab-baseline.md](./runbooks/colab-baseline.md) |
| Colab — full pipeline on real videos | [runbooks/colab-real-data.md](./runbooks/colab-real-data.md) |
| Colab — private-repo SSH clone cells | [runbooks/colab-private.md](./runbooks/colab-private.md) |
| Where data goes when it arrives | [data/layout.md](./data/layout.md) |
| External datasets — curated shortlist | [data/datasets.md](./data/datasets.md) |
| External + synthetic detector training plan | [data/external-detector-training-plan.md](./data/external-detector-training-plan.md) |
| External datasets — deep research + extensions | [data/datasets-research.md](./data/datasets-research.md) |
| Killer feature (stretch): shelf facings/OOS | [strategy.md](./strategy.md) §12 |
| What each git branch is for | [branches.md](./branches.md) |
| Repo-wide orientation for AI agents | [`../AGENTS.md`](../AGENTS.md) |

## The five facts that change decisions

These are extracted from the hackathon docs because they are easy to get wrong
and they reshape priorities:

1. **Metric = "≥80% of *substantive* fields correct per GT-matched tag"**, then
   share of such tags over all GT tags. Technical fields (`filename`,
   `frame_timestamp`, bbox) are *not scored* — only used for matching.
2. **Barcode is the primary matching key.** A row with no/incorrect barcode
   falls back to noisy `frame_timestamp+bbox` matching → may not match at all.
   Barcode recognition is **P0**, not just one field.
3. **Duplicates hurt.** One physical tag emitted N times → one row matches GT,
   the rest are unmatched noise. Tracking + cross-track dedup is **critical**
   and is explicitly the participant's responsibility.
4. **Cloud APIs are banned at inference.** Locally deployable only; lightweight
   / `rknn int8` is rewarded. Heavy training off-box is fine; the *shipped*
   pipeline must run fully local.
5. **`"нет"` vs empty are different.** Field absent on the tag → write `"нет"`.
   Field present but not recognized → leave empty. Confusing them loses points.

## Killer feature (stretch goal)

A planned **shelf-analytics layer** — report per-product **visible facings**
(how many units are on the shelf) + optional empty-slot/gap flags, turning the
tag CSV into a shelf-state report. This targets the *non-metric* finals
criteria (maturity, scalability, business applicability), **not** the technical
score. It is attempted **only if the core graded pipeline is solid**, and its
output stays physically separate from the graded 29-column CSV so it cannot
corrupt the metric. Spec: [strategy.md](./strategy.md) §12; dataset backing:
[data/datasets-research.md](./data/datasets-research.md) (Locount, SKU-110K,
Gap/ROSCH).

## Conventions (apply to all work here)

- **No global pip.** Install into a `uv`-managed `.venv`, never globally.
- **`frame_timestamp` is milliseconds** from video start — never a frame index.
- **Non-ASCII paths must keep working** (the build machine may sit under a
  Cyrillic Windows username). Still-image I/O goes through
  `price_tag_pipeline.cv_io`; never call `cv2.imread/imwrite` directly.
- **Fix root causes, explain them** — don't paper over bugs.
- Dataset is tiny (5 labeled videos, ~274 tag rows) → external data + synthetic
  tags are the main quality lever; see [data/datasets.md](./data/datasets.md).
