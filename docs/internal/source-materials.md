# Source materials registry — heavy organizer decks

> **⚠️ Agents: do NOT open the files listed below.** They are 0.4–10 MB binary
> PDF/PPTX (PPTX = zip archives of full-page JPEG renders). Reading them is
> slow, expensive, and usually impossible in-context. **Everything in them has
> already been distilled into agent-readable markdown** — read the distilled
> doc instead. This page exists only so a human knows what the raw sources are
> and where they physically live.

## Where they live

`real_data/materials/` in the repo root. **Not in git** — `real_data/` is
`.gitignore`d (project convention: it is the untouched local source of truth
and is never committed; see [`../data/layout.md`](../data/layout.md)). It is
therefore also **absent from git worktrees**. The files exist only on a
machine that has the local `real_data/` populated.

## Registry

| File | Size | What it is | Distilled into (read this) |
|---|---|---|---|
| `Расшифровка ценники.pdf` / `.pptx` | ~0.4 MB | Canonical "exploded view": 7 reference tags with per-field callouts. Organizers' **main applied material** (chat id=1615). | [`price-tag-guide.md`](../hackathon/price-tag-guide.md) §3 (field map) |
| `ГМ для ТК.pdf` / `.pptx` | 3.8 / 10.3 MB | Full template catalog: 66 slides, mechanic × size (РПЦ/АПЦ/Распродажа/Скидка ОТ·ДО/BOGOF/ШФ). | [`price-tag-guide.md`](../hackathon/price-tag-guide.md) §2,§4–§12 |
| `Задача Lenta Tech Life Hack.pdf` | 7.0 MB | The official task PDF (fields, metric, constraints, deliverables). | [`task.md`](../hackathon/task.md) + [`briefing.md`](./briefing.md) |

## How the distillation was produced

The `.pdf` files are PowerPoint archives renamed to `.pdf` — internally zip
archives containing one JPEG render per slide plus a `manifest.json`. The
guide was compiled by extracting and reading those JPEGs end-to-end (all 7 +
all 66 slides). If the decks are ever updated, re-distill into
[`price-tag-guide.md`](../hackathon/price-tag-guide.md) — **do not** add the binaries to
git; update the markdown and bump the "Status" line in that guide.
