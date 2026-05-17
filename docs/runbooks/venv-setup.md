# venv setup — make OpenCV (and the pipeline) actually import

**Read this the moment `import cv2` fails.** It always means the same thing.

## Why it keeps breaking

The repo ships **no committed `.venv`** (it's gitignored). A fresh checkout —
**and every new git worktree** — therefore has no environment, so `python`
resolves to the **global Python 3.14**, where `cv2` is not installed and will
not install cleanly (3.14 is too new for the opencv / ultralytics / torch wheel
ecosystem). Installing globally is also forbidden by the project rules.

Fix: give *this* working tree its own `uv`-managed `.venv` on **Python 3.12**.
Each worktree needs its own — `.venv` is not shared across them.

## The one-time setup (run from the worktree root)

```bash
# 3.12 is already installed locally (C:\Python312) — no download.
uv venv --python 3.12.6 .venv

# Lightweight core: makes `import cv2`, cv_io, the data layer, and frame
# extraction work. ~40 MB opencv wheel — the raised timeout is mandatory,
# the default 30 s makes it hang on slow networks.
UV_HTTP_TIMEOUT=600 uv pip install -p .venv/Scripts/python.exe \
  "opencv-python>=4.9,<5.0" "numpy>=1.24,<2.3" "Pillow>=10.0" "PyYAML>=6.0"
```

Verify:

```bash
.venv/Scripts/python.exe -c "import cv2; print(cv2.__version__)"   # -> 4.13.0
```

Then **always invoke `.venv/Scripts/python.exe`** (not bare `python`) for any
script in this tree, or activate first: `source .venv/Scripts/activate`.

## Need the full pipeline (detector / OCR / training), not just OpenCV

Add the heavy stack into the same venv (pulls torch + ultralytics — large,
slow, do it only when you actually run inference/training):

```bash
UV_HTTP_TIMEOUT=600 uv pip install -p .venv/Scripts/python.exe \
  -r projects/price_tag_pipeline/requirements/base.txt
```

`requirements/{ocr,train,dev,demo}.txt` layer on top; the umbrella
`requirements.txt` is everything.

## Gotchas (don't relearn these the hard way)

- **One venv per worktree.** A sibling worktree's working `.venv` does nothing
  for yours.
- **No concurrent `uv pip install` into the same venv** — lock contention
  stalls both. Serialize them.
- **Never `cv2.imread/imwrite`** on non-ASCII paths — route still-image I/O
  through `price_tag_pipeline.cv_io` (cv2 *video* I/O is fine).
- The `Failed to hardlink … falling back to full copy` uv warning is harmless
  here (cache and tree are on different filesystems).
- Pipeline runs should read the **main checkout's** `data/raw`, not an
  ephemeral worktree's. See [data/layout.md](../data/layout.md).
