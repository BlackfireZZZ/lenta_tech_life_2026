"""Bridge from the HTTP contract to the real price-tag pipeline.

This is the ONLY seam between the deployable service and the research
codebase. The model lives in the ``price_tag_pipeline`` package
(``projects/price_tag_pipeline/``); here we call its public entry point
(``PriceTagPipeline(cfg).run``) and render the graded CSV with
``submission.final_tags_to_csv``. Keeping the seam this thin means
training/experiment churn there never breaks the service contract.

Real vs. mock (so the monorepo still boots without GPU/weights — the
architecture.md §5 principle):

* ``ML_MOCK=1``                  → always the fake CSV.
* pipeline import fails          → fall back to the fake CSV, logged once.
* otherwise                      → the real pipeline runs.

Config: ``ML_PIPELINE_CONFIG`` (default ``configs/balanced.yaml``).
Progress is streamed into ``progress_registry`` keyed by ``job_id`` for the
additive ``GET /progress/{job_id}`` endpoint — the ``ProcessRequest`` /
``ProcessResponse`` contract itself is unchanged.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

from app.contract import ProcessRequest, ProcessResponse
from app import progress_registry

LOGGER = logging.getLogger("ml.runner")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PIPELINE_DIR = _REPO_ROOT / "projects" / "price_tag_pipeline"
_PIPELINE_SRC = _PIPELINE_DIR / "src"
if str(_PIPELINE_SRC) not in sys.path:
    # Import the package without installing it — same convention the
    # scripts/ entry points use. Heavy deps (ultralytics, ...) only load
    # when run() actually touches the detector, so import stays cheap.
    sys.path.insert(0, str(_PIPELINE_SRC))

_MOCK_CSV = "filename,frame_timestamp,x1,y1,x2,y2,barcode,name,price,...\n"

# "fast" recognition: how many sharpest crops per tag the heavy VLM OCR
# sees. 1 = just the single best frame per tag (vs balanced.yaml's 5).
_FAST_TOP_K = 1


def _video_meta(video_path: str) -> dict:
    """Raw-clip geometry for the gateway's *non-graded* review overlay.

    The graded CSV's pixel bbox is in **original-frame** coordinates (the
    detector rotates only for the model, then un-projects boxes back —
    pipeline.py / detector.unrotate_box_xyxy). The browser plays that same
    raw clip, so the gateway only needs raw W/H to normalise the box and the
    real duration to map ``frame_timestamp`` (ms) → a seek fraction.

    Best-effort and fully guarded: this rides in ``meta`` (non-graded,
    additive — architecture.md §5.2). Any failure → ``{}``; it must never
    affect the graded result.
    """
    try:
        import cv2  # available on the real ML image

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {}
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        out: dict = {}
        if w > 0 and h > 0:
            out["frame_width"] = w
            out["frame_height"] = h
        if fps > 1.0 and n > 0:
            out["video_duration_s"] = round(n / fps, 3)
        return out
    except Exception as exc:  # pragma: no cover - never fatal
        LOGGER.warning("video meta probe failed: %s", exc)
        return {}

# Cached: did the pipeline import succeed? (None = not yet probed.)
_real_available: bool | None = None


def _mock_forced() -> bool:
    return os.getenv("ML_MOCK", "").strip().lower() in {"1", "true", "yes", "on"}


def _config_path() -> Path:
    env = os.getenv("ML_PIPELINE_CONFIG", "").strip()
    if env:
        p = Path(env).expanduser()
        return p if p.is_absolute() else (_REPO_ROOT / p)
    return _PIPELINE_DIR / "configs" / "balanced.yaml"


def current_mode() -> str:
    """'mock' or 'real' — what the next /process call will do. For /health."""
    if _mock_forced():
        return "mock"
    return "real" if _probe_real() else "mock"


def _probe_real() -> bool:
    """True if the pipeline package imports. Cached; never raises."""
    global _real_available
    if _real_available is None:
        try:
            import price_tag_pipeline.pipeline  # noqa: F401
            import price_tag_pipeline.submission  # noqa: F401

            _real_available = True
        except Exception as exc:  # missing heavy deps → stay mock, but boot
            LOGGER.warning("Pipeline import failed; serving MOCK CSV: %s", exc)
            _real_available = False
    return _real_available


# --- Lenta catalog reconciliation (GT-safe post-step) ----------------------
# Repairs `barcode`/`product_name` against the Lenta master catalog
# (real_data/db_hack.csv). It is a *post-step on the produced CSV*, not part
# of PriceTagPipeline (docs/catalog-reconciliation.md). It re-renders through
# the project's own producer renderer so the graded byte format is unchanged,
# uses the conservative fill-only policy (never overwrites a present name or
# the "нет" sentinel), and is fully guarded: a missing mount or any failure
# serves the raw CSV — it can never break the graded run.
#
# The catalog is mounted (compose) at $CATALOG_CSV (default
# /data/catalog/db_hack.csv). The ~625k-row index is built once per process
# and reused across jobs. `False` = probed and unavailable.
_catalog_index: object | None = None


def _get_catalog_index():
    global _catalog_index
    if _catalog_index is not None:
        return _catalog_index or None

    raw = os.getenv("CATALOG_CSV", "/data/catalog/db_hack.csv").strip()
    path = Path(raw) if raw else None
    if path is None or not path.is_file():
        _catalog_index = False
        LOGGER.info(
            "catalog CSV not mounted (CATALOG_CSV=%s) — reconciliation off; "
            "to enable, mount real_data/db_hack.csv there", raw,
        )
        return None
    try:
        from price_tag_pipeline.catalog import CatalogIndex

        # Cache the pickled index on the persistent, writable HF volume so a
        # restart doesn't rebuild it (the CSV mount is read-only, so we can't
        # cache next to it).
        cache_dir = Path(os.getenv("HF_HOME", "/tmp"))
        cpath = cache_dir / "db_hack.idxcache.pkl"
        t0 = time.time()
        idx = CatalogIndex.load(path, cache=True, cache_path=cpath)
        LOGGER.info(
            "catalog loaded: %d rows in %.1fs", len(idx), time.time() - t0
        )
        _catalog_index = idx
        return idx
    except Exception as exc:  # noqa: BLE001 - never fatal
        LOGGER.warning(
            "catalog load failed (%s) — reconciliation off: %s", path, exc
        )
        _catalog_index = False
        return None


def _apply_catalog(csv_text: str) -> tuple[str, str]:
    """(possibly reconciled CSV, status). Never raises; always GT-safe."""
    idx = _get_catalog_index()
    if idx is None:
        return csv_text, "skipped"
    try:
        import csv as _csv
        import io as _io

        from price_tag_pipeline.catalog import CatalogReconciler
        from price_tag_pipeline.submission import (
            HACK_CSV_COLUMNS,
            hack_rows_to_csv_text,
        )

        reader = _csv.DictReader(_io.StringIO(csv_text))
        if reader.fieldnames != HACK_CSV_COLUMNS:
            # Degraded/mock CSV — leave it exactly as produced.
            return csv_text, "skipped:schema"

        rec = CatalogReconciler(idx)  # name_policy='fill' default = GT-safe
        rows = []
        changed = 0
        for row in reader:
            r = rec.reconcile(row.get("barcode"), row.get("product_name"))
            if r.changed:
                changed += 1
            row["barcode"] = r.barcode if r.barcode is not None else ""
            row["product_name"] = (
                r.product_name if r.product_name is not None else ""
            )
            rows.append(row)
        # Re-render through the *producer* renderer → byte-identical contract
        # (29 cols, '\n', QUOTE_MINIMAL); only barcode/product_name differ.
        out = hack_rows_to_csv_text(rows)
        return out, f"applied:{changed}/{len(rows)}"
    except Exception as exc:  # noqa: BLE001 - graded run must survive
        LOGGER.warning(
            "catalog reconciliation failed — serving raw CSV: %s", exc
        )
        return csv_text, "error"


# --- Local-weights-or-download resolution --------------------------------
# Policy: prefer weights already on disk (bind-mounted by docker-compose),
# else fall back to whatever the config says (the HF refs in balanced.yaml,
# which Ultralytics / transformers then download into the hf_cache volume
# once). The config (balanced.yaml) stays the single canonical profile; this
# only swaps the two model locations when a complete local copy is present.
_DEF_LOCAL_DETECTOR = "/models/local/detector.pt"
_DEF_LOCAL_VLM = "/models/local/qwen3-vl-4b"


def _vlm_dir_complete(d: Path) -> bool:
    """A usable local HF model dir: a config + at least one weight shard.
    (Guards the docker bind-mount footgun where a missing source becomes an
    empty dir — incomplete ⇒ treated as absent ⇒ download fallback.)"""
    return (
        d.is_dir()
        and (d / "config.json").is_file()
        and any(d.glob("*.safetensors"))
    )


def _resolve_local_weights(cfg):
    """Return (cfg, {detector,vlm}: 'local'|'download'). Never raises.

    Frozen dataclasses → rebuild via ``dataclasses.replace``. Only the
    locations that have a complete local copy are swapped; the rest keep the
    config's value (HF ref → downloaded once into hf_cache).
    """
    from dataclasses import replace

    det_src, vlm_src = "download", "download"
    det_path = Path(os.getenv("LOCAL_DETECTOR", _DEF_LOCAL_DETECTOR))
    vlm_path = Path(os.getenv("LOCAL_VLM", _DEF_LOCAL_VLM))

    detector = cfg.detector
    ocr = cfg.ocr
    if det_path.is_file():
        detector = replace(cfg.detector, model_path=str(det_path))
        det_src = "local"
    if _vlm_dir_complete(vlm_path):
        ocr = replace(cfg.ocr, vlm_model=str(vlm_path))
        vlm_src = "local"

    if det_src == "local" or vlm_src == "local":
        cfg = replace(cfg, detector=detector, ocr=ocr)
    LOGGER.info(
        "weights: detector=%s (%s), vlm=%s (%s)",
        det_src, detector.model_path, vlm_src, ocr.vlm_model,
    )
    return cfg, {"detector": det_src, "vlm": vlm_src}


# --- Startup warm-up ------------------------------------------------------
# Loading Qwen3-VL-4B is ~8 GB / tens of seconds. Lazily it happened on the
# *first crop OCR of the first upload* — the user sat watching a stalled
# bar. We instead trigger that load at service start (a background thread —
# see app/main.py), so the model is resident before anyone uploads. The
# process-wide VLM cache (recognition/ocr.py) makes this a one-time cost
# shared by every later request, regardless of per-upload rotation (which
# only touches the cheap detector, never the VLM).
_warm_state = "cold"  # cold | warming | ready | skipped | error


def warm_state() -> str:
    return _warm_state


def warmup() -> None:
    """Force the heavy VLM load now (idempotent, never raises).

    Drives one decode on a blank frame through the *public* recognition
    chain — that runs ``_ensure_loaded`` and fills the process VLM cache. A
    failure just leaves the old lazy path in place for the first real job.
    """
    global _warm_state
    if _mock_forced() or not _probe_real():
        _warm_state = "skipped"
        LOGGER.info("warmup skipped (mock/pipeline unavailable)")
        return
    try:
        _warm_state = "warming"
        from price_tag_pipeline.config import load_config
        from price_tag_pipeline.recognition.ocr import build_ocr_engine

        cfg = load_config(_config_path())
        cfg, _ = _resolve_local_weights(cfg)  # use the same (local) VLM
        engine = build_ocr_engine(cfg.ocr)
        t0 = time.time()
        # Load the model directly — robust regardless of chain/decode
        # heuristics (a blank-image decode can be skipped before the load).
        ensure = getattr(engine, "_ensure_loaded", None)
        if callable(ensure):
            ensure()
        else:  # engine without an explicit loader → tiny real call
            import numpy as np

            engine.recognize_all(np.full((600, 800, 3), 255, np.uint8))
        _warm_state = "ready"
        LOGGER.info("warmup complete in %.1fs — VLM resident", time.time() - t0)
    except Exception as exc:  # noqa: BLE001 - never fatal; lazy path remains
        _warm_state = "error"
        LOGGER.warning("warmup failed (lazy-load on first job): %s", exc)


def run_pipeline(req: ProcessRequest) -> ProcessResponse:
    if _mock_forced() or not _probe_real():
        progress_registry.record(req.job_id, {
            "phase": "done", "fraction": 1.0, "message": "mock", "mock": True,
        })
        return ProcessResponse(csv=_MOCK_CSV, rows=0,
                               meta={"mock": True, "job_id": req.job_id})

    from price_tag_pipeline.config import load_config
    from price_tag_pipeline.pipeline import PriceTagPipeline
    from price_tag_pipeline.progress import ProgressEvent
    from price_tag_pipeline.submission import final_tags_to_csv

    def _on_progress(ev: ProgressEvent) -> None:
        progress_registry.record(req.job_id, ev.as_dict())

    cfg_path = _config_path()
    started = time.time()
    LOGGER.info("job=%s start video=%s config=%s", req.job_id, req.video_path, cfg_path)

    cfg = load_config(cfg_path)
    cfg, weights_src = _resolve_local_weights(cfg)

    # Smoother progress bar. The pipeline emits a ProgressEvent every
    # runtime.log_every_n_frames frames; balanced.yaml uses 50, so on a
    # few-hundred-frame clip the bar jumps ~10% at a time. This knob is
    # purely report/log cadence — zero effect on detection/OCR quality — so
    # the service tightens it (env ML_PROGRESS_EVERY_N, default 10) without
    # touching the canonical config. NOTE: fraction is frame-based; OCR/VLM
    # runs in bursts at track finalization, so equal fraction steps can
    # still arrive at uneven wall-clock intervals — inherent, not a bug.
    try:
        n = int(os.getenv("ML_PROGRESS_EVERY_N", "10") or 0)
    except ValueError:
        n = 0
    if n > 0 and cfg.runtime.log_every_n_frames != n:
        from dataclasses import replace

        cfg = replace(cfg, runtime=replace(cfg.runtime, log_every_n_frames=n))

    # Detector-only frame pre-rotation, chosen in the UI (none|ccw|cw). This
    # ONLY changes how the model sees frames; the detector un-projects boxes
    # back, so the stored video, the review playback and the graded CSV
    # coords stay in the original orientation. The request is authoritative
    # for the product path (the canonical balanced.yaml ccw is for the
    # offline CLI benchmark). Unknown value → keep the config's setting.
    rot = str(getattr(req, "rotation", "none") or "none").strip().lower()
    if rot in ("none", "ccw", "cw") and rot != cfg.detector.frame_rotation:
        from dataclasses import replace

        cfg = replace(cfg, detector=replace(cfg.detector, frame_rotation=rot))
    LOGGER.info("detector frame_rotation = %s", cfg.detector.frame_rotation)

    # Recognition depth, chosen in the UI ("full" | "fast"). The slow part of
    # a run is the per-crop Qwen3-VL OCR: balanced.yaml decodes the top-K
    # sharpest crops *per tag* (K=5) and the aggregator votes across them.
    # "fast" caps that at the single sharpest crop (K=1) → ~5× fewer VLM
    # calls, a bit less voting redundancy. Detection/tracking are untouched
    # (they are cheap — the user explicitly does not want them cut). Unknown
    # / "full" → no override, the canonical config stands.
    mode = str(getattr(req, "mode", "full") or "full").strip().lower()
    if mode == "fast" and cfg.ocr.top_k_crops_per_track > _FAST_TOP_K:
        from dataclasses import replace

        cfg = replace(
            cfg, ocr=replace(cfg.ocr, top_k_crops_per_track=_FAST_TOP_K)
        )
    LOGGER.info(
        "recognition mode=%s top_k_crops_per_track=%d",
        mode, cfg.ocr.top_k_crops_per_track,
    )

    tags = PriceTagPipeline(cfg).run(req.video_path, progress=_on_progress)

    # task.md §3.4: the released Lenta CSVs use a bare stem as `filename`.
    # Prefer the original upload name from the gateway — the bytes on disk
    # are an ASCII-safe "source.<ext>", so video_path.stem would emit the
    # wrong `filename` for every video and break GT matching.
    filename = Path(req.filename or req.video_path).stem
    csv_text = final_tags_to_csv(tags, filename=filename)
    csv_text, catalog_status = _apply_catalog(csv_text)
    elapsed = round(time.time() - started, 2)
    LOGGER.info(
        "job=%s done rows=%d catalog=%s elapsed=%ss",
        req.job_id, len(tags), catalog_status, elapsed,
    )

    return ProcessResponse(
        csv=csv_text,
        rows=len(tags),
        meta={
            "job_id": req.job_id,
            "mock": False,
            "config": cfg_path.name,
            "elapsed_s": elapsed,
            "catalog": catalog_status,
            "weights": weights_src,
            "rotation": cfg.detector.frame_rotation,
            "mode": mode,
            "top_k_crops_per_track": cfg.ocr.top_k_crops_per_track,
            **_video_meta(req.video_path),
        },
    )
