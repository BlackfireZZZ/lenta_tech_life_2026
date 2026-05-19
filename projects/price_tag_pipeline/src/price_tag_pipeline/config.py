"""Pipeline configuration. All sections are loaded from a single YAML file."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


# ---------------------------------------------------------------------------
# Dataclasses (mutable=False — config is immutable after loading)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RuntimeConfig:
    profile_name: str
    output_path: Optional[str]
    log_every_n_frames: int
    seed: int = 42
    fps_override: Optional[float] = None  # if set, ignore container FPS
    audit_path: Optional[str] = None  # JSONL audit trail per OCR call


@dataclass(frozen=True)
class DetectorConfig:
    backend: str  # 'yolo' | 'rfdetr'
    model_path: str
    conf: float
    iou: float
    device: Optional[str]
    classes: Optional[list[int]]  # None = use all classes the model exposes
    tracker_yaml: str
    open_vocab_labels: tuple[str, ...] = ()
    image_size: int = 1280
    # 90° frame rotation applied ONLY for model inference; predictions are
    # un-projected back to original-frame coords so the rest of the pipeline
    # is unchanged. The Lenta scan-robot cam is mounted 90° CW (footage is
    # sideways → the price-tag detector misses badly), so the Lenta runtime
    # profiles set this to "ccw". "none" (default) keeps legacy behavior.
    frame_rotation: str = "none"  # none | ccw | cw


@dataclass(frozen=True)
class RectifierConfig:
    padding_ratio: float
    clahe_clip_limit: float
    clahe_tile_grid_size: int
    rotate_vertical_tags: bool
    keep_color: bool = True  # apply CLAHE on luminance only, preserve RGB
    perspective: bool = False  # try 4-point quad detection + warp before falling back
    super_resolution: bool = False  # upscale tiny crops before OCR
    sr_min_area_px: int = 8_000  # crops smaller than this get the SR boost


@dataclass(frozen=True)
class EnsembleEntry:
    """One member of an OCR ensemble. Backend + optional model override."""
    backend: str
    vlm_model: Optional[str] = None
    weight: float = 1.0


@dataclass(frozen=True)
class OCRConfig:
    backend: str
    # Backends:
    #   classical: noop | tesseract | paddle
    #   VLM (transformers): paddle_vl | glm_ocr | qwen3_vl | dots_ocr |
    #                       hunyuan_ocr | rolm_ocr | monkey_ocr | intern_vl3 |
    #                       transformers_vlm
    #   production server: vllm_server (OpenAI-compatible, any vLLM-served model)
    #   pipeline: mineru
    #   ensemble: runs multiple engines, each output becomes a separate observation
    min_frames_between_ocr_per_track: int
    min_sharpness: float
    min_crop_area_px: int
    min_detection_confidence: float
    top_k_crops_per_track: int = 5
    # Sweep the cheap QR/1D decoders over MORE of the track's best crops than
    # OCR (>top_k to add anything). <=0 = OFF (default). Benchmarked on the
    # Lenta footage: NO GT barcode-recall gain (input-bound ~1%, codes barely
    # decode in motion) while it costs extra decode calls — so off by default
    # to not waste compute. Raise (e.g. 24) only if a future dataset shows a
    # win in scripts/eval_code_reading.py.
    code_decode_top_k: int = 0
    # Level-2: median-fuse this many of the track's sharpest crops into one
    # denoised image and decode THAT too (codes only). Benchmarked: also NO
    # lift on the Lenta footage (motion blur erases the code modules — fusion
    # can't synthesise unsampled detail). 0 = OFF (default); kept behind the
    # flag for if capture quality ever improves. See memory
    # level2-fusion-no-lift-input-bound.
    code_fuse_frames: int = 0
    vlm_model: str = "PaddlePaddle/PaddleOCR-VL"
    vlm_prompt_path: Optional[str] = None
    paddleocr_lang: str = "ru"
    vlm_max_new_tokens: int = 512
    vlm_temperature: float = 0.0
    vllm_url: Optional[str] = None  # e.g. http://localhost:8000/v1
    guided_json: bool = False  # vLLM/SGLang structured-output enforcement
    few_shot_examples_path: Optional[str] = None  # YAML with example crops + JSON
    ensemble_backends: tuple[EnsembleEntry, ...] = ()


@dataclass(frozen=True)
class RecognitionConfig:
    """Crop-decoder chain wiring.

    Defaults keep the full ``QR → barcode → smart OCR`` chain on, so profiles
    without a ``recognition:`` block behave exactly as before. ``merge_policy``
    is documentation/intent today (field reconciliation is the aggregator's
    weighted voting); it exists so a future policy switch is config-only.
    """
    enable_qr: bool = True
    enable_barcode: bool = True
    enable_ocr: bool = True
    merge_policy: str = "qr_first_fill_gaps"


@dataclass(frozen=True)
class ParserConfig:
    min_price: float
    max_price: float


@dataclass(frozen=True)
class AggregationConfig:
    min_observations_per_track: int
    min_final_confidence: float
    track_ttl_frames: int
    price_fuzzy_tolerance: float = 0.5    # bucket prices within ±0.5 RUB
    name_fuzzy_ratio: float = 0.85        # Levenshtein ratio threshold for name bucketing
    dedup_iou_threshold: float = 0.4
    dedup_time_window_frames: int = 150  # legacy; superseded by *_s below
    dedup_time_window_s: float = 8.0     # cross-track dedup window, wall-clock


@dataclass(frozen=True)
class TrainingConfig:
    dataset_yaml: str
    output_dir: str
    epochs: int = 60
    batch_size: int = 8
    image_size: int = 1280
    lr0: float = 1e-3
    workers: int = 4
    pretrained: Optional[str] = None
    project_name: str = "lenta-2026"
    run_name: Optional[str] = None
    seed: int = 42


@dataclass(frozen=True)
class PipelineConfig:
    runtime: RuntimeConfig
    detector: DetectorConfig
    rectifier: RectifierConfig
    ocr: OCRConfig
    parser: ParserConfig
    aggregation: AggregationConfig
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    training: Optional[TrainingConfig] = None


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _opt(d: dict[str, Any], key: str, default: Any = None) -> Any:
    return d.get(key, default)


def _as_runtime(node: dict[str, Any]) -> RuntimeConfig:
    return RuntimeConfig(
        profile_name=str(node["profile_name"]),
        output_path=_opt(node, "output_path") or _opt(node, "output_jsonl"),
        log_every_n_frames=int(node["log_every_n_frames"]),
        seed=int(_opt(node, "seed", 42)),
        fps_override=(float(node["fps_override"]) if node.get("fps_override") is not None else None),
        audit_path=_opt(node, "audit_path"),
    )


def _as_detector(node: dict[str, Any]) -> DetectorConfig:
    classes = node.get("classes")
    if classes is not None:
        classes = [int(v) for v in classes]
    labels = tuple(str(v).strip() for v in (_opt(node, "open_vocab_labels", []) or []) if str(v).strip())
    return DetectorConfig(
        backend=str(_opt(node, "backend", "yolo")).lower(),
        model_path=str(node["model_path"]),
        conf=float(node["conf"]),
        iou=float(node["iou"]),
        device=_opt(node, "device"),
        classes=classes,
        open_vocab_labels=labels,
        tracker_yaml=str(node["tracker_yaml"]),
        image_size=int(_opt(node, "image_size", 1280)),
        frame_rotation=str(_opt(node, "frame_rotation", "none")).lower(),
    )


def _as_rectifier(node: dict[str, Any]) -> RectifierConfig:
    return RectifierConfig(
        padding_ratio=float(node["padding_ratio"]),
        clahe_clip_limit=float(node["clahe_clip_limit"]),
        clahe_tile_grid_size=int(node["clahe_tile_grid_size"]),
        rotate_vertical_tags=bool(node["rotate_vertical_tags"]),
        keep_color=bool(_opt(node, "keep_color", True)),
        perspective=bool(_opt(node, "perspective", False)),
        super_resolution=bool(_opt(node, "super_resolution", False)),
        sr_min_area_px=int(_opt(node, "sr_min_area_px", 8_000)),
    )


def _as_ocr(node: dict[str, Any]) -> OCRConfig:
    entries_raw = node.get("ensemble_backends") or []
    entries = tuple(
        EnsembleEntry(
            backend=str(e["backend"]).lower().strip(),
            vlm_model=_opt(e, "vlm_model"),
            weight=float(_opt(e, "weight", 1.0)),
        )
        for e in entries_raw
    )
    return OCRConfig(
        backend=str(node["backend"]).lower().strip(),
        min_frames_between_ocr_per_track=int(node["min_frames_between_ocr_per_track"]),
        min_sharpness=float(node["min_sharpness"]),
        min_crop_area_px=int(node["min_crop_area_px"]),
        min_detection_confidence=float(node["min_detection_confidence"]),
        top_k_crops_per_track=int(_opt(node, "top_k_crops_per_track", 5)),
        code_decode_top_k=int(_opt(node, "code_decode_top_k", 0)),
        code_fuse_frames=int(_opt(node, "code_fuse_frames", 0)),
        vlm_model=str(_opt(node, "vlm_model", "PaddlePaddle/PaddleOCR-VL")),
        vlm_prompt_path=_opt(node, "vlm_prompt_path"),
        paddleocr_lang=str(_opt(node, "paddleocr_lang", "ru")),
        vlm_max_new_tokens=int(_opt(node, "vlm_max_new_tokens", 512)),
        vlm_temperature=float(_opt(node, "vlm_temperature", 0.0)),
        vllm_url=_opt(node, "vllm_url"),
        guided_json=bool(_opt(node, "guided_json", False)),
        few_shot_examples_path=_opt(node, "few_shot_examples_path"),
        ensemble_backends=entries,
    )


def _as_recognition(node: Optional[dict[str, Any]]) -> RecognitionConfig:
    if not node:
        return RecognitionConfig()
    return RecognitionConfig(
        enable_qr=bool(_opt(node, "enable_qr", True)),
        enable_barcode=bool(_opt(node, "enable_barcode", True)),
        enable_ocr=bool(_opt(node, "enable_ocr", True)),
        merge_policy=str(_opt(node, "merge_policy", "qr_first_fill_gaps")),
    )


def _as_parser(node: dict[str, Any]) -> ParserConfig:
    return ParserConfig(
        min_price=float(node["min_price"]),
        max_price=float(node["max_price"]),
    )


def _as_aggregation(node: dict[str, Any]) -> AggregationConfig:
    return AggregationConfig(
        min_observations_per_track=int(node["min_observations_per_track"]),
        min_final_confidence=float(node["min_final_confidence"]),
        track_ttl_frames=int(node["track_ttl_frames"]),
        price_fuzzy_tolerance=float(_opt(node, "price_fuzzy_tolerance", 0.5)),
        name_fuzzy_ratio=float(_opt(node, "name_fuzzy_ratio", 0.85)),
        dedup_iou_threshold=float(_opt(node, "dedup_iou_threshold", 0.4)),
        dedup_time_window_frames=int(_opt(node, "dedup_time_window_frames", 150)),
        dedup_time_window_s=float(_opt(node, "dedup_time_window_s", 8.0)),
    )


def _as_training(node: Optional[dict[str, Any]]) -> Optional[TrainingConfig]:
    if not node:
        return None
    return TrainingConfig(
        dataset_yaml=str(node["dataset_yaml"]),
        output_dir=str(node["output_dir"]),
        epochs=int(_opt(node, "epochs", 60)),
        batch_size=int(_opt(node, "batch_size", 8)),
        image_size=int(_opt(node, "image_size", 1280)),
        lr0=float(_opt(node, "lr0", 1e-3)),
        workers=int(_opt(node, "workers", 4)),
        pretrained=_opt(node, "pretrained"),
        project_name=str(_opt(node, "project_name", "lenta-2026")),
        run_name=_opt(node, "run_name"),
        seed=int(_opt(node, "seed", 42)),
    )


def load_config(path: str | Path) -> PipelineConfig:
    cfg_path = Path(path).expanduser().resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return PipelineConfig(
        runtime=_as_runtime(data["runtime"]),
        detector=_as_detector(data["detector"]),
        rectifier=_as_rectifier(data["rectifier"]),
        ocr=_as_ocr(data["ocr"]),
        parser=_as_parser(data["parser"]),
        aggregation=_as_aggregation(data["aggregation"]),
        recognition=_as_recognition(data.get("recognition")),
        training=_as_training(data.get("training")),
    )
