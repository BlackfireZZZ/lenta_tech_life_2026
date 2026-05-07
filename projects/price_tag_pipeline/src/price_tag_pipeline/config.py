from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass(frozen=True)
class RuntimeConfig:
    profile_name: str
    output_jsonl: Optional[str]
    log_every_n_frames: int


@dataclass(frozen=True)
class DetectorConfig:
    model_path: str
    conf: float
    iou: float
    device: Optional[str]
    classes: list[int]
    tracker_yaml: str


@dataclass(frozen=True)
class RectifierConfig:
    padding_ratio: float
    clahe_clip_limit: float
    clahe_tile_grid_size: int
    rotate_vertical_tags: bool


@dataclass(frozen=True)
class OCRConfig:
    backend: str
    min_frames_between_ocr_per_track: int
    min_sharpness: float
    min_crop_area_px: int
    min_detection_confidence: float


@dataclass(frozen=True)
class ParserConfig:
    decimal_preferred: bool
    min_price: float
    max_price: float


@dataclass(frozen=True)
class AggregationConfig:
    min_observations_per_track: int
    min_final_confidence: float
    track_ttl_frames: int


@dataclass(frozen=True)
class PipelineConfig:
    runtime: RuntimeConfig
    detector: DetectorConfig
    rectifier: RectifierConfig
    ocr: OCRConfig
    parser: ParserConfig
    aggregation: AggregationConfig


def _as_runtime(node: dict[str, Any]) -> RuntimeConfig:
    return RuntimeConfig(
        profile_name=str(node["profile_name"]),
        output_jsonl=node.get("output_jsonl"),
        log_every_n_frames=int(node["log_every_n_frames"]),
    )


def _as_detector(node: dict[str, Any]) -> DetectorConfig:
    return DetectorConfig(
        model_path=str(node["model_path"]),
        conf=float(node["conf"]),
        iou=float(node["iou"]),
        device=node.get("device"),
        classes=[int(v) for v in node["classes"]],
        tracker_yaml=str(node["tracker_yaml"]),
    )


def _as_rectifier(node: dict[str, Any]) -> RectifierConfig:
    return RectifierConfig(
        padding_ratio=float(node["padding_ratio"]),
        clahe_clip_limit=float(node["clahe_clip_limit"]),
        clahe_tile_grid_size=int(node["clahe_tile_grid_size"]),
        rotate_vertical_tags=bool(node["rotate_vertical_tags"]),
    )


def _as_ocr(node: dict[str, Any]) -> OCRConfig:
    return OCRConfig(
        backend=str(node["backend"]).lower().strip(),
        min_frames_between_ocr_per_track=int(node["min_frames_between_ocr_per_track"]),
        min_sharpness=float(node["min_sharpness"]),
        min_crop_area_px=int(node["min_crop_area_px"]),
        min_detection_confidence=float(node["min_detection_confidence"]),
    )


def _as_parser(node: dict[str, Any]) -> ParserConfig:
    return ParserConfig(
        decimal_preferred=bool(node["decimal_preferred"]),
        min_price=float(node["min_price"]),
        max_price=float(node["max_price"]),
    )


def _as_aggregation(node: dict[str, Any]) -> AggregationConfig:
    return AggregationConfig(
        min_observations_per_track=int(node["min_observations_per_track"]),
        min_final_confidence=float(node["min_final_confidence"]),
        track_ttl_frames=int(node["track_ttl_frames"]),
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
    )

