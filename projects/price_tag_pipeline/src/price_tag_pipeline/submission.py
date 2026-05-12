"""Submission assembly + validation.

Produces a single submission artifact from per-video JSONL outputs.

The exact deliverable format will come from the organizers; this module
emits two complementary forms so we are ready for either:

  * JSON:   one top-level array of {video_id, tags: [...]}.
  * CSV:    flat per-tag rows with video_id as the join key.

Schema validation runs on every tag before it lands in the submission. A
fail-loud option exists (`--strict`) for the final run; the default just
logs and skips invalid rows.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Optional

LOGGER = logging.getLogger(__name__)


# Submission schema — what one tag looks like in the deliverable.
# Mirrors FinalTag.to_dict() output.
SUBMISSION_TAG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "required": ["video_id", "track_id", "bbox", "currency"],
    "properties": {
        "video_id": {"type": "string"},
        "track_id": {"type": "integer"},
        "bbox": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 4,
            "maxItems": 4,
        },
        "timestamp_s": {"type": "number"},
        "source_frames": {"type": "array", "items": {"type": "integer"}},
        "regular_price": {"type": ["number", "null"]},
        "loyalty_price": {"type": ["number", "null"]},
        "product_name": {"type": ["string", "null"]},
        "weight_value": {"type": ["number", "null"]},
        "weight_unit": {"type": ["string", "null"]},
        "price_per_unit_value": {"type": ["number", "null"]},
        "price_per_unit_unit": {"type": ["string", "null"]},
        "promo_flag": {"type": "boolean"},
        "currency": {"type": "string"},
        "overall_confidence": {"type": "number"},
        "n_observations": {"type": "integer"},
    },
}


def _validate_tag(tag: dict[str, Any]) -> Optional[str]:
    """Return None if valid, else an error message."""
    try:
        from jsonschema import validate, ValidationError  # type: ignore
    except ImportError:
        # Lightweight manual fallback so the function still works.
        required = SUBMISSION_TAG_SCHEMA["required"]
        for key in required:
            if key not in tag:
                return f"missing required field: {key}"
        if not (isinstance(tag.get("bbox"), list) and len(tag["bbox"]) == 4):
            return "bbox must be a 4-int list"
        return None
    try:
        validate(instance=tag, schema=SUBMISSION_TAG_SCHEMA)
        return None
    except ValidationError as e:
        return str(e.message)


def collect_predictions(
    inputs_dir: Path,
    video_id_from_filename: bool = True,
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Read every *.jsonl under inputs_dir. Returns list of (video_id, tags)."""
    out: list[tuple[str, list[dict[str, Any]]]] = []
    for p in sorted(inputs_dir.glob("*.jsonl")):
        tags: list[dict[str, Any]] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                LOGGER.warning("Skipping invalid JSONL line in %s: %s", p, e)
                continue
            tags.append(row)
        video_id = p.stem if video_id_from_filename else (tags[0].get("video_id", p.stem) if tags else p.stem)
        out.append((video_id, tags))
    return out


def build_submission(
    inputs_dir: Path,
    out_json: Optional[Path] = None,
    out_csv: Optional[Path] = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Aggregate every per-video JSONL into a single submission artifact.

    Returns the submission dict. Writes JSON and/or CSV if paths are provided.
    Raises ValueError when `strict=True` and any tag fails schema validation.
    """
    pairs = collect_predictions(inputs_dir)
    submission = {"version": "1.0", "videos": []}
    flat_rows: list[dict[str, Any]] = []
    bad = 0
    total = 0
    for video_id, tags in pairs:
        kept: list[dict[str, Any]] = []
        for t in tags:
            t = dict(t)
            t.setdefault("video_id", video_id)
            total += 1
            err = _validate_tag(t)
            if err is not None:
                bad += 1
                msg = f"Schema validation failed for video={video_id}: {err}"
                if strict:
                    raise ValueError(msg)
                LOGGER.warning(msg)
                continue
            kept.append(t)
            flat_rows.append(t)
        submission["videos"].append({"video_id": video_id, "tags": kept})

    LOGGER.info("Aggregated %d tags across %d videos (%d schema failures).",
                total - bad, len(pairs), bad)

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("Wrote JSON submission: %s", out_json)
    if out_csv is not None and flat_rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({k for r in flat_rows for k in r.keys()})
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for r in flat_rows:
                # Flatten list/dict fields for CSV friendliness.
                row_csv = {}
                for k in fields:
                    v = r.get(k)
                    if isinstance(v, (list, dict)):
                        row_csv[k] = json.dumps(v, ensure_ascii=False)
                    else:
                        row_csv[k] = v
                writer.writerow(row_csv)
        LOGGER.info("Wrote CSV submission: %s", out_csv)
    return submission
