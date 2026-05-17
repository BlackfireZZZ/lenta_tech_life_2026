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
import io
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Optional

from .types import FinalTag

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


# ===========================================================================
# Graded hackathon CSV — the single source of truth for the 29-column output
# ===========================================================================
#
# This is the *graded* deliverable (schema: docs/hackathon/task.md §3). It was
# previously implemented only inside the `export_hack_csv.py` CLI script, so
# the ML service had no importable renderer (docs/architecture.md §5 and
# ml/app/runner.py referenced `final_tags_to_csv`, which did not exist). The
# logic now lives here once; `export_hack_csv.py` delegates to it. Behaviour
# is intentionally byte-identical to the old script.
#
# "нет" vs empty (task.md §3.3, scored): this layer is purely a renderer — it
# emits whatever the parser/aggregator decided. A value of None → empty cell
# (present on the tag but not recognized); a literal "нет" string is passed
# through unchanged (field absent on the tag). The distinction is made
# upstream, not here.

HACK_CSV_COLUMNS: list[str] = [
    "filename",
    "product_name",
    "price_default",
    "price_card",
    "price_discount",
    "barcode",
    "discount_amount",
    "id_sku",
    "print_datetime",
    "code",
    "additional_info",
    "color",
    "special_symbols",
    "frame_timestamp",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "qr_code_barcode",
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "wholesale_level_1_count",
    "wholesale_level_1_price",
    "wholesale_level_2_count",
    "wholesale_level_2_price",
    "action_price_qr",
    "action_code_qr",
]


def _hc_to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _hc_fmt_price(v: Any) -> str:
    f = _hc_to_float(v)
    return "" if f is None else f"{f:.2f}"


def _hc_fmt_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _hc_as_int_str(v: Any) -> str:
    if v is None:
        return ""
    try:
        return str(int(round(float(v))))
    except (TypeError, ValueError):
        return ""


def _hc_derive_discount(regular: Optional[float], card: Optional[float]) -> str:
    if regular is None or card is None:
        return ""
    diff = float(regular) - float(card)
    return "0.00" if diff <= 0 else f"{diff:.2f}"


def hack_row_from_tag_dict(tag: dict[str, Any], filename: str) -> dict[str, str]:
    """Map one ``FinalTag.to_dict()`` (== one JSONL row) to a 29-column row.

    Shared by the live ``FinalTag`` path (ML service / Gradio) and the
    JSONL-replay path (``export_hack_csv.py``) so the graded schema has
    exactly one implementation.
    """
    bbox = tag.get("bbox") or [None, None, None, None]
    if len(bbox) != 4:
        bbox = [None, None, None, None]

    regular = _hc_to_float(tag.get("regular_price"))
    card = _hc_to_float(tag.get("loyalty_price"))

    ts_ms: Optional[int] = None
    if tag.get("timestamp_s") is not None:
        try:
            ts_ms = int(round(float(tag["timestamp_s"]) * 1000.0))
        except (TypeError, ValueError):
            ts_ms = None

    price_discount = tag.get("price_discount")
    if price_discount in (None, ""):
        price_discount = _hc_derive_discount(regular, card)

    return {
        "filename": filename,
        "product_name": str(tag.get("product_name") or ""),
        "price_default": _hc_fmt_price(regular),
        "price_card": _hc_fmt_price(card),
        "price_discount": _hc_fmt_cell(price_discount),
        "barcode": _hc_fmt_cell(tag.get("barcode")),
        "discount_amount": _hc_fmt_cell(tag.get("discount_amount")),
        "id_sku": _hc_fmt_cell(tag.get("id_sku")),
        "print_datetime": _hc_fmt_cell(tag.get("print_datetime")),
        "code": _hc_fmt_cell(tag.get("code")),
        "additional_info": _hc_fmt_cell(tag.get("additional_info")),
        "color": _hc_fmt_cell(tag.get("color")),
        "special_symbols": _hc_fmt_cell(tag.get("special_symbols")),
        "frame_timestamp": _hc_as_int_str(ts_ms),
        "x_min": _hc_as_int_str(bbox[0]),
        "y_min": _hc_as_int_str(bbox[1]),
        "x_max": _hc_as_int_str(bbox[2]),
        "y_max": _hc_as_int_str(bbox[3]),
        "qr_code_barcode": _hc_fmt_cell(tag.get("qr_code_barcode")),
        "price1_qr": _hc_fmt_cell(tag.get("price1_qr")),
        "price2_qr": _hc_fmt_cell(tag.get("price2_qr")),
        "price3_qr": _hc_fmt_cell(tag.get("price3_qr")),
        "price4_qr": _hc_fmt_cell(tag.get("price4_qr")),
        "wholesale_level_1_count": _hc_fmt_cell(tag.get("wholesale_level_1_count")),
        "wholesale_level_1_price": _hc_fmt_cell(tag.get("wholesale_level_1_price")),
        "wholesale_level_2_count": _hc_fmt_cell(tag.get("wholesale_level_2_count")),
        "wholesale_level_2_price": _hc_fmt_cell(tag.get("wholesale_level_2_price")),
        "action_price_qr": _hc_fmt_cell(tag.get("action_price_qr")),
        "action_code_qr": _hc_fmt_cell(tag.get("action_code_qr")),
    }


def hack_rows_to_csv_text(rows: Iterable[dict[str, str]]) -> str:
    """Render rows (from :func:`hack_row_from_tag_dict`) as CSV text.

    UTF-8, ``,`` separator, ``.`` decimal, ``\\n`` line terminator,
    ``QUOTE_MINIMAL`` (csv quotes only cells containing ``,``/``"``/newline,
    inner quotes doubled) — task.md §3.3.

    This MUST stay byte-compatible with the gateway's own renderer
    ``backend/app/jobs_mock.py:build_csv`` (same columns via
    ``backend/app/api/v1/schemas/job.py:CSV_COLUMNS``). The gateway serves
    the ML service's CSV *verbatim* as the graded artifact, so a real run
    and the backend mock must produce the identical wire format — they are
    two owners of one contract, like the §5.1 Pydantic mirrors.
    """
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf, fieldnames=HACK_CSV_COLUMNS, extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()


def final_tags_to_csv(tags: Iterable[FinalTag], filename: str) -> str:
    """Render finalized tags as the graded 29-column CSV **text**.

    This is the public entry point the ML service calls (architecture.md §5):
    ``PriceTagPipeline(cfg).run(video) -> list[FinalTag] -> final_tags_to_csv``.
    ``filename`` is the per-video ``filename`` cell (the released Lenta CSVs
    use a bare stem such as ``25_2-10``, not ``25_2-10.mp4``).
    """
    rows = [hack_row_from_tag_dict(t.to_dict(), filename) for t in tags]
    return hack_rows_to_csv_text(rows)
