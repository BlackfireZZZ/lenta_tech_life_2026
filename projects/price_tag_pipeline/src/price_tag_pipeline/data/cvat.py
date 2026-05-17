"""CVAT ⇄ Lenta adapters.

Bridges hand annotation in CVAT with the project's native data layer so the
existing, tested ``prepare_data.py`` path consumes hand-labelled video/photos
**without any pipeline change**.

Direction & format:

* **ours → CVAT** : ``lenta_rows_to_cvat_video_xml`` turns the 29-column Lenta
  CSV of an already-labelled video into *CVAT for video 1.1* XML — one
  ``<track>`` per CSV row (one physical price tag), a single keyframe ``<box>``
  at its frame, every substantive field carried as a ``<attribute>``. The
  annotator imports this and *corrects* noisy boxes instead of drawing from
  scratch.

* **CVAT → ours (video)** : ``cvat_video_to_lenta_rows`` collapses each
  ``<track>`` back to one 29-column row using the largest non-``outside``
  keyframe as the "best frame" (task.md §6.4). ``frame_timestamp`` is rebuilt
  in **milliseconds** with the exact same ms↔frame mapping the training
  pipeline uses (``loaders._parse_lenta_frame_idx``'s inverse), so round-trips
  are stable. The CVAT ``track id`` is preserved in a ``track_id`` column —
  free ground truth for cross-track dedup evaluation (briefing §6.5).

* **CVAT → ours (photos)** : ``cvat_images_to_yolo`` writes the YOLO raw layout
  (``annotations/labels/<set>/*.txt`` + ``frames/<set>/*.jpg`` +
  ``classes.txt``) because ad-hoc store photos have no video / no millisecond
  timestamp and exist purely to augment detector training.

The XML build/parse functions are **pure** (fps/size passed in, no video
opened, no OpenCV) so they unit-test on an OpenCV-less build — same rationale
as ``tests/test_lenta_csv.py``. Only the thin CLI wrappers touch cv2.

Unicode-safe by construction: ``xml.etree.ElementTree`` for I/O, ``cv_io`` for
any still-image copy (cv2.imread/imwrite silently fail on Cyrillic paths).
"""

from __future__ import annotations

import csv
import io
import json
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional
from xml.dom import minidom

from .loaders import _parse_decimal, _parse_lenta_frame_idx, _read_lenta_csv

LOGGER = logging.getLogger(__name__)

LABEL_NAME = "price_tag"

# Canonical 29-column order — must stay identical to
# submission.HACK_CSV_COLUMNS and the released Lenta CSVs.
LENTA_COLUMNS: list[str] = [
    "filename", "product_name", "price_default", "price_card", "price_discount",
    "barcode", "discount_amount", "id_sku", "print_datetime", "code",
    "additional_info", "color", "special_symbols", "frame_timestamp",
    "x_min", "y_min", "x_max", "y_max", "qr_code_barcode", "price1_qr",
    "price2_qr", "price3_qr", "price4_qr", "wholesale_level_1_count",
    "wholesale_level_1_price", "wholesale_level_2_count",
    "wholesale_level_2_price", "action_price_qr", "action_code_qr",
]

# Columns that are NOT per-object CVAT attributes: geometry + technical.
_GEOMETRY_COLS = {"x_min", "y_min", "x_max", "y_max"}
_TECHNICAL_COLS = {"filename", "frame_timestamp"}

# Substantive columns become CVAT attributes. Two are categorical (fast to set
# via a dropdown); the rest are free text (slow — usually only fixed when the
# seed is wrong, see runbook). ``нет`` = field absent on the tag (task.md §3.3).
_SELECT_ATTRS: dict[str, list[str]] = {
    "color": ["white", "yellow", "green", "red", "black", "нет"],
    "special_symbols": ["к", "л", "ш", "нет"],
}
_TEXT_ATTRS: list[str] = [
    c for c in LENTA_COLUMNS
    if c not in _GEOMETRY_COLS
    and c not in _TECHNICAL_COLS
    and c not in _SELECT_ATTRS
]
_ALL_ATTRS: list[str] = list(_SELECT_ATTRS) + _TEXT_ATTRS


# ---------------------------------------------------------------------------
# Label specification (paste into CVAT → Constructor → Raw, or import as JSON)
# ---------------------------------------------------------------------------

def build_label_spec() -> list[dict]:
    """The CVAT label spec: one ``price_tag`` rectangle with our attributes."""
    attributes: list[dict] = []
    for name, values in _SELECT_ATTRS.items():
        attributes.append({
            "name": name,
            "input_type": "select",
            "mutable": True,
            "values": values,
            "default_value": values[0],
        })
    for name in _TEXT_ATTRS:
        attributes.append({
            "name": name,
            "input_type": "text",
            "mutable": True,
            "values": [],
            "default_value": "",
        })
    return [{"name": LABEL_NAME, "type": "rectangle", "attributes": attributes}]


def label_spec_json() -> str:
    return json.dumps(build_label_spec(), ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# ours → CVAT for video 1.1
# ---------------------------------------------------------------------------

def _xml_attr_block(parent: ET.Element) -> None:
    """Append the <labels> spec onto a <task> meta element."""
    labels = ET.SubElement(parent, "labels")
    label = ET.SubElement(labels, "label")
    ET.SubElement(label, "name").text = LABEL_NAME
    ET.SubElement(label, "type").text = "rectangle"
    attrs = ET.SubElement(label, "attributes")
    for spec in build_label_spec()[0]["attributes"]:
        a = ET.SubElement(attrs, "attribute")
        ET.SubElement(a, "name").text = spec["name"]
        ET.SubElement(a, "mutable").text = "true"
        ET.SubElement(a, "input_type").text = spec["input_type"]
        ET.SubElement(a, "default_value").text = spec["default_value"]
        ET.SubElement(a, "values").text = "\n".join(spec["values"])


def _norm_select(col: str, raw: str) -> str:
    """Coerce a seed value into the select's vocabulary (case/locale)."""
    val = (raw or "").strip().lower()
    allowed = _SELECT_ATTRS[col]
    if val in allowed:
        return val
    if not val or val in {"nan", "none"}:
        return "нет"
    return val  # kept verbatim; CVAT shows it, annotator can re-pick


def lenta_rows_to_cvat_video_xml(
    rows: list[dict[str, str]],
    *,
    video_id: str,
    width: int,
    height: int,
    fps: float,
    frame_count: int,
    task_name: Optional[str] = None,
) -> str:
    """Render Lenta CSV rows as *CVAT for video 1.1* XML.

    One ``<track>`` per row, a single keyframe ``<box>`` at the row's frame
    (ms→frame via the training pipeline's exact mapping). Rows whose
    timestamp or bbox cannot be parsed are skipped with a warning — they
    cannot be placed in the frame and would otherwise corrupt the import.
    """
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "id").text = "0"
    ET.SubElement(task, "name").text = task_name or video_id
    ET.SubElement(task, "size").text = str(frame_count)
    ET.SubElement(task, "mode").text = "interpolation"
    ET.SubElement(task, "overlap").text = "0"
    ET.SubElement(task, "bugtracker").text = ""
    ET.SubElement(task, "flipped").text = "False"
    _xml_attr_block(task)
    osz = ET.SubElement(task, "original_size")
    ET.SubElement(osz, "width").text = str(width)
    ET.SubElement(osz, "height").text = str(height)

    kept = 0
    for track_id, row in enumerate(rows):
        frame = _parse_lenta_frame_idx(
            row.get("frame_timestamp"), fps=fps, frame_count=frame_count
        )
        box = _parse_bbox(row, width, height)
        if frame is None or box is None:
            LOGGER.warning(
                "Seed: skipping row %d of %s (bad frame_timestamp/bbox)",
                track_id, video_id,
            )
            continue
        xtl, ytl, xbr, ybr = box
        tr = ET.SubElement(
            root, "track",
            {"id": str(track_id), "label": LABEL_NAME, "source": "manual"},
        )
        b = ET.SubElement(tr, "box", {
            "frame": str(frame),
            "xtl": f"{xtl:.2f}", "ytl": f"{ytl:.2f}",
            "xbr": f"{xbr:.2f}", "ybr": f"{ybr:.2f}",
            "outside": "0", "occluded": "0", "keyframe": "1",
        })
        for col in _ALL_ATTRS:
            val = (row.get(col) or "").strip()
            if col in _SELECT_ATTRS:
                val = _norm_select(col, val)
            ET.SubElement(b, "attribute", {"name": col}).text = val
        # A trailing 'outside' box one frame later bounds the single-frame
        # track so CVAT does not extrapolate the seed box forever.
        last = min(frame_count - 1, frame + 1) if frame_count else frame + 1
        if last > frame:
            ET.SubElement(tr, "box", {
                "frame": str(last),
                "xtl": f"{xtl:.2f}", "ytl": f"{ytl:.2f}",
                "xbr": f"{xbr:.2f}", "ybr": f"{ybr:.2f}",
                "outside": "1", "occluded": "0", "keyframe": "1",
            })
        kept += 1

    LOGGER.info("Seed %s: %d/%d rows -> tracks", video_id, kept, len(rows))
    raw = ET.tostring(root, encoding="utf-8")
    return minidom.parseString(raw).toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")


def _parse_bbox(
    row: dict[str, str], width: int, height: int
) -> Optional[tuple[float, float, float, float]]:
    vals = [_parse_decimal(row.get(k)) for k in ("x_min", "y_min", "x_max", "y_max")]
    if any(v is None for v in vals):
        return None
    x0, y0, x1, y1 = (float(v) for v in vals)  # type: ignore[arg-type]
    x0, x1 = (max(0.0, min(float(width), v)) for v in (x0, x1))
    y0, y1 = (max(0.0, min(float(height), v)) for v in (y0, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


# ---------------------------------------------------------------------------
# CVAT → ours
# ---------------------------------------------------------------------------

@dataclass
class CvatBox:
    frame: int
    xtl: float
    ytl: float
    xbr: float
    ybr: float
    outside: bool
    keyframe: bool
    attrs: dict[str, str] = field(default_factory=dict)


@dataclass
class CvatTrack:
    track_id: int
    label: str
    boxes: list[CvatBox] = field(default_factory=list)


@dataclass
class CvatImage:
    name: str
    width: int
    height: int
    boxes: list[CvatBox] = field(default_factory=list)


@dataclass
class CvatDoc:
    mode: str  # "video" | "images"
    tracks: list[CvatTrack] = field(default_factory=list)
    images: list[CvatImage] = field(default_factory=list)
    width: int = 0
    height: int = 0


def _box_from_elem(el: ET.Element, *, with_frame: bool) -> CvatBox:
    attrs = {a.get("name", ""): (a.text or "") for a in el.findall("attribute")}
    return CvatBox(
        frame=int(el.get("frame", "0")) if with_frame else 0,
        xtl=float(el.get("xtl", "0")),
        ytl=float(el.get("ytl", "0")),
        xbr=float(el.get("xbr", "0")),
        ybr=float(el.get("ybr", "0")),
        outside=el.get("outside", "0") == "1",
        keyframe=el.get("keyframe", "1") == "1",
        attrs=attrs,
    )


def parse_cvat_xml(text: str) -> CvatDoc:
    """Parse *CVAT for video 1.1* or *CVAT for images 1.1* XML."""
    root = ET.fromstring(text)
    osz = root.find("./meta/task/original_size")
    width = int(osz.findtext("width", "0")) if osz is not None else 0
    height = int(osz.findtext("height", "0")) if osz is not None else 0

    tracks = root.findall("track")
    if tracks:
        doc = CvatDoc(mode="video", width=width, height=height)
        for tr in tracks:
            t = CvatTrack(int(tr.get("id", "0")), tr.get("label", LABEL_NAME))
            for bx in tr.findall("box"):
                t.boxes.append(_box_from_elem(bx, with_frame=True))
            doc.tracks.append(t)
        return doc

    doc = CvatDoc(mode="images", width=width, height=height)
    for im in root.findall("image"):
        ci = CvatImage(
            name=im.get("name", ""),
            width=int(im.get("width", str(width))),
            height=int(im.get("height", str(height))),
        )
        for bx in im.findall("box"):
            ci.boxes.append(_box_from_elem(bx, with_frame=False))
        doc.images.append(ci)
    return doc


def _frame_to_ms(frame: int, fps: float) -> int:
    eff = fps if fps and fps > 0 else 30.0
    return int(round(frame / eff * 1000.0))


def _best_box(track: CvatTrack) -> Optional[CvatBox]:
    """Pick the largest visible keyframe — the "best frame" (task.md §6.4)."""
    visible = [b for b in track.boxes if not b.outside]
    if not visible:
        return None
    keyed = [b for b in visible if b.keyframe] or visible
    return max(keyed, key=lambda b: (b.xbr - b.xtl) * (b.ybr - b.ytl))


def cvat_video_to_lenta_rows(
    doc: CvatDoc, *, fps: float, filename: str
) -> list[dict[str, str]]:
    """Collapse tracks to one 29-column Lenta row each (+ ``track_id``)."""
    rows: list[dict[str, str]] = []
    for tr in sorted(doc.tracks, key=lambda t: t.track_id):
        box = _best_box(tr)
        if box is None:
            LOGGER.warning("Track %d has no visible box — skipped", tr.track_id)
            continue
        row = {c: "" for c in LENTA_COLUMNS}
        row["filename"] = filename
        row["frame_timestamp"] = str(_frame_to_ms(box.frame, fps))
        row["x_min"] = f"{box.xtl:.2f}"
        row["y_min"] = f"{box.ytl:.2f}"
        row["x_max"] = f"{box.xbr:.2f}"
        row["y_max"] = f"{box.ybr:.2f}"
        for col in _ALL_ATTRS:
            if col in box.attrs:
                row[col] = box.attrs[col].strip()
        row["track_id"] = str(tr.track_id)
        rows.append(row)
    return rows


def lenta_rows_to_csv_text(rows: Iterable[dict[str, str]]) -> str:
    """29 canonical columns + trailing ``track_id``. UTF-8, ``,``, ``.``."""
    cols = LENTA_COLUMNS + ["track_id"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue()


def read_lenta_csv(path: Path) -> list[dict[str, str]]:
    """Public re-export of the header-typo-normalizing CSV reader."""
    return _read_lenta_csv(path)
