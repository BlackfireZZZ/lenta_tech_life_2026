"""Pure round-trip tests for the CVAT adapters.

No OpenCV: the XML build/parse functions take fps/size as arguments and
never open a video, so this runs on an opencv-less build (same rationale as
test_lenta_csv.py — it is where a silent ms↔frame regression would hide).
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from price_tag_pipeline.data.cvat import (  # noqa: E402
    LENTA_COLUMNS,
    build_label_spec,
    cvat_video_to_lenta_rows,
    lenta_rows_to_cvat_video_xml,
    parse_cvat_xml,
)

FPS = 25.0
FRAMES = 1000


def _row(**kw: str) -> dict[str, str]:
    r = {c: "" for c in LENTA_COLUMNS}
    r.update(kw)
    return r


def test_label_spec_has_price_tag_and_categorical_attrs() -> None:
    spec = build_label_spec()
    assert len(spec) == 1 and spec[0]["name"] == "price_tag"
    by_name = {a["name"]: a for a in spec[0]["attributes"]}
    assert by_name["color"]["input_type"] == "select"
    assert "red" in by_name["color"]["values"]
    assert by_name["special_symbols"]["input_type"] == "select"
    assert by_name["product_name"]["input_type"] == "text"


def test_video_round_trip_preserves_bbox_attrs_and_ms() -> None:
    rows = [
        _row(filename="25_2-10", frame_timestamp="200", product_name="Вино, сухое",
             x_min="100", y_min="150", x_max="300", y_max="400",
             color="red", special_symbols="К", barcode="3760094282559"),
        _row(filename="25_2-10", frame_timestamp="2000", product_name="Молоко",
             x_min="50", y_min="60", x_max="120", y_max="240",
             color="white", special_symbols="нет"),
    ]
    xml = lenta_rows_to_cvat_video_xml(
        rows, video_id="25_2-10", width=2160, height=3840, fps=FPS, frame_count=FRAMES
    )
    doc = parse_cvat_xml(xml)
    assert doc.mode == "video"
    assert len(doc.tracks) == 2

    out = cvat_video_to_lenta_rows(doc, fps=FPS, filename="25_2-10")
    assert len(out) == 2
    a, b = out

    # ms survives the ms→frame→ms hop for clean values (200ms@25fps=frame5).
    assert a["frame_timestamp"] == "200"
    assert b["frame_timestamp"] == "2000"
    # bbox preserved.
    assert (float(a["x_min"]), float(a["y_min"]), float(a["x_max"]), float(a["y_max"])) == (100.0, 150.0, 300.0, 400.0)
    # text attribute with a comma survives the XML hop.
    assert a["product_name"] == "Вино, сухое"
    assert a["barcode"] == "3760094282559"
    # select attribute is locale/case-normalized into the vocabulary.
    assert a["color"] == "red"
    assert a["special_symbols"] == "к"
    # track id is carried through as dedup-eval ground truth.
    assert a["track_id"] == "0" and b["track_id"] == "1"


def test_bad_rows_are_skipped_not_fatal() -> None:
    rows = [
        _row(frame_timestamp="нет", x_min="1", y_min="1", x_max="2", y_max="2"),
        _row(frame_timestamp="100", x_min="0", y_min="0", x_max="0", y_max="0"),  # zero-area
        _row(frame_timestamp="500", x_min="10", y_min="10", x_max="90", y_max="90"),
    ]
    doc = parse_cvat_xml(
        lenta_rows_to_cvat_video_xml(
            rows, video_id="v", width=200, height=200, fps=FPS, frame_count=FRAMES
        )
    )
    assert len(doc.tracks) == 1  # only the valid row became a track


def test_image_export_parses_as_photos() -> None:
    xml = """<?xml version="1.0"?>
    <annotations><version>1.1</version>
      <meta><task><original_size><width>4000</width><height>3000</height></original_size></task></meta>
      <image id="0" name="shot1.jpg" width="4000" height="3000">
        <box label="price_tag" xtl="10" ytl="20" xbr="110" ybr="220" occluded="0">
          <attribute name="color">yellow</attribute>
        </box>
      </image>
    </annotations>"""
    doc = parse_cvat_xml(xml)
    assert doc.mode == "images"
    assert len(doc.images) == 1
    im = doc.images[0]
    assert im.name == "shot1.jpg" and im.width == 4000
    assert len(im.boxes) == 1 and im.boxes[0].attrs["color"] == "yellow"
