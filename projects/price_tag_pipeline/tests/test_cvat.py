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
    MODEL_BACKED_SOURCES,
    candidates_csv_to_image_boxes,
    cvat_images_to_yolo,
    cvat_video_to_lenta_rows,
    image_boxes_to_cvat_images_xml,
    cvat_video_to_yolo,
    has_substantive_annotations,
    interpolate_track_boxes,
    lenta_rows_to_cvat_video_xml,
    parse_cvat_xml,
)


def _video_xml(tracks_xml: str, w: int = 100, h: int = 100) -> str:
    return f"""<?xml version="1.0"?>
    <annotations><version>1.1</version>
      <meta><task><original_size><width>{w}</width><height>{h}</height>
      </original_size></task></meta>
      {tracks_xml}
    </annotations>"""

FPS = 25.0
FRAMES = 1000


def _row(**kw: str) -> dict[str, str]:
    r = {c: "" for c in LENTA_COLUMNS}
    r.update(kw)
    return r


def test_label_spec_is_bare_price_tag_rectangle() -> None:
    spec = build_label_spec()
    assert spec == [{"name": "price_tag", "type": "rectangle", "attributes": []}]


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


def test_interpolation_makes_dense_boxes_between_keyframes() -> None:
    # Two keyframes 4 frames apart -> a box on every frame 0..4.
    xml = _video_xml("""
      <track id="0" label="price_tag" source="manual">
        <box frame="0" xtl="0" ytl="0" xbr="10" ybr="10" outside="0" occluded="0" keyframe="1"/>
        <box frame="4" xtl="40" ytl="0" xbr="50" ybr="10" outside="0" occluded="0" keyframe="1"/>
      </track>""")
    doc = parse_cvat_xml(xml)
    pts = interpolate_track_boxes(doc.tracks[0])
    frames = {p[0] for p in pts}
    assert frames == {0, 1, 2, 3, 4}
    by_f = {p[0]: p[1:] for p in pts}
    assert by_f[0] == (0.0, 0.0, 10.0, 10.0)
    assert by_f[4] == (40.0, 0.0, 50.0, 10.0)
    assert by_f[2][0] == 20.0  # x0 linearly interpolated at midpoint

    labels = cvat_video_to_yolo(doc, width=100, height=100)
    assert sorted(labels) == [0, 1, 2, 3, 4]
    assert labels[2] == ["0 0.250000 0.050000 0.100000 0.100000"]


def test_single_keyframe_track_yields_one_frame() -> None:
    # The seed style (one keyframe + an 'outside' terminator) -> exactly 1 box.
    xml = _video_xml("""
      <track id="0" label="price_tag" source="manual">
        <box frame="7" xtl="1" ytl="1" xbr="9" ybr="9" outside="0" occluded="0" keyframe="1"/>
        <box frame="8" xtl="1" ytl="1" xbr="9" ybr="9" outside="1" occluded="0" keyframe="1"/>
      </track>""")
    doc = parse_cvat_xml(xml)
    pts = interpolate_track_boxes(doc.tracks[0])
    assert [p[0] for p in pts] == [7]


def test_boxes_only_is_detected_as_non_substantive() -> None:
    boxes_only = parse_cvat_xml(_video_xml("""
      <track id="0" label="price_tag" source="manual">
        <box frame="0" xtl="0" ytl="0" xbr="5" ybr="5" outside="0" occluded="0" keyframe="1"/>
      </track>"""))
    assert has_substantive_annotations(boxes_only) is False

    with_field = parse_cvat_xml(_video_xml("""
      <track id="0" label="price_tag" source="manual">
        <box frame="0" xtl="0" ytl="0" xbr="5" ybr="5" outside="0" occluded="0" keyframe="1">
          <attribute name="color">red</attribute>
        </box>
      </track>"""))
    assert has_substantive_annotations(with_field) is True


def test_candidates_csv_filters_type_and_round_trips_images() -> None:
    rows = [
        # merged_final, absolute coords, portrait
        {"image_key": "a.jpg", "image_width": "4080", "image_height": "3060",
         "candidate_type": "merged_final", "confidence": "0.9",
         "x_min": "100", "y_min": "200", "x_max": "300", "y_max": "500"},
        # second box on same image, via YOLO-normalized fallback (no absolutes)
        {"image_key": "a.jpg", "image_width": "4080", "image_height": "3060",
         "candidate_type": "merged_final", "confidence": "",
         "x_min": "", "y_min": "", "x_max": "", "y_max": "",
         "yolo_x_center": "0.5", "yolo_y_center": "0.5",
         "yolo_width": "0.1", "yolo_height": "0.2"},
        # raw candidate — must be ignored
        {"image_key": "a.jpg", "image_width": "4080", "image_height": "3060",
         "candidate_type": "model_raw", "confidence": "0.8",
         "x_min": "1", "y_min": "1", "x_max": "9", "y_max": "9"},
        # landscape image
        {"image_key": "b.jpg", "image_width": "8160", "image_height": "6120",
         "candidate_type": "merged_final", "confidence": "0.7",
         "x_min": "10", "y_min": "20", "x_max": "60", "y_max": "120"},
    ]
    boxes = candidates_csv_to_image_boxes(rows)  # default {"merged_final"}
    assert set(boxes) == {"a.jpg", "b.jpg"}
    assert len(boxes["a.jpg"].boxes) == 2  # absolute + yolo fallback, not raw
    assert boxes["a.jpg"].width == 4080 and boxes["b.jpg"].height == 6120

    xml = image_boxes_to_cvat_images_xml(
        [boxes["a.jpg"], boxes["b.jpg"]], task_name="friends_scene_1"
    )
    doc = parse_cvat_xml(xml)
    assert doc.mode == "images" and len(doc.images) == 2
    yolo = cvat_images_to_yolo(doc)
    aw, ah, alines = yolo["a.jpg"]
    assert (aw, ah) == (4080, 3060) and len(alines) == 2
    # first box centre: ((100+300)/2)/4080, ((200+500)/2)/3060
    assert alines[0].startswith("0 0.049020 0.114379 ")


def test_candidates_csv_source_filter_drops_color_cv() -> None:
    rows = [
        {"image_key": "i.jpg", "image_width": "100", "image_height": "100",
         "candidate_type": "merged_final", "source": "openfoodfacts_yolo",
         "x_min": "1", "y_min": "1", "x_max": "9", "y_max": "9"},
        {"image_key": "i.jpg", "image_width": "100", "image_height": "100",
         "candidate_type": "merged_final", "source": "classic_color_cv",
         "x_min": "2", "y_min": "2", "x_max": "8", "y_max": "8"},
        {"image_key": "i.jpg", "image_width": "100", "image_height": "100",
         "candidate_type": "merged_final", "source": "merged_yolo_color_overlap",
         "x_min": "3", "y_min": "3", "x_max": "7", "y_max": "7"},
    ]
    keep = candidates_csv_to_image_boxes(rows, sources=MODEL_BACKED_SOURCES)
    assert len(keep["i.jpg"].boxes) == 2  # yolo + overlap, color_cv dropped
    assert "classic_color_cv" not in MODEL_BACKED_SOURCES
    assert len(candidates_csv_to_image_boxes(rows)["i.jpg"].boxes) == 3  # no filter


def test_candidates_csv_min_conf_filter() -> None:
    rows = [
        {"image_key": "x.jpg", "image_width": "100", "image_height": "100",
         "candidate_type": "merged_final", "confidence": "0.3",
         "x_min": "1", "y_min": "1", "x_max": "9", "y_max": "9"},
    ]
    assert candidates_csv_to_image_boxes(rows, min_conf=0.5) == {}
    assert "x.jpg" in candidates_csv_to_image_boxes(rows, min_conf=0.2)


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

    yolo = cvat_images_to_yolo(doc)
    w, h, lines = yolo["shot1.jpg"]
    assert (w, h) == (4000, 3000) and len(lines) == 1
    assert lines[0].startswith("0 ")
