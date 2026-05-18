"""The gateway↔ML unifying-contract seam (architecture.md §5.6): the real
ML service returns ONLY the verbatim 29-column CSV; the gateway reconstructs
the non-graded review payload from it. This pins the round-trip:

    FinalTag --(real producer)--> CSV --(gateway)--> JobPredictions

so a change to either owner that breaks the seam fails here, without Docker,
a DB, a GPU or the network.
"""

from uuid import uuid4

from app.predictions import build_predictions_from_csv
from price_tag_pipeline.submission import final_tags_to_csv
from price_tag_pipeline.types import FinalTag


def _tag(track_id: int, ts_s: float, bbox, color: str, extra: dict) -> FinalTag:
    return FinalTag(
        track_id=track_id,
        bbox_xyxy=bbox,
        timestamp_s=ts_s,
        source_frames=[1, 2],
        regular_price=129.99,
        loyalty_price=99.99,
        product_name="Молоко Простоквашино 2.5% 1л",
        promo_flag=True,
        currency="RUB",
        extra_fields={"color": color, **extra},
    )


def _csv_of(tags):
    return final_tags_to_csv(tags, filename="25_2-10")


def test_roundtrip_preserves_states_bbox_timestamp():
    # "нет" = field absent on the tag; a missing extra (None) renders ""
    # = present but unrecognised. Both must survive to the review payload
    # distinctly — confusing them loses points (task.md §3.3/§5.3).
    tags = [
        _tag(
            7, 5.0, (100, 200, 400, 500), "red",
            {"barcode": "4601234567890", "qr_code_barcode": "нет",
             "additional_info": "нет"},  # 'нет' absent; price*_qr left None -> ""
        ),
        _tag(
            8, 12.5, (10, 10, 60, 40), "white",
            {"barcode": "4609999999999"},
        ),
    ]
    csv_text = _csv_of(tags)
    meta = {"frame_width": 1920, "frame_height": 1080,
            "video_duration_s": 20.0}

    preds = build_predictions_from_csv(
        job_id=uuid4(), filename="25_2-10", csv_text=csv_text, meta=meta,
        video_url="/v", csv_url="/c",
    )

    assert len(preds.tags) == 2
    t0 = preds.tags[0]
    assert t0.index == 0
    assert t0.color == "red"
    assert t0.fields["barcode"] == "4601234567890"
    assert t0.fields["qr_code_barcode"] == "нет"      # absent preserved
    assert t0.fields["additional_info"] == "нет"
    assert t0.fields["price1_qr"] == ""               # unrecognised preserved
    assert t0.fields["product_name"] == "Молоко Простоквашино 2.5% 1л"

    # frame_timestamp is real ms; t_frac maps it onto clip duration.
    assert t0.frame_timestamp == 5000
    assert abs(t0.t_frac - (5000 / 1000) / 20.0) < 1e-6

    # bbox: pixel / raw-frame-size, clamped to [0,1].
    assert abs(t0.bbox.x1 - 100 / 1920) < 1e-4
    assert abs(t0.bbox.y1 - 200 / 1080) < 1e-4
    assert abs(t0.bbox.x2 - 400 / 1920) < 1e-4
    assert abs(t0.bbox.y2 - 500 / 1080) < 1e-4

    # The graded CSV the gateway would serve is byte-identical to producer.
    assert preds.columns == list(preds.columns)
    assert preds.tags[1].frame_timestamp == 12500


def test_degrades_safely_without_meta():
    # cv2 probe failed → no frame dims/duration. Review must still render
    # every field (the graded CSV download is independent and unaffected).
    tags = [_tag(1, 3.0, (1, 2, 3, 4), "green", {"barcode": "460"})]
    preds = build_predictions_from_csv(
        job_id=uuid4(), filename="x", csv_text=_csv_of(tags), meta=None,
        video_url="/v", csv_url="/c",
    )
    assert len(preds.tags) == 1
    t = preds.tags[0]
    assert t.fields["barcode"] == "460"
    assert (t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2) == (0.0, 0.0, 0.0, 0.0)
    assert t.t_frac == 0.0
    assert t.frame_timestamp == 3000


def test_degraded_csv_yields_no_tags_not_a_crash():
    # The runner's guarded mock fallback emits a 1-line stub when a heavy
    # import fails. Reconstruction must tolerate it (empty review, no 500);
    # the verbatim stub is still downloadable as the (degraded) CSV.
    stub = "filename,frame_timestamp,x1,y1,x2,y2,barcode,name,price,...\n"
    preds = build_predictions_from_csv(
        job_id=uuid4(), filename="x", csv_text=stub, meta={},
        video_url="/v", csv_url="/c",
    )
    assert preds.tags == []
