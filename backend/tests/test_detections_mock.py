"""The mock detector trace powering the review screen's detector view
without a GPU. Pins the contract shape the SPA consumes and that it never
disagrees with the mock tags it is derived from.
"""

from uuid import uuid4

from app.jobs_mock import build_detections, generate_tags


def _trace():
    job_id = uuid4()
    tags = generate_tags(job_id, "25_2-10")
    return build_detections(tags), tags


def test_trace_has_the_contract_shape() -> None:
    trace, _ = _trace()
    assert set(trace) == {
        "frame_width",
        "frame_height",
        "conf_threshold",
        "sampled",
        "frames",
    }
    assert trace["frame_width"] > 0 and trace["frame_height"] > 0
    assert 0.0 < trace["conf_threshold"] < 1.0
    assert trace["sampled"] is False
    assert len(trace["frames"]) > 0


def test_frames_are_time_ordered_and_boxes_normalised() -> None:
    trace, _ = _trace()
    times = [f["t_ms"] for f in trace["frames"]]
    assert times == sorted(times)
    assert all(t >= 0 for t in times)
    for f in trace["frames"]:
        assert f["boxes"], "a recorded frame must carry at least one box"
        for x1, y1, x2, y2, score in f["boxes"]:
            assert 0.0 <= x1 < x2 <= 1.0
            assert 0.0 <= y1 < y2 <= 1.0
            assert trace["conf_threshold"] <= score <= 1.0


def test_is_deterministic_for_the_same_job() -> None:
    job_id = uuid4()
    a = build_detections(generate_tags(job_id, "v"))
    b = build_detections(generate_tags(job_id, "v"))
    assert a == b  # same job id → identical trace (polling must not reshuffle)
