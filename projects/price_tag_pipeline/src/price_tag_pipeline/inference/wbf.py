"""Weighted Box Fusion (WBF) for ensembling detectors.

A pure-Python WBF that does not pull in the `ensemble-boxes` package, but
falls back to it if available (for a small speed/accuracy bump).

WBF clusters boxes by IoU, then fuses each cluster as a confidence-weighted
average of its coordinates. Unlike NMS, WBF combines information from all
boxes in a cluster instead of picking the single highest-confidence one.
That typically gains +1–4 mAP when combining 2+ detectors.

Inputs (per detector):
    boxes:  [(x1, y1, x2, y2), ...]   normalized to [0,1] or pixel — be consistent
    scores: [s1, s2, ...]
    labels: [c1, c2, ...]
    weight: float (relative importance of this detector)
"""

from __future__ import annotations

from typing import Iterable


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def weighted_box_fusion(
    detectors_boxes: list[list[tuple[float, float, float, float]]],
    detectors_scores: list[list[float]],
    detectors_labels: list[list[int]],
    detectors_weights: list[float] | None = None,
    iou_threshold: float = 0.55,
    skip_box_threshold: float = 0.001,
):
    """Pure-Python WBF. Returns (boxes, scores, labels).

    If the `ensemble_boxes` package is installed, prefer it.
    """
    try:
        from ensemble_boxes import weighted_boxes_fusion as _wbf  # type: ignore
        weights = detectors_weights or [1.0] * len(detectors_boxes)
        return _wbf(
            detectors_boxes, detectors_scores, detectors_labels,
            weights=weights, iou_thr=iou_threshold, skip_box_thr=skip_box_threshold,
        )
    except ImportError:
        pass

    weights = detectors_weights or [1.0] * len(detectors_boxes)

    # Gather all boxes (with detector weight applied to score).
    pool: list[dict] = []
    for det_idx, (boxes, scores, labels) in enumerate(
        zip(detectors_boxes, detectors_scores, detectors_labels)
    ):
        w = weights[det_idx]
        for b, s, lab in zip(boxes, scores, labels):
            if s < skip_box_threshold:
                continue
            pool.append({"box": tuple(b), "score": float(s) * w, "raw_score": float(s),
                         "label": int(lab), "weight": w})

    # Sort by raw score descending, then greedy-cluster by IoU + label.
    pool.sort(key=lambda x: x["raw_score"], reverse=True)
    clusters: list[list[dict]] = []
    for box in pool:
        placed = False
        for cluster in clusters:
            rep = cluster[0]
            if rep["label"] != box["label"]:
                continue
            if _iou(rep["box"], box["box"]) >= iou_threshold:
                cluster.append(box)
                placed = True
                break
        if not placed:
            clusters.append([box])

    fused_boxes: list[tuple[float, float, float, float]] = []
    fused_scores: list[float] = []
    fused_labels: list[int] = []
    n_models = max(1, len(detectors_boxes))
    for cluster in clusters:
        total_w = sum(b["score"] for b in cluster)
        if total_w <= 0:
            continue
        x1 = sum(b["box"][0] * b["score"] for b in cluster) / total_w
        y1 = sum(b["box"][1] * b["score"] for b in cluster) / total_w
        x2 = sum(b["box"][2] * b["score"] for b in cluster) / total_w
        y2 = sum(b["box"][3] * b["score"] for b in cluster) / total_w
        # WBF scoring: scale by (cluster size / n_models).
        score = (sum(b["raw_score"] for b in cluster) / len(cluster)) * (len(cluster) / n_models)
        fused_boxes.append((x1, y1, x2, y2))
        fused_scores.append(min(1.0, score))
        fused_labels.append(cluster[0]["label"])
    return fused_boxes, fused_scores, fused_labels
