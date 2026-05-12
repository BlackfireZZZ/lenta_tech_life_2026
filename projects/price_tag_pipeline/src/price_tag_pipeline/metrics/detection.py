"""Detection mAP via torchmetrics.

Inputs are lists of dicts with shape:
    pred = {"boxes": Nx4 xyxy float, "scores": N float, "labels": N int}
    target = {"boxes": Mx4 xyxy float, "labels": M int}
"""

from __future__ import annotations

from typing import Any


def compute_detection_map(
    preds: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    iou_thresholds: list[float] | None = None,
) -> dict[str, float]:
    """Wraps torchmetrics.detection.MeanAveragePrecision.

    Returns a dict with at least: map, map_50, map_75, map_small, map_medium, map_large.
    """
    try:
        import torch
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
    except ImportError as e:
        raise RuntimeError(
            "compute_detection_map needs torch + torchmetrics. "
            "Install with: pip install torch torchmetrics"
        ) from e

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_thresholds=iou_thresholds,
        class_metrics=False,
    )

    def _to_tensor_dict(d: dict[str, Any], with_scores: bool) -> dict[str, "torch.Tensor"]:
        out = {
            "boxes": torch.as_tensor(d["boxes"], dtype=torch.float32),
            "labels": torch.as_tensor(d["labels"], dtype=torch.int64),
        }
        if with_scores:
            out["scores"] = torch.as_tensor(d["scores"], dtype=torch.float32)
        return out

    pred_t = [_to_tensor_dict(p, with_scores=True) for p in preds]
    tgt_t = [_to_tensor_dict(t, with_scores=False) for t in targets]
    metric.update(pred_t, tgt_t)
    result = metric.compute()
    return {k: float(v) for k, v in result.items()}
