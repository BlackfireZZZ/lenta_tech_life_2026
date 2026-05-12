"""End-to-end scoring for the submission target.

Inputs are aligned lists of (predicted_tag_dict, ground_truth_tag_dict).
The caller is responsible for matching predictions to ground truth (e.g. by
track_id, or by bbox-IoU + price-equality). This module assumes the alignment
is already done.

Per-field rules:
- Prices: exact equality within ±0.005 RUB (i.e. 1 kopeck).
- weight_value: equality within ±0.001 of the smaller unit.
- weight_unit, currency, price_per_unit_unit: exact string equality after .strip().lower().
- promo_flag: exact bool equality.
- product_name: CER ≤ τ counts as correct (τ defaults to 0.15).

`e2e_field_accuracy` returns the fraction of tags where ALL graded fields are
correct simultaneously (the hackathon's likely grading rule).
`per_field_report` returns per-field accuracy for diagnosis.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from .ocr_metrics import cer


GRADED_FIELDS = (
    "regular_price",
    "loyalty_price",
    "product_name",
    "weight_value",
    "weight_unit",
    "price_per_unit_value",
    "price_per_unit_unit",
    "promo_flag",
    "currency",
)


def _approx_eq_float(a: Optional[float], b: Optional[float], tol: float) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol


def _eq_str_lc(a: Optional[str], b: Optional[str]) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return a.strip().lower() == b.strip().lower()


def _eq_text_cer(a: Optional[str], b: Optional[str], tau: float) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return cer(a.strip(), b.strip()) <= tau


def _field_match(pred: dict[str, Any], gt: dict[str, Any], field: str, name_tau: float) -> bool:
    p = pred.get(field)
    g = gt.get(field)
    if field in ("regular_price", "loyalty_price", "price_per_unit_value"):
        return _approx_eq_float(p, g, tol=0.005)
    if field == "weight_value":
        return _approx_eq_float(p, g, tol=0.001)
    if field in ("weight_unit", "currency", "price_per_unit_unit"):
        return _eq_str_lc(p, g)
    if field == "promo_flag":
        return bool(p) == bool(g)
    if field == "product_name":
        return _eq_text_cer(p, g, tau=name_tau)
    raise ValueError(f"Unknown graded field: {field}")


def e2e_field_accuracy(
    paired: Iterable[tuple[dict[str, Any], dict[str, Any]]],
    fields: tuple[str, ...] = GRADED_FIELDS,
    name_tau: float = 0.15,
) -> float:
    """Fraction of tags where ALL graded fields are simultaneously correct."""
    paired = list(paired)
    if not paired:
        return 0.0
    correct = 0
    for pred, gt in paired:
        if all(_field_match(pred, gt, f, name_tau) for f in fields):
            correct += 1
    return correct / len(paired)


def per_field_report(
    paired: Iterable[tuple[dict[str, Any], dict[str, Any]]],
    fields: tuple[str, ...] = GRADED_FIELDS,
    name_tau: float = 0.15,
) -> dict[str, float]:
    """Per-field accuracy across the paired list."""
    paired = list(paired)
    out: dict[str, float] = {}
    if not paired:
        return {f: 0.0 for f in fields}
    for f in fields:
        n_correct = sum(1 for pred, gt in paired if _field_match(pred, gt, f, name_tau))
        out[f] = n_correct / len(paired)
    return out
