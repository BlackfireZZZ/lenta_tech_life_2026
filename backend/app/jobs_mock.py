"""Deterministic mock price-tag predictions for the gateway skeleton.

The ML service is mocked (docs/architecture.md §5.4). Until it is live, the
gateway fabricates a realistic set of tags here so the whole product —
upload → poll → review → CSV — is reviewable end to end.

Determinism matters: the *same* job id always yields the *same* tags, so the
CSV (``build_csv``) and the review JSON (``build_predictions``) never
disagree, and polling the job repeatedly doesn't reshuffle the result.

Realism follows docs/hackathon/task.md: the five shooting zones (alcohol /
dairy / honey / jams / syrups), the white/red/yellow/green ``color`` logic
(§4), the <100 ₽ → percent / ≥100 ₽ → ruble ``discount_amount`` rule (§4),
EAN-13 barcodes with the occasional 12-digit partial (§6.6), tags with no QR
at all, and the three field states (value / ``"нет"`` / ``""``) from §3.3.
Nothing here is graded — it only makes the mock honest to look at.
"""

from __future__ import annotations

import csv
import io
from random import Random
from uuid import UUID

from app.api.v1.schemas.job import (
    CSV_COLUMNS,
    SUBSTANTIVE_FIELDS,
    BBoxNorm,
    JobPredictions,
    TagPrediction,
)

# Nominal frame the pixel bbox / CSV coords are written against. The robot
# camera is mounted rotated 90° (task.md §2), so a decoded frame is portrait:
# 3840×2160 rotated → 2160 wide × 3840 tall.
FRAME_WIDTH = 2160
FRAME_HEIGHT = 3840

ABSENT = "нет"  # field is not present on this tag (task.md §3.3)
UNREC = ""  # present on the tag but not recognized (task.md §3.3)

# Curated catalogue across the five filmed zones. (name, default ₽, card ₽,
# zone-color). card == default means "no loyalty price on this tag".
_CATALOGUE: list[tuple[str, str, str, str]] = [
    ("Вино красное сухое Киндзмараули 0.75л", "899.00", "749.00", "white"),
    ("Водка Пшеничная 0.5л", "529.00", "479.00", "white"),
    ("Коньяк армянский Арарат 5 лет 0.5л", "1899.00", "1699.00", "red"),
    ("Виски Bell's 0.7л", "1499.00", "1299.00", "red"),
    ("Молоко Домик в деревне 3.2% 0.93л", "94.99", "84.99", "white"),
    ("Кефир Простоквашино 1% 0.93л", "89.99", "79.99", "white"),
    ("Сметана Брест-Литовск 20% 350г", "129.00", "109.00", "white"),
    ("Творог Савушкин 5% 300г", "159.00", "139.00", "red"),
    ("Сыр Российский 45% весовой", "749.00", "699.00", "white"),
    ("Мёд цветочный натуральный 500г", "399.00", "349.00", "white"),
    ("Мёд гречишный 350г", "459.00", "459.00", "white"),
    ("Джем абрикосовый Махеевъ 300г", "139.00", "119.00", "white"),
    ("Джем клубничный домашний 320г", "189.00", "159.00", "red"),
    ("Сироп клёновый канадский 250мл", "699.00", "629.00", "white"),
    ("Сироп Monin Карамель 0.7л", "899.00", "799.00", "white"),
    ("Хлеб бородинский нарезка 390г", "65.99", "59.99", "yellow"),
    ("Багет французский 250г", "79.99", "69.99", "yellow"),
    ("Яблоки Голден весовые", "129.99", "109.99", "green"),
    ("Бананы весовые", "89.99", "79.99", "green"),
    ("Помидоры Черри 250г", "199.00", "169.00", "green"),
]

_DISPLAY_SYMBOLS = ["к", "л", "ш"]


def _ean13(rng: Random) -> str:
    """A plausible 13-digit barcode; occasionally a 12-digit partial — a
    partial barcode is better than none (task.md §6.6)."""
    digits = "460" + "".join(str(rng.randint(0, 9)) for _ in range(10))
    if rng.random() < 0.18:  # one digit lost in motion blur
        return digits[:12]
    return digits


def _price_num(s: str) -> float:
    return float(s)


def _build_one(rng: Random, index: int, filename: str, n: int) -> TagPrediction:
    name, price_default, price_card, color = rng.choice(_CATALOGUE)
    has_card = price_card != price_default

    # Promo tags (red) carry a discount marker; the <100 ₽ → % / ≥100 ₽ → ₽
    # rule is task.md §4.
    if color == "red":
        diff = round(_price_num(price_default) - _price_num(price_card), 2)
        price_discount = price_card
        discount_amount = f"-{int(diff)}₽" if diff >= 100 else f"-{int(round(diff / _price_num(price_default) * 100))}%"
    else:
        price_discount = ABSENT
        discount_amount = ABSENT

    barcode = _ean13(rng)

    # ~30% of tags have no QR block at all → every QR field is "нет".
    # Of the rest, a few have a QR that simply didn't decode → "" (unrec).
    has_qr = rng.random() > 0.30
    qr_unrecognized = has_qr and rng.random() < 0.20

    def qr(value: str) -> str:
        if not has_qr:
            return ABSENT
        if qr_unrecognized:
            return UNREC
        return value

    fields: dict[str, str] = {
        "filename": filename,
        "product_name": name,
        "price_default": price_default,
        "price_card": price_card if has_card else ABSENT,
        "price_discount": price_discount,
        "barcode": barcode,
        "discount_amount": discount_amount,
        "id_sku": str(rng.randint(100000, 999999)),
        "print_datetime": f"2026-05-{rng.randint(10, 17):02d} {rng.randint(6, 21):02d}:{rng.randint(0, 59):02d}",
        "code": f"{rng.choice('ABCDEF')}{rng.randint(1, 24):02d}",
        "additional_info": ABSENT if rng.random() < 0.6 else "акция 1+1",
        "color": color,
        "special_symbols": rng.choice(_DISPLAY_SYMBOLS) if rng.random() < 0.35 else ABSENT,
        "qr_code_barcode": qr(barcode),
        "price1_qr": qr(price_default),
        "price2_qr": qr(price_card if has_card else price_default),
        "price3_qr": qr(price_discount if price_discount != ABSENT else price_default),
        "price4_qr": qr(ABSENT) if has_qr else ABSENT,
        "wholesale_level_1_count": qr("6"),
        "wholesale_level_1_price": qr(str(round(_price_num(price_default) * 0.95, 2))),
        "wholesale_level_2_count": qr("12"),
        "wholesale_level_2_price": qr(str(round(_price_num(price_default) * 0.90, 2))),
        "action_price_qr": qr(price_discount) if price_discount != ABSENT else qr(ABSENT),
        "action_code_qr": qr(str(rng.randint(1000, 9999))) if price_discount != ABSENT else (ABSENT if not has_qr else ABSENT),
    }

    # A couple of substantive fields left unrecognized on some tags, so the
    # per-tag completeness shown in the UI is honest about the 80% threshold
    # (task.md §5.1) rather than a perfect 100% everywhere.
    if rng.random() < 0.45:
        victim = rng.choice(["product_name", "id_sku", "print_datetime", "code"])
        fields[victim] = UNREC

    # Bounding box in normalized [0,1] coords. Tags sit across the shelf width
    # and in the lower ~75% of the portrait frame.
    bw = rng.uniform(0.16, 0.30)
    bh = rng.uniform(0.09, 0.17)
    x1 = rng.uniform(0.04, 1.0 - bw - 0.04)
    y1 = rng.uniform(0.18, 1.0 - bh - 0.06)
    bbox = BBoxNorm(x1=round(x1, 4), y1=round(y1, 4), x2=round(x1 + bw, 4), y2=round(y1 + bh, 4))

    # Pixel coords for the CSV, derived from the same normalized box.
    fields["x_min"] = str(round(bbox.x1 * FRAME_WIDTH))
    fields["y_min"] = str(round(bbox.y1 * FRAME_HEIGHT))
    fields["x_max"] = str(round(bbox.x2 * FRAME_WIDTH))
    fields["y_max"] = str(round(bbox.y2 * FRAME_HEIGHT))

    # One timestamp per unique tag (task.md §6.3): spread across the clip.
    t_frac = round((index + 0.5) / n * 0.9 + rng.uniform(-0.02, 0.02) + 0.03, 4)
    t_frac = min(0.97, max(0.02, t_frac))
    frame_timestamp = int(t_frac * 60000)  # nominal ms; UI seeks by t_frac
    fields["frame_timestamp"] = str(frame_timestamp)

    return TagPrediction(
        index=index,
        color=color,
        frame_timestamp=frame_timestamp,
        t_frac=t_frac,
        bbox=bbox,
        fields=fields,
    )


def generate_tags(job_id: UUID, filename: str) -> list[TagPrediction]:
    """Deterministic per-job tag set (8–14 unique tags)."""
    rng = Random(job_id.int)
    n = rng.randint(8, 14)
    return [_build_one(rng, i, filename, n) for i in range(n)]


def build_predictions(
    job_id: UUID, filename: str, tags: list[TagPrediction], video_url: str, csv_url: str
) -> JobPredictions:
    return JobPredictions(
        job_id=job_id,
        filename=filename,
        columns=CSV_COLUMNS,
        substantive_fields=SUBSTANTIVE_FIELDS,
        video_url=video_url,
        csv_url=csv_url,
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
        tags=tags,
    )


def build_csv(tags: list[TagPrediction]) -> str:
    """The graded 29-column CSV (task.md §3): comma separator, ``.`` decimal,
    UTF-8, minimal quoting (a field with a comma/quote is wrapped in ``"`` and
    inner quotes doubled). One row per unique tag, columns in CSV_COLUMNS
    order. Values are written exactly as predicted, so ``"нет"`` and empty
    (unrecognized) are preserved — confusing them loses points (task.md §5.3).
    """
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(CSV_COLUMNS)
    for tag in tags:
        writer.writerow([tag.fields.get(col, UNREC) for col in CSV_COLUMNS])
    return buf.getvalue()
