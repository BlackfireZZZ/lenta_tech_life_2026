"""Normalization + validation primitives shared by the catalog index and the
reconciler.

Two field-value semantics matter here and are easy to get wrong (see
``docs/index.md`` fact #5):

* **missing** — ``None`` / ``""`` : the recognizer did not read the field.
  Eligible to be *filled* from the catalog.
* **absent sentinel** — the literal ``"нет"`` : the field is *intentionally*
  not on the tag. It is a *correct* answer and must **never** be overwritten
  by a catalog guess, nor fed into fuzzy name matching.

:func:`is_absent_sentinel` is the single source of truth for that distinction.
"""

from __future__ import annotations

import re
import unicodedata

# Literal "field is not present on the tag" marker used by the hackathon CSV.
ABSENT_SENTINEL = "нет"

_DIGITS_RE = re.compile(r"\D+")
_NAME_KEEP_RE = re.compile(r"[^0-9a-zа-я ]+")
_WS_RE = re.compile(r"\s+")


def is_absent_sentinel(raw: str | None) -> bool:
    """True iff ``raw`` is the deliberate "not on the tag" marker (``"нет"``).

    Distinct from missing (``None``/``""``): a sentinel is a *correct* value and
    is left untouched by reconciliation.
    """
    return raw is not None and raw.strip().casefold() == ABSENT_SENTINEL


def is_missing(raw: str | None) -> bool:
    """True iff the recognizer produced no value (``None`` or blank).

    The sentinel ``"нет"`` is **not** missing — it is a real answer.
    """
    return raw is None or raw.strip() == ""


def normalize_barcode(raw: str | None) -> str:
    """Reduce a raw barcode reading to its bare digit string.

    Strips spaces (incl. NBSP), separators and any stray glyphs the OCR added,
    after NFKC folding (full-width digits → ASCII). The GT contains values like
    ``"4 607018 308355"`` — these normalize to ``"4607018308355"``.
    Returns ``""`` when nothing digit-like remains.
    """
    if not raw:
        return ""
    s = unicodedata.normalize("NFKC", raw)
    return _DIGITS_RE.sub("", s)


def gtin_checksum_ok(digits: str) -> bool | None:
    """Validate the GS1 mod-10 check digit (EAN-8/UPC-A/EAN-13/GTIN-14).

    Returns ``True``/``False`` for standard lengths {8, 12, 13, 14}, or
    ``None`` when the length is non-standard (cannot judge — e.g. truncated
    OCR or a non-GTIN internal code). ``None`` means "unknown", not "bad".
    """
    if not digits.isdigit() or len(digits) not in (8, 12, 13, 14):
        return None
    *body, check = (int(c) for c in digits)
    # GS1: weight alternates 3,1,... from the rightmost body digit leftward.
    total = 0
    for i, d in enumerate(reversed(body)):
        total += d * (3 if i % 2 == 0 else 1)
    return (10 - total % 10) % 10 == check


def normalize_name(raw: str | None) -> str:
    """Canonical comparison form for a product name.

    NFKC → casefold → ``ё``→``е`` → keep only Cyrillic/Latin letters, digits and
    single spaces. Digits are kept on purpose: volume/percent ("молоко 3 2",
    "0 5 л") is a strong product discriminator for fuzzy matching.
    """
    if not raw:
        return ""
    s = unicodedata.normalize("NFKC", raw).casefold().replace("ё", "е")
    s = _NAME_KEEP_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def first_token(name_norm: str) -> str:
    """First whitespace token of an already-normalized name (blocking key)."""
    return name_norm.split(" ", 1)[0] if name_norm else ""
