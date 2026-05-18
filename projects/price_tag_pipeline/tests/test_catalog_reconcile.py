"""Catalog reconciliation: normalization, GTIN checksum, and the
:class:`CatalogReconciler` decision policy.

No 625 k load — a tiny in-memory :meth:`CatalogIndex.from_pairs` fixture.
The barcodes below have *real* GS1 check digits so checksum behaviour is
exercised honestly:

* ``4607018308355`` valid EAN-13   (Коктейль)
* ``4690228006432`` valid EAN-13   (Молоко)
* ``96385074``      valid EAN-8
* ``4607018308350`` **invalid** EAN-13 (wrong check digit, still indexed)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from price_tag_pipeline.catalog import (  # noqa: E402
    CatalogIndex,
    CatalogReconciler,
    gtin_checksum_ok,
    is_absent_sentinel,
    is_missing,
    normalize_barcode,
    normalize_name,
)

KOKTEIL = "Коктейль высокобелковый к/м ACTIVE ENERGY Тропический микс 230мл"
MOLOKO = "Молоко ПРОСТОКВАШИНО отборное пастеризованное 3,4-6% 930мл"
SYR = "Сыр ХОХЛАНД сливочный 140г"
PECHENYE = "Печенье ЮБИЛЕЙНОЕ молочное 232г"


@pytest.fixture(scope="module")
def index() -> CatalogIndex:
    return CatalogIndex.from_pairs([
        (KOKTEIL, "4607018308355"),
        (MOLOKO, "4690228006432"),
        (SYR, "96385074"),
        (PECHENYE, "4607018308350"),  # bad checksum, still a catalog member
    ])


@pytest.fixture(scope="module")
def rec(index: CatalogIndex) -> CatalogReconciler:
    """Default conservative policy: fill missing only, never override."""
    return CatalogReconciler(index)


@pytest.fixture(scope="module")
def rec_fix(index: CatalogIndex) -> CatalogReconciler:
    """Aggressive policy: also rewrite a present name to canonical."""
    return CatalogReconciler(index, name_policy="fix")


# --------------------------------------------------------------- primitives
def test_gtin_checksum():
    assert gtin_checksum_ok("4607018308355") is True
    assert gtin_checksum_ok("4690228006432") is True
    assert gtin_checksum_ok("96385074") is True            # EAN-8
    assert gtin_checksum_ok("4607018308350") is False       # wrong check digit
    assert gtin_checksum_ok("12345") is None                # non-standard length
    assert gtin_checksum_ok("abc") is None


def test_normalize_barcode():
    assert normalize_barcode("4 607018 308355") == "4607018308355"
    assert normalize_barcode(" 4607018308355\xa0") == "4607018308355"
    assert normalize_barcode("ШК: 4607018308355") == "4607018308355"
    assert normalize_barcode(None) == ""
    assert normalize_barcode("нет") == ""


def test_normalize_name():
    assert normalize_name("Сыр  ХОХЛАНД, сливочный (140г)") == \
        "сыр хохланд сливочный 140г"
    assert normalize_name("Ёлка") == "елка"
    assert normalize_name(None) == ""


def test_sentinels():
    assert is_absent_sentinel("нет")
    assert is_absent_sentinel("  НЕТ ")
    assert not is_absent_sentinel("нет в наличии")
    assert is_missing(None) and is_missing("   ")
    assert not is_missing("нет")  # a real answer, not a miss


# --------------------------------------------------------------- reconcile
def test_barcode_in_catalog_fills_missing_name(rec):
    r = rec.reconcile("4690228006432", "")
    assert r.barcode == "4690228006432"
    assert r.product_name == MOLOKO
    assert r.name_source == "filled_from_barcode"
    assert r.barcode_in_catalog and not r.conflict


def test_spaced_barcode_normalized_default_keeps_present_name(rec):
    # Default "fill" policy: barcode is cleaned (safe), but a present (typo'd)
    # name is NOT rewritten — the noisy-GT scorer makes that double-edged.
    r = rec.reconcile("4 690228 006432", "Молоко ПРОСТКВАШИНО отборн 3,4-6% 930мл")
    assert r.barcode == "4690228006432"
    assert r.barcode_source == "normalized"
    assert r.product_name == "Молоко ПРОСТКВАШИНО отборн 3,4-6% 930мл"
    assert r.name_source == "kept"
    assert not r.conflict


def test_spaced_barcode_name_typo_fixed_under_fix_policy(rec_fix):
    r = rec_fix.reconcile("4 690228 006432", "Молоко ПРОСТКВАШИНО отборн 3,4-6% 930мл")
    assert r.barcode == "4690228006432"
    assert r.product_name == MOLOKO
    assert r.name_source == "corrected_by_barcode"


def test_correct_pair_is_left_unchanged(rec):
    r = rec.reconcile("4607018308355", KOKTEIL)
    assert r.product_name == KOKTEIL
    assert r.name_source == "kept"
    assert not r.changed or r.barcode_source != "kept"  # at most cosmetic


def test_leading_zero_barcode_still_resolves(rec):
    r = rec.reconcile("04690228006432", "")
    assert r.barcode_in_catalog
    assert r.product_name == MOLOKO


def test_recover_missing_barcode_from_strong_name_is_policy_independent(rec):
    # Recovering a *missing* barcode is a pure win and happens under the
    # default policy; the present name is left as recognized.
    q = "Коктейль высокобелковый ACTIVE ENERGY Тропический микс 230мл"
    r = rec.reconcile("", q)
    assert r.barcode == "4607018308355"
    assert r.barcode_source == "recovered_from_name"
    assert r.product_name == q
    assert r.name_source == "kept"


def test_recover_barcode_and_canonicalize_name_under_fix(rec_fix):
    r = rec_fix.reconcile(
        "", "Коктейль высокобелковый ACTIVE ENERGY Тропический микс 230мл"
    )
    assert r.barcode == "4607018308355"
    assert r.barcode_source == "recovered_from_name"
    assert r.product_name == KOKTEIL
    assert r.name_source == "corrected_by_name"


def test_garbage_name_no_barcode_unchanged(rec):
    r = rec.reconcile("", "zzz qwerty 99999")
    assert r.barcode == ""
    assert not r.changed


def test_name_sentinel_not_overwritten(rec):
    r = rec.reconcile("4690228006432", "нет")
    assert r.product_name == "нет"          # deliberate absence preserved
    assert r.name_source == "kept"
    assert r.barcode_in_catalog
    assert any("нет" in n for n in r.notes)


def test_barcode_sentinel_never_recovered(rec_fix):
    # Even the aggressive policy must not invent a barcode over a "нет".
    r = rec_fix.reconcile("нет", KOKTEIL)
    assert r.barcode == "нет"
    assert r.barcode_source == "kept"
    assert r.product_name == KOKTEIL


def test_conflict_is_flagged_but_default_keeps_name(rec):
    # Valid-checksum barcode for Молоко, name says Сыр: conflict is raised,
    # but the default policy does not rewrite the present name.
    r = rec.reconcile("4690228006432", SYR)
    assert r.conflict
    assert r.product_name == SYR
    assert r.name_source == "kept"


def test_conflict_valid_barcode_wins_over_name_under_fix(rec_fix):
    r = rec_fix.reconcile("4690228006432", SYR)
    assert r.conflict
    assert r.product_name == MOLOKO         # barcode is the primary key
    assert r.name_source == "corrected_by_barcode"


def test_bad_checksum_barcode_does_not_override_disagreeing_name(rec):
    # Barcode is in the catalog but its own GTIN check digit is wrong AND the
    # name disagrees -> the barcode is the likely misread; keep the name.
    r = rec.reconcile("4607018308350", "Совершенно другое название XYZ")
    assert r.conflict
    assert r.product_name == "Совершенно другое название XYZ"
    assert r.name_source == "kept"
    assert any("misread" in n for n in r.notes)
