"""Catalog reconciliation: cross-check / repair recognized ``barcode`` and
``product_name`` against the Lenta master catalog (``real_data/db_hack.csv``).

Local, network-free post-processing for the recognition output. Typical use::

    from price_tag_pipeline.catalog import CatalogIndex, CatalogReconciler

    rec = CatalogReconciler(CatalogIndex.load("real_data/db_hack.csv"))
    out = rec.reconcile(barcode="4 607018 308355", name_raw="Коктейль высокб")
    # out.barcode -> "4607018308355", out.product_name -> canonical fullname

Importing this package is cheap (RapidFuzz is the only heavy-ish dep and it is
light); the 625 k-row index is built lazily by :meth:`CatalogIndex.load`.
"""

from __future__ import annotations

from .index import CatalogEntry, CatalogIndex, NameMatch
from .normalize import (
    gtin_checksum_ok,
    is_absent_sentinel,
    is_missing,
    normalize_barcode,
    normalize_name,
)
from .reconcile import CatalogReconciler, ReconcileResult

__all__ = (
    "CatalogIndex",
    "CatalogEntry",
    "NameMatch",
    "CatalogReconciler",
    "ReconcileResult",
    "normalize_barcode",
    "normalize_name",
    "gtin_checksum_ok",
    "is_absent_sentinel",
    "is_missing",
)
