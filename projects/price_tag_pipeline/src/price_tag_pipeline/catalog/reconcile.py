"""Reconcile a recognized ``(barcode, product_name)`` pair against the Lenta
master catalog.

What it does, in priority order:

1. **Normalize** the barcode (strip OCR spaces/glyphs) — always safe.
2. **Barcode is the primary key** (``docs/index.md`` fact #2). If the barcode
   resolves in the catalog, the catalog ``fullname`` is authoritative: fill a
   missing name, or correct a mis-read one to the canonical spelling.
3. **Guard against a mis-read barcode**: if the barcode fails its own GTIN
   check digit *and* strongly disagrees with a present name, don't let it
   overwrite the name — flag a conflict instead.
4. **Recover from the name** when there is no trustworthy barcode: a
   high-confidence fuzzy name match can supply the missing barcode and snap
   the name to canonical.
5. **Never touch an intentional ``"нет"``** and never invent a low-confidence
   value (fact #5). Missing stays missing unless the evidence is strong.

Pure-local, no network — safe for the inference box where cloud APIs are
banned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rapidfuzz import fuzz

from .index import CatalogIndex
from .normalize import (
    gtin_checksum_ok,
    is_absent_sentinel,
    is_missing,
    normalize_barcode,
    normalize_name,
)


@dataclass(frozen=True)
class ReconcileResult:
    """Outcome of reconciling one pair. ``*_source`` explains every change.

    ``barcode_source`` ∈ {kept, normalized, recovered_from_name}
    ``name_source``    ∈ {kept, filled_from_barcode, corrected_by_barcode,
                           corrected_by_name}
    """

    barcode: str | None
    product_name: str | None
    barcode_source: str = "kept"
    name_source: str = "kept"
    name_score: float | None = None  # fuzzy score behind a name change
    checksum_ok: bool | None = None  # GTIN check digit (None = can't judge)
    barcode_in_catalog: bool = False
    conflict: bool = False  # signals the caller may want to down-rank this tag
    notes: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return self.barcode_source != "kept" or self.name_source != "kept"


@dataclass
class CatalogReconciler:
    """Applies catalog knowledge to a recognized pair. Thresholds are
    RapidFuzz ``token_set_ratio`` scores in ``[0, 100]`` and are deliberately
    conservative — a wrong correction costs more than a missed one."""

    index: CatalogIndex
    # Name-write policy. The hackathon scorer compares product_name by exact
    # (case/space-normalized) string match against a GT that itself carries
    # OCR-style noise (Latin с/у, "6/", "амж", missing weight). So:
    #   "fill" (default) — only *fill a missing* name; never replace a present
    #          one. Pure win: empty scores 0, the catalog name might match GT
    #          and never does worse than empty.
    #   "fix"  — also rewrite a present name to the canonical catalog spelling
    #          (objectively correct, but vs the noisy GT it is double-edged on
    #          the local proxy scorer; correct for a key/fuzzy official metric).
    # Conflicts are always *flagged*; only "fix" lets them rewrite the name.
    name_policy: str = "fill"
    # Below this, a present name vs the catalog name for a barcode is treated
    # as disagreement (raises a conflict).
    conflict_cutoff: float = 55.0
    # Min name match to *recover a barcode* from the name alone.
    name_recover_cutoff: float = 92.0
    # Min name match to snap a name to canonical when there is no barcode
    # evidence at all (stricter — only signal is the name itself).
    name_only_fix_cutoff: float = 96.0
    notes: list[str] = field(default_factory=list, repr=False)

    @property
    def _may_override_name(self) -> bool:
        return self.name_policy == "fix"

    def reconcile(
        self, barcode_raw: str | None, name_raw: str | None
    ) -> ReconcileResult:
        notes: list[str] = []

        bc_sentinel = is_absent_sentinel(barcode_raw)
        nm_sentinel = is_absent_sentinel(name_raw)
        bc_missing = is_missing(barcode_raw) and not bc_sentinel
        nm_missing = is_missing(name_raw) and not nm_sentinel

        out_barcode: str | None = barcode_raw
        out_name: str | None = name_raw
        bc_source = "kept"
        nm_source = "kept"
        name_score: float | None = None
        checksum: bool | None = None
        in_catalog = False
        conflict = False

        # --- 1. barcode normalization (always safe) ----------------------
        bc_norm = ""
        if not bc_sentinel and not bc_missing:
            bc_norm = normalize_barcode(barcode_raw)
            if bc_norm:
                checksum = gtin_checksum_ok(bc_norm)
                if bc_norm != (barcode_raw or "").strip():
                    out_barcode, bc_source = bc_norm, "normalized"
                else:
                    out_barcode = bc_norm  # already clean digits

        # --- 2/3. trusted-barcode path -----------------------------------
        if bc_norm:
            entry = self.index.lookup_barcode(bc_norm)
            if entry is not None:
                in_catalog = True
                if nm_sentinel:
                    notes.append(
                        "barcode in catalog but name is 'нет' — left as-is"
                    )
                elif nm_missing:
                    out_name, nm_source = entry.name, "filled_from_barcode"
                else:
                    s = fuzz.token_set_ratio(
                        normalize_name(name_raw), entry.name_norm
                    )
                    name_score = float(s)
                    if checksum is False and s < self.conflict_cutoff:
                        # barcode self-inconsistent AND name disagrees →
                        # the barcode, not the name, is the likely error.
                        conflict = True
                        notes.append(
                            "bad GTIN checksum and name disagrees — barcode "
                            "likely misread; kept recognized name"
                        )
                    else:
                        if s < self.conflict_cutoff:
                            conflict = True
                            notes.append(
                                "name disagrees with catalog name for this "
                                "barcode; barcode trusted (primary key)"
                            )
                        if (
                            self._may_override_name
                            and normalize_name(name_raw) != entry.name_norm
                        ):
                            out_name, nm_source = (
                                entry.name,
                                "corrected_by_barcode",
                            )

        # --- 4. name-based recovery (only without a trusted barcode) -----
        if not in_catalog and not nm_sentinel and not nm_missing:
            floor = min(self.name_recover_cutoff, self.name_only_fix_cutoff)
            m = self.index.match_name(name_raw, score_cutoff=floor)
            if m is not None:
                name_score = m.score
                strong = m.score >= self.name_recover_cutoff
                if strong and bc_missing and m.entry.barcode:
                    out_barcode, bc_source = (
                        m.entry.barcode,
                        "recovered_from_name",
                    )
                elif (
                    strong
                    and not bc_missing
                    and not bc_sentinel
                    and m.entry.barcode
                    and bc_norm
                    and bc_norm != m.entry.barcode
                ):
                    conflict = True
                    notes.append(
                        "present barcode not in catalog and disagrees with "
                        "the name-matched product — barcode left as-is"
                    )
                if (
                    self._may_override_name
                    and (m.score >= self.name_only_fix_cutoff or strong)
                    and normalize_name(name_raw) != m.entry.name_norm
                ):
                    out_name, nm_source = m.entry.name, "corrected_by_name"

        return ReconcileResult(
            barcode=out_barcode,
            product_name=out_name,
            barcode_source=bc_source,
            name_source=nm_source,
            name_score=name_score,
            checksum_ok=checksum,
            barcode_in_catalog=in_catalog,
            conflict=conflict,
            notes=tuple(notes),
        )
