"""The Lenta master-catalog index.

Loads ``real_data/db_hack.csv`` (``fullname;code``, cp1251, ~625 k rows) once
and exposes two lookups the reconciler needs:

* exact **barcode → entry** (leading-zero tolerant), and
* fuzzy **name → entry** (RapidFuzz ``token_set_ratio`` with first-token
  blocking + an optional full-scan fallback).

Building the fuzzy structures over 625 k rows takes a couple of seconds, so the
core is pickled next to the CSV and reused while the CSV is unchanged
(cache key = file size + mtime).
"""

from __future__ import annotations

import csv
import pickle
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz, process

from .normalize import first_token, normalize_barcode, normalize_name

_CACHE_VERSION = 2  # bump when the pickled layout or normalization changes


@dataclass(frozen=True)
class CatalogEntry:
    """One catalog row. ``name`` keeps the original casing for output;
    ``name_norm`` is the precomputed comparison form."""

    barcode: str  # normalized digits ("" if the row had no usable code)
    name: str  # original fullname, original case
    name_norm: str


@dataclass(frozen=True)
class NameMatch:
    entry: CatalogEntry
    score: float  # RapidFuzz token_set_ratio in [0, 100]


class CatalogIndex:
    """In-memory catalog with exact-barcode and fuzzy-name lookup."""

    def __init__(
        self,
        entries: list[CatalogEntry],
        by_barcode: dict[str, int],
        by_barcode_int: dict[int, int],
        block: dict[str, list[int]],
        *,
        barcode_collisions: int = 0,
    ) -> None:
        self.entries = entries
        self._by_barcode = by_barcode  # exact digits  -> entry index
        self._by_barcode_int = by_barcode_int  # int(digits) -> entry index
        self._block = block  # first name token -> entry indices
        self._names_norm = [e.name_norm for e in entries]  # aligned, full scan
        self.barcode_collisions = barcode_collisions

    # ----------------------------------------------------------------- build
    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[str, str]]) -> "CatalogIndex":
        """Build from ``(fullname, code)`` pairs. First occurrence of a
        barcode wins; later collisions are counted, not overwritten."""
        entries: list[CatalogEntry] = []
        by_barcode: dict[str, int] = {}
        by_barcode_int: dict[int, int] = {}
        block: dict[str, list[int]] = {}
        collisions = 0

        for name, code in pairs:
            name = (name or "").strip()
            bc = normalize_barcode(code)
            nnorm = normalize_name(name)
            idx = len(entries)
            entries.append(CatalogEntry(barcode=bc, name=name, name_norm=nnorm))

            if bc:
                if bc in by_barcode:
                    collisions += 1
                else:
                    by_barcode[bc] = idx
                    by_barcode_int.setdefault(int(bc), idx)

            tok = first_token(nnorm)
            if tok:
                block.setdefault(tok, []).append(idx)

        return cls(
            entries,
            by_barcode,
            by_barcode_int,
            block,
            barcode_collisions=collisions,
        )

    @classmethod
    def load(
        cls,
        csv_path: str | Path,
        *,
        encoding: str = "cp1251",
        delimiter: str = ";",
        cache: bool = True,
        cache_path: str | Path | None = None,
    ) -> "CatalogIndex":
        """Load the catalog CSV, using an on-disk pickle cache when fresh."""
        csv_path = Path(csv_path)
        st = csv_path.stat()
        cache_key = (_CACHE_VERSION, st.st_size, int(st.st_mtime))
        cpath = (
            Path(cache_path)
            if cache_path is not None
            else csv_path.with_suffix(csv_path.suffix + ".idxcache.pkl")
        )

        if cache and cpath.exists():
            try:
                with cpath.open("rb") as fh:
                    blob = pickle.load(fh)
                if blob.get("key") == cache_key:
                    return cls(
                        blob["entries"],
                        blob["by_barcode"],
                        blob["by_barcode_int"],
                        blob["block"],
                        barcode_collisions=blob.get("collisions", 0),
                    )
            except Exception:  # noqa: BLE001 - a stale/corrupt cache just rebuilds
                pass

        with csv_path.open("r", encoding=encoding, newline="") as fh:
            reader = csv.reader(fh, delimiter=delimiter)
            header = next(reader, None)
            # Tolerate either a real header ("fullname;code") or a headerless
            # file: if row 0 doesn't look like the header, treat it as data.
            rows: Iterable[tuple[str, str]]
            first_is_header = bool(
                header and header[0].strip().lower() in ("fullname", "name")
            )
            seed = [] if first_is_header else ([tuple(header[:2])] if header else [])

            def _iter() -> Iterable[tuple[str, str]]:
                for r in seed:
                    yield (r[0], r[1] if len(r) > 1 else "")
                for r in reader:
                    if not r:
                        continue
                    yield (r[0], r[1] if len(r) > 1 else "")

            index = cls.from_pairs(_iter())

        if cache:
            try:
                tmp = cpath.with_suffix(cpath.suffix + ".tmp")
                with tmp.open("wb") as fh:
                    pickle.dump(
                        {
                            "key": cache_key,
                            "entries": index.entries,
                            "by_barcode": index._by_barcode,
                            "by_barcode_int": index._by_barcode_int,
                            "block": index._block,
                            "collisions": index.barcode_collisions,
                        },
                        fh,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                tmp.replace(cpath)
            except Exception:  # noqa: BLE001 - caching is best-effort
                pass

        return index

    # ---------------------------------------------------------------- lookup
    def __len__(self) -> int:
        return len(self.entries)

    def lookup_barcode(self, raw: str | None) -> CatalogEntry | None:
        """Exact barcode lookup, tolerant of leading-zero / zero-pad drift
        (``0460…`` vs ``460…``). Returns ``None`` if not found."""
        bc = normalize_barcode(raw)
        if not bc:
            return None
        i = self._by_barcode.get(bc)
        if i is None:
            i = self._by_barcode_int.get(int(bc))
        return None if i is None else self.entries[i]

    def match_name(
        self,
        raw: str | None,
        *,
        score_cutoff: float = 88.0,
        full_scan_on_miss: bool = True,
    ) -> NameMatch | None:
        """Best fuzzy catalog match for a (possibly OCR-mangled) name.

        Blocks on the first normalized token first (fast, covers the common
        case); falls back to a full scan when the block misses or its best
        score is under ``score_cutoff`` and the leading word may itself be a
        typo. Returns ``None`` when nothing clears ``score_cutoff``.
        """
        q = normalize_name(raw)
        if not q:
            return None

        scorer = fuzz.token_set_ratio
        best_idx: int | None = None
        best_score = -1.0

        tok = first_token(q)
        block_idxs = self._block.get(tok)
        if block_idxs:
            choices = {i: self._names_norm[i] for i in block_idxs}
            hit = process.extractOne(q, choices, scorer=scorer)
            if hit is not None:
                _name, score, key = hit
                best_idx, best_score = key, score

        if (best_score < score_cutoff) and full_scan_on_miss:
            hit = process.extractOne(q, self._names_norm, scorer=scorer)
            if hit is not None:
                _name, score, idx = hit
                if score > best_score:
                    best_idx, best_score = idx, score

        if best_idx is None or best_score < score_cutoff:
            return None
        return NameMatch(entry=self.entries[best_idx], score=float(best_score))
