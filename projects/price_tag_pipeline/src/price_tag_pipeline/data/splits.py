"""Video-level GroupKFold split, optionally stratified by store/aisle metadata.

Why video-level: adjacent frames are near-duplicates. Frame-level splits leak
visual style AND sometimes the same physical tag across train/val, inflating
the val score.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger(__name__)


def load_metadata(metadata_path: Optional[Path]) -> dict[str, dict[str, str]]:
    """Load optional per-video metadata. Columns: video_id, store_id, aisle_id, camera_id, recorded_at."""
    if metadata_path is None or not metadata_path.exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    with metadata_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row.get("video_id")
            if not vid:
                continue
            out[vid] = {k: (v or "") for k, v in row.items() if k != "video_id"}
    return out


def build_video_level_folds(
    video_ids: list[str],
    n_splits: int = 5,
    metadata: Optional[dict[str, dict[str, str]]] = None,
    seed: int = 42,
) -> list[dict[str, list[str]]]:
    """Return a list of {"train": [...], "val": [...]} dicts of video_ids."""
    metadata = metadata or {}
    video_ids = sorted(set(video_ids))
    if len(video_ids) < n_splits:
        LOGGER.warning(
            "Only %d videos for n_splits=%d; reducing n_splits to %d",
            len(video_ids), n_splits, max(2, len(video_ids))
        )
        n_splits = max(2, min(n_splits, len(video_ids)))

    strat_key = _build_stratification_key(video_ids, metadata)
    try:
        from sklearn.model_selection import StratifiedKFold, KFold
        if strat_key is not None and len(set(strat_key)) > 1:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            indices = list(splitter.split(video_ids, strat_key))
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            indices = list(splitter.split(video_ids))
    except ImportError:
        LOGGER.warning("sklearn not available; falling back to round-robin K-fold")
        indices = _manual_kfold(len(video_ids), n_splits, seed=seed)

    folds = []
    for train_idx, val_idx in indices:
        folds.append({
            "train": [video_ids[i] for i in train_idx],
            "val": [video_ids[i] for i in val_idx],
        })
    return folds


def _build_stratification_key(
    video_ids: list[str],
    metadata: dict[str, dict[str, str]],
) -> Optional[list[str]]:
    if not metadata:
        return None
    keys: list[str] = []
    for vid in video_ids:
        meta = metadata.get(vid, {})
        store = meta.get("store_id", "")
        aisle = meta.get("aisle_id", "")
        keys.append(f"{store}|{aisle}")
    if len(set(keys)) <= 1:
        return None
    return keys


def _manual_kfold(n: int, k: int, seed: int = 42) -> list[tuple[list[int], list[int]]]:
    import random
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    folds: list[list[int]] = [[] for _ in range(k)]
    for i, j in enumerate(idx):
        folds[i % k].append(j)
    out: list[tuple[list[int], list[int]]] = []
    for i in range(k):
        val = sorted(folds[i])
        train = sorted([j for f in folds[:i] + folds[i + 1:] for j in f])
        out.append((train, val))
    return out


def write_folds(folds: list[dict[str, list[str]]], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, fold in enumerate(folds):
        path = out_dir / f"fold_{i}.json"
        path.write_text(json.dumps(fold, indent=2, ensure_ascii=False), encoding="utf-8")
        paths.append(path)
    return paths


def load_fold(fold_path: Path) -> dict[str, list[str]]:
    return json.loads(fold_path.read_text(encoding="utf-8"))
