#!/usr/bin/env python3
"""Bootstrap a ready-to-annotate CVAT instance via the CVAT SDK.

Idempotently creates:

1. project "Lenta price tags" with the exact ``price_tag`` label schema
   the importer expects (``cvat.build_label_spec``);
2. one **video task per released video** (``real_data/dataset/<id>/<id>.mp4``)
   inside that project;
3. the matching ``cvat_seeds/<id>.cvat.xml`` imported into each task, so the
   noisy released boxes show up as tracks to *correct* rather than redraw.

After this you only open http://localhost:8080 and annotate.

Re-running skips a project / task / seed that already exists, so it is safe
to run again after adding more videos.

    .venv/Scripts/python.exe projects/price_tag_pipeline/scripts/cvat_bootstrap.py \\
        --src "E:/Hackatons/lenta_tech_life_2026/real_data/dataset" \\
        --seeds cvat_seeds --user admin --password '***'

Credentials may instead come from $CVAT_USER / $CVAT_PASSWORD.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
SRC = THIS.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from price_tag_pipeline.data.cvat import build_label_spec  # noqa: E402

LOGGER = logging.getLogger("cvat_bootstrap")
PROJECT_NAME = "Lenta price tags"
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def _find_video(folder: Path, stem: str) -> Path | None:
    return next((folder / f"{stem}{e}" for e in VIDEO_EXTS
                 if (folder / f"{stem}{e}").exists()), None)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=Path("real_data/dataset"))
    p.add_argument("--seeds", type=Path, default=Path("cvat_seeds"))
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--user", default=os.environ.get("CVAT_USER", ""))
    p.add_argument("--password", default=os.environ.get("CVAT_PASSWORD", ""))
    p.add_argument("--only", help="Comma-separated video_ids to limit to")
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    logging.basicConfig(level=getattr(logging, a.log_level), format="%(levelname)s: %(message)s")

    if not a.user or not a.password:
        LOGGER.error("Need --user/--password (or $CVAT_USER/$CVAT_PASSWORD).")
        return 2

    try:
        from cvat_sdk import make_client
        from cvat_sdk.core.proxies.tasks import ResourceType
    except ModuleNotFoundError:
        LOGGER.error("cvat-sdk not installed. In the venv: "
                     "uv pip install --python .venv cvat-sdk")
        return 2

    only = {s.strip() for s in a.only.split(",")} if a.only else None

    with make_client(host=a.host, port=a.port, credentials=(a.user, a.password)) as client:
        # 1. project (reuse if present) -----------------------------------
        project = next(
            (pr for pr in client.projects.list() if pr.name == PROJECT_NAME), None
        )
        if project is None:
            project = client.projects.create({
                "name": PROJECT_NAME,
                "labels": build_label_spec(),
            })
            LOGGER.info("Created project %r (id=%d)", PROJECT_NAME, project.id)
        else:
            LOGGER.info("Reusing project %r (id=%d)", PROJECT_NAME, project.id)

        existing = {t.name for t in client.tasks.list() if t.project_id == project.id}

        # 2/3. one task per video + seed import ---------------------------
        made = 0
        for entry in sorted(a.src.iterdir()):
            if not entry.is_dir() or entry.name.lower() == "unlabeled":
                continue
            vid = entry.name
            if only and vid not in only:
                continue
            video = _find_video(entry, vid)
            if video is None:
                LOGGER.warning("No video for %s — skipped", vid)
                continue
            if vid in existing:
                LOGGER.info("Task %r already exists — skipped", vid)
                continue

            LOGGER.info("Creating task %r (uploading %s)…", vid, video.name)
            task = client.tasks.create_from_data(
                spec={"name": vid, "project_id": project.id},
                resource_type=ResourceType.LOCAL,
                resources=[str(video)],
            )

            seed = a.seeds / f"{vid}.cvat.xml"
            if seed.exists():
                LOGGER.info("Importing seed %s into task %d…", seed.name, task.id)
                task.import_annotations(format_name="CVAT 1.1", filename=str(seed))
            else:
                LOGGER.warning("No seed %s — task created without pre-labels", seed)
            made += 1

        LOGGER.info(
            "Done. project=%r tasks_created=%d -> http://%s:%d",
            PROJECT_NAME, made, a.host, a.port,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
