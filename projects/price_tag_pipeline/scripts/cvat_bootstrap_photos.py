#!/usr/bin/env python3
"""Create CVAT image tasks (scenes) from a cvat_from_external manifest.

For each scene: one **image** task inside the existing "Lenta price tags"
project, all the scene's photos uploaded, the scene's seed XML imported so
the friend's boxes are there to validate. No tracks (image task). Idempotent
— a scene whose task name already exists is skipped.

    .venv/Scripts/python.exe projects/price_tag_pipeline/scripts/cvat_bootstrap_photos.py \\
        --manifest cvat_seeds_friends/manifest.json --user admin --password ***
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
SRC = THIS.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from price_tag_pipeline.data.cvat import build_label_spec  # noqa: E402

LOGGER = logging.getLogger("cvat_bootstrap_photos")
PROJECT_NAME = "Lenta price tags"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--user", default=os.environ.get("CVAT_USER", ""))
    p.add_argument("--password", default=os.environ.get("CVAT_PASSWORD", ""))
    p.add_argument("--only", help="comma list of scene names to limit to")
    p.add_argument("--replace", action="store_true",
                   help="for an existing scene: clear its annotations and "
                        "re-import the (re-filtered) seed — no photo re-upload")
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    logging.basicConfig(level=getattr(logging, a.log_level), format="%(levelname)s: %(message)s")

    if not a.user or not a.password:
        LOGGER.error("Need --user/--password (or $CVAT_USER/$CVAT_PASSWORD).")
        return 2
    manifest = json.loads(a.manifest.read_text(encoding="utf-8"))
    only = {s.strip() for s in a.only.split(",")} if a.only else None

    try:
        from cvat_sdk import make_client
        from cvat_sdk.core.proxies.tasks import ResourceType
    except ModuleNotFoundError:
        LOGGER.error("cvat-sdk not installed: uv pip install --python .venv cvat-sdk")
        return 2

    with make_client(host=a.host, port=a.port, credentials=(a.user, a.password)) as client:
        project = next((pr for pr in client.projects.list() if pr.name == PROJECT_NAME), None)
        if project is None:
            project = client.projects.create({"name": PROJECT_NAME, "labels": build_label_spec()})
            LOGGER.info("Created project %r (id=%d)", PROJECT_NAME, project.id)
        else:
            LOGGER.info("Reusing project %r (id=%d)", PROJECT_NAME, project.id)
        by_name = {t.name: t for t in client.tasks.list() if t.project_id == project.id}

        made = repl = 0
        for scene in manifest:
            name = scene["scene"]
            if only and name not in only:
                continue
            xml = Path(scene["xml"])
            if name in by_name:
                if not a.replace:
                    LOGGER.info("Task %r exists — skipped (use --replace)", name)
                    continue
                task = by_name[name]
                LOGGER.info("Replacing annotations on %r (task %d): clear + re-import…",
                            name, task.id)
                task.remove_annotations()
                if scene["boxed_images"] and xml.exists():
                    task.import_annotations(format_name="CVAT 1.1", filename=str(xml))
                    LOGGER.info("  re-imported %s (%d boxes)", xml.name, scene["boxes"])
                repl += 1
                continue
            imgs = [str(Path(p)) for p in scene["images"]]
            LOGGER.info("Creating image task %r (%d photos, %d boxed)…",
                        name, scene["total_images"], scene["boxed_images"])
            task = client.tasks.create_from_data(
                spec={"name": name, "project_id": project.id},
                resource_type=ResourceType.LOCAL,
                resources=imgs,
            )
            xml = Path(scene["xml"])
            if scene["boxed_images"] and xml.exists():
                LOGGER.info("Importing %s (%d boxes) into task %d…",
                            xml.name, scene["boxes"], task.id)
                task.import_annotations(format_name="CVAT 1.1", filename=str(xml))
            made += 1

        LOGGER.info("Done. project=%r created=%d replaced=%d -> http://%s:%d",
                    PROJECT_NAME, made, repl, a.host, a.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
