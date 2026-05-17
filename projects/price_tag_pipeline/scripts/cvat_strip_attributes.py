#!/usr/bin/env python3
"""Remove all attributes from the live project's ``price_tag`` label.

Detector-first workflow: only boxes are drawn, no OCR fields — so the CVAT
per-object "details" panel is pure noise. ``build_label_spec`` is already
bare for new projects/tasks; this fixes the *already running* instance by
PATCHing the existing label (token auth, stdlib only — no extra deps).
Existing photo boxes carry no attribute values, so nothing is lost.

    .venv/Scripts/python.exe projects/price_tag_pipeline/scripts/cvat_strip_attributes.py \\
        --user admin --password ***
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import urllib.request

LOGGER = logging.getLogger("cvat_strip_attributes")
PROJECT_NAME = "Lenta price tags"


def _req(method: str, url: str, token: str | None, body: dict | None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header("Content-Type", "application/json")
    if token:
        r.add_header("Authorization", f"Token {token}")
    with urllib.request.urlopen(r) as resp:  # noqa: S310 (localhost, trusted)
        return json.loads(resp.read() or "null")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--user", default=os.environ.get("CVAT_USER", ""))
    p.add_argument("--password", default=os.environ.get("CVAT_PASSWORD", ""))
    p.add_argument("--project", default=PROJECT_NAME)
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    logging.basicConfig(level=getattr(logging, a.log_level), format="%(levelname)s: %(message)s")
    if not a.user or not a.password:
        LOGGER.error("Need --user/--password (or $CVAT_USER/$CVAT_PASSWORD).")
        return 2

    base = f"http://{a.host}:{a.port}/api"
    token = _req("POST", f"{base}/auth/login", None,
                 {"username": a.user, "password": a.password})["key"]

    projects = _req("GET", f"{base}/projects?page_size=1000", token, None)["results"]
    proj = next((p for p in projects if p["name"] == a.project), None)
    if proj is None:
        LOGGER.error("Project %r not found", a.project)
        return 1

    labels = _req("GET", f"{base}/labels?project_id={proj['id']}&page_size=1000",
                   token, None)["results"]
    stripped = 0
    for lab in labels:
        attrs = lab.get("attributes") or []
        if not attrs:
            LOGGER.info("Label %r already bare", lab["name"])
            continue
        LOGGER.info("Stripping %d attribute(s) from label %r (id=%d)…",
                    len(attrs), lab["name"], lab["id"])
        _req("PATCH", f"{base}/labels/{lab['id']}", token,
             {"attributes": [{"id": at["id"], "deleted": True} for at in attrs]})
        after = _req("GET", f"{base}/labels/{lab['id']}", token, None)
        if after.get("attributes"):
            LOGGER.error("Label %r still has attributes: %s",
                         lab["name"], [x["name"] for x in after["attributes"]])
            return 1
        stripped += 1

    LOGGER.info("Done. project=%r labels_stripped=%d — details panel gone.",
                a.project, stripped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
