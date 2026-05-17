#!/usr/bin/env python3
"""Gradio demo: upload robot video -> annotated video + hackathon CSV.

This is the mandated upload-video -> download-CSV UI (task.md §9) and it
satisfies the §13 "progress bar (videos take a while)" requirement: the
pipeline's per-frame/phase progress is streamed live into a Gradio bar.

Bar budget: the pipeline run owns 0..85 %, CSV export 85..92 %, the
annotated-video render 92..100 % — so the bar reflects *total* work, not
just detection.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

THIS = Path(__file__).resolve().parent
SRC = THIS.parent / "src"
REPO_ROOT = THIS.parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:  # keep `--help` working when gradio isn't installed (it's a demo-only dep)
    import gradio as gr
except ImportError:
    gr = None  # type: ignore


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def process_video(
    video_path: str,
    config_path: str,
    output_dir: str,
    progress=gr.Progress() if gr is not None else None,
) -> tuple[str, str, str]:
    if not video_path:
        raise ValueError("Upload a video first.")
    out_root = Path(output_dir or "outputs/demo").expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    src_video = Path(video_path)
    work_video = out_root / src_video.name
    if src_video.resolve() != work_video.resolve():
        shutil.copy2(src_video, work_video)

    stem = work_video.stem
    jsonl = out_root / f"{stem}.jsonl"
    csv_path = out_root / f"{stem}_hack_submission.csv"
    annotated = out_root / f"{stem}_annotated.mp4"
    audit = out_root / f"{stem}_audit.jsonl"

    # Reuse config immutability safely by writing an audit-enabled temp copy.
    tmp_cfg = out_root / f"{stem}_runtime.yaml"
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    data.setdefault("runtime", {})["audit_path"] = str(audit)
    tmp_cfg.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")

    from price_tag_pipeline.config import load_config
    from price_tag_pipeline.pipeline import PriceTagPipeline
    from price_tag_pipeline.progress import ProgressEvent
    from price_tag_pipeline.submission import final_tags_to_csv

    def _on_progress(ev: ProgressEvent) -> None:
        if progress is None:
            return
        label = f"{ev.phase} — {ev.message}" if ev.message else ev.phase
        # Pipeline owns the first 85 % of the bar (see module docstring).
        progress(0.85 * ev.fraction, desc=label)

    cfg = load_config(tmp_cfg)
    pipe = PriceTagPipeline(cfg)
    tags = pipe.run(video_path=str(work_video), output_path=str(jsonl), progress=_on_progress)

    if progress is not None:
        progress(0.88, desc="Building hackathon CSV")
    # In-process, single-source 29-column renderer (no subprocess).
    csv_text = final_tags_to_csv(tags, filename=work_video.name)
    csv_path.write_text(csv_text, encoding="utf-8", newline="")

    if progress is not None:
        progress(0.92, desc="Rendering annotated video")
    vis_cmd = [
        sys.executable,
        "projects/price_tag_pipeline/scripts/visualize_predictions.py",
        "--video",
        str(work_video),
        "--pred",
        str(jsonl),
        "--out",
        str(annotated),
    ]
    if audit.exists():
        vis_cmd.extend(["--audit", str(audit)])
    _run(vis_cmd)

    if progress is not None:
        progress(1.0, desc="Done")
    summary = f"Detected {len(tags)} final unique price tags. CSV: {csv_path.name}"
    return str(annotated), str(csv_path), summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="projects/price_tag_pipeline/configs/zeroshot_nolabel.yaml")
    p.add_argument("--outputs-dir", default="outputs/demo")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()

    if gr is None:
        raise SystemExit("Install Gradio first: pip install gradio>=4.0")

    with gr.Blocks(title="Lenta price-tag recognition") as demo:
        gr.Markdown("# Lenta price-tag recognition\nUpload a robot video, then download annotated MP4 and CSV.")
        with gr.Row():
            video = gr.Video(label="Input robot video")
            with gr.Column():
                config = gr.Textbox(value=args.config, label="Pipeline config")
                outputs = gr.Textbox(value=args.outputs_dir, label="Outputs directory")
                run = gr.Button("Run pipeline", variant="primary")
        annotated = gr.Video(label="Annotated result")
        csv_file = gr.File(label="Hackathon CSV")
        summary = gr.Textbox(label="Run summary")
        run.click(process_video, inputs=[video, config, outputs], outputs=[annotated, csv_file, summary])

    demo.queue().launch(server_name=args.host, server_port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
