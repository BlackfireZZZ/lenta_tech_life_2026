"""Bridge from the HTTP contract to the real pipeline.

This is the ONLY seam between the deployable service and the research
codebase. The actual model lives in the ``price_tag_pipeline`` package
(``projects/price_tag_pipeline/``); here we just call its public entry point
and hand the CSV back. Keeping the seam this thin means training/experiment
churn in that package never breaks the service contract.

Status: MOCKED. The real call is written below and commented out so the
wiring is unambiguous when weights/config are ready.
"""

from app.contract import ProcessRequest, ProcessResponse

# Real wiring (enable when the pipeline + weights are available in the image):
#
#     from price_tag_pipeline.config import load_config
#     from price_tag_pipeline.pipeline import PriceTagPipeline
#     from price_tag_pipeline.submission import final_tags_to_csv
#
# `PriceTagPipeline(cfg).run(video_path)` returns list[FinalTag]; the
# submission module renders the graded 29-column CSV. See
# docs/pipeline-reference.md for CLI/profile equivalents.

_MOCK_CSV = "filename,frame_timestamp,x1,y1,x2,y2,barcode,name,price,...\n"


def run_pipeline(req: ProcessRequest) -> ProcessResponse:
    # --- MOCK ---
    return ProcessResponse(csv=_MOCK_CSV, rows=0, meta={"mock": True, "job_id": req.job_id})

    # --- REAL (uncomment, drop the return above) ---
    # cfg = load_config("configs/balanced.yaml")
    # tags = PriceTagPipeline(cfg).run(req.video_path)
    # csv_text = final_tags_to_csv(tags)
    # return ProcessResponse(csv=csv_text, rows=len(tags),
    #                        meta={"job_id": req.job_id})
