# Speed-optimization research — make a clip process faster without losing quality

Date: 2026-05-19. Branch: `worktree-speed-research`. Method: code +
config + existing-report analysis (no GPU on the research box; every claim
is tagged either **[measured]** from a prior report, **[code-fact]** from
this tree, or **[estimate]** with the reasoning shown). Every lever carries
a **quality-risk** class and a **how-to-verify** so nothing is adopted on
faith — per the repo rule "verify, don't assert; fix root cause and explain".

---

## 0. TL;DR — the one number that decides everything

A robot clip is processed in two GPU phases. Their costs are **not** the
same order of magnitude:

| Phase | What runs | Per-clip cost (T4-class GPU) |
|---|---|---|
| Detect | YOLO11x@1280 + BoT-SORT, **every frame** | minutes (~3–6 min for a ~2-min clip) **[estimate]** |
| Recognise | Qwen3-VL-4B, **`top_k` crops × every track**, serial | **tens of minutes to hours** **[estimate, anchored to a measurement]** |

Anchor: Qwen3-VL-4B measured **~18 s/crop on an RTX 4070 Ti 12 GB**
(`ocr-campaign-log.md` exp. #3/#9) **[measured]**. The deployment GPU is a
**T4-class 8 GB** card (`memory: rented-hackathon-server`;
`docker-compose.yaml` ml service is the real CUDA Qwen3-VL image) — slower
than a 4070 Ti, and Qwen3-VL-4B in bf16 (~8.5 GB weights, per
`hq_qwen3_vl.yaml`) barely fits 8 GB, so it runs at best ~18 s and likely
**25–45 s/crop** with memory pressure **[estimate]**.

`balanced.yaml` (the deployed config, `docker-compose.yaml:60`) uses
`top_k_crops_per_track: 5`; `hq_qwen3_vl.yaml` uses `8`. A clip with ~80
finalised tracks ⇒ **400–640 serial Qwen forward passes** ⇒ on the order of
**2–7 hours** in the worst case, vs single-digit minutes for detection.

> **Conclusion: > 90 % of wall-clock is the serial Qwen3-VL forward
> passes.** Every speed lever that matters attacks OCR *throughput*.
> Detector tweaks are real but second-order — do them, but don't expect
> them to move the headline. The job queue serialises one clip at a time
> (`memory: job-queue-worktree`), so per-clip latency *is* the user's wait;
> there is no cross-clip parallelism to hide behind.

---

## 1. Cost model (where the time goes) — [code-fact]

`pipeline.py` flow:

```
_run():    per frame → detector.stream_video → rectify → push_crop
           (NO OCR here; only QR/OCR for tracks that expire mid-clip)
_finalize(): for every track still alive at clip end:
             _recognize_track → top_k crops × chain.decode
             chain = QR(ms) → barcode(ms) → OCRDecoder(Qwen3-VL, seconds)
```

- **Detection** runs on **every frame**. Production profiles set
  `frame_rotation: ccw`, so the `_stream_rotated` path is used: a
  synchronous `cap.read()` → `cv2.rotate` → **single-image** `model.track()`
  per frame (`detector.py:340-410`). This defeats Ultralytics' native
  threaded streaming dataloader (which the non-rotated path gets via
  `source=video, stream=True`): the GPU stalls on CPU video-decode + rotate
  every frame. **[code-fact]**
- **Recognition** is the wall. `_finalize` loops tracks **serially**, each
  track loops its `top_k` crops **serially**, each crop calls
  `self.recognition.decode` → `OCRDecoder` → one `model.generate()`
  (`ocr.py:533-565`). No batching anywhere; one image per forward pass.
  **[code-fact]**
- QR + barcode are ~milliseconds and already de-duplicated: the shared
  engine memoises one decode per crop so QR+barcode don't decode twice
  (`recognition-pipeline.md`). The cheap code-sweep passes (`code_decode_top_k`,
  `code_fuse_frames`) are **OFF by default** and benchmarked as no-lift
  (`memory: level2-fusion-no-lift-input-bound`). The crop→fields cheap path
  is already lean — **the cost is purely the VLM.** **[code-fact]**

The single Qwen call's latency = **prefill** (image tokens + the prompt) +
**decode** (output tokens, autoregressive). Measured: prompt ≈ 2512 chars ≈
**~1000–1400 text tokens, byte-identical for every crop** (`DEFAULT_VLM_PROMPT`);
a complete 29-key answer ≈ **~280 tokens** of decode **[estimate, measured
char counts in this branch]**. Two facts fall out of this and drive Tier 1:

1. The prompt is the same ~1k+ tokens re-encoded on **every** call — pure
   waste without a prefix cache.
2. `vlm_max_new_tokens: 640` (`balanced.yaml`/`hq_qwen3_vl.yaml`) is ~2.3×
   the ~280 tokens a good answer needs.

---

## 2. Tier 1 — big wins, quality-neutral (same model, same prompt, same outputs)

These change *how* the model is called, not *what* it computes, so the
extracted fields are bit-for-bit (or distribution-) identical. Highest
priority.

### 1.1 Batch the VLM forward passes — **the single highest-leverage change**

**Now:** `_finalize` → per-track → per-crop serial `model.generate()`
(`pipeline.py:210-218` + `ocr.py:558-559`). The GPU processes one ~280-token
generation at a time; the GEMMs are tiny and the card is starved.

**Change:** all finalize-phase crops are independent. Collect them, run
`model.generate()` on a **left-padded batch** of B images (HF transformers
supports batched multimodal generation). The vision encoder and the LM
prefill/decode then amortise across B — typical **4–6× throughput** at
B=8 on an unstarved card **[estimate, standard batched-decode behaviour]**.

- **Quality risk: none.** Identical model, prompt, sampling
  (`do_sample=False`); batching only groups independent calls. Left-padding
  + correct attention mask is the only correctness concern (unit-testable
  against the serial path on the same crops).
- **T4 caveat:** 8 GB caps batch size with a 4B bf16 model — realistic
  B≈2–4 in-process. The full multiplier needs §1.2 (vLLM paged KV) or §1.3
  (quant). Even B=3 is a ~2.5–3× win for free.
- **Seam:** add `recognize_batch(list[np.ndarray])` to `BaseOCREngine`
  (default = map `recognize`), implement the real batched path in
  `TransformersVLMEngine`, and have `_finalize` gather crops across tracks
  before decoding. The frozen `CropDecoder`/`RecognitionResult` seam and
  chain order are untouched (batching lives *inside* the OCR engine + the
  finalize orchestration, not the decoder contract).
- **Verify:** `eval_hack_csv.py` on the 5 labelled videos must be
  byte-identical (greedy float compare) batched vs serial; wall-clock from
  the existing progress timestamps.

### 1.2 vLLM / SGLang server — continuous batching + prefix cache + quant

`VLLMServerEngine`, `hq_vllm.yaml`, and `guided_json` **already exist**
(`ocr.py:679-751`). What's missing: `vllm` is commented out in
`requirements/ocr.txt`, no Qwen3-VL vLLM service in `docker-compose.yaml`,
and the pipeline submits **serially** so continuous batching is never
exercised. Wiring this in gives three independent multipliers:

1. **Continuous batching** across all crops/tracks — same idea as §1.1 but
   the server schedules it optimally; combine with concurrent submission
   from `_finalize` (a bounded thread pool of `recognize` calls, or an
   async client) so the server always has a full batch.
2. **Automatic prefix caching** — the ~1k+ identical prompt tokens are
   encoded **once** and the KV reused for every crop. This is exact (not
   lossy) and removes the bigger half of prefill on a short-decode workload.
3. **FP8 / AWQ-int4 Qwen3-VL weights** — FP8 is near-lossless and makes the
   4B model fit T4 8 GB *comfortably* (room for a real KV-cache batch);
   int4-AWQ is a small accuracy risk → benchmark-gate it.

- **Quality risk: none → positive**, holding the model + FP8. `guided_json`
  with `PRICE_TAG_JSON_SCHEMA` *removes* JSON-parse failures (a quality win
  the in-process path doesn't get). AWQ-int4 is the only sub-lever needing
  a gate.
- **Verify:** `eval_hack_csv.py` parity (Qwen3-VL bf16 in-process) vs
  (Qwen3-VL FP8 vLLM + guided_json); expect ≥ parity. Then measure
  throughput at concurrency 8/16.
- **Effort:** medium (dep + a compose service + concurrent submission), but
  the building blocks are already in the tree. This is the strategic target.

### 1.3 FlashAttention-2 / SDPA attention for the in-process path

`TransformersVLMEngine._load_locked` (`ocr.py:476-518`) never passes
`attn_implementation`, so it uses eager attention. SDPA is exact; FA2 is
exact. **1.2–1.6× on prefill+decode, zero quality change**, one kwarg.
The right **stopgap before vLLM lands** and it stacks with §1.1.

- **Quality risk: none** (numerically equivalent attention).
- **Verify:** load succeeds with `attn_implementation="sdpa"` (fallback to
  eager on failure); `eval_hack_csv.py` unchanged.

---

## 3. Tier 2 — medium wins, near-zero quality risk

### 2.1 Cap `vlm_max_new_tokens` 640 → ~320

Decode time is linear in generated tokens and is half the per-call latency.
Generation already stops at EOS (`do_sample=False`), so a complete answer
(~280 tokens) is unaffected by the cap — **the cap only bounds runaway /
degenerate generations, which are exactly the slowest calls.** 320 leaves
headroom over the ~280-token typical answer.

- **Quality risk: ~zero**, *if* no legitimate answer exceeds 320 tokens
  (long product names are the only candidate). **Verify before adopting:**
  enable `runtime.audit_path`, run the 5 videos, histogram `raw_text` token
  lengths; set the cap at p99 + margin. Pairs with §1.2 `guided_json`
  (schema-constrained output cannot run away at all).

### 2.2 Async frame prefetch for the rotated detect path

`_stream_rotated` is synchronous: decode+rotate a frame, *then* the GPU
runs, *then* decode the next. A 1-deep producer thread (decode + `cv2.rotate`
of frame N+1 while the GPU runs frame N) overlaps CPU and GPU and recovers
the throughput the rotated path loses vs Ultralytics' native streaming
dataloader. Detection is second-order overall, but this is free and the
rotated path is *every* production run (`frame_rotation: ccw` everywhere).

- **Quality risk: none** — identical frames, identical order, identical
  tracker state (frames still fed in sequence with `persist`).
- **Verify:** detection output identical frame-for-frame; wall-clock delta.

### 2.3 Detector FP16 (`half=True`)

`model.track(...)` / `model.predict(...)` kwargs (`detector.py:152-162,
326-332`) don't set `half`. FP16 inference is the industry standard for
YOLO deployment; **~1.3–1.8× detector throughput on T4, negligible mAP
delta**.

- **Quality risk: negligible** (FP16 detection is standard practice).
- **Verify:** `eval_detector.py` / `eval_qa.py` recall@conf unchanged on a
  held-out video before/after.

---

## 4. Tier 3 — large potential, but a real quality trade → benchmark-gated

These can be big but are **not** "no quality loss" by inspection. They are
listed because they're tempting and the team should know the gate, not skip
them blindly.

### 3.1 `top_k_crops_per_track`: 5→3 (balanced), 8→4 (hq)

A direct **40–50 % linear cut** in Qwen calls. But more observations make
the aggregator's per-field voting more robust — especially **barcode (P0
GT-matching key)**. Diminishing returns past K=3 are *plausible* (crops are
already sharpness-ranked; #4–8 are the worst) but **unverified** on the
labelled set. fast-mode already proved K=1 is acceptable *for the fast tier*
(`memory: fast-mode-worktree`) — that's evidence the curve flattens, not
proof K=3 == K=5 on the graded metric.

- **Quality risk: real, must measure.** Gate: `eval_hack_csv.py` on all 5
  videos at K ∈ {2,3,4,5,8}; adopt the smallest K whose headline + barcode
  recall is within run-to-run noise of K=5. This is a *measured trade*, not
  a free win — label it that way in any config change.

### 3.2 Two-stage OCR (cheap gate → escalate to Qwen-4B only on gaps)

Most tags are clean; a fast small model could clear them and Qwen-4B is
reserved for hard crops. Biggest theoretical speedup. **But**: classical
PaddleOCR returned empty on these crops (`inference-current-issues.md` §3)
and GLM-0.9B was "too weak / hallucinates" (`ocr-campaign-log.md` #3). So a
naive gate risks silently shipping the weak model's answer.

- **Quality risk: high if ungated.** Frame as an *experiment*: candidate
  fast stage = HunyuanOCR-1B (already wrapped), gate on "all P0 fields
  present + confidence ≥ τ", escalate otherwise; accept only if
  `eval_hack_csv.py` ≥ Qwen-only within noise while cutting Qwen calls
  measurably. Lower priority than Tier 1/2.

### 3.3 Detection frame stride (process every 2nd frame)

~2× detection. But the tracker (BoT-SORT/ByteTrack) needs reasonably dense
frames for association; striding risks track fragmentation (→ duplicate
rows, the metric's #1 enemy) and may skip the single sharpest frame of a
fast tag pass.

- **Quality risk: real.** Gate with `eval_tracking.py` (fragmentation /
  ID-switch) **and** `eval_hack_csv.py`. Only if both hold at stride 2.
  Detection is second-order anyway — low priority vs OCR levers.

---

## 5. The tempting trap — do NOT treat detector `imgsz` as a free speed knob

It is the most obvious "speed" lever and it is **wrong** here. The
detector-fine-tuning report's #1 finding **[measured]**: *resolution is the
single biggest recall lever* — yolo11s recall **0.48 → 0.82 going 1280 →
1536**, and it explicitly recommends inferring at **≥1536**. Production runs
at **1280** — already *below* the recommended resolution. `fast.yaml`'s 960
is a deliberate quality sacrifice for smoke runs, not a template.

Dropping `imgsz` trades real recall; **missed tags are unrecoverable** (no
detection ⇒ no track ⇒ no crop ⇒ no OCR ⇒ that GT tag is simply unscored).
Fewer detections *does* shrink the OCR bill — but by *deleting output*, not
by going faster at equal quality. Classify `imgsz` reduction as a
**quality-costing** knob and keep it out of the "no quality loss" basket.
(FP16/TensorRT, §2.3, are the *quality-neutral* detector speedups.)

---

## 6. Recommended order of execution (impact ÷ risk ÷ effort)

1. **§1.3 FA2/SDPA attn** — one kwarg, exact, ships today, stacks with all.
2. **§2.1 max_new_tokens cap** — config + a one-run audit histogram.
3. **§2.3 detector `half=True`** + **§2.2 async prefetch** — cheap,
   quality-neutral, removes the second-order detector cost.
4. **§1.1 in-process batched generate** — the biggest single quality-neutral
   recognition win; B≈3 even on T4.
5. **§1.2 vLLM + prefix cache + FP8 + guided_json + concurrent submit** —
   the strategic end-state; subsumes much of §1.1/§1.3 and lifts the T4
   memory ceiling. Largest total payoff.
6. **§3.1 top_k sweep** — run the gate; likely a safe extra 30–40 % once
   measured.
7. §3.2 / §3.3 — only if 1–6 are insufficient; both fully benchmark-gated.

Expected stacked effect of 1–5 (all quality-neutral): the recognition phase
— today's > 90 % of wall-clock — is the part being multiplied, so a
realistic target is a **single-digit-multiple reduction in per-clip latency
with the graded metric held within measurement noise**, before touching any
Tier-3 trade. Exact numbers require a GPU run; this document deliberately
states mechanism + expected direction + the gate, not invented speedups.

## 7. What is already optimal (don't re-litigate)

- QR/barcode decode is memoised once per crop; the wide code-sweep and
  Level-2 fusion are benchmarked no-lift and correctly default OFF.
- The "OCR only the top-K sharpest buffered crops, not every Nth frame"
  design is itself the foundational recognition-cost optimisation — these
  levers tune *within* it, they don't replace it.
- The process-wide `_VLM_CACHE` already prevents reloading the ~8.5 GB model
  per request/rotation. Good; keep it.
