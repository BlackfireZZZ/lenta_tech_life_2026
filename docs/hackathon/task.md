# Lenta Tech Life Hack — "Shelf Under Control"

Automatic price tag recognition from video captured by a robot moving along supermarket shelves in Lenta hypermarkets.

> This document combines the official hackathon materials (PDFs: "Task", "Price Tag Legend", "Hypermarket Design Guide for Retail") **with ~230 messages from organizers and participants in the "Task Questions" Telegram chat** dated May 12–17, 2026. Organizers significantly clarified and reworded parts of the PDF requirements in the chat — both versions are captured here.
>
> **Online round deadline: May 19, 2026, 15:00 MSK. Finals: May 24, Moscow.**

---

## 1. The task

A robot moves along a shelf and records video. The solution must:

1. Accept this video as input.
2. Detect price tags in the frames.
3. For each unique price tag, extract as many fields as possible: **19 fields from the tag itself + 11 fields from the QR code = 30 fields total**.
4. Export results as `.csv` (one row = one unique price tag).
5. Wrap everything in a working UI (upload video → download CSV).
6. Deploy publicly, ship with a README and a presentation.

Business context: automation of shelf compliance checks, faster shelf audits, foundation for further digital monitoring scenarios in the sales floor.

---

## 2. What the robot captures (input data)

A public set of videos shot along shelves in real-world conditions. Zones: alcohol, dairy, honey, jams, syrups.

**Two motion modes:**
- **With stops** — the robot pauses in front of each shelf section; the tag is sharp for 1–2 seconds (essentially a photo).
- **Without stops** — continuous motion, frames are motion-blurred.

**Shooting characteristics:**
- Mixed lighting (daylight + store lights), glare and shadows.
- Tags at different heights and angles relative to the camera.
- Partial occlusions by product or shelf elements.
- Some zones (alcohol) are shot **through a glass partition** — reflections, double images.

**Camera parameters** (important — not in the PDF, only in the chat, id=1410):
- Resolution: **3840×2160**.
- Sensor diagonal: 16/2.8 mm.
- Focal length: **2.8 mm** — ultra-wide, noticeable edge distortion.
- Camera is mounted **rotated 90° counter-clockwise** — the raw frame is vertical and needs to be rotated.
- Distortion coefficients are **not provided**. Organizers: *"you can estimate the correction approach yourself from the available videos."*

---

## 3. Output format — CSV

Each row = one unique price tag.

### 3.1 Fields from the tag itself (18 fields; not all of them count toward the metric — see §5)

| Field | Description |
|---|---|
| `filename` | video filename |
| `product_name` | product name |
| `price_default` | price without loyalty card |
| `price_card` | price with loyalty card |
| `price_discount` | promo price |
| `barcode` | barcode (digits under the bars) |
| `discount_amount` | discount amount |
| `id_sku` | SKU id |
| `print_datetime` | date and time the tag was printed |
| `code` | display zone code |
| `additional_info` | additional info on the tag |
| `color` | tag color (see §4) |
| `special_symbols` | display type (`к` / `л` / `ш`) |
| `frame_timestamp` | time in **milliseconds** from the start of the video |
| `x_min`, `y_min`, `x_max`, `y_max` | bbox coordinates in pixels |

### 3.2 Fields from the QR code (11 fields)

| CSV field | QR field name |
|---|---|
| `qr_code_barcode` | `barcode` / `b` |
| `price1_qr` | `price1` / `p1` |
| `price2_qr` | `price2` / `p2` |
| `price3_qr` | `price3` / `p3` |
| `price4_qr` | `price4` / `p4` |
| `wholesale_level_1_count` | `wholesaleLevel1Count` / `wL1C` |
| `wholesale_level_1_price` | `wholesaleLevel1Price` / `wL1P` |
| `wholesale_level_2_count` | `wholesaleLevel2Count` / `wL2C` |
| `wholesale_level_2_price` | `wholesaleLevel2Price` / `wL2P` |
| `action_price_qr` | `actionPrice` / `aP` |
| `action_code_qr` | `actionCode` / `aC` |

### 3.3 Technical formatting requirements

Confirmed by organizers in the chat (the question came up repeatedly):

- Field separator — `,` (comma).
- Decimal separator — `.` (period).
- Encoding — UTF-8.
- String fields are wrapped in `"` if they contain commas or quotes.
- **Parameter is absent on the tag** → write `"нет"` (Russian for "no").
- **Parameter exists but is not recognized** → leave the field empty.

These are two distinct cases — confusing them is critical (see §5).

### 3.4 Heads-up: `sample.csv` vs the provided labels

Organizers published `sample.csv` several days after the start. Before that, the format of the labels in the provided dataset **differed** from the final required format (e.g., the decimal separator in the labels was `,`).

Organizer quote: *"Format differences will still be handled on the metric side. The main thing is to preserve the required structure and correctly fill the substantive fields."* — meaning **don't take the label format as a template**, follow the spec in the PDF + `sample.csv`.

---

## 4. Price tag types and color logic

From "Price Tag Legend" and "Hypermarket Design Guide" — 7 base types.

**Sizes and orientations:**
- A4 vertical / A4 horizontal.
- A3 horizontal, A2 horizontal.
- 6×6 (compact), 6×12 with shelf-talker.
- МНЦ (small new price tags) — white / yellow / green / red.

**Color logic** — this is the value of the `color` field:
- **White** — regular shelf price (РПЦ), standard item.
- **Yellow** — bakery.
- **Green** — fruits and vegetables (often with scale number).
- **Red** — promo / promotional shelf price (АПЦ), discount.
- **Black circle with `-N%` or `-N₽`** — promo marker, a **separate element on the tag**, not the color of the whole tag. Rule: if discount < 100 ₽ → shown as a percentage, if ≥ 100 ₽ → shown in rubles.

**Display type `special_symbols`:** possible values — `к` (box), `л` (tray), `ш` (piece). Not always shown.

**Per-weight vs per-piece marker:** the tag shows "per 1 kg" / "per 100 g" / "per 1 piece" — may require a separate classification step.

**Organizer recommendation** (id=1615): *"`Расшифровка ценники.pptx` is the main applied material. `ГМ для ТК.pptx` is an additional reference for understanding tag variability and building a more scalable solution."*

---

## 5. The metric — detailed breakdown

This is the most important section. Without understanding how scoring works, it's easy to misallocate effort.

### 5.1 The headline number

**Share of price tags for which at least 80% of substantive fields were recognized correctly, out of the total number of price tags in GT.**

**Critical clarification** (id=1352, reworded later): "80%" is **not the share of detected tags**, as many teams initially assumed. It's the **per-tag threshold on fields**: within a single matched tag, at least 80% of its substantive fields must be correct.

### 5.2 Two-stage matching (id=1219)

**Stage A. Matching your row against a GT row:**
1. **Priority #1 — barcode.** If the barcode in your row matches GT, the rows are matched directly. This is the most reliable key.
2. **Priority #2 — spatio-temporal key.** If barcode is missing or unrecognized — matching uses `frame_timestamp + bbox` with tolerances.

**If neither key matches your row to a GT row, the row doesn't count.**

**Stage B. Substantive field evaluation:**
- A tag is considered "successfully recognized" if it was matched **and** at least 80% of its substantive fields are correct.
- "Substantive" = everything **except** `filename`, `frame_timestamp`, `x_min`, `y_min`, `x_max`, `y_max`. These technical fields are used only for matching, not scored.
- The final metric = share of "successfully recognized" tags out of the total in GT.

### 5.3 Open questions about scoring (unanswered in the chat)

- **How text fields are compared** (`product_name`, `print_datetime`, etc.) — Levenshtein? Exact match? Fuzzy with threshold? **No direct answer.** Hint from id=1613: *"the product name doesn't have to be recognized character-perfectly, but the closer to the actual text on the tag, the better."* Plan for fuzzy comparison with normalization (case, whitespace, punctuation).
- **Is QR counted as 1 field or as 11?** konstanta (id=1548) asked this directly. If each of the 11 QR fields counts separately — passing the 80% threshold on 19+11=30 fields **without QR is nearly impossible**. If QR=1 — passable. **Organizers gave no direct answer**, redirected to the criteria. Plan for the worst case: each QR field counts separately.
- **`"нет"` vs empty.** Critical — if you write an empty field where `"нет"` is expected, the field will likely count as not recognized.

### 5.4 Strategic implications

- **Recognizing the barcode is critical.** It's the primary matching key. Without it you depend on bbox+timestamp accuracy, and GT bboxes are noisy (see §6) — higher chance of failing to match.
- **Duplicates hurt the metric.** If one tag appears 5 times in the CSV, only one row will match GT; the other 4 may not match and drag the score down. *(Inferred from the metric formulation — logically follows.)*
- **Skipping a tag is better than submitting noise.** If confidence in substantive fields is low — don't emit the row at all, rather than emit a row with errors.

---

## 6. Pitfalls and gotchas (from the chat)

### 6.1 GT bbox coordinates are noisy
Confirmed by organizers (id=1446): *"the source labels contain non-ideal bboxes, some frames may be shifted relative to the actual tag position. We account for this risk, and bbox matching tolerance will not be too tight."*

Team Max Agafonov visually overlaid the CSV labels onto frames — **the bboxes often don't sit on the tag** (id=1424, 1275). Some labels point to top-shelf tags that the camera physically can't resolve (too small / occluded).

**Takeaway:** build your own tag detection, don't try to reproduce GT bboxes.

### 6.2 bbox coordinates in labels are decimal, not integer
Confirmed (id=1274): *"in the pixel grid the coordinates effectively correspond to integer values. The decimals are an artifact of label processing format. Not critical for the solution."* You can round.

### 6.3 `frame_timestamp` — label frames don't match a straight video decode
Raised by Ars (id=1667, May 17, right before the deadline): *"It's unclear how the `frame_timestamp` values in the labels were obtained. When decoding frames manually, the labels shift by a small delta."* The decoding script and the pre-decoded frames are **not provided**.

**Mitigation.** Organizers (id=1219): *"matching uses tolerances — for example, if you pick any frame from an interval where the tag is well-visible, you almost certainly fall within these tolerances."* So per unique tag, submit **one** timestamp (the frame where the tag was best recognized), and rely on the time tolerance.

### 6.4 "Best frame" — what does it actually mean?
A single tag passes through many frames from edge to edge of the camera view, and it's unclear which one to write into the CSV. Organizers (id=1180, 1219): *"fields are filled based on the best frame, where the tag was most clearly visible"*, *"you should specify the moment of final determination, i.e., the frame with the best recognition quality."*

Marie (id=1204) framed the problem precisely: *"Video with a stop. The same tag, same clarity, for 2 full seconds. How do you guess the required timestamp?"* No direct answer — the time tolerance from §6.3 covers this case.

### 6.5 Tag deduplication is on the participant's side
Organizers, plain text: *"The logic of video analysis, frame selection, candidate filtering for recognition, duplicate filtering, and result aggregation remains on the participant's side and is part of the proposed solution"* (id=1219).

**Tracking + deduplication is a critical component.** Without it, each tag will appear in the CSV dozens of times → the metric drops (see §5.4).

### 6.6 Partial barcode is better than no barcode
*"If you recognize 12 out of 13 digits, it's better to submit 12 digits than to omit the field entirely"* (Checkpoint FAQ, id=1613).

### 6.7 QR can be skipped without disaster (with caveats)
*"Not recognizing the QR code by itself won't be a critical minus. You can focus on the substantive fields of the tag"* (FAQ, id=1613).

But: QR contains **11 out of 30 fields**. If you don't fill them at all and each QR field counts separately in the metric (see §5.3), the 80% threshold for a matched tag becomes very hard to clear.

### 6.8 Actual dataset quality vs expectations
Sarcastic but accurate remark from Maksim (id=1293): *"Such a great dataset. Such a cool 4K camera. Excellent contrast on the tag text… Tesseract recognizes 99% of the text out of the box… The QR is just a song 😂"* — in reality, barcodes and QR **don't get recognized in motion** even by experienced participants. *"At least one in motion"* — that's about QR (id=1331).

Mark_Borodin: *"reconstructing something from blur is very expensive and ineffective"* — no point chasing washed-out tags; focus on the ones where the model is confident.

### 6.9 Stops vs no-stops, strategic angle
At stops, the tag is sharp for 1–2 seconds (essentially a photo). In motion — blurred. **Strategically easier to target tags during stops** or those that briefly fall into focus during slow motion.

---

## 7. Important update on manual labeling (rule change!)

At the May 12 kickoff, organizers said "manual labeling is forbidden." This caused mass confusion. **The updated position** (id=1436, 1458, 1497):

**Allowed:**
- Manually labeling data for training / fine-tuning models.
- Adding labels to the provided dataset.
- Using external datasets (any open ones with a suitable license) and self-collected ones.
- Building an SKU directory / your own product database (FAQ, id=1613).
- Using data from Lenta's public website (FAQ, id=1613).

**Forbidden:**
- Any manual work **at the inference stage**: editing the CSV, manually picking frames, manually adjusting bboxes, operator-in-the-loop.
- Using observations from previous runs on the test video to tune the service for subsequent runs (id=1454).

**Mandatory:**
- In the solution materials, **describe in detail**: what data was labeled, in what volume, how it was used, what the fine-tuning pipeline looked like.

Quote (id=1436): *"If the data used, labeling, or training stages are not disclosed in the solution materials, this may affect the evaluation"* — transparency here is a hard requirement.

---

## 8. The organizers' "ideal solution" (between the lines)

Compiled from the checkpoint FAQ (id=1613), Olga Dubovaya's replies (id=1177, 1178), and what gets explicitly praised or asked for.

### 8.1 What was explicitly named as a priority

1. **Desktop solution, not on-robot.** *"The priority is a regular desktop solution. Running on the robot, distributed architecture, or on-robot processing can be a plus but is not mandatory"* (FAQ).
2. **Recognition quality > speed.** Repeated across dozens of replies.
3. **Substantive fields > QR.** *"Priority is the substantive fields that help understand which product the tag belongs to and what its price is. Fields that allow linking the tag to a product matter most"* (FAQ).
4. **Applicability, not lab work.** *"It's important to be able to explain how this will scale and be applied in the business process"* (FAQ).

### 8.2 The architecture they're already considering

Olga Dubovaya (id=1177): *"We're considering various architecture options. With deployment on the robot, and without it — sending photos via API to an external server."*

Meaning Lenta hasn't decided yet — either a heavy model on the server fed by a stream of photos from the robot, or a light model directly on the robot. **A solution that explicitly addresses both scenarios** (e.g., a lightweight on-device detector + optional heavy OCR on the server) is likely to win on "scalability and applicability."

### 8.3 Indirect hardware signals

*"Right now it's a prototype, the configuration can be changed to fit the solution"* (id=1178) — they refuse to name specific hardware. But the mention of `rknn` (int8) in the task = **Rockchip NPU** (RK3568 / RK3588), a typical board for edge AI in robotics. Baseline: ARM, NPU ~6 TOPS, typical models — YOLOv5/v8 nano/small, MobileNet, MobileSAM.

### 8.4 What gets praised in their language

- "Maturity of approach" — understanding of limitations, explicit discussion of what doesn't work.
- Describing edge cases in the README and presentation.
- Ideas for scaling and improvements "for industrial deployment" (id=1372).

---

## 9. UI and deployment

- **UI is mandatory:** upload video → download CSV. Any open-source framework (Gradio, Streamlit, FastAPI + HTML — your call).
- **Public deployment without authentication is a plus** for evaluation.
- If login/password is required for verification — **document it in the README** (id=1613): *"If checking the solution requires login, password, or other access, provide all necessary credentials and instructions."*
- README must cover: architecture, models used, what data was labeled (if any), training pipeline, reproduction instructions.
- Processing time: no hard limit. *"A good guideline is a few seconds per frame, but longer is acceptable"* (FAQ, id=1613). An hour per video would be a minus.

---

## 10. Constraints (from the PDF, reaffirmed)

1. **No cloud APIs, no external online services** that can't be reproduced inside Lenta's internal infrastructure. Strict ban.
2. Pretrained models and libraries with **open licenses are allowed**, provided they can be deployed locally.
3. The model should be **relatively lightweight** and runnable on constrained hardware. Heavy models aren't fully banned, but their use must be justified.
4. **No manual labeling at inference** (see §7 for the full nuance).
5. **Optimization for `rknn (int8)` is encouraged.**

---

## 11. Deliverables

1. **Source code repository** — full source, detailed `README.md` (architecture, run instructions, hardware requirements, model licenses, dataset descriptions including self-labeled ones, "Limitations" and "Future Work" sections).
2. **Deployed solution** — public URL, no auth required (or credentials documented in the README).
3. **Output CSV** for the control video in the required format.
4. **Presentation:**
   - Title slide (solution name, team name).
   - Team composition (names, roles, contacts).
   - Input data and feature analysis.
   - Architecture and pipeline logic.
   - Examples of the UI and the output CSV.
   - Approach limitations.
   - Scaling ideas.
   - Link to repo and to the deployed solution.

---

## 12. Timeline

- **May 12, 17:00 MSK** — kickoff, task release, Q&A.
- **May 15, 17–19** — checkpoint 1 (FAQ with 14 questions published afterwards, id=1613).
- **May 17, 12–14** — checkpoint 2.
- **May 19, 15:00 MSK** — online round submission deadline (upload to the personal cabinet).
- **May 21, 21:00** — finalist announcement.
- **May 24** — finals in Moscow (remote participation possible by prior arrangement, request by May 17 14:00).

**At the finals**, beyond the technical metric, evaluation includes solution defense and presentation, architecture and scalability, and README and repository quality.

**Prize pool: 600,000 RUB.** Approximate scale of active teams: 30–40 (39 slots at checkpoint 1, 30 at checkpoint 2).

---

## 13. Suggested pipeline (consolidated checklist)

**Processing:**
- [ ] Rotate frame 90° clockwise (camera is mounted 90° counter-clockwise).
- [ ] Optional distortion correction (focal length 2.8 mm, ultra-wide).
- [ ] Tag detection (YOLO / RT-DETR / something lightweight).
- [ ] Tracking + deduplication across frames (ByteTrack / IoU-tracker).
- [ ] "Best frame" selection per unique tag (by sharpness / confidence / centrality).
- [ ] OCR of substantive fields (PaddleOCR / EasyOCR / TrOCR — whichever performs best on Russian).
- [ ] Barcode detection and decoding (**priority #1** — it's the matching key).
- [ ] QR decoding (zbar / pyzbar / Dynamsoft) — desirable but not critical.
- [ ] Tag color classification (`color`).
- [ ] Display-type extraction (`к` / `л` / `ш`).
- [ ] Output structuring to match the exact CSV format.

**CSV format:**
- [ ] All 30 fields in the correct order.
- [ ] UTF-8, separator `,`, decimal `.`.
- [ ] `"нет"` if absent on tag, empty if not recognized.
- [ ] `frame_timestamp` in **milliseconds** from video start.
- [ ] One unique tag = one row (deduplication works).

**UI / deployment:**
- [ ] Video upload + "Process" button.
- [ ] Progress bar (videos take a while).
- [ ] CSV download.
- [ ] Optional: bbox visualization on frames.
- [ ] Public URL without auth.
- [ ] If password required — document in the README.

**Documentation:**
- [ ] README with architecture, run instructions, hardware requirements.
- [ ] Description of all models used and their licenses.
- [ ] Description of datasets (including self-collected / manually labeled — mandatory!).
- [ ] "Limitations" and "Future Work" sections.

**Presentation:**
- [ ] Title + team (names, roles, contacts).
- [ ] Data and feature analysis.
- [ ] Architecture and pipeline.
- [ ] UI and CSV examples.
- [ ] Limitations.
- [ ] Scaling ideas.
- [ ] Repo link + deployed solution link.

---

## 14. Contacts

- **Hackathon chat** (Telegram): Lenta Tech Life Hack group, "Task Questions" topic — all technical questions go here.
- **Organizer email:** `LentaTechLifeHack@cl-friends.com` — for individual / sensitive questions (e.g., additional data, remote defense).
- **DMs to organizers about the task are not accepted** — public chat or email only (id=1624).

---

*Sources: official hackathon PDFs (Задача_Lenta_Tech_Life_Hack.pdf, Расшифровка_ценники.pdf, ГМ_для_ТК.pdf) + Telegram chat export from 17.05.2026 (topics: "Information", "Task Questions", "Tasks", "Task Responses").*
