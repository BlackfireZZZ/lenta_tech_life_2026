# Lenta Tech Life Hack — Agent Briefing

> This document combines the official task materials (task PDF, price tag legend, price tag catalog) **with ~230 messages from the "Task Questions" chat** covering May 12–17, 2026. The goal is to give the agent context that isn't in the PDFs: clarifications from organizers, requirements that got reworded mid-hackathon, non-obvious nuances of the scoring metric, common mistakes other teams made, and indirect "leaks" about how evaluation works.
>
> **Finals: May 24, Moscow. Online round deadline: May 19, 15:00 MSK.**

---

## 1. The task in one page

**What needs to be built.** Given video from a robot moving along supermarket shelves at Lenta hypermarkets, automatically detect price tags on shelves and extract as many fields as possible from each tag (19 fields from the tag itself + 11 fields from the QR code = 30 fields total). Output the result as a `.csv`, wrap it in a working UI (upload video → download CSV), deploy publicly, write a README, prepare a presentation.

**What's forbidden.**
- Using cloud APIs and external online services that can't be reproduced inside Lenta's internal infrastructure (strict ban).
- Manual labeling/correction **at the inference stage** (in the final pipeline) is forbidden. See §6 for an important clarification.
- Heavy models aren't fully banned, but you need to justify them.

**What's encouraged.**
- Locally deployable solution (any open-source models/weights with a suitable license — fine).
- Lightweight and applicable on constrained hardware. Bonus: optimization for `rknn (int8)`.
- Publicly deployed UI without authentication — gives **extra points**.
- Describing the limitations of your approach and ideas for scaling (organizers explicitly say: "we value maturity of approach").

**Final metric.**
- Share of price tags for which **at least 80% of substantive fields are recognized correctly**, out of the total number of price tags in GT.
- 80% is **not about the share of detected tags**, as many initially assumed. It's the **threshold for fields within a single tag**.
- Technical fields (`filename`, `frame_timestamp`, `x_min/y_min/x_max/y_max`) are **not counted in the metric**; they're only used for matching (see §5).

---

## 2. Output CSV: exact format

**From the official organizer reply** (the format question came up multiple times):
- Field separator — `,` (comma)
- Decimal separator in numbers — `.` (period)
- Encoding — UTF-8
- String fields wrapped in `"` if they contain commas or quotes
- If a parameter is **absent on the tag** → write `"нет"` (Russian for "no")
- If a parameter is **not recognized** → leave the field empty

**Important note about the sample.** Organizers posted `sample.csv` after several days of confusion. Before that, the format of the labels in the provided data **differed** from what's required (e.g., decimal separator there was `,`). Organizer quote: *"Format differences will still be handled on the metric side. The main thing is to preserve the required structure and correctly fill the substantive fields."* So don't use the provided label format as a template — go by the field spec in the PDF + sample.csv.

**Field list** (from the "Lenta Tech Life Hack Task" PDF, page 11):

From the tag:
- `filename`, `product_name`, `price_default`, `price_card`, `price_discount`, `barcode`, `discount_amount`, `id_sku`, `print_datetime`, `code` (display zone code), `additional_info`, `color` (tag color), `special_symbols` (display type), `frame_timestamp` (ms from video start), `x_min`, `y_min`, `x_max`, `y_max`

From the QR code:
- `qr_code_barcode` (`b`), `price1_qr` (`p1`), `price2_qr` (`p2`), `price3_qr` (`p3`), `price4_qr` (`p4`), `wholesale_level_1_count` (`wL1C`), `wholesale_level_1_price` (`wL1P`), `wholesale_level_2_count` (`wL2C`), `wholesale_level_2_price` (`wL2P`), `action_price_qr` (`aP`), `action_code_qr` (`aC`)

---

## 3. ⚠️ Main pitfalls and gotchas (from the chat)

### 3.1 bbox coordinates in the source labels are not clean
Confirmed by organizers (id=1446): *"the source labels contain non-ideal bboxes, some frames may be shifted relative to the actual tag position. We do account for this risk and the strictness of bbox matching will not be too tight."*

**Team Max Agafonov visually overlaid the CSV labels on frames — frames are often not on the tag** (id=1424, 1275). Additionally, some labels point to tags on the top shelf that the camera physically can't resolve (too small/occluded).

**Takeaway:** build your own tag detection, don't try to reproduce GT bboxes.

### 3.2 bbox coordinates in labels are decimals, not ints
Confirmed: *"in pixel grid the coordinates effectively correspond to integer values. The decimals appeared due to label processing format. Not critical for the solution"* (id=1274). You can round them.

### 3.3 frame_timestamp — frames from labels don't match when you decode the video
This is the last big problem raised by Ars (id=1667, on May 17, right before the deadline): *"It's not clear how exactly the frame_timestamps in the labels were obtained. When trying to decode frames, the labeling shifts by a small delta."* The decoding script or the frames themselves are **not provided**.

**What to do about it.** Organizers in id=1219 wrote: *"matching will use tolerances — for example, if you pick any frame from an interval where the tag is visible well, you almost certainly fall within these tolerances."* So for each unique tag you should submit **one** timestamp (the frame where it was best recognized), and a time tolerance applies.

### 3.4 "Best frame" — what does it actually mean
One tag passes through frames from edge to edge, and it's unclear which frame to write into the CSV. Organizers formulated (id=1180, 1219): *"fields are filled based on the best frame, where the tag was most clearly visible"*, *"you need to specify the moment of final determination, i.e., the frame with the best recognition quality."*

Marie (id=1204) framed the problem precisely: *"Video with a stop. The same tag with the same clarity for exactly 2 seconds. How do you guess the required timestamp?"* No direct answer to this specifically — the tolerance from §3.3 covers it.

### 3.5 Tag deduplication across frames — on the participant's side
Organizers state directly: *"The logic of video analysis, frame selection, candidate filtering for recognition, duplicate filtering, and result aggregation remains on the participant's side and is part of the proposed solution"* (id=1219).

This means: **tracking + deduplication is a critical component**. Without it, each tag will appear in the CSV dozens of times → the metric drops, because for each "duplicate" the bbox/timestamp match may miss.

### 3.6 Barcode — partially recognized is better than nothing
*"If you recognize 12 digits out of 13, it's better to submit 12 digits than to omit the field entirely"* (Checkpoint FAQ, id=1613).

### 3.7 QR can be skipped without disaster
*"Not recognizing the QR code will not be a critical minus by itself. You can focus on the substantive fields of the tag"* (FAQ, id=1613).

But: QR contains **11 out of 30 fields**. If you don't fill them at all, the "≥80% of fields correct" metric for the matched tag drops. See §5 on the metric — there's a subtlety about how much QR weighs.

### 3.8 Actual dataset quality
A sarcastic remark from Maksim (id=1293) that accurately reflects reality: *"Such a great dataset. Such a cool 4K camera. Excellent contrast on the tag text... Tesseract out of the box recognizes 99% of the text... QR is just a song 😂"* — in reality, barcodes and QR **don't get recognized in motion** even by experienced participants. *"At least one in motion"* — that's about QR (id=1331).

Mark_Borodin: *"reconstructing something from blur is very expensive and ineffective"* — there's no point chasing faded tags; better to focus on the ones where the model is confident.

### 3.9 Video with stops vs without stops
Two shooting modes. At stops, the tag is sharp for 1-2 seconds (essentially a photo). In motion — blurred. Strategically, it's **easier to target tags during stops** or those that fall into focus during slow motion.

### 3.10 Camera parameters — known, but not distortion coefficients
From id=1410:
- Resolution: **3840×2160**
- Sensor diagonal: 16/2.8 mm
- Focal length: **2.8 mm** (ultra-wide → noticeable edge distortion)
- The camera shoots **rotated 90° counter-clockwise** (so the raw frame is vertical, needs to be rotated)
- Distortion coefficients are not provided — *"you can estimate the correction approach yourself from the available videos"*

### 3.11 Glass partition
From the PDF: *"In some zones the robot shoots through a glass partition"* (alcohol section). Glare/double reflections — a separate case.

---

## 4. What matters for UI and deployment

- **UI is mandatory**: upload video → download CSV. Any open-source framework (Gradio, Streamlit, FastAPI+HTML — your choice).
- **Public deployment without authentication is a plus for evaluation.**
- If login/password is needed for verification — be sure to specify in the README (id=1613, FAQ): *"If checking the solution requires a login, password, or other access method, provide all necessary credentials and instructions."*
- The README should cover: architecture, which models are used, what data was labeled (if any), training pipeline, reproduction instructions.
- Processing time: no hard limit, *"a good guideline is a few seconds per frame, but longer options are acceptable"* (FAQ id=1613). But an hour per video would be a minus.

---

## 5. The metric — detailed breakdown (pulled out of the chat)

This is the **most important section**, because without understanding how scoring works, it's easy to misallocate effort.

### 5.1 Two-stage matching (id=1219)

**Stage A. Matching your tag with a GT tag:**
1. **Priority #1 — barcode.** If the barcode in your row matches GT, the rows are matched directly. This is the most reliable key.
2. **Priority #2 — spatio-temporal key.** If barcode isn't recognized or absent — matching uses `frame_timestamp + bbox` with tolerances.

**If neither key matches your row to GT, it doesn't count.**

### 5.2 Stage B. Substantive field evaluation

From id=1352 (key clarification): *"A tag is considered successfully recognized if it was matched with GT and **at least 80% of substantive fields were correctly recognized**."*

"Substantive" = everything except `filename`, `frame_timestamp`, `x_min/y_min/x_max/y_max`.

Final metric = share of such "successfully recognized" tags out of the total in GT.

### 5.3 Subtleties and unanswered questions

- **How text is compared** (`product_name`, `print_datetime`, etc) — Levenshtein, exact match, something else? **No direct answer in the chat**, but there's a hint: *"the product name doesn't have to be perfectly recognized character-by-character, but the closer to the actual text on the tag, the better"* (id=1613). Probably fuzzy with a threshold, but that's speculation.
- **QR as one field or 11.** konstanta (id=1548) asked directly: is QR counted as 1 field or each of its 11 separately? If each separately — passing the 80% threshold on 19+11=30 fields without QR is nearly impossible. If QR=1 field — passable. **Organizers gave no direct answer**, redirecting to the criteria. Plan for each QR field counting separately — it's safer.
- **What to write when the field is absent** on the tag vs not recognized: `"нет"` vs empty (see §2). This is critical — if you put empty where `"нет"` is expected, the field may count as not recognized.

### 5.4 Strategic conclusions

- **Recognizing the barcode is critical.** It's the primary key that guarantees matching. Without barcode you depend on bbox+timestamp accuracy, and bbox in GT is messy (see §3.1) → higher chance of not matching.
- **Duplicates kill the metric.** If one tag appears 5 times in the CSV, organizers match **one** row as GT, the other 4 may not match → counted as garbage and drag the overall score down. *(This is inferred from the metric formulation — but logically follows.)*
- **Skipping a tag is better than submitting noise.** If confidence is low on substantive fields — don't submit it, rather than submit with errors.

---

## 6. ⚠️ Updated rule about manual labeling (important change!)

At the May 12 opening, organizers said "manual labeling is forbidden". This caused mass confusion. **The updated position** (id=1436, 1458, 1497):

**ALLOWED:**
- Manually labeling data for training/fine-tuning models
- Adding labels to the provided dataset
- Using external datasets (any open ones with a suitable license) and self-collected ones
- Building an SKU directory / your own product database (id=1613, FAQ)
- Using data from Lenta's website (id=1613, FAQ)

**FORBIDDEN:**
- Any manual work **at the inference stage**: editing the CSV, manually picking frames, manually adjusting bboxes, operator-in-the-loop
- Using observations from previous runs on the test video to tune the service for subsequent runs (id=1454)

**REQUIRED:**
- In the solution materials, **describe in detail**: what data was labeled, what volume, how it was used, what fine-tuning pipeline. Otherwise the score will drop.

Quote id=1436: *"If the data used, labeling, or training stages are not disclosed in the solution materials, this may affect the evaluation"* — so transparency here is at the level of a mandatory requirement.

---

## 7. The "ideal solution" vision from organizers (between the lines)

Compiled from the checkpoint FAQ (id=1613), Olga Dubovaya's replies (id=1177, 1178), general framings, and what gets explicitly praised/requested.

### 7.1 What was explicitly named as a priority

1. **Desktop solution, not on-robot deployment.** *"Priority is a regular desktop solution. Running on the robot, distributed architecture, or processing right on the robot can be a plus, but it's not a mandatory requirement"* (FAQ).
2. **Recognition quality > speed.** This is repeated in dozens of replies.
3. **Substantive fields > QR.** *"Priority is substantive fields that help understand which product the tag is for and what its price is. Fields that allow linking the tag to a product matter."* (FAQ).
4. **Applicability, not lab work.** *"It's important to be able to explain how this will scale and be applied in the business process"* (FAQ).

### 7.2 The architecture they're already considering

Olga Dubovaya's remark (id=1177): *"We're considering various architecture options. With deployment on the robot, and without it — sending photos via API to an external server."*

So at Lenta they haven't decided yet — either a heavy model on the server with a stream of photos from the robot, or a light model right on the robot. **A solution that explicitly addresses both scenarios** (e.g., light on-device detector + optional heavy OCR on the server) is likely to win on "scalability and applicability".

### 7.3 Indirect hardware signals

*"Right now it's a prototype, and the configuration can be changed to fit the solution"* (id=1178) — they refuse to name specific hardware. But the mention of `rknn` (int8) in the task = **Rockchip NPU** (RK3568/RK3588), a typical board for edge AI in robotics. This gives a baseline: ARM, NPU ~6 TOPS, typical models for NPU inference — YOLOv5/v8 nano/small, MobileNet, MobileSAM.

### 7.4 What gets praised in their language

- "Maturity of approach" — understanding of limitations, explicit discussion of what doesn't work.
- Describing edge cases in README/presentation.
- Ideas for scaling and improvements "for industrial deployment" (id=1372).

---

## 8. Price tag types and their "color logic"

From the materials (`ГМ для ТК.pdf` (Hypermarket for retail design), `Расшифровка ценники.pdf` (Tag legend)) — 7 base tag types.

**Sizes/orientations:**
- A4 vertical / A4 horizontal
- A3 horizontal, A2 horizontal
- 6×6 (compact), 6×12 with shelf-talker
- МНЦ (small new price tags) — white/yellow/green/red

**Color logic (important for the `color` field):**
- **White** — regular shelf price (РПЦ), standard item
- **Yellow** — bakery
- **Green** — fruits and vegetables (often with scale number)
- **Red** — promo / promotional shelf price (АПЦ), discount
- **Black circle with -N%** or **-N₽** — promo marker (a separate element on the tag, not the color of the whole tag)

**Display type (`special_symbols`):** possible values — `к` (box), `л` (tray), `ш` (piece). Not always shown.

**Per-weight vs per-piece marker:** present in the format (per 1 kg / per 100g / per 1 piece), may require separate classification.

**Organizer recommendation:** *"Расшифровка ценники.pptx is the main applied material... ГМ для ТК.pptx can be used as an additional reference if you want to better understand tag variability and build a more scalable solution"* (id=1615).

---

## 9. "Leaks" and tips from participants (what others let slip)

These aren't hints from organizers but observations from other teams that can save time.

- **Tesseract works decently out of the box on static tags** (Maksim's comment, id=1293, ironic but essentially true for frames at stops). But not enough for motion.
- **YOLO** was discussed as the obvious choice for detection (id=1499). Organizers confirmed: allowed (id=1584), note the license in the README.
- **Mark_Borodin** got a calm answer to a skeptical question "how do you stuff this into a Raspberry Pi" — meaning edge deployment is a plus, but not required.
- **Repeated capture of the same tag across many frames** is the norm, teams are expected to build deduplication themselves (see §3.5).
- **Team "ТаР" (Andrey K.)** talked about diligence and imperfection — realistic about expectations.
- **Team "Три Лося́" (Three Mooses)** mentioned working in audio and CV, testing hypotheses in real stores — they may have an advantage with field data.
- **Team ALOLA** emphasizes confidence scores — relevant to §5.4: filter by confidence.
- **Team snoopy dog (Yuna, Vika, Nikita)** — flexible, multi-profile.
- **PriceVision AI (Yuri Tkenov)** — active posters, emphasizing OCR.
- **МтусиТоп** — focus on BI/analytics (Chibi — RFM, KPI), may produce a strong UI/dashboard.

### Teams spotted at checkpoints
~39 slots at the first checkpoint, ~30 at the second. This indicates the approximate size of "active" hackathon teams — around 30-40. Prize fund: 600,000 RUB.

### What NotFound was asking in DMs (id=1595-1624)
Team TeamNotFound tried to ask "closed" questions to organizers in DMs — organizers refused to answer privately, requiring public chat or email. **Signal:** all answers are public, everything discussed in the chat is everything other teams know.

---

## 10. What we couldn't pull from the chat (open questions)

These remained without a clear answer, and worth planning for **robustness** to different interpretations:

1. **Exact text-field comparison function.** Levenshtein? Exact? Fuzzy with threshold? — plan for fuzzy, normalize case/spaces/punctuation.
2. **QR as one field or 11.** See §5.3 — plan for "11".
3. **How many rows of tags "bottom to top" are counted.** Answer id=1245 — "all visible", but "if occluded or blurred — will be taken into account at check". This is vague. Pragmatically: top shelves are often unreadable → don't try.
4. **Whether the frame-by-frame labels will be recomputed.** Ars's request for the decoding script (id=1667) on May 17 still pending.
5. **Which exact videos are in the private set.** Private only, no public set at evaluation (id=896).

---

## 11. Timeline and checkpoints

- **May 12, 17:00** — opening, task statement, Q&A
- **May 15, 17-19** — checkpoint 1 (FAQ with 14 questions published after, see id=1613)
- **May 17, 12-14** — checkpoint 2
- **May 19, 15:00 MSK** — deadline for online round submission (upload to personal cabinet)
- **May 21, 21:00** — finalist announcement
- **May 24** — finals in Moscow (remote possible by arrangement, request by May 17 14:00)

**At the finals**, in addition to the technical metric, evaluation includes:
- Solution defense and presentation
- Architecture and scalability
- Quality of README and repo

The presentation must include: title slide with solution and team name, team composition slide (names, roles, contacts), task analysis, architecture, examples of UI and output CSV, **approach limitations**, scaling ideas, repo link, deployed solution link.

---

## 12. Checklist (short summary for the agent)

When generating code or solutions — keep in mind:

**Pipeline:**
- [ ] Rotate frame 90° clockwise (camera is rotated 90° counter-clockwise)
- [ ] Optional distortion correction (focal length 2.8mm, ultra-wide)
- [ ] Tag detection (YOLO/RT-DETR/something lightweight)
- [ ] Tracking + deduplication across frames (ByteTrack/IoU-tracker)
- [ ] "Best frame" selection per unique tag (by sharpness/confidence/centrality)
- [ ] OCR of substantive fields (PaddleOCR/EasyOCR/TrOCR — whichever works best on Russian)
- [ ] Barcode detection and decoding (priority #1 — it's the matching key)
- [ ] QR decoding (zbar/pyzbar/Dynamsoft) — desirable but not critical
- [ ] Tag color classification (`color`)
- [ ] Display type extraction (к/л/ш)
- [ ] Structuring output to match the exact CSV format

**CSV format:**
- [ ] All 30 fields in the correct order
- [ ] UTF-8, separator `,`, decimal `.`
- [ ] `"нет"` if field absent on tag, empty if not recognized
- [ ] frame_timestamp in **milliseconds** from video start
- [ ] One unique tag = one row (deduplication works)

**UI/Deployment:**
- [ ] Video upload + "Process" button
- [ ] Progress bar (videos take long to process)
- [ ] CSV download
- [ ] Optional: bbox visualization on frames
- [ ] Public URL without authentication
- [ ] If password needed — specify in README

**Documentation:**
- [ ] README with architecture, run instructions, hardware requirements
- [ ] Description of all used models and their licenses
- [ ] Description of datasets (including self-collected/manually labeled — mandatory!)
- [ ] "Limitations" and "What to improve" sections

**Presentation:**
- [ ] Title + team (names, roles, contacts)
- [ ] Data and feature analysis
- [ ] Architecture and pipeline
- [ ] Examples of UI and CSV
- [ ] Limitations
- [ ] Scaling ideas
- [ ] Link to repo and deployed solution

---

## 13. Contacts and channels

- **Hackathon chat** (Telegram): Lenta Tech Life Hack group, "Task Questions" topic — all technical questions go there
- **Organizer email**: `LentaTechLifeHack@cl-friends.com` (for individual/sensitive questions — e.g., additional data, remote defense)
- DMs to organizers about the task **are not accepted**, only public chat or email (id=1624)

---

*Document compiled based on the chat export from 17.05.2026 (4 topics from the Lenta Tech Life Hack chat: "Information", "Task Questions", "Tasks", "Task Responses") and official hackathon PDF materials.*
