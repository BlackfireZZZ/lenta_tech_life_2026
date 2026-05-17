# Lenta Price Tag Field Guide

A field-by-field guide to how price tags are physically laid out in Lenta hypermarkets, intended for agents building OCR pipelines. Compiled from two source decks: **Расшифровка ценники** (canonical "exploded view" of 7 reference tags with field callouts) and **ГМ для ТК** (full catalog of tag templates by mechanic × size).

> **Status:** complete. All 66 slides of `ГМ для ТК` and all 7 slides of `Расшифровка ценники` have been read end-to-end.

---

## 1. The two source decks and how to use them

**`Расшифровка ценники` (7 slides) — the canonical reference.** Each slide shows one real tag with arrows pointing to each field. This is the ground truth for **where each CSV field is physically located** on the tag. Organizers explicitly named this deck as "the main applied material" (chat id=1615).

**`ГМ для ТК` (66 slides) — the full catalog.** Organized as a matrix of **mechanic × size**:

| Mechanic | Slides |
|---|---|
| **РПЦ** (Регулярная полочная цена / Regular shelf price) | 1–12 |
| **АПЦ** (Акционная полочная цена / Promo shelf price) | 13–30 |
| **Распродажа** (Clearance) | 31–32 |
| **Скидка при покупке ОТ N** (Discount from N units) | 33–44 |
| **Скидка при покупке до N шт** (Discount up to N units) | 45–54 |
| **BOGOF** (Buy One Get One Free) | 55–65 |
| **РПЦ/АПЦ + ШФ** (with shelf-talker linkage) | 66 |

Within each mechanic, slides walk through the sizes: 6×6, 6×12, А5, А4 vertical, А4 horizontal, А3 vertical, А3 horizontal, А2 vertical, А2 horizontal, МНЦ. Use this deck as additional reference for tag variability — same logic, different layouts and physical sizes.

---

## 2. Tag size taxonomy

| Size | Orientation | Notes |
|---|---|---|
| **6×6** | square, compact | Most common small tag. Has QR top-right. Used for РПЦ, АПЦ, sales, threshold discounts. |
| **6×12** | horizontal, compact | Wider variant of 6×6. **Barcode is printed as text ("ШК: 160404_633570") rather than as a graphical barcode in many variants.** Has QR top-left. |
| **А5** | horizontal | Mid-size. **No QR code.** Price is large on the right; small graphical barcode bottom-right; promo marker is a **black half-circle in the top-left** for АПЦ. |
| **А4 vertical** | vertical | Large. Two prices stacked (без карты top / по карте bottom, larger). QR small at bottom-right. |
| **А4 horizontal** | horizontal | Large. Same fields as А4 vertical, horizontal layout. |
| **А3 vertical / А3 horizontal** | both | Larger versions of А4, same logic. |
| **А2 vertical / А2 horizontal** | both | Largest, same logic as А3/А4. |
| **МНЦ** (small new tag) | narrow horizontal strip | **Very few fields:** product name, price без карты, price по карте, date. **No barcode, no QR, no id_sku visible.** Color encodes category (white/yellow/green/red). |

**A note about button numbers** (`number on scales`, "номер на весах"): variants are labeled with the suffix "**без кнопки**" (without button) and "**с кнопкой**"/no suffix (with button). When present, this number appears in a rectangular frame near the title or in a yellow/grey circle for МНЦ. It's part of `additional_info`.

---

## 3. Canonical field map (from "Расшифровка ценники")

Each subsection below = one reference slide. Spatial layout uses a 3×3 grid metaphor where helpful (TL = top-left, BR = bottom-right, etc.).

### 3.1 Slide 1 — Compact 6×6 РПЦ with discount circle (`-32%`)
**Example product:** Кофе NESCAFE Classic (Россия) 500g — price 250.09 / 168.90 with -32% from non-card price.

```
┌─────────────────────────────────┐
│ Product name (TL)      QR (TR)  │
│                                 │
│                  Без карты, ₽   │
│                    250⁰⁹        │
│  ┌─────┐                        │
│  │-32% │      С картой, ₽       │
│  │ от  │       168⁹⁰            │
│  │цены │                        │
│  │б/к  │                        │
│  └─────┘                        │
│ id_sku                          │
│ 24.12.2025 12:25  barcode (BR)  │
└─────────────────────────────────┘
```

| Field | Where on the tag |
|---|---|
| `product_name` | Top-left, multiline |
| `qr_code_barcode` (and other QR fields) | Top-right, QR code |
| `price_default` | Right side, mid-top, "Без карты, ₽" label |
| `price_card` | Right side, large, "С картой, ₽" label |
| `discount_amount` | Left side, **black circle** containing `-N%` (e.g. `-32%`) with text "от цены без карты" |
| `id_sku` | Bottom-left, above the date |
| `print_datetime` | Bottom-left, format `DD.MM.YYYY HH:MM` |
| `barcode` (graphical + digits) | Bottom-right, EAN-style barcode with digits below |

### 3.2 Slide 2 — Compact 6×6 АПЦ weight-priced, with display type and zone code
**Example product:** Креветки Королевские с/м с/г 50/70 вес (Россия) — 72.59 / 55.39 за 100г, -23%.

Adds two fields not present in slide 1:

| New field | Where |
|---|---|
| `code` (display zone code) | Bottom-left, **line directly above the date**, format like `06_062 003` |
| `special_symbols` (display type) | Bottom-center, **letter in a circle** between date and barcode: `Ш` (piece / штука), `К` (box / короб), `Л` (tray / лоток) |

Also note: prices have the suffix "за 100г" (per 100g) in the label for weighted goods.

### 3.3 Slide 3 — 6×6 АПЦ wine with `additional_info`
**Example product:** Вино HAUT MARIN Colombard Ugni-blanc (Франция) 0.75L — 1747.36 / 1104.99, -36%.

| New field | Where |
|---|---|
| `additional_info` | Below the product name, **in a rounded rectangle**. Example value: `"Сухое"` (dry). For wine this denotes sweetness category; for other categories it can hold other free-text info. |

All other fields identical to slide 2 (with `code`, `special_symbols`, etc.).

### 3.4 Slide 4 — А4 vertical "Discount from N" (three stacked prices)
**Example product:** Напиток безалкогольный ЧЕРНОГОЛОВКА НеЛимонад (Россия) 2L — three prices: 527.39 (без карты), 500.99 (по карте), **168.76** (по карте от 5 шт).

```
┌──────────────────────────┐
│ Product name (top-left)  │
│ multiline                │
│                          │
│ ₽/1шт                    │
│        527³⁹ Без карты   │
│        500⁹⁹ По карте    │
│  168⁷⁶ ┌─────────┐       │
│        │По карте │       │
│        │от 5 шт  │       │
│        └─────────┘       │
│                          │
│  date  id_sku       QR   │
└──────────────────────────┘
```

| Field | Where on the tag |
|---|---|
| `product_name` | Top, multiline, bold |
| `price_default` | Mid-right, smaller "Без карты" label |
| `price_card` | Mid, "По карте" label |
| `price_discount` | Large bottom price with rectangular label "По карте от N шт" |
| `print_datetime`, `id_sku` | Bottom-center, small text |
| QR code | Bottom-right, small |

**Critical:** on this template there is **no visible "Без карты" graphical barcode** in the standard place. Recognition of `barcode` digits will rely on the QR `b` field or fail.

### 3.5 Slide 5 — Compact 6×6 РПЦ without discount circle
**Example product:** Шоколад FAZER Geisha (Финляндия) 100г — 345.09 / 303.79, no discount.

The plainest variant. **No discount circle**, the two prices sit one above the other on the right, **barcode is on the LEFT** (rotated from slide 1).

| Field | Where |
|---|---|
| `product_name` | Top-left |
| QR | Top-right |
| `id_sku` | Above barcode, left side |
| `barcode` (graphical + digits) | Left-center |
| `price_default` | Top-right area, "Без карты, ₽" |
| `price_card` | Bottom-right, large, "По карте, ₽" |
| `print_datetime` | Bottom-center |

### 3.6 Slide 6 — 6×6 РПЦ with shelf-talker (right side panel)
**Example product:** Орехи грецкие очищ. 1 сорт вес — 1284.29 / 1029.99 за 1 кг, **with shelf-talker showing "номер на весах: 214"**.

A **shelf-talker** is a separate white rectangular panel attached to the right of the main tag. It carries:
- "Номер на весах" (scale number) for weighted goods
- "Удачная упаковка" promo text
- Other auxiliary product info

This is `additional_info`. The main tag part is otherwise identical to slide 5. When the robot's view of a tag also includes a shelf-talker, **both are part of the same logical tag** — the scale number on the talker maps to `additional_info`.

### 3.7 Slide 7 — Same product as 6, without the shelf-talker
Confirms that the shelf-talker is **optional**: same tag, no right panel. All other fields in the same positions.

---

## 4. The `discount_amount` encoding — read this carefully

This field is shape-coded and content-coded simultaneously, and the encoding **differs across tag formats**.

### 4.1 Compact 6×6 АПЦ → black circle on the LEFT
Black filled circle, text inside reads `-N%` (top line) + "от цены без карты" (bottom). Always a percentage on compact tags.

### 4.2 А4 / А3 / А2 АПЦ → black half-circle ("dome") on top
For vertical layouts: dome is **centered at the top above the product name**.
For horizontal layouts: dome is in the **top-right corner**.

**Critical rule (printed at the bottom of every АПЦ slide):**

> *Если скидка в рублях меньше 100 ₽, то идёт отражение в %, а если больше или равно — то в рублях.*

> **"If the discount in rubles is < 100 ₽, it's shown as a percentage. If ≥ 100 ₽, it's shown in rubles."**

So `discount_amount` can be `-18%`, `-23%`, `-32%`, OR `-100₽`, `-248₽`, `-286₽`. **Both encodings are valid values for the same field.** Your OCR must handle both. Storing the raw string ("-18%", "-286₽") is the safest — parsing into a numeric value is downstream.

### 4.3 6×12 АПЦ → rectangular outline frame on the right
`-45%` etc. **inside a rounded rectangle** in the top-right, smaller than the dome but same content rules.

### 4.4 А5 АПЦ → black half-circle in the top-left
With `-33%` etc.

### 4.5 МНЦ АПЦ (cheap promotional small tag) → red arrow + crossed-out old price
Most distinctive layout. **Red downward-pointing arrow** on the left containing `-N%`. **Old price `1 029.99` is printed top-right with a red diagonal strikethrough across it.** New price is small on the bottom-left as "цена без карты". This is the only mechanic where the **old price is visually crossed out**.

### 4.6 A4/A3/A2 "Скидка ОТ N" → no discount circle on the main tag
Instead the discount is encoded as the **third stacked price** with a rectangle label "По карте от N шт/кг". `discount_amount` here either has to be **computed from prices** or extracted from the small "-N%" labels inside the threshold blocks (see §5).

### 4.7 Clearance ("Распродажа") А4 → yellow half-circle + big "АКЦИЯ" word
Yellow dome on top with `-N%`, then the word "АКЦИЯ" (very large), then the product name. Completely different visual identity from АПЦ — use it as a strong feature for mechanic classification.

### 4.8 ZB21 variants → discount marker may be ABSENT
The "ZB21" subtype of threshold discounts replaces the dome with a **black half-circle containing "При покупке от N шт"** — the wording of the **promo mechanic**, not a percentage. So `discount_amount` for ZB21 tags may be `"нет"` if no percentage is shown on the tag.

---

## 5. The "Скидка ОТ N" mechanic — two simultaneous discounts

Slides 33–44 show tags with **two side-by-side discount panels**, separated by a thin vertical dotted line:

```
┌────────────────────────────────────┐
│ Product name              QR       │
│ Номер на весах: 81                 │
│ с картой по акции, ₽/кг:  от 5 кг  │
│  -53% от цены б/к | -85% от цены   │
│      5⁴¹          |      1⁶⁸       │
│ date  id_sku  Ш  Ш: barcode_digits │
└────────────────────────────────────┘
```

| Left panel | Right panel |
|---|---|
| Regular "with card" discount: `-53%`, price 5.41 | Threshold discount "from N units": `-85%`, price 1.68 |
| → maps to `price_card` + `discount_amount` | → maps to `price_discount` + the threshold N (in `additional_info`?) |

**This is genuinely ambiguous to map.** Pragmatic recommendation: put `discount_amount = "-53%"` (the everyday discount), `price_card = 5.41`, `price_discount = 1.68`. The threshold "от 5 кг" is part of the visual but doesn't map cleanly to any single CSV field.

Note also: **`barcode` digits on these tags are written as text `"Ш:2099999041253"`** (Ш = sokrashchenie for ШК = shtrikh-kod). No graphical EAN bars. You'll need a text OCR for them, not a barcode decoder.

### 5.1 ZB21 subtype of "Скидка ОТ"
Slides 34, 36, 38, 60 — has prefix `ZB21_` in the template name. Visual differences:
- **Discount dome is replaced** by text "При покупке от N шт" (no percentage)
- Layout otherwise similar
- `discount_amount` will often be **absent** as a printed value

---

## 6. The "Скидка ДО N шт" mechanic (inverse threshold)

Slides 45–54. **Logic is reversed from "Скидка ОТ":** the discount applies only **up to** N units; everything beyond N is at regular price.

### 6.1 Compact 6×6 layout (slide 45)
Two side-by-side panels, just like "Скидка ОТ", but the labels above each panel are inverted:

```
с картой по акции, ₽/кг:  с картой за каждый
не более 5кг              следующий кг, ₽/кг:
 -69% от цены б/к
   195⁸²                       200⁰⁰
```

- **Left panel:** discount applies **"не более 5 кг"** — 195.82 at -69% off
- **Right panel:** "**за каждый следующий кг**" — 200.00 (no discount)

### 6.2 6×12 layout (slide 46)
Single panel layout, **discount marker `-21%` is a small framed rectangle in the top-right** (same as АПЦ 6×12). Two prices:
- "**Более 5 кг, с картой:** 129.99 ₽/кг" (mid-left, small)
- "**Не более 5 кг с картой:** 106.99" (large right)
- "Без карты, ₽/кг: 136.84" (smallest, below the "более 5 кг" line)

### 6.3 А5 layout (slide 47)
- 3 small prices stacked on the left: "С картой / 1 кг ₽ 129.99", "Без карты / 1 кг ₽ 136.84"
- **Large promo price on the right:** "С картой по акции / 1 кг ₽ 106.99"
- Under the promo: italic "Цена по акции действительна при покупке не более 5 кг"
- "Акция действует с DD.MM.YYYY по DD.MM.YYYY" date range mid-bottom
- Graphical barcode bottom-center
- **No QR code** on А5

### 6.4 А4 / А3 / А2 layouts (slides 48–53, both vertical and horizontal)
**Three stacked prices, mirror image of "Скидка ОТ N"**:
- "**Без карты**" — top
- "**По карте**" — middle
- "**По карте до N кг/шт**" — bottom, large, with rectangular label `[По карте до 5 кг]`

**Visually almost identical to "Скидка ОТ"** — distinguishable only by reading "ДО" vs "ОТ" in the rectangular label under the third price. **Use the text of the label as the discriminator.**

### 6.5 МНЦ layout (slide 54)
- **No red arrow, no crossed-out price** (very different from МНЦ АПЦ — §4.5)
- Top-right: large "Цена по карте за 1 кг 109.99 руб"
- Bottom-left: "Цена без карты 115.79 руб"
- Bottom-right: "Цена по карте за 1 кг, при покупке **не более 5 кг** руб:" (label may be without a printed value — it lives in the template)
- Optional yellow circle / rectangle "Номер на весах: 81" on the left for weighted goods
- "Акция действует 23.12.2025 по: " date line at the bottom

### 6.6 Field mapping for "Скидка ДО"
- `price_default` → "Без карты" (cheapest case, regular non-loyalty price)
- `price_card` → "По карте" / "С картой" (regular loyalty price)
- `price_discount` → the **promo price valid up to N units** (large, with "до N" label)
- `discount_amount` → "-N%" if printed (panels) or computable from prices
- `additional_info` → may contain the "не более N кг/шт" threshold and date range

---

## 7. The BOGOF mechanic ("Buy One Get One Free" / "Скидка при покупке X")

Slides 55–65. The promo activates only when the customer buys **N units** — typically expressed as **"3=2"** (buy 3, pay as for 2) or **"при покупке от N шт"**. Every size has a slightly different visual encoding.

### 7.1 6×6 BOGOF (slide 55) — distinctive "3=2" rectangle
```
┌──────────────────────────────┐
│ Product name        QR       │
│                              │
│              с картой, ₽:  без карты, ₽: │
│                54⁰⁶          115⁷⁹       │
│ ┌─────┐                                  │
│ │АКЦИЯ│  С картой по акции при покупке   │
│ │ 3=2 │  3шт., ₽:                        │
│ └─────┘     52¹⁰                         │
│                                          │
│ id_sku  graphical barcode                │
└──────────────────────────────────────────┘
```

- **Bordered rectangle on the left containing "АКЦИЯ! 3=2"** — distinctive feature
- Three prices: "с картой 54.06", "без карты 115.79" (both small, top-right), promo "52.10" (large, center)
- **No discount circle / dome / arrow** — the "3=2" rectangle replaces them

### 7.2 6×12 BOGOF (slide 56) — "3=2" rectangle top-right
- **Rounded rectangle with "3=2"** in the **top-right corner**
- Three prices: "За 1 кг с картой 89.99" (small), "Без карты, ₽/кг: 189.49" (smaller), promo "126.49" (large, right) with label "При покупке 3 кг с картой, ₽/1кг"
- "К" / "Л" / "Ш" display type marker still applies

### 7.3 А5 BOGOF (slide 57) — **no rectangular marker at all**
- 2 small prices stacked left: "С картой / 1 шт ₽ 24.99", "Без карты / 1 шт ₽ 26.31"
- 1 large promo price right: "С картой по акции / 1 шт ₽ 17.62"
- Italic line: "Цена по акции действительна при покупке 3 штук"
- "Акция действует с DD.MM.YYYY по DD.MM.YYYY" line
- Graphical barcode bottom, **no QR**
- **Identification cue:** the italic "при покупке N штук/кг" line is what marks this as BOGOF, not the visual marker

### 7.4 А4 / А3 / А2 BOGOF (slides 58, 59, 61, 62, 63, 64) — visually indistinguishable from "Скидка ОТ/ДО"
**Three stacked prices**, rectangular label on the third one reads "По карте при покупке 3 кг" or "По карте при покупке 3 шт":

```
₽/1кг
        168⁴⁹ Без карты
        119⁹⁹ По карте
  112⁸⁸ ┌─────────┐
        │ По карте│
        │ при по- │
        │ купке   │
        │ 3 кг    │
        └─────────┘
       QR (bottom-right)
```

**Critical:** the layout is **identical** to "Скидка ОТ" (§5) and "Скидка ДО" (§6) on А4/А3/А2. The **only way to distinguish the three mechanics is by reading the wording of the third-price label**:
- "По карте **от** N кг/шт" → Скидка ОТ
- "По карте **до** N кг/шт" → Скидка ДО
- "По карте **при покупке** N кг/шт" → BOGOF

### 7.5 ZB21 BOGOF А4 (slide 60)
- **Black half-dome in the top-right** containing text "при покупке от 3 шт" (resembles АПЦ marker, but text instead of percent)
- Large promo price center: "72.98" with label "с картой по акции при покупке от 3 шт"
- Two small prices bottom-left: "без карты 115.79 ₽/шт", "**с картой без акции** 54.06 ₽/шт"
- Date range "Акция действует с DD.MM.YYYY" bottom-right
- **Important pricing quirk:** the BOGOF promo price (72.98) may be **higher than the regular loyalty price** (54.06). This is correct — BOGOF activates only at threshold N, and the structure is "buy N+1 pay for N". Per-unit math at the threshold isn't always cheaper than the everyday loyalty price for buying 1. **Don't reject values just because `price_discount > price_card`.**

### 7.6 МНЦ BOGOF (slide 65) — minimal layout
- "Цена по карте за шт., руб: 109.99" (top-right, large)
- "Цена без карты: 115.79" (bottom-left)
- "Цена по акции за шт., руб:" (bottom-right, label only — value may be empty in template)
- "Акция действует с DD.MM.YYYY по: " (bottom-left)
- **No discount marker, no crossed-out price, no QR, no barcode**

### 7.7 Field mapping for BOGOF
- `price_default` → "Без карты" (regular non-loyalty)
- `price_card` → "По карте" / "С картой" / "С картой без акции"
- `price_discount` → promo price valid at N units
- `discount_amount` → typically **absent as a printed value**. The promo is encoded as "3=2" or "при покупке от N" — neither maps cleanly to a percentage or ruble amount. **Expect `"нет"` here often.** If the tag does print a "-N%" anywhere, capture it; otherwise leave empty.
- `additional_info` → threshold "N кг/шт" and date range

---

## 8. РПЦ/АПЦ with shelf-talker (ШФ) — slide 66

The very last slide of the deck shows a separate **"linked" mechanic** where a regular РПЦ or АПЦ tag is paired with a **shelf-talker** (ШФ — short for "шелфтокер") — a small square panel attached **to the right of the main tag**.

```
┌──────────────────┬─────────────────┐
│ Main tag (6×6)   │  Shelf-talker   │
│ Product name  QR │                 │
│                  │  номер на весах │
│ Без карты 1 284²⁹│                 │
│ С картой за 1кг  │       214       │
│   1 029⁹⁹        │                 │
│ id_sku  barcode  │                 │
└──────────────────┴─────────────────┘
```

### 8.1 What the shelf-talker holds
- "**Номер на весах**" (scale button number) — most common content for weighted goods. Critical for cashier ergonomics.
- "**Удачная упаковка**" (good-deal package size) text
- Other free-text product info

### 8.2 Why this matters for the pipeline
- The shelf-talker and the main tag are **a single logical price tag** in the CSV. One row per pair.
- The button number on the talker maps to `additional_info`.
- **Visually**, the talker can be **detected as a separate object** by a generic detector — your tracking/aggregation logic must **glue** the talker to its parent tag (by horizontal adjacency).

### 8.3 The `color` field — a critical clarification from slide 66
Slide 66 has two examples captioned:
- **АПЦ + shelf-talker → "Красный ценник"** ("red tag")
- **РПЦ + shelf-talker → "Жёлтый ценник"** ("yellow tag")

Yet in both photos the main tag is white! The **red/yellow color refers to the shelf-talker / accompanying signage**, not the main tag body. This means the `color` field in the CSV may follow a logic that **isn't fully captured by the main tag's body color** — it may reflect:
- The accompanying talker color
- The category (red = promo, yellow = regular weighted)
- A system attribute not visually obvious

**Pragmatic recommendation:** for the OCR pipeline, classify `color` based on the most visually salient color on the tag composition (talker + main tag together), defaulting to "белый" if everything is white. Don't trust a heuristic of "the main tag is white → color = белый" alone.

---

## 9. Updated color logic (`color` field)

Replacing the simplified table from earlier:

| Color | Where it appears | Likely categories |
|---|---|---|
| **Белый** (white) | Default body color of most tag formats | All РПЦ; many АПЦ |
| **Жёлтый** (yellow) | (a) МНЦ bakery body color, (b) clearance А4 "АКЦИЯ" dome, (c) **shelf-talker / linked sign** indication on weighted products | Bakery, weighted regulars |
| **Зелёный** (green) | МНЦ body color for fruits and vegetables | F&V (always weighted) |
| **Красный** (red) | (a) МНЦ promotional body color, (b) **red downward arrow** on МНЦ АПЦ, (c) **shelf-talker / linked sign** indication on promo products | Promotions, МНЦ АПЦ |

**Critical rule (from slide 66):** the `color` field may be determined by **the linked shelf-talker or the system attribute** rather than the main tag's body. If you can detect a colored shelf-talker or accompanying sign, use that color.

The **black/red/yellow promo markers** (discount circles, domes, arrows, "АКЦИЯ" rectangles) are **never** the `color` field — they're graphical overlays.

---

## 10. Field-by-field cheat sheet (across all tag types)

| CSV field | Where to look (in priority order) |
|---|---|
| `product_name` | Top of the tag, multiline, bold; usually 2-4 lines. For МНЦ "набор по товарам" → two product names "название в строке 1/2" |
| `price_default` | Smaller price labeled "Без карты" / "₽/1шт без карты" / "₽/100г без карты" |
| `price_card` | Larger price labeled "С картой" / "По карте" / "С картой без акции" (on BOGOF ZB21) |
| `price_discount` | Promo price valid under conditions — typically a third price with one of these labels: "По карте от N шт/кг" (Скидка ОТ), "По карте до N шт/кг" (Скидка ДО), "По карте при покупке N кг/шт" (BOGOF) |
| `barcode` | (1) Graphical EAN with digits below at bottom; (2) text starting with "ШК:" or "Ш:" on 6×12 and threshold-discount layouts; (3) absent on СТОЛОТО, МНЦ, BOGOF МНЦ |
| `discount_amount` | Black circle / dome (АПЦ), red arrow (МНЦ АПЦ), yellow dome (Clearance). Value: `-32%` or `-286₽` (rule: <100₽ → %, ≥100₽ → ₽). **Often `"нет"` for BOGOF and ZB21** — no explicit percentage printed. |
| `id_sku` | Small text near bottom-left, usually 12+ digit alphanumeric. Example: `220301 664884`. **Absent on МНЦ** |
| `print_datetime` | Format `DD.MM.YYYY HH:MM`, near bottom of tag |
| `code` | Display zone code. Format `06_062 003` (digits + underscores). On compact 6×6 sits on the line directly above the date. On 6×12 appears below the "ШК:" text line as `07_073 001 - 074` |
| `additional_info` | Multiple sources, **collect any/all and concatenate**: (a) text in rounded rectangle below product name like `"Сухое"`; (b) "номер на весах: N" in a rectangle/circle (frame or yellow circle on МНЦ); (c) shelf-talker text "номер на весах N", "Удачная упаковка"; (d) promo date range "Акция действует с DD.MM.YYYY по DD.MM.YYYY"; (e) threshold "от N кг/шт" / "до N кг/шт" / "при покупке N шт"; (f) "3=2" / "купи N плати как за M" text |
| `color` | Tag composition color including shelf-talker. White by default. See §9. **Black/yellow/red promo markers are not `color`** |
| `special_symbols` | Single letter in a circle between date and barcode at the bottom: `Ш` (штука/piece) / `К` (короб/box) / `Л` (лоток/tray). Often absent |
| QR fields (11) | Decode from QR code if present. **Absent on А5, МНЦ, BOGOF МНЦ** — write `"нет"` for all 11 |
| `frame_timestamp`, `x_min/y_min/x_max/y_max` | Computed by your pipeline, not read off the tag |

---

## 11. QR code presence by format

| Tag size | QR? | Position |
|---|---|---|
| 6×6 РПЦ/АПЦ/Распродажа | **YES** | Top-right corner |
| 6×6 BOGOF | **YES** | Top-right corner |
| 6×12 РПЦ/АПЦ | **YES** | Top-left or center-left |
| 6×12 BOGOF, threshold | **YES** | Top-left |
| А5 (any mechanic) | **NO** | — Don't expect to fill the 11 QR fields |
| А4 vertical / horizontal | **YES** | Bottom-right, small |
| А3 vertical / horizontal | **YES** | Bottom-right, small |
| А2 vertical / horizontal | **YES** | Bottom-right, small |
| МНЦ (any mechanic, any color) | **NO** | — One of the most QR-deprived formats |

If your detector classifies the tag as А5 or МНЦ, you can skip QR decoding entirely and write `"нет"` for all 11 QR fields.

---

## 12. Visual cheat sheet — how to recognize the mechanic at a glance

| Visual feature | Mechanic |
|---|---|
| No discount marker at all, single set of prices | **РПЦ** (regular) |
| Black filled **circle** with `-N%` on the left of compact tag | **АПЦ** compact (6×6) |
| Black **half-circle (dome)** on top with `-N%` or `-N₽` | **АПЦ** A4/A3/A2 vertical |
| Black half-circle in **top-right corner** | **АПЦ** horizontal (A4/A3/A2) |
| Black half-circle in **top-left**, large tag, no QR | **АПЦ** А5 |
| Rounded rectangle on right with `-N%` | **АПЦ** 6×12 |
| **Red arrow** + diagonal strikethrough on old price | **МНЦ АПЦ** |
| **Yellow dome** + big word "АКЦИЯ" | **Clearance** ("Распродажа") A4 |
| **Two side-by-side panels** with dotted separator + "от N кг" wording | **Скидка ОТ N** (compact) |
| Three stacked prices, rectangular label "**По карте от N** кг/шт" | **Скидка ОТ N** (large formats) |
| "**Не более N кг**" + "**за каждый следующий**" wording (compact) | **Скидка ДО N шт** (compact) |
| Three stacked prices, rectangular label "**По карте до N** кг/шт" | **Скидка ДО N** (large formats) |
| Black half-circle with text "**При покупке от N шт**" (no %) | **ZB21** threshold variant |
| Bordered rectangle "**АКЦИЯ! 3=2**" on the left (compact) | **BOGOF** 6×6 |
| Rounded rectangle "**3=2**" in top-right (compact) | **BOGOF** 6×12 |
| Three stacked prices, rectangular label "**По карте при покупке N** кг/шт" | **BOGOF** large formats (A4/A3/A2) |
| Italic line "**Цена по акции действительна при покупке N шт/кг**" | **BOGOF** A5 (no rectangular marker) |
| Black half-dome with text "**при покупке от 3 шт**" + promo price can be > regular | **ZB21 BOGOF** A4 horizontal |
| **Two-line "название в строке 1/2"** layout | **МНЦ набор по товарам** (set discount) |
| Main tag + separate **right-side panel** with "номер на весах N" | Any mechanic **with shelf-talker** |

---

## 13. Open questions and edge cases (unresolved)

These details aren't fully spelled out in the decks and may need to be resolved empirically against the test video or by asking organizers:

- **`code` on large formats:** the display zone code (`06_062 003`) is verified on compact 6×6 / 6×12 tags. Its position on А3/А2 isn't explicitly called out — likely small-text bottom region, but verify.
- **Clearance A4 `color`:** the body of clearance tags appears white with a yellow dome. Whether `color = "жёлтый"` or `"белый"` for these depends on system attribution. Default to **"белый"** unless the whole body is yellow.
- **"Удачная упаковка"** shelf-talker text — needs an example from real data to know exact wording and how to fold into `additional_info`.
- **`Δ` triangle marker** on slide 31 (clearance kids' tag) near `id_sku` — small downward-pointing triangle. Purpose unclear; possibly distinguishes kids' age-restricted promos. Capture it as part of `additional_info` if visible.
- **МНЦ "набор по товарам"** (set discount, slide 30) is the only template with **two product names** on one tag. The CSV format expects one `product_name` per row — pragmatic approach: concatenate as `"Product A + Product B"` or pick the first name.
- **BOGOF promo > regular price** edge case (§7.5) — pricing isn't always monotonic. Don't reject `price_discount > price_card`.
- **Multiple "additional_info" sources on a single tag** — e.g. wine ценник with "Сухое" + "Акция действует с/по" + "номер на весах" all could co-exist. Concatenate with a separator (`;` or `|`) and write to one CSV field.

---

## 14. Sources

- `Расшифровка_ценники.pdf` — **7 slides, all covered.** The canonical reference with field callouts.
- `ГМ_для_ТК.pdf` — **66 slides, all covered.** Full template catalog organized as mechanic × size.
- The original files are actually **PowerPoint archives (`.pptx`) renamed to `.pdf`** — internally they're zip archives with JPEG renders of each slide plus a `manifest.json`. Extracting and reading the JPEGs directly was used to compile this guide.
