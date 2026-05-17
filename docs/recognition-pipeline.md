# Recognition pipeline — the crop → fields chain

How one rectified price-tag crop becomes structured fields, and the **stable
seam** the separate QR/barcode branch plugs into. This is the single source of
truth for the recognition restructure (2026-05-17).

## Where it sits

```
detector ─▶ tracker ─▶ rectifier ─▶  recognition  ─▶ aggregator ─▶ CSV
                       (crop)     (QR→barcode→OCR)   (voting+dedup)
```

`detector` is the fixed base of the solution: the OpenFoodFacts price-tag
detector (YOLO11x), which we **fine-tune** — see
[`projects/price_tag_pipeline/experiments/`](../projects/price_tag_pipeline/experiments/README.md).
Everything from the crop onward is the **recognition** package
(`src/price_tag_pipeline/recognition/`).

## The chain

For each of the top-K-sharpest crops of a track the pipeline runs an ordered
chain of `CropDecoder`s:

1. **QR** (`recognition/qr.py` → `QRDecoder`) — decode QR payload to fields.
2. **barcode** (`recognition/barcode.py` → `BarcodeDecoder`) — 1D barcode /
   `ШК:`-text GTIN. **STUB today** (owned by a separate branch).
3. **smart OCR** (`recognition/ocr.py` → `OCRDecoder`) — classical/VLM OCR +
   parser; an ensemble engine yields several readings per crop.

Order is fixed and QR is **not** a short-circuit: every enabled decoder runs
on every crop. Each non-empty reading is emitted as one `TagObservation`.

### Merge policy: `qr_first_fill_gaps`

The agreed semantics ("QR-first, OCR fills the gaps") are realized by emitting
QR/barcode/OCR readings as **separate observations** and letting the
`TrackAggregator`'s existing per-field weighted voting reconcile them:

- QR fields carry confidence ≈ 0.97 → they win where QR speaks.
- OCR fields carry their parser confidence → they fill fields QR is silent on
  (often `product_name`, `weight`), and formats with no QR fall through to
  barcode + OCR entirely.

Field-level reconciliation is **deliberately not** re-implemented in the
chain. The chain owns *which decoders run and in what order*; the aggregator
owns *field-level merge*. Keeping that boundary is what lets decoder internals
change without touching voting. `merge_policy` in config is intent/forward
-compat today; changing voting weights is an aggregator change, out of scope
for the recognition seam.

## The seam (frozen interface)

`recognition/base.py`:

```python
class CropDecoder(ABC):
    name: str
    def decode(self, crop_bgr: np.ndarray) -> list[RecognitionResult]: ...

@dataclass(frozen=True)
class RecognitionResult:
    parsed: ParsedTag      # fields go in core attrs / extra_fields
    decoder: str           # "qr" | "barcode" | "<ocr backend>"
    confidence: float      # audit + (OCR) parser confidence
    text: str              # raw payload / OCR text, for audit
    found: bool            # did this decoder recognize anything?

def parsed_is_empty(parsed: ParsedTag) -> bool: ...   # the ONE emptiness rule
```

Decoder rules: cheap to construct; **never raise** on a bad crop (return
`[]`); return `[]` when nothing is recognized so the chain reads correctly and
no empty observation is emitted.

## Contract for the separate QR/barcode branch

The ultimate QR and barcode readers are being built on a **separate branch**.
They plug in here **without changing the seam or the chain order**:

- **QR** → replace the internals behind `recognition/qr.py::QRDecoder`
  (`QRCodeExtractor` / `parse_qr_payload` are the current baseline). Keep
  `QRDecoder.decode` returning `list[RecognitionResult]`.
- **barcode** → implement `recognition/barcode.py::BarcodeDecoder.decode`.
  On success, `parsed.extra_fields["barcode"] = "<digits>"` (also
  `"qr_code_barcode"` when it is the same GTIN), set `extra_confidences`,
  `found=True`. Barcode is the metric's **primary GT-matching key — P0**
  ([`index.md`](./index.md) "five facts"), not just one column.
- **Do NOT change** `CropDecoder` / `RecognitionResult` / `parsed_is_empty` /
  the QR→barcode→OCR order / the `recognition:` config keys.
- Merge is the aggregator's job — do not add field-merge logic to decoders.

Because this restructure moved `qr.py`/`ocr.py` into the package, the other
branch should land its work in `recognition/qr.py` / `recognition/barcode.py`
(not the old top-level `qr.py`).

## Configuration

Optional `recognition:` block (defaults keep the full chain on, so existing
profiles are unchanged):

```yaml
recognition:
  enable_qr: true
  enable_barcode: true       # currently a no-op stub; safe to leave on
  enable_ocr: true
  merge_policy: qr_first_fill_gaps
```

## Tests

`tests/test_recognition_chain.py` locks: chain order, no-short-circuit,
shared emptiness rule, and that `build_recognition_chain` honours the enable
flags / defaults. No GPU, models, or data. `tests/test_qr.py` keeps covering
QR payload parsing (now `recognition.qr`).
