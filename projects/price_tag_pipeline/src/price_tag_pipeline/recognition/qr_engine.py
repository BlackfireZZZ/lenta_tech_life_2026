"""Robust, ML-free QR + 1D-barcode decoding for Lenta price-tag crops.

Why this module exists
----------------------
The barcode is the **primary GT-matching key** (see docs/index.md): a row with
no/incorrect barcode barely matches, so the rest of the recognised fields are
wasted. Reading *something* — the QR payload or the printed 1D barcode — off a
small, blurry, skewed, glare-hit shelf crop is therefore P0.

Strategy: an **ensemble of independent decoders** run through an **escalating
classical-CV preprocessing cascade**. Each tier is more expensive than the last;
we stop as soon as the caller's goal is met (default: any symbol decoded). No
training and no heavy models — every backend is an off-the-shelf C/C++ decoder
or a pure-OpenCV operation, so a crop that decodes early costs ~1 ms.

Decoders (all optional, lazily imported, soft-failing):
  * zxing-cpp        — fast, strong on QR + EAN/UPC/Code128, has its own
                        rotate/downscale/harder modes.
  * pyzbar (ZBar)    — the reference 1D EAN decoder; also QR.
  * OpenCV barcode   — `cv2.barcode.BarcodeDetector`, classical 1D detect+decode.
  * OpenCV QR        — `cv2.QRCodeDetector` detect/decode (+ multi).
  * OpenCV WeChat QR — `cv2.wechat_qrcode.WeChatQRCode`; the strongest open QR
                        detector on degraded images. Ships a ~1 MB bundled model
                        and runs in milliseconds on CPU (not trained here, not a
                        bottleneck). Toggle with `use_wechat=False` if undesired.

The cascade (classical CV only): grayscale, integral upscaling, CLAHE, unsharp
mask, bilateral denoise, Otsu / adaptive / inverted binarisation, morphological
bar-bridging, gamma, fixed + small-angle rotation, and gradient-based barcode /
detector-based QR **region localisation** with perspective rectification — the
single biggest lever on wide shelf photos where the code is tiny.

Public entry point: :func:`decode_image`.
"""

from __future__ import annotations

import contextlib
import logging
import os
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np


@contextlib.contextmanager
def _suppress_native_stderr():
    """Silence C-level stderr (fd 2) for the duration of the block.

    ZBar writes decoder assertions ("pdf417.c:89 ... Assertion failed") straight
    to fd 2 on noisy input — Python-level redirection can't catch them. We dup
    the fd over os.devnull and restore it after. Best-effort: if the platform
    refuses the dup we just run without suppression.
    """
    try:
        saved = os.dup(2)
    except OSError:
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)

LOGGER = logging.getLogger(__name__)

# Ground truth from real Lenta tags (see scripts/eval_qr.py output): the
# "barcode" is **GS1 DataBar** (Stacked/Expanded, carrying a `(01)` GTIN — the
# P0 matching key), NOT EAN-13; the small square is **DataMatrix**, NOT a QR.
# So the meaningful split is linear/stacked product barcode ("1d") vs. 2D
# matrix code ("2d"). EAN/UPC still appear on some product packaging.
_1D_FORMATS = {
    "EAN13", "EAN8", "UPCA", "UPCE",
    "CODE128", "CODE39", "CODE93", "ITF", "I25", "CODABAR",
    "DATABAR", "DATABARSTACKED", "DATABAREXPANDED",
    "DATABAREXPANDEDSTACKED", "DATABARLIMITED", "RSS14", "RSSEXPANDED",
}
_2D_FORMATS = {
    "QRCODE", "MICROQR", "RMQR", "DATAMATRIX", "AZTEC", "PDF417", "MAXICODE",
}


@dataclass
class DecodedSymbol:
    """One decoded barcode/QR symbol plus provenance for debugging/voting."""

    text: str
    kind: str                       # '1d' (product barcode) | '2d' (matrix)
    fmt: str                        # normalised symbology, e.g. 'DATABAR', 'DATAMATRIX'
    decoder: str                    # which backend produced it
    stage: str = "raw"              # which preprocessing stage decoded it
    confidence: float = 0.9
    checksum_ok: Optional[bool] = None   # GS1 mod-10 check-digit validation
    gtin: Optional[str] = None      # normalised GTIN for 1d (GS1 AI-01 stripped)

    def key(self) -> tuple[str, str]:
        return (self.kind, self.text.strip())


# ---------------------------------------------------------------------------
# Symbology / checksum helpers
# ---------------------------------------------------------------------------

def _norm_fmt(raw: object) -> str:
    s = str(raw).upper().split(".")[-1].strip()
    return s.replace("-", "_").replace(" ", "_")


def _kind_for(fmt: str) -> str:
    """'1d' = linear/stacked product barcode (the GT key); '2d' = matrix code."""
    f = fmt.replace("_", "")
    if f in _1D_FORMATS or f.startswith(("EAN", "UPC", "DATABAR", "RSS")):
        return "1d"
    if f in _2D_FORMATS:
        return "2d"
    return "2d"


def gs1_extract_gtin(text: str) -> Optional[str]:
    """Pull the GTIN from a GS1 element string, e.g. ``(01)04147130070834``.

    Lenta's DataBar carries the product GTIN under Application Identifier 01.
    zxing emits it parenthesised; ZBar may emit it bare with a leading ']e0' or
    '01' prefix. Returns the GTIN digits (14, or 13/8 if that's all there is).
    """
    import re as _re
    m = _re.search(r"\(01\)(\d{8,14})", text)
    if m:
        return m.group(1)
    # Bare GS1: starts with AI '01' then 14 digits (DataBar omni/stacked).
    m = _re.match(r"\s*01(\d{14})", text)
    if m:
        return m.group(1)
    return None


def ean_checksum_ok(digits: str) -> Optional[bool]:
    """Validate a GS1 mod-10 check digit (EAN-8/UPC-A/EAN-13/GTIN-14).

    The algorithm is length-agnostic: weight the body right-to-left by
    3,1,3,1,…; the check digit makes the weighted sum a multiple of 10.
    Returns None when the string isn't a plausible GTIN.
    """
    if not digits.isdigit() or len(digits) not in (8, 12, 13, 14):
        return None
    body, check = digits[:-1], int(digits[-1])
    total = 0
    for i, ch in enumerate(reversed(body)):
        total += int(ch) * (3 if i % 2 == 0 else 1)
    return (10 - total % 10) % 10 == check


# ---------------------------------------------------------------------------
# Decoder backends — each: ndarray -> list[DecodedSymbol]; never raises.
# ---------------------------------------------------------------------------

class _Backends:
    """Lazily-probed decoder backends. Construction never fails."""

    def __init__(self, enabled: tuple[str, ...] = ("zxing", "pyzbar")) -> None:
        self.enabled = set(enabled)
        self.use_wechat = "wechat" in self.enabled
        self._zxing = None
        self._zx_fmts = None
        self._zx_fmts_done = False
        self._pyzbar = None
        self._zbar_allowed = None
        self._cv2 = None
        self._cv_qr = None
        self._cv_bar = None
        self._wechat = "unset"

    # -- cv2 + detectors are cached singletons (constructing them is not free) --
    @property
    def cv2(self):
        if self._cv2 is None:
            try:
                import cv2  # type: ignore
                self._cv2 = cv2
            except Exception:  # pragma: no cover
                self._cv2 = False
        return self._cv2 or None

    @property
    def cv_qr(self):
        cv2 = self.cv2
        if cv2 is None:
            return None
        if self._cv_qr is None:
            try:
                self._cv_qr = cv2.QRCodeDetector()
            except Exception:
                self._cv_qr = False
        return self._cv_qr or None

    @property
    def cv_bar(self):
        cv2 = self.cv2
        if cv2 is None:
            return None
        if self._cv_bar is None:
            try:
                self._cv_bar = cv2.barcode.BarcodeDetector()
            except Exception:
                self._cv_bar = False
        return self._cv_bar or None

    @property
    def wechat(self):
        if not self.use_wechat:
            return None
        cv2 = self.cv2
        if cv2 is None:
            return None
        if self._wechat == "unset":
            try:
                self._wechat = cv2.wechat_qrcode.WeChatQRCode()
            except Exception:
                LOGGER.debug("WeChatQRCode unavailable", exc_info=True)
                self._wechat = None
        return self._wechat or None

    @property
    def zxing(self):
        if self._zxing is None:
            try:
                import zxingcpp  # type: ignore
                self._zxing = zxingcpp
            except Exception:
                self._zxing = False
        return self._zxing or None

    @property
    def pyzbar(self):
        if self._pyzbar is None:
            try:
                from pyzbar import pyzbar  # type: ignore
                from pyzbar.pyzbar import ZBarSymbol  # type: ignore
                self._pyzbar = pyzbar
                # Restrict ZBar to the symbologies a Lenta tag actually
                # carries. Eval proved I25/CODABAR/CODE39 only ever produced
                # short spurious reads off shelf-rail texture (e.g. I25
                # "076765") — and a *wrong* barcode mis-keys the GT match, so
                # it is worse than none. Keep QR + EAN/UPC + Code128 + DataBar
                # (the real GTIN carrier); drop the rest. PDF417 stays off too
                # (never present; its native decoder is the fd-2 spammer).
                wanted = ("QRCODE", "EAN13", "EAN8", "UPCA", "UPCE",
                          "CODE128", "DATABAR", "DATABAR_EXP")
                self._zbar_allowed = [getattr(ZBarSymbol, n)
                                      for n in wanted
                                      if hasattr(ZBarSymbol, n)] or None
            except Exception:
                self._pyzbar = False
        return self._pyzbar or None

    # ------------------------------------------------------------------ #
    def available(self) -> list[str]:
        """Enabled backends that actually imported/constructed OK."""
        names = []
        if "zxing" in self.enabled and self.zxing:
            names.append("zxingcpp")
        if "pyzbar" in self.enabled and self.pyzbar:
            names.append("pyzbar")
        if "cv2bar" in self.enabled and self.cv_bar:
            names.append("cv2.barcode")
        if "cv2qr" in self.enabled and self.cv_qr:
            names.append("cv2.qr")
        if "wechat" in self.enabled and self.wechat:
            names.append("cv2.wechat")
        return names

    # ------------------------------------------------------------------ #
    def _zxing_formats(self, zx):
        """Restricted format mask: only symbologies real Lenta tags carry.

        Letting zxing try everything yielded short spurious ITF/Codabar reads
        on shelf-rail texture. A wrong barcode mis-keys the GT match, so this
        precision guard matters. Falls back to Any if the enum is unexpected.
        """
        if self._zx_fmts_done:
            return self._zx_fmts
        self._zx_fmts_done = True
        bf = getattr(zx, "BarcodeFormat", None)
        if bf is None:
            return None
        wanted = ("QRCode", "MicroQRCode", "DataMatrix", "EAN13", "EAN8",
                  "UPCA", "UPCE", "Code128", "DataBar", "DataBarExpanded",
                  "DataBarLimited")
        mask = None
        for name in wanted:
            m = getattr(bf, name, None)
            if m is None:
                continue
            mask = m if mask is None else (mask | m)
        if mask is None:
            mask = getattr(bf, "Any", None)
        self._zx_fmts = mask
        return self._zx_fmts

    def decode_zxing(self, img: np.ndarray, stage: str) -> list[DecodedSymbol]:
        zx = self.zxing
        if zx is None:
            return []
        reader = getattr(zx, "read_barcodes", None) or getattr(zx, "read_barcode", None)
        if reader is None:
            return []
        # Build only kwargs this zxing-cpp version actually accepts — the
        # Python binding's signature changed across 1.x/2.x.
        kw: dict = {}
        fmts = self._zxing_formats(zx)
        if fmts is not None:
            kw["formats"] = fmts
        for opt in ("try_rotate", "try_downscale", "try_invert"):
            kw[opt] = True
        results = None
        for attempt_kw in (kw, {"formats": kw.get("formats")} if "formats" in kw else {}, {}):
            try:
                results = reader(img, **{k: v for k, v in attempt_kw.items()
                                         if v is not None})
                break
            except TypeError:
                continue
            except Exception:
                LOGGER.debug("zxingcpp decode failed", exc_info=True)
                return []
        if results is None:
            return []
        if not isinstance(results, (list, tuple)):
            results = [results]
        out: list[DecodedSymbol] = []
        for r in results or []:
            text = (getattr(r, "text", "") or "").strip()
            if not text:
                continue
            fmt = _norm_fmt(getattr(r, "format", ""))
            out.append(self._mk(text, fmt, "zxingcpp", stage, 0.95))
        return out

    def decode_pyzbar(self, gray: np.ndarray, stage: str) -> list[DecodedSymbol]:
        pz = self.pyzbar
        if pz is None:
            return []
        try:
            with _suppress_native_stderr():
                symbols = pz.decode(gray, symbols=self._zbar_allowed)
        except Exception:
            LOGGER.debug("pyzbar decode failed", exc_info=True)
            return []
        out: list[DecodedSymbol] = []
        for s in symbols:
            try:
                text = s.data.decode("utf-8", errors="ignore").strip()
            except Exception:
                continue
            if not text:
                continue
            out.append(self._mk(text, _norm_fmt(s.type), "pyzbar", stage, 0.92))
        return out

    def decode_cv_barcode(self, img: np.ndarray, stage: str) -> list[DecodedSymbol]:
        bd = self.cv_bar
        if bd is None:
            return []
        try:
            res = bd.detectAndDecode(img)
        except Exception:
            LOGGER.debug("cv2.barcode decode failed", exc_info=True)
            return []
        # API varies: (ok, infos, types, pts) on most builds, (infos, types,
        # pts) on others.
        if len(res) == 4:
            ok, infos, types, _ = res
            if not ok:
                return []
        elif len(res) == 3:
            infos, types, _ = res
        else:
            return []
        if infos is None:
            return []
        out: list[DecodedSymbol] = []
        for text, t in zip(infos, types or [""] * len(infos)):
            text = (text or "").strip()
            if text:
                out.append(self._mk(text, _norm_fmt(t) or "EAN13",
                                    "cv2.barcode", stage, 0.85))
        return out

    def decode_cv_qr(self, img: np.ndarray, stage: str) -> list[DecodedSymbol]:
        qd = self.cv_qr
        if qd is None:
            return []
        out: list[DecodedSymbol] = []
        try:
            ok, decoded, _, _ = qd.detectAndDecodeMulti(img)
            if ok:
                out += [self._mk(s.strip(), "QRCODE", "cv2.qr", stage, 0.85)
                        for s in decoded if s and s.strip()]
        except Exception:
            LOGGER.debug("cv2.qr multi failed", exc_info=True)
        if not out:
            try:
                s, _, _ = qd.detectAndDecode(img)
                if s and s.strip():
                    out.append(self._mk(s.strip(), "QRCODE", "cv2.qr", stage, 0.85))
            except Exception:
                LOGGER.debug("cv2.qr single failed", exc_info=True)
        return out

    def decode_wechat(self, img: np.ndarray, stage: str) -> list[DecodedSymbol]:
        det = self.wechat
        if det is None:
            return []
        try:
            texts, _ = det.detectAndDecode(img)
        except Exception:
            LOGGER.debug("wechat decode failed", exc_info=True)
            return []
        return [self._mk(t.strip(), "QRCODE", "cv2.wechat", stage, 0.9)
                for t in (texts or []) if t and t.strip()]

    def _mk(self, text: str, fmt: str, decoder: str, stage: str,
            conf: float) -> DecodedSymbol:
        kind = _kind_for(fmt)
        gtin = chk = None
        if kind == "1d":
            # GS1 DataBar emits "(01)<gtin>"; plain EAN/UPC emit bare digits.
            gtin = gs1_extract_gtin(text) or (text if text.isdigit() else None)
            if gtin:
                chk = ean_checksum_ok(gtin)
            # A barcode that fails its own check digit is almost certainly
            # garbage — a wrong barcode is worse than none (it mis-keys the
            # GT match), so down-weight it hard.
            if chk is False:
                conf *= 0.35
        return DecodedSymbol(text=text, kind=kind, fmt=fmt or kind.upper(),
                             decoder=decoder, stage=stage, confidence=conf,
                             checksum_ok=chk, gtin=gtin)


# ---------------------------------------------------------------------------
# Preprocessing primitives (classical CV only — no learned models)
# ---------------------------------------------------------------------------

def _to_gray(cv2, img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _to_bgr(cv2, img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return img
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _upscale(cv2, gray: np.ndarray, factor: float) -> np.ndarray:
    h, w = gray.shape[:2]
    interp = cv2.INTER_CUBIC if factor > 1 else cv2.INTER_AREA
    return cv2.resize(gray, (max(1, int(w * factor)), max(1, int(h * factor))),
                      interpolation=interp)


def _fit_long_side(cv2, gray: np.ndarray, target: int) -> np.ndarray:
    """Scale so the long side ≈ target (both up- and down-scaling help)."""
    h, w = gray.shape[:2]
    long = max(h, w)
    if long == 0:
        return gray
    return _upscale(cv2, gray, target / long)


def _clahe(cv2, gray: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)


def _unsharp(cv2, gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    return cv2.addWeighted(gray, 1.6, blur, -0.6, 0)


def _bilateral(cv2, gray: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(gray, 7, 60, 60)


def _otsu(cv2, gray: np.ndarray, invert: bool = False) -> np.ndarray:
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, th = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)
    return th


def _adaptive(cv2, gray: np.ndarray, block: int = 31) -> np.ndarray:
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block, 5)


def _gamma(cv2, gray: np.ndarray, g: float) -> np.ndarray:
    lut = np.array([((i / 255.0) ** (1.0 / g)) * 255 for i in range(256)],
                   dtype=np.uint8)
    return cv2.LUT(gray, lut)


def _close(cv2, binimg: np.ndarray, kx: int, ky: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    return cv2.morphologyEx(binimg, cv2.MORPH_CLOSE, k)


def _rotate(cv2, img: np.ndarray, angle: float) -> np.ndarray:
    if angle in (90, 180, 270):
        code = {90: cv2.ROTATE_90_CLOCKWISE,
                180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE}[angle]
        return cv2.rotate(img, code)
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    border = cv2.BORDER_REPLICATE
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_CUBIC,
                          borderMode=border)


# ---------------------------------------------------------------------------
# Region localisation — find the code, decode it big & upright
# ---------------------------------------------------------------------------

def _order_quad(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32).reshape(-1, 2)
    s = pts.sum(1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def _warp_quad(cv2, img: np.ndarray, quad: np.ndarray,
               pad: float = 0.12) -> Optional[np.ndarray]:
    q = _order_quad(quad)
    (tl, tr, br, bl) = q
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if w < 12 or h < 12:
        return None
    px, py = int(w * pad), int(h * pad)
    W, H = w + 2 * px, h + 2 * py
    dst = np.array([[px, py], [px + w, py], [px + w, py + h], [px, py + h]],
                   dtype=np.float32)
    m = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(img, m, (W, H), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def find_barcode_regions(cv2, gray: np.ndarray,
                         max_regions: int = 6) -> list[np.ndarray]:
    """Classic gradient-based 1D-barcode localiser → upright crops.

    A barcode is a region of strong horizontal gradient and weak vertical
    gradient. Scharr-x minus Scharr-y, blur, threshold, close into a blob,
    take the largest rectangular contours, deskew via minAreaRect.
    """
    h, w = gray.shape[:2]
    scale = 900.0 / max(h, w) if max(h, w) > 900 else 1.0
    g = _upscale(cv2, gray, scale) if scale != 1.0 else gray
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=-1)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=-1)
    grad = cv2.convertScaleAbs(cv2.subtract(np.abs(gx), np.abs(gy)))
    grad = cv2.blur(grad, (9, 9))
    _, th = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = _close(cv2, th, 21, 7)
    th = cv2.erode(th, None, iterations=4)
    th = cv2.dilate(th, None, iterations=4)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:max_regions * 2]
    crops: list[np.ndarray] = []
    for c in cnts:
        if cv2.contourArea(c) < (g.shape[0] * g.shape[1]) * 0.0015:
            continue
        rect = cv2.minAreaRect(c)
        (rw, rh) = rect[1]
        if rw < 16 or rh < 16:
            continue
        ar = max(rw, rh) / max(1.0, min(rw, rh))
        if ar < 1.4 or ar > 14:        # barcodes are wide strips, not blobs
            continue
        box = cv2.boxPoints(rect) / scale
        warped = _warp_quad(cv2, gray, box, pad=0.18)
        if warped is not None:
            if warped.shape[0] > warped.shape[1]:      # make bars vertical
                warped = _rotate(cv2, warped, 90)
            crops.append(warped)
        if len(crops) >= max_regions:
            break
    return crops


def find_qr_regions(backends: "_Backends", gray: np.ndarray,
                    bgr: np.ndarray, max_regions: int = 6) -> list[np.ndarray]:
    """Locate QR finder-pattern regions and return perspective-rectified crops.

    Uses the detectors' *detect-only* path (works even when decode fails),
    plus a contour fallback for square modular blobs.
    """
    cv2 = backends.cv2
    crops: list[np.ndarray] = []
    quads: list[np.ndarray] = []

    qd = backends.cv_qr
    if qd is not None:
        try:
            ok, pts = qd.detectMulti(gray)
            if ok and pts is not None:
                quads += [p.reshape(-1, 2) for p in pts]
        except Exception:
            LOGGER.debug("cv2.qr detectMulti failed", exc_info=True)

    det = backends.wechat
    if det is not None:
        try:
            _, points = det.detectAndDecode(bgr)
            quads += [np.asarray(p).reshape(-1, 2) for p in (points or [])]
        except Exception:
            LOGGER.debug("wechat detect failed", exc_info=True)

    # Contour fallback: near-square high-frequency blobs after adaptive thresh.
    try:
        th = _adaptive(cv2, gray, 31)
        th = _close(cv2, 255 - th, 9, 9)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        area_img = gray.shape[0] * gray.shape[1]
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:30]:
            a = cv2.contourArea(c)
            if a < area_img * 0.0008 or a > area_img * 0.5:
                continue
            x, y, ww, hh = cv2.boundingRect(c)
            if 0.55 <= ww / max(1, hh) <= 1.8:
                quads.append(np.array([[x, y], [x + ww, y],
                                       [x + ww, y + hh], [x, y + hh]],
                                      dtype=np.float32))
    except Exception:
        LOGGER.debug("qr contour fallback failed", exc_info=True)

    for q in quads[: max_regions * 2]:
        if q is None or len(q) < 4:
            continue
        warped = _warp_quad(cv2, bgr, np.asarray(q, dtype=np.float32), pad=0.25)
        if warped is not None:
            crops.append(warped)
        if len(crops) >= max_regions:
            break
    return crops


# Doc-informed spatial prior. From docs/hackathon/price-tag-guide.md:
#   * 6×6 (most common): DataMatrix top-right, barcode bottom-right
#   * 6×12: code top-left
#   * А4/А3/А2: small code bottom-right
#   * slide layouts: code top-right, barcode bottom-right, sometimes left
# Decoding these known sub-windows (upscaled) instead of blindly scanning the
# whole tag is both faster and more accurate — the code fills the ROI and we
# don't pick up neighbouring product barcodes. Windows are generous and
# overlapping so a slightly-off (non-final) YOLO box still contains the code.
_LAYOUT_ROIS: tuple[tuple[str, float, float, float, float], ...] = (
    ("TR",   0.42, 0.00, 1.00, 0.58),   # DataMatrix: 6×6 / slides
    ("BR",   0.42, 0.42, 1.00, 1.00),   # barcode (slide1/6×6) / QR (A4)
    ("TL",   0.00, 0.00, 0.58, 0.58),   # code: 6×12
    ("BL",   0.00, 0.42, 0.58, 1.00),   # barcode rotated (slide5) / id_sku
    ("Rcol", 0.55, 0.00, 1.00, 1.00),   # right column (DataMatrix + price)
    ("Brow", 0.00, 0.52, 1.00, 1.00),   # bottom band (barcode line)
    ("Lcol", 0.00, 0.00, 0.45, 1.00),   # left column (slide5 barcode on left)
)


def layout_rois(gray: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Canonical code sub-windows of a *tag crop* per the price-tag guide."""
    h, w = gray.shape[:2]
    out: list[tuple[str, np.ndarray]] = []
    for name, fx0, fy0, fx1, fy1 in _LAYOUT_ROIS:
        x0, y0 = int(w * fx0), int(h * fy0)
        x1, y1 = int(w * fx1), int(h * fy1)
        if x1 - x0 >= 12 and y1 - y0 >= 12:
            out.append((name, gray[y0:y1, x0:x1]))
    return out


# ---------------------------------------------------------------------------
# Orchestration: the escalating cascade
# ---------------------------------------------------------------------------

@dataclass
class CascadeConfig:
    # Only zxing-cpp and ZBar can read Lenta's actual symbologies (DataMatrix +
    # GS1 DataBar); cv2.qr / WeChat / cv2.barcode cannot and were pure overhead
    # on every variant — off by default, opt in for non-Lenta inputs.
    decoders: tuple[str, ...] = ("zxing", "pyzbar")
    use_wechat: bool = False        # back-compat shim; adds 'wechat' if True
    # Stop policy (ignored when `exhaustive`):
    #   "any"  -> stop at the first decoded symbol (fastest).
    #   "both" -> keep escalating until both the 1D barcode (GTIN matching key)
    #             and the 2D DataMatrix are read (max coverage; slower).
    stop_when: str = "any"
    exhaustive: bool = False        # run every tier + FULL transform set (eval)
    max_side: int = 1600            # downscale huge inputs before tier-0
    enable_regions: bool = True     # tier-2 localisation
    enable_rotations: bool = True   # tier-3 small-angle / 90s

    def resolved_decoders(self) -> tuple[str, ...]:
        d = list(self.decoders)
        if self.use_wechat and "wechat" not in d:
            d.append("wechat")
        return tuple(d)


# (name, fn) transforms operating on a grayscale image; decoders that take BGR
# get a 3-channel view of the same data.
def _build_transforms(cv2, full: bool = False
                       ) -> list[tuple[str, Callable[[np.ndarray], np.ndarray]]]:
    """Preprocessing variants. The lean set (default) is the six that actually
    cracked real Lenta tags in eval — small-code upscaling, local-contrast +
    sharpen, adaptive/Otsu binarisation, and bar-gap bridging for DataBar. The
    `full` set adds redundant variants and is only for the exhaustive ceiling
    measurement (`scripts/eval_qr.py` without --fast).
    """
    lean: list[tuple[str, Callable[[np.ndarray], np.ndarray]]] = [
        ("up2", lambda g: _upscale(cv2, g, 2.0)),
        ("up3", lambda g: _upscale(cv2, g, 3.0)),
        ("up2_clahe_sharp",
         lambda g: _unsharp(cv2, _clahe(cv2, _upscale(cv2, g, 2.0)))),
        ("up2_adaptive", lambda g: _adaptive(cv2, _upscale(cv2, g, 2.0), 41)),
        ("clahe_otsu", lambda g: _otsu(cv2, _clahe(cv2, g))),
        ("bars_close",
         lambda g: _close(cv2, _otsu(cv2, _upscale(cv2, g, 2.0)), 1, 5)),
    ]
    if not full:
        return lean
    extra: list[tuple[str, Callable[[np.ndarray], np.ndarray]]] = [
        ("gray", lambda g: g),
        ("fit1024", lambda g: _fit_long_side(cv2, g, 1024)),
        ("clahe", lambda g: _clahe(cv2, g)),
        ("up2_clahe", lambda g: _clahe(cv2, _upscale(cv2, g, 2.0))),
        ("up2_sharp", lambda g: _unsharp(cv2, _upscale(cv2, g, 2.0))),
        ("bilateral_clahe", lambda g: _clahe(cv2, _bilateral(cv2, g))),
        ("otsu", lambda g: _otsu(cv2, g)),
        ("otsu_inv", lambda g: _otsu(cv2, g, invert=True)),
        ("up2_otsu", lambda g: _otsu(cv2, _upscale(cv2, g, 2.0))),
        ("adaptive", lambda g: _adaptive(cv2, g, 31)),
        ("gamma_lo", lambda g: _gamma(cv2, g, 0.5)),
        ("gamma_hi", lambda g: _gamma(cv2, g, 1.8)),
    ]
    return lean + extra


class QRBarcodeEngine:
    """Stateful, reusable decoder (caches cv2 detectors). Thread-unsafe."""

    def __init__(self, config: Optional[CascadeConfig] = None) -> None:
        self.cfg = config or CascadeConfig()
        self.b = _Backends(enabled=self.cfg.resolved_decoders())

    # --- run every backend on one prepared variant ---------------------- #
    def _decode_variant(self, gray: np.ndarray, stage: str) -> list[DecodedSymbol]:
        cv2 = self.b.cv2
        if cv2 is None:
            return []
        en = self.b.enabled
        out: list[DecodedSymbol] = []
        if "zxing" in en:
            out += self.b.decode_zxing(gray, stage)
        if "pyzbar" in en:
            out += self.b.decode_pyzbar(gray, stage)
        if en & {"cv2bar", "cv2qr", "wechat"}:
            bgr = _to_bgr(cv2, gray)
            if "cv2bar" in en:
                out += self.b.decode_cv_barcode(bgr, stage)
            if "cv2qr" in en:
                out += self.b.decode_cv_qr(bgr, stage)
            if "wechat" in en:
                out += self.b.decode_wechat(bgr, stage)
        return out

    def _satisfied(self, found: dict[tuple[str, str], DecodedSymbol]) -> bool:
        if self.cfg.exhaustive or not found:
            return False
        if self.cfg.stop_when == "both":
            kinds = {k for k, _ in found}
            return {"1d", "2d"}.issubset(kinds)
        return True  # "any": first decoded symbol is enough

    def _ingest(self, found: dict, syms: Iterable[DecodedSymbol]) -> None:
        for s in syms:
            if not s.text:
                continue
            prev = found.get(s.key())
            if prev is None or s.confidence > prev.confidence:
                found[s.key()] = s

    def decode(self, image: np.ndarray) -> list[DecodedSymbol]:
        """Full escalating cascade. Returns de-duplicated symbols, best first."""
        cv2 = self.b.cv2
        if cv2 is None:
            LOGGER.warning("OpenCV unavailable — QR/barcode decoding disabled")
            return []
        if image is None or image.size == 0:
            return []

        found: dict[tuple[str, str], DecodedSymbol] = {}
        gray0 = _to_gray(cv2, image)
        if max(gray0.shape[:2]) > self.cfg.max_side:
            gray0 = _fit_long_side(cv2, gray0, self.cfg.max_side)

        full = self.cfg.exhaustive
        transforms = _build_transforms(cv2, full=full)

        # ---- Tier 0: whole crop, raw + a quick upscale --------------- #
        self._ingest(found, self._decode_variant(gray0, "tier0_raw"))
        if self._satisfied(found):
            return self._finalize(found)

        # ---- Tier L: doc-informed layout ROIs (the main fast path) --- #
        # The code is tiny inside a tag crop; the price-tag guide tells us
        # *where*. Decode each known corner/strip, upscaled, before falling
        # back to blind scanning of the whole crop.
        for roi_name, roi in layout_rois(gray0):
            self._ingest(found, self._decode_variant(roi, f"tierL_{roi_name}"))
            if self._satisfied(found):
                return self._finalize(found)
            for name, fn in transforms:
                try:
                    var = fn(roi)
                except Exception:
                    continue
                self._ingest(found,
                             self._decode_variant(var, f"tierL_{roi_name}_{name}"))
                if self._satisfied(found):
                    return self._finalize(found)

        # ---- Tier 1: whole crop through the preprocessing cascade ---- #
        for name, fn in transforms:
            try:
                var = fn(gray0)
            except Exception:
                LOGGER.debug("transform %s failed", name, exc_info=True)
                continue
            self._ingest(found, self._decode_variant(var, f"tier1_{name}"))
            if self._satisfied(found):
                return self._finalize(found)

        # ---- Tier 2: localise code regions, decode each big & upright - #
        if self.cfg.enable_regions:
            regions: list[tuple[str, np.ndarray]] = []
            try:
                regions += [("bar_region", r)
                            for r in find_barcode_regions(cv2, gray0)]
            except Exception:
                LOGGER.debug("barcode localisation failed", exc_info=True)
            try:
                bgr0 = _to_bgr(cv2, gray0)
                regions += [("qr_region", r)
                            for r in find_qr_regions(self.b, gray0, bgr0)]
            except Exception:
                LOGGER.debug("qr localisation failed", exc_info=True)

            for tag, reg in regions:
                rg = _to_gray(cv2, reg)
                self._ingest(found, self._decode_variant(rg, f"tier2_{tag}"))
                if self._satisfied(found):
                    return self._finalize(found)
                # A localised region is small — the cascade on it is cheap and
                # is exactly where degraded codes finally fall.
                for name, fn in transforms:
                    try:
                        var = fn(rg)
                    except Exception:
                        continue
                    self._ingest(found,
                                 self._decode_variant(var, f"tier2_{tag}_{name}"))
                    if self._satisfied(found):
                        return self._finalize(found)

        # ---- Tier 3: rotations / skew on the whole image ------------- #
        # zxing already tries 90° steps internally, so explicit rotation is
        # mainly about small skew. Keep it short unless measuring the ceiling.
        if self.cfg.enable_rotations:
            angles = ((90, 270, 180, 7, -7, 15, -15, 30, -30) if full
                      else (7, -7, 15, -15))
            for ang in angles:
                try:
                    rot = _rotate(cv2, gray0, ang)
                except Exception:
                    continue
                for name, fn in (("gray", lambda g: g),
                                 ("up2_clahe_sharp",
                                  lambda g: _unsharp(cv2, _clahe(
                                      cv2, _upscale(cv2, g, 2.0))))):
                    try:
                        var = fn(rot)
                    except Exception:
                        continue
                    self._ingest(found,
                                 self._decode_variant(var, f"tier3_rot{ang}_{name}"))
                    if self._satisfied(found):
                        return self._finalize(found)

        return self._finalize(found)

    @staticmethod
    def _finalize(found: dict) -> list[DecodedSymbol]:
        syms = list(found.values())
        # Prefer checksum-valid 1D, then confidence; QR keeps natural order.
        syms.sort(key=lambda s: (s.checksum_ok is True, s.confidence),
                  reverse=True)
        return syms


# Module-level convenience (a shared engine instance is fine for the pipeline,
# which calls this sequentially per crop).
_DEFAULT_ENGINE: Optional[QRBarcodeEngine] = None


def decode_image(image: np.ndarray,
                 config: Optional[CascadeConfig] = None) -> list[DecodedSymbol]:
    """Decode all QR/1D symbols in `image` (BGR or gray ndarray).

    With the default config the cascade stops at the first decoded symbol
    (production: fast). Pass ``CascadeConfig(exhaustive=True)`` to run every
    tier and surface everything (evaluation / diagnostics).
    """
    global _DEFAULT_ENGINE
    if config is not None:
        return QRBarcodeEngine(config).decode(image)
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = QRBarcodeEngine()
    return _DEFAULT_ENGINE.decode(image)


# ---------------------------------------------------------------------------
# Shared crop decode for the recognition chain.
#
# RecognitionChain runs QR then barcode on the *same* crop back-to-back. Both
# need the decoded symbols, but the cascade is the expensive part — so a shared
# engine with a 1-entry memo means a crop is decoded once and the second
# decoder reuses it. Keyed defensively (id can be recycled after GC, so shape /
# size / a strided checksum guard against a stale hit).
# ---------------------------------------------------------------------------

_CHAIN_ENGINE: Optional[QRBarcodeEngine] = None
_CROP_CACHE: tuple[object, list[DecodedSymbol]] = (None, [])


def _crop_key(img: np.ndarray):
    flat = img.reshape(-1)
    step = max(1, flat.size // 512)
    return (id(img), img.shape, int(img.nbytes),
            int(flat[::step].sum(dtype=np.int64)))


def decode_crop(crop_bgr: np.ndarray) -> list[DecodedSymbol]:
    """Decode one rectified crop for the recognition chain (memoised).

    Production cascade settings: escalate until *both* a 1D barcode and a 2D
    code are found (or the cascade is exhausted) — the chain wants the GTIN
    matching key *and* the QR fields, not just whichever decodes first.
    """
    global _CHAIN_ENGINE, _CROP_CACHE
    if crop_bgr is None or crop_bgr.size == 0:
        return []
    if _CHAIN_ENGINE is None:
        _CHAIN_ENGINE = QRBarcodeEngine(CascadeConfig(stop_when="both"))
    key = _crop_key(crop_bgr)
    if _CROP_CACHE[0] == key:
        return _CROP_CACHE[1]
    syms = _CHAIN_ENGINE.decode(crop_bgr)
    _CROP_CACHE = (key, syms)
    return syms
