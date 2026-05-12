"""OCR / VLM engines.

All engines implement `BaseOCREngine.recognize(image) -> OCRResult`.

For VLM engines the `OCRResult.text` is a JSON STRING (the model is prompted
to return structured JSON) and `structured == True`. The pipeline routes this
to `TagParser.parse_vlm_json`. Classical OCR engines emit free-form text and
route to `TagParser.parse_text`.

Backends:
- `paddle_vl`   — PaddleOCR-VL 1.5 (0.9B VLM). Primary path for 2026.
- `paddle`      — Classical PaddleOCR 3.x with Russian recognition model.
- `tesseract`   — Fallback, real per-word confidence via image_to_data.
- `noop`        — No-op for smoke tests and CI; never use in production.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from .config import OCRConfig
from .types import OCRResult

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VLM JSON schema prompt
# ---------------------------------------------------------------------------

DEFAULT_VLM_PROMPT = """You are reading a Russian supermarket price tag from the Lenta retail chain.
Extract structured fields and respond with ONLY a single JSON object — no commentary, no markdown fences.

Required JSON keys (use null if a field is not visible or you are uncertain):
{
  "regular_price": "number with two decimals (rubles.kopecks) or null",
  "loyalty_price": "number with two decimals (card / discount price) or null",
  "product_name": "product name in Russian, exactly as printed, or null",
  "weight_value": "number or null",
  "weight_unit": "one of: кг, г, л, мл, шт, or null",
  "price_per_unit_value": "number or null",
  "price_per_unit_unit": "string like 'руб/кг' or null",
  "promo_flag": "true if the tag visually marks a promotion (yellow/red highlight, words АКЦИЯ, СКИДКА, ПО КАРТЕ, -NN%), false otherwise",
  "currency": "RUB"
}

Disambiguation rules:
- The LARGER printed price is usually the loyalty/card price.
- A crossed-out, smaller, or grey price is the regular price.
- If only one price is shown and the tag has promo highlights, that price is the loyalty price.
- Numbers are roubles. Two-digit superscripts after a price are kopecks.

Do not invent. If unsure, output null for that field."""


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseOCREngine(ABC):
    structured: bool = False  # True for VLM engines that emit JSON

    @abstractmethod
    def recognize(self, image: np.ndarray) -> OCRResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# No-op (for tests / CI)
# ---------------------------------------------------------------------------

class NoOpOCREngine(BaseOCREngine):
    structured = False

    def recognize(self, image: np.ndarray) -> OCRResult:
        return OCRResult(text="", confidence=0.0, backend="noop")


# ---------------------------------------------------------------------------
# Tesseract (fallback)
# ---------------------------------------------------------------------------

class TesseractOCREngine(BaseOCREngine):
    structured = False

    def __init__(self, lang: str = "rus+eng"):
        import pytesseract  # noqa: F401
        self._pt = __import__("pytesseract")
        self.lang = lang

    def recognize(self, image: np.ndarray) -> OCRResult:
        # image_to_data gives per-word text + confidence.
        data = self._pt.image_to_data(
            image,
            lang=self.lang,
            config="--oem 1 --psm 6",
            output_type=self._pt.Output.DICT,
        )
        words: list[tuple[str, float]] = []
        for txt, conf in zip(data["text"], data["conf"]):
            if not txt or not txt.strip():
                continue
            try:
                c = float(conf) / 100.0
            except ValueError:
                continue
            if c < 0:
                continue
            words.append((txt.strip(), c))
        if not words:
            return OCRResult(text="", confidence=0.0, backend="tesseract", lines=())
        full = " ".join(w for w, _ in words)
        avg = sum(c for _, c in words) / len(words)
        return OCRResult(
            text=full,
            confidence=avg,
            backend="tesseract",
            lines=tuple(words),
        )


# ---------------------------------------------------------------------------
# Classical PaddleOCR (3.x API)
# ---------------------------------------------------------------------------

class PaddleOCREngine(BaseOCREngine):
    """Classical PaddleOCR 3.x with Russian recognition model.

    Note: PaddleOCR 3.x changed parameter names. We use the new API explicitly:
        PaddleOCR(use_textline_orientation=True, lang='ru')
    and call `.predict(image)` which returns a list of result objects with
    `.rec_texts` and `.rec_scores`. The legacy `.ocr(image, cls=True)` is gone.
    """

    structured = False

    def __init__(self, lang: str = "ru"):
        from paddleocr import PaddleOCR  # type: ignore
        self._engine = PaddleOCR(
            use_textline_orientation=True,
            lang=lang,
        )

    def recognize(self, image: np.ndarray) -> OCRResult:
        # PaddleOCR 3.x preferred API: .predict(input).
        result_list = self._engine.predict(image)
        if not result_list:
            return OCRResult(text="", confidence=0.0, backend="paddle", lines=())
        result = result_list[0]
        texts = list(getattr(result, "rec_texts", []) or [])
        scores = list(getattr(result, "rec_scores", []) or [])
        if not texts:
            return OCRResult(text="", confidence=0.0, backend="paddle", lines=())
        pairs = list(zip(texts, [float(s) for s in scores]))
        full = " ".join(texts)
        avg = sum(s for _, s in pairs) / len(pairs) if pairs else 0.0
        return OCRResult(
            text=full,
            confidence=avg,
            backend="paddle",
            lines=tuple(pairs),
        )


# ---------------------------------------------------------------------------
# PaddleOCR-VL 1.5 (primary VLM path)
# ---------------------------------------------------------------------------

class PaddleVLMEngine(BaseOCREngine):
    """PaddleOCR-VL 1.5 — 0.9B VLM with native Russian + structured JSON output.

    The engine prompts the model with a JSON-schema instruction and returns the
    raw JSON text. Downstream, `TagParser.parse_vlm_json` decodes the JSON and
    builds a `ParsedTag` directly.

    Loading happens lazily so importing this module does not pull in the model.
    """

    structured = True

    def __init__(
        self,
        model_name: str = "PaddlePaddle/PaddleOCR-VL",
        prompt: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.prompt = prompt or DEFAULT_VLM_PROMPT
        self.device = device
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Two possible install paths: paddleocr package (recommended) or
        # transformers (when supported). We prefer paddleocr.
        try:
            from paddleocr import PaddleOCRVL  # type: ignore
            self._impl = "paddleocr"
            self._model = PaddleOCRVL.from_pretrained(self.model_name)
        except ImportError:
            try:
                from transformers import AutoModelForVision2Seq, AutoProcessor  # type: ignore
                self._impl = "transformers"
                self._processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
                self._model = AutoModelForVision2Seq.from_pretrained(
                    self.model_name, trust_remote_code=True
                )
                if self.device:
                    self._model = self._model.to(self.device)
            except ImportError as e:
                raise RuntimeError(
                    "PaddleOCR-VL requires either 'paddleocr>=3.0' (preferred) "
                    "or 'transformers' to be installed. Install with:\n"
                    "    pip install paddleocr>=3.0\n"
                    "or\n"
                    "    pip install transformers torch"
                ) from e

    def recognize(self, image: np.ndarray) -> OCRResult:
        self._ensure_loaded()
        if self._impl == "paddleocr":
            # PaddleOCRVL accepts a numpy image + a prompt string and returns JSON.
            out = self._model.predict(image, prompt=self.prompt)  # type: ignore[attr-defined]
            text = self._extract_text(out)
            conf = self._extract_conf(out)
            return OCRResult(text=text, confidence=conf, backend="paddle_vl")
        # transformers path
        from PIL import Image  # type: ignore
        import torch  # type: ignore

        rgb = image[:, :, ::-1] if image.ndim == 3 else image
        pil = Image.fromarray(rgb)
        inputs = self._processor(images=pil, text=self.prompt, return_tensors="pt")  # type: ignore
        if self.device:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self._model.generate(  # type: ignore[union-attr]
                **inputs,
                max_new_tokens=384,
                do_sample=False,
            )
        text = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0]  # type: ignore
        # Best-effort: strip the prompt back out if the model echoes it.
        if self.prompt and text.startswith(self.prompt):
            text = text[len(self.prompt):].strip()
        return OCRResult(text=text, confidence=0.85, backend="paddle_vl")

    @staticmethod
    def _extract_text(out) -> str:
        if isinstance(out, str):
            return out
        if isinstance(out, list) and out:
            first = out[0]
            if isinstance(first, str):
                return first
            if isinstance(first, dict):
                for k in ("text", "json", "output", "result"):
                    if k in first and isinstance(first[k], str):
                        return first[k]
        if isinstance(out, dict):
            for k in ("text", "json", "output", "result"):
                if k in out and isinstance(out[k], str):
                    return out[k]
        return str(out)

    @staticmethod
    def _extract_conf(out) -> float:
        if isinstance(out, dict) and "confidence" in out:
            try:
                return float(out["confidence"])
            except (TypeError, ValueError):
                pass
        if isinstance(out, list) and out and isinstance(out[0], dict):
            return float(out[0].get("confidence", 0.85))
        return 0.85


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_ocr_engine(cfg: OCRConfig) -> BaseOCREngine:
    key = cfg.backend.lower().strip()
    if key == "noop":
        return NoOpOCREngine()
    if key == "tesseract":
        return TesseractOCREngine()
    if key in {"paddle", "paddleocr"}:
        return PaddleOCREngine(lang=cfg.paddleocr_lang)
    if key in {"paddle_vl", "paddleocr_vl", "vlm"}:
        prompt = DEFAULT_VLM_PROMPT
        if cfg.vlm_prompt_path:
            p = Path(cfg.vlm_prompt_path)
            if p.exists():
                prompt = p.read_text(encoding="utf-8")
        return PaddleVLMEngine(model_name=cfg.vlm_model, prompt=prompt)
    raise ValueError(f"Unsupported OCR backend: {cfg.backend}")
