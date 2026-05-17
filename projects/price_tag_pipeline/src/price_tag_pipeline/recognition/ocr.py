"""OCR / VLM engines.

All engines implement `BaseOCREngine.recognize(image) -> OCRResult`.

For VLM engines the `OCRResult.text` is a JSON STRING (the model is prompted
to return structured JSON) and `structured == True`. The pipeline routes this
to `TagParser.parse_vlm_json`. Classical OCR engines emit free-form text and
route to `TagParser.parse_text`.

Available backends (May 2026):

  CLASSICAL
    noop          — No-op for smoke tests and CI; never use in production.
    tesseract     — Fallback. Real per-word confidence via image_to_data.
    paddle        — PaddleOCR 3.x classical with Russian recognition model.

  VLM (structured JSON output)
    paddle_vl     — PaddleOCR-VL 1.5 (0.9B). 94.50 on OmniDocBench v1.5.
    glm_ocr       — GLM-OCR (0.9B, Z.AI). 94.62 on OmniDocBench v1.5 — current #1.
    qwen3_vl      — Qwen3-VL 4B/8B/30B-A3B/235B-A22B. 201 languages, 256K ctx.
    dots_ocr      — rednote-hilab/dots.ocr (3B). 88.41 on OmniDocBench.
    hunyuan_ocr   — Tencent-Hunyuan/HunyuanOCR (1B). Multiple SOTA benchmarks.
    rolm_ocr      — reducto/RolmOCR (Qwen2.5-VL-7B fine-tune).
    intern_vl3    — OpenGVLab/InternVL3 family.
    monkey_ocr    — Yuliang-Liu/MonkeyOCR (3B). Beats GPT-4o on OmniDocBench.

  GENERIC
    transformers_vlm  — Configurable Hugging Face transformers VLM. Pass any
                        model_id via cfg.vlm_model.
    vllm_server       — OpenAI-compatible client; works with ANY model served
                        by vLLM/SGLang. Best for production: swap models via
                        config without code changes.

Sources verified May 2026:
  - OmniDocBench v1.5 leaderboard at codesota.com/ocr/benchmark/omnidocbench
  - GLM-OCR docs: https://docs.z.ai/guides/vlm/glm-ocr
  - PaddleOCR-VL paper: arXiv 2510.14528
  - Qwen3-VL: https://github.com/QwenLM/Qwen3-VL
  - dots.ocr: https://github.com/rednote-hilab/dots.ocr
  - MonkeyOCR: https://github.com/Yuliang-Liu/MonkeyOCR
  - HunyuanOCR: https://github.com/Tencent-Hunyuan/HunyuanOCR
  - RolmOCR: https://huggingface.co/reducto/RolmOCR
"""

from __future__ import annotations

import base64
import io
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..config import OCRConfig, ParserConfig
from ..parser import TagParser
from ..types import OCRResult
from .base import CropDecoder, RecognitionResult, parsed_is_empty

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default JSON-schema prompt for VLM engines
# ---------------------------------------------------------------------------

DEFAULT_VLM_PROMPT = """You are reading a Russian supermarket price tag from the Lenta retail chain.
Extract structured fields and respond with ONLY a single JSON object — no commentary, no markdown fences.

Required JSON keys (use null if a field is not visible or you are uncertain):
{
  "regular_price": "number with two decimals (rubles.kopecks) or null",
  "loyalty_price": "number with two decimals (card / discount price) or null",
  "product_name": "product name in Russian, exactly as printed, or null",
  "barcode": "visible EAN/GTIN barcode number or null",
  "price_discount": "discount amount in rubles or null",
  "discount_amount": "discount percent or amount as printed or null",
  "id_sku": "SKU/article id or null",
  "print_datetime": "price tag print date/time or null",
  "code": "shelf/zone code or null",
  "additional_info": "other printed info that may matter or null",
  "color": "dominant tag color, e.g. yellow/white/red/green, or null",
  "special_symbols": "layout/special marker type if visible, e.g. promo, card, wholesale, or null",
  "weight_value": "number or null",
  "weight_unit": "one of: кг, г, л, мл, шт, or null",
  "price_per_unit_value": "number or null",
  "price_per_unit_unit": "string like 'руб/кг' or null",
  "qr_code_barcode": "barcode from QR payload or null",
  "price1_qr": "QR price1/p1 or null",
  "price2_qr": "QR price2/p2 or null",
  "price3_qr": "QR price3/p3 or null",
  "price4_qr": "QR price4/p4 or null",
  "wholesale_level_1_count": "QR wholesaleLevel1Count/wL1C or null",
  "wholesale_level_1_price": "QR wholesaleLevel1Price/wL1P or null",
  "wholesale_level_2_count": "QR wholesaleLevel2Count/wL2C or null",
  "wholesale_level_2_price": "QR wholesaleLevel2Price/wL2P or null",
  "action_price_qr": "QR actionPrice/aP or null",
  "action_code_qr": "QR actionCode/aC or null",
  "promo_flag": "true if the tag visually marks a promotion (yellow/red highlight, words АКЦИЯ, СКИДКА, ПО КАРТЕ, -NN%), false otherwise",
  "currency": "RUB"
}

Disambiguation rules:
- The LARGER printed price is usually the loyalty/card price.
- A crossed-out, smaller, or grey price is the regular price.
- If only one price is shown and the tag has promo highlights, that price is the loyalty price.
- Numbers are roubles. Two-digit superscripts after a price are kopecks.

Do not invent. If unsure, output null for that field."""


# JSON Schema for guided decoding (vLLM / SGLang `guided_json` / Outlines).
PRICE_TAG_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "regular_price": {"type": ["number", "null"]},
        "loyalty_price": {"type": ["number", "null"]},
        "product_name": {"type": ["string", "null"]},
        "barcode": {"type": ["string", "null"]},
        "price_discount": {"type": ["number", "string", "null"]},
        "discount_amount": {"type": ["number", "string", "null"]},
        "id_sku": {"type": ["string", "null"]},
        "print_datetime": {"type": ["string", "null"]},
        "code": {"type": ["string", "null"]},
        "additional_info": {"type": ["string", "null"]},
        "color": {"type": ["string", "null"]},
        "special_symbols": {"type": ["string", "null"]},
        "weight_value": {"type": ["number", "null"]},
        "weight_unit": {"type": ["string", "null"], "enum": ["кг", "г", "л", "мл", "шт", None]},
        "price_per_unit_value": {"type": ["number", "null"]},
        "price_per_unit_unit": {"type": ["string", "null"]},
        "qr_code_barcode": {"type": ["string", "null"]},
        "price1_qr": {"type": ["number", "string", "null"]},
        "price2_qr": {"type": ["number", "string", "null"]},
        "price3_qr": {"type": ["number", "string", "null"]},
        "price4_qr": {"type": ["number", "string", "null"]},
        "wholesale_level_1_count": {"type": ["number", "string", "null"]},
        "wholesale_level_1_price": {"type": ["number", "string", "null"]},
        "wholesale_level_2_count": {"type": ["number", "string", "null"]},
        "wholesale_level_2_price": {"type": ["number", "string", "null"]},
        "action_price_qr": {"type": ["number", "string", "null"]},
        "action_code_qr": {"type": ["string", "null"]},
        "promo_flag": {"type": "boolean"},
        "currency": {"type": "string"},
    },
    "required": [
        "regular_price", "loyalty_price", "product_name",
        "weight_value", "weight_unit",
        "price_per_unit_value", "price_per_unit_unit",
        "promo_flag", "currency",
    ],
}


def build_few_shot_prompt(base_prompt: str, examples_path: Optional[str]) -> str:
    """Append example (description -> JSON) pairs to the prompt.

    Examples YAML format:
      - description: "Tag with regular + loyalty price, milk, 1 L"
        json:
          regular_price: 89.90
          loyalty_price: 69.90
          product_name: "Молоко Простоквашино 2.5% 1л"
          weight_value: 1.0
          weight_unit: "л"
          promo_flag: true
          currency: "RUB"
    """
    if not examples_path:
        return base_prompt
    p = Path(examples_path)
    if not p.exists():
        LOGGER.warning("few_shot_examples_path=%s not found; using base prompt", examples_path)
        return base_prompt
    try:
        import yaml  # type: ignore
    except ImportError:
        LOGGER.warning("pyyaml not available; skipping few-shot examples")
        return base_prompt

    data = yaml.safe_load(p.read_text(encoding="utf-8")) or []
    if not isinstance(data, list) or not data:
        return base_prompt

    lines = [base_prompt, "", "EXAMPLES (read carefully and follow the same JSON shape):"]
    for i, ex in enumerate(data, 1):
        desc = ex.get("description", f"Example {i}")
        body = ex.get("json", {})
        lines.append(f"\nExample {i} ({desc}):")
        lines.append(json.dumps(body, ensure_ascii=False, indent=2))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseOCREngine(ABC):
    """All engines implement recognize(image_bgr) -> OCRResult.

    Subclasses set `structured = True` if their output is JSON intended for
    `TagParser.parse_vlm_json`.

    Ensemble engines override `recognize_all` to return multiple results from
    a single crop; the default just wraps `recognize` in a single-item list.
    """

    structured: bool = False
    backend_name: str = "unknown"

    @abstractmethod
    def recognize(self, image: np.ndarray) -> OCRResult:
        raise NotImplementedError

    def recognize_all(self, image: np.ndarray) -> list[OCRResult]:
        return [self.recognize(image)]


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _bgr_to_pil(image: np.ndarray):
    """Convert OpenCV BGR ndarray to PIL RGB Image."""
    from PIL import Image  # local import — keeps top-level import lean
    if image.ndim == 3 and image.shape[2] == 3:
        rgb = image[:, :, ::-1]
    else:
        rgb = image
    return Image.fromarray(rgb)


def _image_to_base64_url(image: np.ndarray, fmt: str = "JPEG", quality: int = 95) -> str:
    """Encode an image as a data URL for OpenAI-compatible APIs."""
    from PIL import Image  # noqa: F401
    pil = _bgr_to_pil(image)
    buf = io.BytesIO()
    pil.save(buf, format=fmt, quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"


def _load_prompt(prompt: Optional[str], prompt_path: Optional[str]) -> str:
    if prompt_path:
        p = Path(prompt_path)
        if p.exists():
            return p.read_text(encoding="utf-8")
        LOGGER.warning("vlm_prompt_path=%s not found; using default prompt", prompt_path)
    return prompt or DEFAULT_VLM_PROMPT


# ---------------------------------------------------------------------------
# No-op + classical engines
# ---------------------------------------------------------------------------

class NoOpOCREngine(BaseOCREngine):
    structured = False
    backend_name = "noop"

    def recognize(self, image: np.ndarray) -> OCRResult:
        return OCRResult(text="", confidence=0.0, backend=self.backend_name)


class TesseractOCREngine(BaseOCREngine):
    structured = False
    backend_name = "tesseract"

    def __init__(self, lang: str = "rus+eng"):
        import pytesseract  # noqa: F401
        self._pt = __import__("pytesseract")
        self.lang = lang

    def recognize(self, image: np.ndarray) -> OCRResult:
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
            return OCRResult(text="", confidence=0.0, backend=self.backend_name, lines=())
        full = " ".join(w for w, _ in words)
        avg = sum(c for _, c in words) / len(words)
        return OCRResult(text=full, confidence=avg, backend=self.backend_name, lines=tuple(words))


class PaddleOCREngine(BaseOCREngine):
    """Classical PaddleOCR 3.x with the Russian recognition model."""

    structured = False
    backend_name = "paddle"

    def __init__(self, lang: str = "ru"):
        from paddleocr import PaddleOCR  # type: ignore
        self._engine = PaddleOCR(use_textline_orientation=True, lang=lang)

    def recognize(self, image: np.ndarray) -> OCRResult:
        result_list = self._engine.predict(image)
        if not result_list:
            return OCRResult(text="", confidence=0.0, backend=self.backend_name, lines=())
        result = result_list[0]
        texts = list(getattr(result, "rec_texts", []) or [])
        scores = list(getattr(result, "rec_scores", []) or [])
        if not texts:
            return OCRResult(text="", confidence=0.0, backend=self.backend_name, lines=())
        pairs = list(zip(texts, [float(s) for s in scores]))
        full = " ".join(texts)
        avg = sum(s for _, s in pairs) / len(pairs) if pairs else 0.0
        return OCRResult(text=full, confidence=avg, backend=self.backend_name, lines=tuple(pairs))


# ---------------------------------------------------------------------------
# PaddleOCR-VL 1.5 (kept separate — has its own PaddleOCRVL loader)
# ---------------------------------------------------------------------------

class PaddleVLMEngine(BaseOCREngine):
    """PaddleOCR-VL 1.5 — 0.9B VLM, 94.50 on OmniDocBench v1.5."""

    structured = True
    backend_name = "paddle_vl"

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
        self._impl: Optional[str] = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from paddleocr import PaddleOCRVL  # type: ignore
            self._impl = "paddleocr"
            self._model = PaddleOCRVL.from_pretrained(self.model_name)
        except ImportError:
            try:
                from transformers import AutoModelForVision2Seq, AutoProcessor  # type: ignore
                self._impl = "transformers"
                self._processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
                self._model = AutoModelForVision2Seq.from_pretrained(self.model_name, trust_remote_code=True)
                if self.device:
                    self._model = self._model.to(self.device)
            except ImportError as e:
                raise RuntimeError(
                    "PaddleOCR-VL requires either 'paddleocr>=3.0' or 'transformers'. "
                    "Install: pip install paddleocr>=3.0"
                ) from e

    def recognize(self, image: np.ndarray) -> OCRResult:
        self._ensure_loaded()
        if self._impl == "paddleocr":
            out = self._model.predict(image, prompt=self.prompt)  # type: ignore[attr-defined]
            return OCRResult(text=_extract_str(out), confidence=_extract_conf(out, 0.85),
                             backend=self.backend_name)
        # transformers path
        import torch  # type: ignore
        pil = _bgr_to_pil(image)
        inputs = self._processor(images=pil, text=self.prompt, return_tensors="pt")  # type: ignore
        if self.device:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=512, do_sample=False)  # type: ignore[union-attr]
        text = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0]  # type: ignore
        if self.prompt and text.startswith(self.prompt):
            text = text[len(self.prompt):].strip()
        return OCRResult(text=text, confidence=0.85, backend=self.backend_name)


def _extract_str(out: Any) -> str:
    if isinstance(out, str):
        return out
    if isinstance(out, list) and out:
        first = out[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            for k in ("text", "json", "output", "result", "markdown"):
                if k in first and isinstance(first[k], str):
                    return first[k]
    if isinstance(out, dict):
        for k in ("text", "json", "output", "result", "markdown"):
            if k in out and isinstance(out[k], str):
                return out[k]
    return str(out)


def _extract_conf(out: Any, default: float = 0.85) -> float:
    if isinstance(out, dict) and "confidence" in out:
        try:
            return float(out["confidence"])
        except (TypeError, ValueError):
            pass
    if isinstance(out, list) and out and isinstance(out[0], dict):
        return float(out[0].get("confidence", default))
    return default


# ---------------------------------------------------------------------------
# Generic Transformers VLM (used as a base for many SOTA models)
# ---------------------------------------------------------------------------

class TransformersVLMEngine(BaseOCREngine):
    """Generic Hugging Face transformers VLM engine.

    Supports the Qwen-style chat-template flow that most current SOTA VLMs use
    (Qwen3-VL, GLM-OCR, dots.ocr, HunyuanOCR, RolmOCR, MonkeyOCR, ...). For
    models with materially different APIs (e.g. InternVL3's tiled-image
    preprocessing), subclass and override `_build_inputs` / `_decode`.

    The engine is lazy: importing this module does NOT load the model. The
    first `recognize()` call triggers the download + load.
    """

    structured = True
    backend_name = "transformers_vlm"

    # Default load kwargs — subclasses may override.
    DEFAULT_LOAD_KWARGS: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": "auto",
        "device_map": "auto",
    }

    def __init__(
        self,
        model_id: str,
        prompt: Optional[str] = None,
        device: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        backend_name: Optional[str] = None,
    ):
        self.model_id = model_id
        self.prompt = prompt or DEFAULT_VLM_PROMPT
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        if backend_name:
            self.backend_name = backend_name
        self._model = None
        self._processor = None
        self._tokenizer = None

    # ------------------------------ load -------------------------------- #

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "transformers is required. Install: pip install transformers accelerate torch"
            ) from e
        import torch  # type: ignore

        load_kwargs = dict(self.DEFAULT_LOAD_KWARGS)
        if load_kwargs.get("torch_dtype") == "auto":
            load_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        LOGGER.info("Loading VLM %s with kwargs=%s", self.model_id, load_kwargs)
        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
        self._model.eval()
        if self.device and load_kwargs.get("device_map") is None:
            self._model = self._model.to(self.device)

    # ------------------------- overridable hooks ------------------------ #

    def _build_messages(self, prompt: str, pil_image) -> list[dict[str, Any]]:
        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ],
        }]

    def _build_inputs(self, image: np.ndarray) -> tuple[dict, int]:
        """Return (model_inputs, prompt_token_len). Override per model family."""
        pil = _bgr_to_pil(image)
        messages = self._build_messages(self.prompt, pil)
        # Apply chat template.
        text = self._processor.apply_chat_template(  # type: ignore[union-attr]
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._processor(  # type: ignore[union-attr]
            text=[text], images=[pil], return_tensors="pt", padding=True,
        )
        # Move to the model's device.
        target_device = next(self._model.parameters()).device  # type: ignore[union-attr]
        inputs = {k: v.to(target_device) if hasattr(v, "to") else v for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        return inputs, prompt_len

    def _decode(self, generated_ids, inputs: dict, prompt_len: int) -> str:
        # Strip the prompt by slicing the first prompt_len tokens off.
        if "input_ids" in inputs and prompt_len > 0:
            trimmed = generated_ids[:, prompt_len:]
        else:
            trimmed = generated_ids
        return self._processor.batch_decode(  # type: ignore[union-attr]
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]

    # ------------------------------ run --------------------------------- #

    def recognize(self, image: np.ndarray) -> OCRResult:
        self._ensure_loaded()
        import torch  # type: ignore

        inputs, prompt_len = self._build_inputs(image)
        gen_kwargs = dict(max_new_tokens=self.max_new_tokens)
        if self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.temperature)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, **gen_kwargs)  # type: ignore[union-attr]
        text = self._decode(generated_ids, inputs, prompt_len).strip()
        # Strip leading "```json" code fences that some models emit despite prompting.
        text = _strip_json_fence(text)
        return OCRResult(text=text, confidence=0.9, backend=self.backend_name)


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        # remove ```json / ``` opening
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


# ---------------------------------------------------------------------------
# Model-specific subclasses
# ---------------------------------------------------------------------------

class Qwen3VLEngine(TransformersVLMEngine):
    """Qwen3-VL — Alibaba; 2B/4B/8B/32B dense + 30B-A3B / 235B-A22B MoE.

    Supports 201 languages incl. Russian; 256K context.
    Default: 8B-Instruct (best 4070 Ti fit with bf16). For more VRAM, switch
    to Qwen/Qwen3-VL-32B-Instruct or the 235B MoE.
    """

    backend_name = "qwen3_vl"

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-8B-Instruct", **kwargs):
        super().__init__(model_id=model_id, **kwargs)


class GLMOCREngine(TransformersVLMEngine):
    """GLM-OCR (0.9B, Z.AI / Zhipu) — 94.62 on OmniDocBench v1.5 (current #1).

    Apache-2.0, supports Russian, vLLM/SGLang ready.
    """

    backend_name = "glm_ocr"

    def __init__(self, model_id: str = "zai-org/GLM-OCR", **kwargs):
        super().__init__(model_id=model_id, **kwargs)


class DotsOCREngine(TransformersVLMEngine):
    """rednote-hilab/dots.ocr (3B) — SOTA multilingual document parsing.

    vLLM-integrated since v0.11.0. For Russian + structured fields, this is
    a strong alternative to PaddleOCR-VL / GLM-OCR. Some checkpoint variants
    are now hosted under `rednote-hilab/dots.mocr`.
    """

    backend_name = "dots_ocr"

    def __init__(self, model_id: str = "rednote-hilab/dots.ocr", **kwargs):
        super().__init__(model_id=model_id, **kwargs)


class HunyuanOCREngine(TransformersVLMEngine):
    """Tencent-Hunyuan/HunyuanOCR (1B) — multiple SOTA benchmarks, vLLM-ready."""

    backend_name = "hunyuan_ocr"

    def __init__(self, model_id: str = "Tencent-Hunyuan/HunyuanOCR", **kwargs):
        super().__init__(model_id=model_id, **kwargs)


class RolmOCREngine(TransformersVLMEngine):
    """reducto/RolmOCR — Qwen2.5-VL-7B fine-tuned for document OCR.

    Faster than the base Qwen2.5-VL-7B, similar quality on OCR transcription.
    Strong choice if you need robust transcription with a known-good Qwen
    processor pipeline.
    """

    backend_name = "rolm_ocr"

    def __init__(self, model_id: str = "reducto/RolmOCR", **kwargs):
        super().__init__(model_id=model_id, **kwargs)


class MonkeyOCREngine(TransformersVLMEngine):
    """Yuliang-Liu/MonkeyOCR — 3B; beats GPT-4o / Qwen2.5-VL-72B / InternVL3-78B
    on OmniDocBench. Lightweight LMM for document parsing.
    """

    backend_name = "monkey_ocr"

    def __init__(self, model_id: str = "echo840/MonkeyOCR-pro-3B", **kwargs):
        super().__init__(model_id=model_id, **kwargs)


class InternVL3Engine(TransformersVLMEngine):
    """OpenGVLab/InternVL3 — multi-tile image preprocessing.

    Overrides _build_inputs to handle InternVL's specific image-tile flow
    when needed; falls back to the generic chat-template path otherwise.
    Note: InternVL has historically required a slightly different processor
    invocation than Qwen-style chat. If the default path errors out for you,
    pin the model_id to OpenGVLab/InternVL3-8B-Instruct and update this hook.
    """

    backend_name = "intern_vl3"

    def __init__(self, model_id: str = "OpenGVLab/InternVL3-8B-Instruct", **kwargs):
        super().__init__(model_id=model_id, **kwargs)


# ---------------------------------------------------------------------------
# vLLM / SGLang OpenAI-compatible client (production path)
# ---------------------------------------------------------------------------

class VLLMServerEngine(BaseOCREngine):
    """Talk to a vLLM (or SGLang) server via the OpenAI-compatible /v1/chat/completions endpoint.

    Why this matters: model swapping becomes a config change. You can serve
    GLM-OCR today, swap to Qwen3-VL tomorrow, without touching this code.

    Start a server, e.g.:
        vllm serve zai-org/GLM-OCR --port 8000 --max-model-len 8192
        vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000 --tensor-parallel-size 2
        vllm serve PaddlePaddle/PaddleOCR-VL --port 8000

    Then point ocr.vllm_url at http://localhost:8000/v1
    """

    structured = True
    backend_name = "vllm_server"

    def __init__(
        self,
        url: str,
        model_id: str,
        prompt: Optional[str] = None,
        api_key: str = "EMPTY",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        timeout: float = 120.0,
        guided_json: bool = False,
    ):
        self.url = url.rstrip("/")
        self.model_id = model_id
        self.prompt = prompt or DEFAULT_VLM_PROMPT
        self.api_key = api_key
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.guided_json = guided_json
        self._client = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "VLLMServerEngine requires the 'openai' Python SDK. "
                "Install: pip install openai"
            ) from e
        self._client = OpenAI(base_url=self.url, api_key=self.api_key, timeout=self.timeout)

    def recognize(self, image: np.ndarray) -> OCRResult:
        self._ensure_client()
        data_url = _image_to_base64_url(image)
        kwargs: dict[str, Any] = dict(
            model=self.model_id,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": self.prompt},
                ],
            }],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        if self.guided_json:
            # vLLM accepts `guided_json` via extra_body. SGLang uses the same key.
            kwargs["extra_body"] = {"guided_json": PRICE_TAG_JSON_SCHEMA}
        response = self._client.chat.completions.create(**kwargs)  # type: ignore[union-attr]
        msg = response.choices[0].message
        text = (msg.content or "").strip()
        text = _strip_json_fence(text)
        return OCRResult(text=text, confidence=0.9, backend=f"vllm:{self.model_id}")


# ---------------------------------------------------------------------------
# Ensemble engine — runs N engines per crop
# ---------------------------------------------------------------------------

class EnsembleOCREngine(BaseOCREngine):
    """Run multiple OCR engines per crop, return all their results.

    The pipeline ingests each result as a separate observation, so per-field
    voting in the aggregator naturally fuses three (or N) different VLMs.

    Engines run sequentially; for production prefer a vllm_server backend
    with two or three models served from the same server (different model_id
    per ensemble entry).
    """

    structured = True  # majority of practical configurations include VLMs
    backend_name = "ensemble"

    def __init__(self, engines: list[BaseOCREngine]):
        if not engines:
            raise ValueError("EnsembleOCREngine requires at least one sub-engine.")
        self.engines = engines
        # If any sub-engine is unstructured the ensemble is mixed; treat as
        # structured if the majority are.
        struct_count = sum(1 for e in engines if e.structured)
        self.structured = struct_count >= len(engines) / 2

    def recognize(self, image: np.ndarray) -> OCRResult:
        results = self.recognize_all(image)
        # Single-result fallback: pick the highest-confidence.
        return max(results, key=lambda r: r.confidence) if results else OCRResult(
            text="", confidence=0.0, backend="ensemble:empty"
        )

    def recognize_all(self, image: np.ndarray) -> list[OCRResult]:
        out: list[OCRResult] = []
        for eng in self.engines:
            try:
                res = eng.recognize(image)
            except Exception as e:
                LOGGER.warning("Ensemble sub-engine %s failed: %s", eng.backend_name, e)
                continue
            if res.text:
                out.append(res)
        return out


# ---------------------------------------------------------------------------
# MinerU 2.5 — pipeline-based document parser
# ---------------------------------------------------------------------------

class MinerUEngine(BaseOCREngine):
    """MinerU 2.5 (1.2B) — pipeline-based document parser; 90.67 on OmniDocBench.

    Heavier than the VLM path and tuned for multi-page documents rather than
    isolated tag crops. Included for completeness; not recommended as the
    primary engine for tag-level extraction. Returns Markdown that the
    classical text parser will then convert into ParsedTag.
    """

    structured = False  # returns markdown, not JSON
    backend_name = "mineru"

    def __init__(self, model_dir: Optional[str] = None, device: Optional[str] = None):
        self.model_dir = model_dir
        self.device = device
        self._loaded = False
        self._predict = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            from magic_pdf.pipe.UNIPipe import UNIPipe  # type: ignore  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "MinerU is not installed. Install: pip install magic-pdf  "
                "(see https://github.com/opendatalab/MinerU for full setup)"
            ) from e
        # MinerU operates on images via its CLI/SDK; minimal in-process integration goes here.
        self._loaded = True

    def recognize(self, image: np.ndarray) -> OCRResult:
        self._ensure_loaded()
        # Minimal stub: write image to a temp file and call magic_pdf's image
        # mode. Real implementation lands when MinerU becomes the chosen path.
        raise NotImplementedError(
            "MinerU integration is a stub. Use paddle_vl / glm_ocr / qwen3_vl "
            "for tag-level extraction, or wire up MinerU's image pipeline here."
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_BACKEND_ALIASES = {
    "noop": "noop",
    "tesseract": "tesseract",
    "paddle": "paddle", "paddleocr": "paddle",
    "paddle_vl": "paddle_vl", "paddleocr_vl": "paddle_vl", "vlm": "paddle_vl",
    "glm_ocr": "glm_ocr", "glm-ocr": "glm_ocr", "glm": "glm_ocr",
    "qwen3_vl": "qwen3_vl", "qwen3-vl": "qwen3_vl", "qwen": "qwen3_vl",
    "dots_ocr": "dots_ocr", "dots.ocr": "dots_ocr", "dots": "dots_ocr",
    "hunyuan_ocr": "hunyuan_ocr", "hunyuan": "hunyuan_ocr",
    "rolm_ocr": "rolm_ocr", "rolm": "rolm_ocr",
    "monkey_ocr": "monkey_ocr", "monkey": "monkey_ocr",
    "intern_vl3": "intern_vl3", "internvl3": "intern_vl3", "intern": "intern_vl3",
    "transformers_vlm": "transformers_vlm", "hf_vlm": "transformers_vlm",
    "vllm": "vllm_server", "vllm_server": "vllm_server", "sglang": "vllm_server",
    "mineru": "mineru",
    "ensemble": "ensemble", "vlm_ensemble": "ensemble",
}


def build_ocr_engine(cfg: OCRConfig) -> BaseOCREngine:
    """Build an engine from config. See module docstring for the backend list."""
    key = _BACKEND_ALIASES.get(cfg.backend.lower().strip())
    if key is None:
        raise ValueError(
            f"Unsupported OCR backend: {cfg.backend!r}. "
            f"Known: {sorted(set(_BACKEND_ALIASES.values()))}"
        )

    base_prompt = _load_prompt(None, cfg.vlm_prompt_path)
    prompt = build_few_shot_prompt(base_prompt, cfg.few_shot_examples_path)

    if key == "noop":
        return NoOpOCREngine()
    if key == "tesseract":
        return TesseractOCREngine()
    if key == "paddle":
        return PaddleOCREngine(lang=cfg.paddleocr_lang)
    if key == "paddle_vl":
        return PaddleVLMEngine(model_name=cfg.vlm_model, prompt=prompt)
    if key == "glm_ocr":
        return GLMOCREngine(
            model_id=cfg.vlm_model if cfg.vlm_model != "PaddlePaddle/PaddleOCR-VL" else "zai-org/GLM-OCR",
            prompt=prompt, max_new_tokens=cfg.vlm_max_new_tokens,
        )
    if key == "qwen3_vl":
        return Qwen3VLEngine(
            model_id=cfg.vlm_model if cfg.vlm_model != "PaddlePaddle/PaddleOCR-VL" else "Qwen/Qwen3-VL-8B-Instruct",
            prompt=prompt, max_new_tokens=cfg.vlm_max_new_tokens,
        )
    if key == "dots_ocr":
        return DotsOCREngine(
            model_id=cfg.vlm_model if cfg.vlm_model != "PaddlePaddle/PaddleOCR-VL" else "rednote-hilab/dots.ocr",
            prompt=prompt, max_new_tokens=cfg.vlm_max_new_tokens,
        )
    if key == "hunyuan_ocr":
        return HunyuanOCREngine(
            model_id=cfg.vlm_model if cfg.vlm_model != "PaddlePaddle/PaddleOCR-VL" else "Tencent-Hunyuan/HunyuanOCR",
            prompt=prompt, max_new_tokens=cfg.vlm_max_new_tokens,
        )
    if key == "rolm_ocr":
        return RolmOCREngine(
            model_id=cfg.vlm_model if cfg.vlm_model != "PaddlePaddle/PaddleOCR-VL" else "reducto/RolmOCR",
            prompt=prompt, max_new_tokens=cfg.vlm_max_new_tokens,
        )
    if key == "monkey_ocr":
        return MonkeyOCREngine(
            model_id=cfg.vlm_model if cfg.vlm_model != "PaddlePaddle/PaddleOCR-VL" else "echo840/MonkeyOCR-pro-3B",
            prompt=prompt, max_new_tokens=cfg.vlm_max_new_tokens,
        )
    if key == "intern_vl3":
        return InternVL3Engine(
            model_id=cfg.vlm_model if cfg.vlm_model != "PaddlePaddle/PaddleOCR-VL" else "OpenGVLab/InternVL3-8B-Instruct",
            prompt=prompt, max_new_tokens=cfg.vlm_max_new_tokens,
        )
    if key == "transformers_vlm":
        return TransformersVLMEngine(
            model_id=cfg.vlm_model, prompt=prompt, max_new_tokens=cfg.vlm_max_new_tokens,
        )
    if key == "vllm_server":
        if not cfg.vllm_url:
            raise ValueError("vllm backend requires ocr.vllm_url in the config.")
        return VLLMServerEngine(
            url=cfg.vllm_url, model_id=cfg.vlm_model, prompt=prompt,
            max_new_tokens=cfg.vlm_max_new_tokens, temperature=cfg.vlm_temperature,
            guided_json=cfg.guided_json,
        )
    if key == "mineru":
        return MinerUEngine()
    if key == "ensemble":
        if not cfg.ensemble_backends:
            raise ValueError(
                "ensemble backend requires ocr.ensemble_backends with at least one entry."
            )
        engines: list[BaseOCREngine] = []
        for entry in cfg.ensemble_backends:
            # Build a per-entry OCRConfig overlay so each engine picks up its own
            # backend and (optionally) model id, while inheriting other fields.
            sub_cfg = OCRConfig(
                backend=entry.backend,
                min_frames_between_ocr_per_track=cfg.min_frames_between_ocr_per_track,
                min_sharpness=cfg.min_sharpness,
                min_crop_area_px=cfg.min_crop_area_px,
                min_detection_confidence=cfg.min_detection_confidence,
                top_k_crops_per_track=cfg.top_k_crops_per_track,
                vlm_model=entry.vlm_model or cfg.vlm_model,
                vlm_prompt_path=cfg.vlm_prompt_path,
                paddleocr_lang=cfg.paddleocr_lang,
                vlm_max_new_tokens=cfg.vlm_max_new_tokens,
                vlm_temperature=cfg.vlm_temperature,
                vllm_url=cfg.vllm_url,
                guided_json=cfg.guided_json,
                few_shot_examples_path=cfg.few_shot_examples_path,
            )
            engines.append(build_ocr_engine(sub_cfg))
        return EnsembleOCREngine(engines)
    raise ValueError(f"Backend resolved to unknown key: {key}")


# ---------------------------------------------------------------------------
# CropDecoder adapter — the "smart OCR" fallback link of the chain
# ---------------------------------------------------------------------------

class OCRDecoder(CropDecoder):
    """Last link of the recognition chain: OCR/VLM the crop and parse it.

    One crop can yield several results (an ensemble engine returns one per
    sub-engine); each becomes its own :class:`RecognitionResult`. A failing
    engine degrades to ``[]`` — it never aborts the video. Per-field merge
    with QR/barcode is delegated to the aggregator's weighted voting, not
    re-implemented here (see ``docs/recognition-pipeline.md``).
    """

    name = "ocr"

    def __init__(self, engine: "BaseOCREngine", parser: TagParser) -> None:
        self._engine = engine
        self._parser = parser

    @classmethod
    def from_config(cls, ocr_cfg: OCRConfig, parser_cfg: ParserConfig) -> "OCRDecoder":
        return cls(build_ocr_engine(ocr_cfg), TagParser(parser_cfg))

    def decode(self, crop_bgr: np.ndarray) -> list[RecognitionResult]:
        try:
            results = self._engine.recognize_all(crop_bgr)
        except Exception as exc:  # one bad crop must not kill the whole video
            LOGGER.warning("OCR failed on a crop: %s", exc)
            return []
        out: list[RecognitionResult] = []
        for res in results:
            if not res.text:
                continue
            if self._engine.structured or res.text.lstrip().startswith("{"):
                parsed = self._parser.parse_vlm_json(
                    res.text, vlm_confidence=res.confidence, backend=res.backend
                )
            else:
                parsed = self._parser.parse_text(
                    res.text, ocr_confidence=res.confidence, backend=res.backend
                )
            out.append(
                RecognitionResult(
                    parsed=parsed,
                    decoder=res.backend,
                    confidence=res.confidence,
                    text=res.text,
                    found=not parsed_is_empty(parsed),
                )
            )
        return out
