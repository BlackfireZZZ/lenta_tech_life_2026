from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .types import OCRResult


class BaseOCREngine(ABC):
    @abstractmethod
    def recognize(self, image: np.ndarray) -> OCRResult:
        raise NotImplementedError


class NoOpOCREngine(BaseOCREngine):
    def recognize(self, image: np.ndarray) -> OCRResult:
        return OCRResult(text="", confidence=0.0, backend="noop")


class TesseractOCREngine(BaseOCREngine):
    def __init__(self):
        import pytesseract

        self._tesseract = pytesseract

    def recognize(self, image: np.ndarray) -> OCRResult:
        text = self._tesseract.image_to_string(image, config="--oem 1 --psm 6")
        return OCRResult(text=text.strip(), confidence=0.55 if text.strip() else 0.0, backend="tesseract")


class PaddleOCREngine(BaseOCREngine):
    def __init__(self):
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(use_angle_cls=True, lang="ru")

    def recognize(self, image: np.ndarray) -> OCRResult:
        lines = self._ocr.ocr(image, cls=True)
        if not lines or not lines[0]:
            return OCRResult(text="", confidence=0.0, backend="paddle")
        fragments: list[str] = []
        confs: list[float] = []
        for node in lines[0]:
            text, conf = node[1]
            fragments.append(text)
            confs.append(float(conf))
        avg = sum(confs) / len(confs) if confs else 0.0
        return OCRResult(text=" ".join(fragments).strip(), confidence=avg, backend="paddle")


def build_ocr_engine(name: str) -> BaseOCREngine:
    key = name.lower().strip()
    if key == "noop":
        return NoOpOCREngine()
    if key == "tesseract":
        return TesseractOCREngine()
    if key == "paddle":
        return PaddleOCREngine()
    raise ValueError(f"Unsupported OCR backend: {name}")

