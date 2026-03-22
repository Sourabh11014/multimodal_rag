from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

from PIL import Image


class ImageTextExtractor:
    """Turns image assets into retrievable text via captions, OCR, and metadata."""

    def extract(self, image_path: Path, caption: str | None = None) -> str:
        parts: list[str] = []
        if caption:
            parts.append(caption.strip())

        parts.append(image_path.stem.replace("_", " ").replace("-", " "))

        with Image.open(image_path) as image:
            width, height = image.size
            parts.append(f"image dimensions {width}x{height}")
            pytesseract = self._load_pytesseract()
            if pytesseract is not None:
                ocr_text = pytesseract.image_to_string(image).strip()
                if ocr_text:
                    parts.append(ocr_text)

        return ". ".join(part for part in parts if part)

    @staticmethod
    def _load_pytesseract():
        if importlib.util.find_spec("pytesseract") is None:
            return None
        return importlib.import_module("pytesseract")
