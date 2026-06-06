import re
from datetime import datetime, timezone

import cv2
import numpy as np
import pytesseract

from cleardrive.core.config import (
    DEFAULT_PLATE_FORMAT,
    DEFAULT_PLATE_PREFIX_VALUES,
    env_csv_list,
    env_int,
    env_optional_str,
    env_str,
)
from cleardrive.core.module import Module
from cleardrive.core.types import ImageFrame

PatternSegment = str | tuple[str, int]


def parse_pattern_segments(pattern: str) -> list[PatternSegment]:
    """Split a pattern into literal strings and token runs (X, 0, F)."""
    segments: list[PatternSegment] = []
    index = 0

    while index < len(pattern):
        char = pattern[index]
        if char in {"X", "0", "F"}:
            token = char
            end = index + 1
            while end < len(pattern) and pattern[end] == token:
                end += 1
            segments.append((token, end - index))
            index = end
            continue

        end = index + 1
        while end < len(pattern) and pattern[end] not in {"X", "0", "F"}:
            end += 1
        segments.append(pattern[index:end])
        index = end

    return segments


def pattern_to_regex(pattern: str, prefix_values: list[str]) -> re.Pattern[str]:
    """Convert a plate format pattern to a regex (X = letter, 0 = digit, F = env prefix)."""
    parts: list[str] = []

    for segment in parse_pattern_segments(pattern):
        if isinstance(segment, str):
            parts.append(re.escape(segment))
            continue

        token, length = segment
        if token == "X":
            parts.append(r"[A-Z]" if length == 1 else rf"[A-Z]{{{length}}}")
        elif token == "0":
            parts.append(r"\d" if length == 1 else rf"\d{{{length}}}")
        elif token == "F":
            allowed = [value for value in prefix_values if len(value) == length]
            if not allowed:
                raise ValueError(
                    f"No prefix values with length {length} for pattern segment '{'F' * length}'"
                )
            alternatives = "|".join(re.escape(value) for value in allowed)
            parts.append(f"(?:{alternatives})")

    return re.compile("^" + "".join(parts) + "$")


def format_to_pattern(text: str, pattern: str, prefix_values: list[str]) -> str | None:
    """Insert spacing from *pattern* into a compact alphanumeric *text*."""
    compact = re.sub(r"[^A-Z0-9]", "", text.upper())
    result: list[str] = []
    index = 0

    for segment in parse_pattern_segments(pattern):
        if isinstance(segment, str):
            result.append(segment)
            continue

        token, length = segment
        if index + length > len(compact):
            return None

        chunk = compact[index : index + length]
        if token == "F":
            allowed = {value for value in prefix_values if len(value) == length}
            if chunk not in allowed:
                return None
        elif token == "X":
            if not chunk.isalpha():
                return None
        elif token == "0":
            if not chunk.isdigit():
                return None

        result.append(chunk)
        index += length

    if index != len(compact):
        return None

    return "".join(result)


class PlateOCRModule(Module):
    """Reads license plate text from a cropped plate image using OCR on black characters."""

    name = "plate_ocr"

    def __init__(
        self,
        plate_format: str | None = None,
        prefix_values: list[str] | None = None,
        black_v_max: int | None = None,
        black_s_max: int | None = None,
        min_plate_height: int | None = None,
        tesseract_cmd: str | None = None,
        tesseract_psm: int | None = None,
    ) -> None:
        self.plate_format = plate_format or env_str("OCR_PLATE_FORMAT", DEFAULT_PLATE_FORMAT)
        self.prefix_values = prefix_values or env_csv_list(
            "OCR_PLATE_PREFIX_VALUES", DEFAULT_PLATE_PREFIX_VALUES
        )
        self.black_v_max = black_v_max if black_v_max is not None else env_int("OCR_BLACK_V_MAX", 80)
        self.black_s_max = black_s_max if black_s_max is not None else env_int("OCR_BLACK_S_MAX", 100)
        self.min_plate_height = (
            min_plate_height if min_plate_height is not None else env_int("OCR_MIN_PLATE_HEIGHT", 50)
        )
        self.tesseract_cmd = tesseract_cmd or env_optional_str("TESSERACT_CMD")
        self.tesseract_psm = (
            tesseract_psm if tesseract_psm is not None else env_int("OCR_TESSERACT_PSM", 7)
        )
        compact_pattern = self.plate_format.replace(" ", "")
        self._format_regex = pattern_to_regex(compact_pattern, self.prefix_values)

    def setup(self) -> None:
        if self.tesseract_cmd is not None:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

    def teardown(self) -> None:
        pass

    def process(self, frame: ImageFrame | None = None) -> ImageFrame | None:
        """Return an ImageFrame with recognized plate text in metadata, or None on failure."""
        if frame is None:
            return None

        text = self.recognize(frame.data)
        if text is None:
            return None

        return ImageFrame(
            data=frame.data,
            timestamp=datetime.now(timezone.utc),
            source=self.name,
            metadata={
                **frame.metadata,
                "text": text,
                "plate_format": self.plate_format,
                "input_source": frame.source,
            },
        )

    def recognize(self, image: np.ndarray) -> str | None:
        """Recognize and validate plate text from a BGR image. Returns None if OCR fails validation."""
        if image is None or image.size == 0:
            return None

        binary = self._isolate_black_characters(image)
        scaled = self._scale_for_ocr(binary)
        raw_text = self._run_ocr(scaled)
        if not raw_text:
            return None

        return self._validate_text(raw_text)

    def _isolate_black_characters(self, image: np.ndarray) -> np.ndarray:
        """Keep only dark, low-saturation pixels so colored plate text is ignored."""
        if image.ndim == 2:
            gray = image
        else:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower = np.array([0, 0, 0], dtype=np.uint8)
            upper = np.array([180, self.black_s_max, self.black_v_max], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            gray = np.full(image.shape[:2], 255, dtype=np.uint8)
            gray[mask > 0] = 0
            return gray

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _scale_for_ocr(self, binary: np.ndarray) -> np.ndarray:
        height = binary.shape[0]
        if height >= self.min_plate_height:
            return binary

        scale = self.min_plate_height / float(height)
        width = int(binary.shape[1] * scale)
        return cv2.resize(binary, (width, self.min_plate_height), interpolation=cv2.INTER_CUBIC)

    def _run_ocr(self, binary: np.ndarray) -> str:
        config = (
            f"--psm {self.tesseract_psm} --oem 3 "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
        )
        text = pytesseract.image_to_string(binary, config=config)
        return text.strip()

    def _validate_text(self, raw_text: str) -> str | None:
        compact = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
        if not compact:
            return None

        if not self._format_regex.fullmatch(compact):
            return None

        return format_to_pattern(compact, self.plate_format, self.prefix_values)

    def __enter__(self) -> "PlateOCRModule":
        self.setup()
        return self

    def __exit__(self, *_: object) -> None:
        self.teardown()
