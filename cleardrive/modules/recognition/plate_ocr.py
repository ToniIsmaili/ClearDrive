import re
from datetime import datetime, timezone
from collections import deque
from itertools import combinations

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


def pattern_compact_length(pattern: str) -> int:
    """Return the number of alphanumeric characters in a plate format pattern."""
    length = 0
    for segment in parse_pattern_segments(pattern):
        if isinstance(segment, str):
            continue
        _, segment_length = segment
        length += segment_length
    return length


def pattern_segment_starts(pattern: str) -> set[int]:
    """Return compact-string indices where a new pattern segment begins."""
    starts = {0}
    index = 0

    for segment in parse_pattern_segments(pattern):
        if isinstance(segment, str):
            continue

        _, segment_length = segment
        index += segment_length

        if index < pattern_compact_length(pattern):
            starts.add(index)

    return starts


def extract_plate_compact(
    compact: str,
    *,
    expected_len: int,
    format_regex: re.Pattern[str],
    plate_format: str,
    prefix_values: list[str],
    max_extra_chars: int = 4,
    segment_starts: set[int] | None = None,
) -> str | None:
    """Pull a valid compact plate out of noisy OCR (extra edge or inserted characters)."""

    def validate(candidate: str) -> bool:
        return (
            format_regex.fullmatch(candidate) is not None
            and format_to_pattern(candidate, plate_format, prefix_values) is not None
        )

    if validate(compact):
        return compact

    boundaries = segment_starts if segment_starts is not None else pattern_segment_starts(plate_format)
    length = len(compact)
    if length < expected_len or length > expected_len + max_extra_chars:
        return None

    for start in range(length - expected_len + 1):
        candidate = compact[start : start + expected_len]
        if validate(candidate):
            return candidate

    best: tuple[tuple[int, ...], str] | None = None
    best_rank: tuple[int, int, int, int] | None = None

    for indices in combinations(range(length), expected_len):
        candidate = "".join(compact[index] for index in indices)
        if not validate(candidate):
            continue

        skipped = [index for index in range(length) if index not in indices]
        duplicate_skips = sum(
            1
            for index in skipped
            if (index > 0 and compact[index - 1] == compact[index])
            or (index + 1 < length and compact[index + 1] == compact[index])
        )
        boundary_skips = sum(1 for index in skipped if index in boundaries)

        if indices[-1] - indices[0] + 1 == expected_len:
            rank = (1, -boundary_skips, -duplicate_skips, min(skipped))
        else:
            rank = (2, -boundary_skips, -duplicate_skips, min(skipped))

        if best_rank is None or rank < best_rank:
            best_rank = rank
            best = (indices, candidate)

    return best[1] if best is not None else None


# Common Tesseract confusions on low-resolution plate crops.
OCR_SUBSTITUTIONS: dict[str, str] = {
    "O": "08",
    "0": "O8",
    "Q": "O0",
    "D": "0",
    "Z": "24",
    "2": "Z",
    "S": "5",
    "5": "S",
    "B": "8",
    "8": "B6",
    "G": "6",
    "6": "G8",
    "I": "1",
    "1": "I",
    "L": "1",
    "T": "7",
    "7": "T",
    "X": "KS",
    "K": "X",
}


def _ocr_substitution_variants(char: str) -> set[str]:
    variants = {char}
    for alt in OCR_SUBSTITUTIONS.get(char, ""):
        variants.add(alt)
    return variants


def _validate_compact(
    candidate: str,
    *,
    format_regex: re.Pattern[str],
    plate_format: str,
    prefix_values: list[str],
) -> bool:
    return (
        format_regex.fullmatch(candidate) is not None
        and format_to_pattern(candidate, plate_format, prefix_values) is not None
    )


def _prefixes_matching_second_letter(compact: str, prefix_values: list[str]) -> list[str]:
    """Return prefixes whose second letter matches the first character of *compact*."""
    if not compact or not compact[0].isalpha():
        return []

    return [prefix for prefix in prefix_values if len(prefix) >= 2 and compact[0] == prefix[1]]


def _prefix_completion_candidates(
    compact: str,
    prefix_values: list[str],
    expected_len: int,
) -> set[str]:
    """Rebuild plates missing exactly one known prefix letter when OCR still hints at it."""
    candidates: set[str] = set()
    missing = expected_len - len(compact)

    if missing != 1:
        return candidates

    matching = _prefixes_matching_second_letter(compact, prefix_values)
    if len(matching) != 1:
        return candidates

    prefix = matching[0]
    candidates.add(prefix[0] + compact)
    return candidates


def _repair_neighbors(
    compact: str,
    prefix_values: list[str],
    expected_len: int,
) -> set[str]:
    """Apply one OCR repair step to *compact*."""
    neighbors: set[str] = set()

    for index in range(len(compact)):
        neighbors.add(compact[:index] + compact[index + 1 :])

        for alt in _ocr_substitution_variants(compact[index]):
            if alt != compact[index]:
                neighbors.add(compact[:index] + alt + compact[index + 1 :])

    neighbors.update(_prefix_completion_candidates(compact, prefix_values, expected_len))

    return {value for value in neighbors if value}


def repair_plate_compact(
    compact: str,
    *,
    expected_len: int,
    format_regex: re.Pattern[str],
    plate_format: str,
    prefix_values: list[str],
    max_extra_chars: int = 4,
    max_repairs: int = 3,
    segment_starts: set[int] | None = None,
) -> str | None:
    """Recover a valid plate from missing or misread OCR characters."""

    def validate(candidate: str) -> bool:
        return _validate_compact(
            candidate,
            format_regex=format_regex,
            plate_format=plate_format,
            prefix_values=prefix_values,
        )

    extracted = extract_plate_compact(
        compact,
        expected_len=expected_len,
        format_regex=format_regex,
        plate_format=plate_format,
        prefix_values=prefix_values,
        max_extra_chars=max_extra_chars,
        segment_starts=segment_starts,
    )
    if extracted is not None:
        return extracted

    min_len = expected_len - max_repairs
    max_len = expected_len + max_extra_chars
    if len(compact) < min_len or len(compact) > max_len:
        return None

    if validate(compact):
        return compact

    best: str | None = None
    best_cost: tuple[int, int] | None = None
    seen: set[str] = {compact}
    queue: deque[tuple[str, int]] = deque([(compact, 0)])

    while queue:
        current, repairs = queue.popleft()

        if validate(current):
            cost = (repairs, abs(len(current) - expected_len))
            if best_cost is None or cost < best_cost:
                best = current
                best_cost = cost
            continue

        if repairs >= max_repairs:
            continue

        for neighbor in _repair_neighbors(current, prefix_values, expected_len):
            if neighbor in seen:
                continue
            if len(neighbor) < min_len or len(neighbor) > max_len:
                continue
            seen.add(neighbor)
            queue.append((neighbor, repairs + 1))

        if len(current) > expected_len:
            nested = extract_plate_compact(
                current,
                expected_len=expected_len,
                format_regex=format_regex,
                plate_format=plate_format,
                prefix_values=prefix_values,
                max_extra_chars=max_extra_chars,
                segment_starts=segment_starts,
            )
            if nested is not None and nested not in seen:
                seen.add(nested)
                queue.append((nested, repairs + 1))

    return best


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
        self.black_v_max = black_v_max if black_v_max is not None else env_int("OCR_BLACK_V_MAX", 110)
        self.black_s_max = black_s_max if black_s_max is not None else env_int("OCR_BLACK_S_MAX", 35)
        self.min_plate_height = (
            min_plate_height if min_plate_height is not None else env_int("OCR_MIN_PLATE_HEIGHT", 50)
        )
        self.tesseract_cmd = tesseract_cmd or env_optional_str("TESSERACT_CMD")
        self.tesseract_psm = (
            tesseract_psm if tesseract_psm is not None else env_int("OCR_TESSERACT_PSM", 7)
        )
        compact_pattern = self.plate_format.replace(" ", "")
        self._format_regex = pattern_to_regex(compact_pattern, self.prefix_values)
        self._expected_compact_length = pattern_compact_length(self.plate_format)
        self._segment_starts = pattern_segment_starts(self.plate_format)
        self._last_log_message: str | None = None

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

    def preprocess(self, image: np.ndarray) -> np.ndarray | None:
        """Return the binary image fed to the OCR engine, or None for empty input."""
        if image is None or image.size == 0:
            return None

        binary = self._isolate_black_characters(image)
        return self._scale_for_ocr(binary)

    def recognize(self, image: np.ndarray) -> str | None:
        """Recognize and validate plate text from a BGR image. Returns None if OCR fails validation."""
        scaled = self.preprocess(image)
        if scaled is None:
            self._log("empty or unusable plate crop")
            return None

        raw_text = self._run_ocr(scaled)
        if not raw_text:
            self._log("tesseract returned nothing")
            return None

        compact = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
        if not compact:
            self._log(f"raw {raw_text!r} -> no alphanumerics")
            return None

        matched = repair_plate_compact(
            compact,
            expected_len=self._expected_compact_length,
            format_regex=self._format_regex,
            plate_format=self.plate_format,
            prefix_values=self.prefix_values,
            segment_starts=self._segment_starts,
        )
        if matched is None:
            self._log(
                f"raw {raw_text!r} -> compact {compact!r} "
                f"does not match {self.plate_format}"
            )
            return None

        formatted = format_to_pattern(matched, self.plate_format, self.prefix_values)
        if formatted is None:
            self._log(
                f"raw {raw_text!r} -> compact {compact!r} "
                f"failed prefix check for {self.plate_format}"
            )
            return None

        if matched == compact:
            self._log(f"raw {raw_text!r} -> {formatted}")
        else:
            self._log(f"raw {raw_text!r} -> {formatted} (repaired from {compact!r})")
        return formatted

    def _log(self, message: str) -> None:
        if message == self._last_log_message:
            return
        self._last_log_message = message
        print(f"[OCR] {message}", flush=True)

    def _isolate_black_characters(self, image: np.ndarray) -> np.ndarray:
        """Separate black/gray ink from the plate background; ignore colored artwork."""
        if image.ndim == 2:
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, a_channel, b_channel = cv2.split(lab)
        a_dev = np.abs(a_channel.astype(np.int16) - 128)
        b_dev = np.abs(b_channel.astype(np.int16) - 128)
        chroma = np.maximum(a_dev, b_dev)
        achromatic = chroma <= self.black_s_max

        # Split ink from background on neutral pixels only; colored regions stay white.
        work = np.where(achromatic, lightness, 255).astype(np.uint8)
        _, binary = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = np.where(achromatic, binary, 255).astype(np.uint8)

        if np.count_nonzero(binary == 0) > binary.size * 0.45:
            binary = cv2.bitwise_not(binary)
            binary = np.where(achromatic, binary, 255).astype(np.uint8)

        # Drop anything too bright to be black ink (plate surface, shadows, anti-alias halos).
        binary[(lightness > self.black_v_max) & (binary == 0)] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

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

    def __enter__(self) -> "PlateOCRModule":
        self.setup()
        return self

    def __exit__(self, *_: object) -> None:
        self.teardown()
