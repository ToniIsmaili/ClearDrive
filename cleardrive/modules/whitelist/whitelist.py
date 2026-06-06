import re
from datetime import datetime, timezone

from cleardrive.core.config import (
    DEFAULT_PLATE_FORMAT,
    DEFAULT_PLATE_PREFIX_VALUES,
    DEFAULT_WHITELIST_PLATES,
    env_csv_list,
    env_str,
)
from cleardrive.core.module import Module
from cleardrive.core.types import ImageFrame
from cleardrive.modules.recognition.plate_ocr import format_to_pattern, pattern_to_regex


class WhiteListModule(Module):
    """Checks whether a license plate text is on the configured whitelist."""

    name = "whitelist"

    def __init__(
        self,
        plate_format: str | None = None,
        prefix_values: list[str] | None = None,
        whitelist_plates: list[str] | None = None,
    ) -> None:
        self.plate_format = plate_format or env_str("OCR_PLATE_FORMAT", DEFAULT_PLATE_FORMAT)
        self.prefix_values = prefix_values or env_csv_list(
            "OCR_PLATE_PREFIX_VALUES", DEFAULT_PLATE_PREFIX_VALUES
        )
        compact_pattern = self.plate_format.replace(" ", "")
        self._format_regex = pattern_to_regex(compact_pattern, self.prefix_values)
        self._whitelist = self._build_whitelist(whitelist_plates)

    def setup(self) -> None:
        pass

    def teardown(self) -> None:
        pass

    def process(self, frame: ImageFrame | None = None) -> ImageFrame | None:
        """Check plate text from *frame* metadata and attach whitelist results."""
        if frame is None:
            return None

        text = frame.metadata.get("text")
        if not isinstance(text, str) or not text.strip():
            return None

        whitelisted, plate = self.check(text)

        return ImageFrame(
            data=frame.data,
            timestamp=datetime.now(timezone.utc),
            source=self.name,
            metadata={
                **frame.metadata,
                "whitelisted": whitelisted,
                "plate": plate,
                "input_source": frame.source,
            },
        )

    def check(self, text: str) -> tuple[bool, str | None]:
        """Return whether *text* is whitelisted and the processed plate (or None if invalid)."""
        plate = self._normalize_plate(text)
        if plate is None:
            return False, None

        return plate in self._whitelist, plate

    def _build_whitelist(self, whitelist_plates: list[str] | None) -> set[str]:
        raw_plates = whitelist_plates or env_csv_list("WHITELIST_PLATES", DEFAULT_WHITELIST_PLATES)
        whitelist: set[str] = set()

        for entry in raw_plates:
            plate = self._normalize_plate(entry)
            if plate is not None:
                whitelist.add(plate)

        return whitelist

    def _normalize_plate(self, text: str) -> str | None:
        compact = re.sub(r"[^A-Z0-9]", "", text.upper())
        if not compact:
            return None

        if not self._format_regex.fullmatch(compact):
            return None

        return format_to_pattern(compact, self.plate_format, self.prefix_values)

    def __enter__(self) -> "WhiteListModule":
        self.setup()
        return self

    def __exit__(self, *_: object) -> None:
        self.teardown()
