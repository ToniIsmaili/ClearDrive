import re
from datetime import datetime, timezone

from cleardrive.core.config import (
    DEFAULT_PLATE_FORMAT,
    DEFAULT_PLATE_PREFIX_VALUES,
    env_csv_list,
    env_str,
)
from cleardrive.core.module import Module
from cleardrive.core.types import ImageFrame
from cleardrive.modules.recognition.plate_ocr import format_to_pattern, pattern_to_regex
from cleardrive.modules.whitelist.influx_cache import InfluxWhitelistCache


class WhiteListModule(Module):
    """Checks whether a license plate text is on the InfluxDB whitelist."""

    name = "whitelist"

    def __init__(
        self,
        plate_format: str | None = None,
        prefix_values: list[str] | None = None,
        influx_cache: InfluxWhitelistCache | None = None,
    ) -> None:
        self.plate_format = plate_format or env_str("OCR_PLATE_FORMAT", DEFAULT_PLATE_FORMAT)
        self.prefix_values = prefix_values or env_csv_list(
            "OCR_PLATE_PREFIX_VALUES", DEFAULT_PLATE_PREFIX_VALUES
        )
        compact_pattern = self.plate_format.replace(" ", "")
        self._format_regex = pattern_to_regex(compact_pattern, self.prefix_values)
        self._influx_cache = influx_cache or InfluxWhitelistCache()

    def setup(self) -> None:
        self._influx_cache.setup()

    def teardown(self) -> None:
        self._influx_cache.teardown()

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

        return plate in self._influx_cache.get_plates(self._normalize_plate), plate

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
