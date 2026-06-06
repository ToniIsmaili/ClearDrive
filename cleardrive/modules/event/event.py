from datetime import datetime, timezone
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError

from cleardrive.core.module import Module
from cleardrive.core.types import ImageFrame
from cleardrive.modules.event.sns_publisher import SnsPublisher


class EventModule(Module):
    """Publishes an SNS event when a whitelisted license plate is detected."""

    name = "event"

    def __init__(self, sns_publisher: SnsPublisher | None = None) -> None:
        self._publisher = sns_publisher or SnsPublisher()
        self._last_published_plate: str | None = None

    def setup(self) -> None:
        self._publisher.setup()

    def teardown(self) -> None:
        self._publisher.teardown()

    def process(self, frame: ImageFrame | None = None) -> ImageFrame | None:
        """Publish an SNS event when *frame* contains a whitelisted plate."""
        if frame is None:
            return None

        if not frame.metadata.get("whitelisted"):
            return frame

        plate = frame.metadata.get("plate")
        if not isinstance(plate, str) or not plate.strip():
            return frame

        published = False
        message_id: str | None = None
        event_error: str | None = None

        if plate != self._last_published_plate:
            try:
                message_id = self.publish(plate, frame.metadata)
                self._last_published_plate = plate
                published = True
            except (BotoCoreError, ClientError, ValueError) as exc:
                event_error = str(exc)
                self._last_published_plate = plate

        return ImageFrame(
            data=frame.data,
            timestamp=datetime.now(timezone.utc),
            source=self.name,
            metadata={
                **frame.metadata,
                "event_published": published,
                "sns_message_id": message_id,
                "event_error": event_error,
                "input_source": frame.source,
            },
        )

    def publish(self, plate: str, metadata: dict[str, Any] | None = None) -> str:
        """Publish an SNS event for *plate* and return the message ID."""
        event_metadata: dict[str, Any] = {}
        if metadata is not None:
            for key in ("confidence", "bbox", "text", "input_source"):
                value = metadata.get(key)
                if value is not None:
                    event_metadata[key] = value

        return self._publisher.publish(plate, event_metadata)

    def __enter__(self) -> "EventModule":
        self.setup()
        return self

    def __exit__(self, *_: object) -> None:
        self.teardown()
