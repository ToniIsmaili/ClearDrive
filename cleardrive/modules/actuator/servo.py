import threading
from datetime import datetime, timezone

from cleardrive.core.config import (
    DEFAULT_SERVO_CLOSE_ANGLE,
    DEFAULT_SERVO_CLOSE_DELAY_SECONDS,
    DEFAULT_SERVO_OPEN_ANGLE,
    env_int,
)
from cleardrive.core.module import Module
from cleardrive.core.types import ImageFrame
from cleardrive.modules.actuator.servo_driver import ServoDriver, create_servo_driver


class ServoModule(Module):
    """Opens the ramp for whitelisted plates and closes it after a delay."""

    name = "servo"

    def __init__(
        self,
        open_angle: float | None = None,
        close_angle: float | None = None,
        close_delay_seconds: int | None = None,
        driver: ServoDriver | None = None,
    ) -> None:
        self.open_angle = open_angle if open_angle is not None else env_int(
            "SERVO_OPEN_ANGLE", DEFAULT_SERVO_OPEN_ANGLE
        )
        self.close_angle = close_angle if close_angle is not None else env_int(
            "SERVO_CLOSE_ANGLE", DEFAULT_SERVO_CLOSE_ANGLE
        )
        self.close_delay_seconds = (
            close_delay_seconds
            if close_delay_seconds is not None
            else env_int("SERVO_CLOSE_DELAY_SECONDS", DEFAULT_SERVO_CLOSE_DELAY_SECONDS)
        )
        self._driver = driver or create_servo_driver(
            open_angle=self.open_angle,
            close_angle=self.close_angle,
        )
        self._last_opened_plate: str | None = None
        self._close_timer: threading.Timer | None = None
        self._timer_lock = threading.Lock()

    def setup(self) -> None:
        self._driver.setup()

    def teardown(self) -> None:
        with self._timer_lock:
            if self._close_timer is not None:
                self._close_timer.cancel()
                self._close_timer = None

        try:
            self._driver.move_to(self.close_angle)
        except Exception:
            pass

        self._driver.teardown()

    def process(self, frame: ImageFrame | None = None) -> ImageFrame | None:
        """Open the ramp when *frame* contains a whitelisted plate."""
        if frame is None:
            return None

        if not frame.metadata.get("whitelisted"):
            return frame

        plate = frame.metadata.get("plate")
        if not isinstance(plate, str) or not plate.strip():
            return frame

        opened = False
        servo_error: str | None = None

        if plate != self._last_opened_plate:
            try:
                self._open_ramp()
                self._last_opened_plate = plate
                opened = True
            except Exception as exc:
                servo_error = str(exc)
                self._last_opened_plate = plate

        return ImageFrame(
            data=frame.data,
            timestamp=datetime.now(timezone.utc),
            source=self.name,
            metadata={
                **frame.metadata,
                "servo_opened": opened,
                "servo_close_delay_seconds": self.close_delay_seconds if opened else None,
                "servo_error": servo_error,
                "input_source": frame.source,
            },
        )

    def _open_ramp(self) -> None:
        self._driver.move_to(self.open_angle)
        self._schedule_close()

    def _schedule_close(self) -> None:
        with self._timer_lock:
            if self._close_timer is not None:
                self._close_timer.cancel()

            self._close_timer = threading.Timer(
                self.close_delay_seconds,
                self._close_ramp,
            )
            self._close_timer.daemon = True
            self._close_timer.start()

    def _close_ramp(self) -> None:
        try:
            self._driver.move_to(self.close_angle)
        except Exception:
            pass

        with self._timer_lock:
            self._close_timer = None

    def __enter__(self) -> "ServoModule":
        self.setup()
        return self

    def __exit__(self, *_: object) -> None:
        self.teardown()
