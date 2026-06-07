import logging
from typing import Any

from cleardrive.core.config import (
    DEFAULT_SERVO_CLOSE_ANGLE,
    DEFAULT_SERVO_ENABLED,
    DEFAULT_SERVO_GPIO_PIN,
    DEFAULT_SERVO_MAX_PULSE_WIDTH,
    DEFAULT_SERVO_MIN_PULSE_WIDTH,
    DEFAULT_SERVO_OPEN_ANGLE,
    env_bool,
    env_int,
)

logger = logging.getLogger(__name__)


class ServoDriver:
    """Controls an SG90-style hobby servo for ramp open/close positions."""

    def __init__(
        self,
        pin: int | None = None,
        open_angle: float | None = None,
        close_angle: float | None = None,
        min_pulse_width: float | None = None,
        max_pulse_width: float | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.pin = pin if pin is not None else env_int("SERVO_GPIO_PIN", DEFAULT_SERVO_GPIO_PIN)
        self.open_angle = open_angle if open_angle is not None else env_int(
            "SERVO_OPEN_ANGLE", DEFAULT_SERVO_OPEN_ANGLE
        )
        self.close_angle = close_angle if close_angle is not None else env_int(
            "SERVO_CLOSE_ANGLE", DEFAULT_SERVO_CLOSE_ANGLE
        )
        self.min_pulse_width = (
            min_pulse_width
            if min_pulse_width is not None
            else DEFAULT_SERVO_MIN_PULSE_WIDTH
        )
        self.max_pulse_width = (
            max_pulse_width
            if max_pulse_width is not None
            else DEFAULT_SERVO_MAX_PULSE_WIDTH
        )
        self.enabled = enabled if enabled is not None else env_bool("SERVO_ENABLED", DEFAULT_SERVO_ENABLED)
        self._servo: Any | None = None

    def setup(self) -> None:
        if not self.enabled:
            logger.info("Servo driver disabled (SERVO_ENABLED=false)")
            return

        if self._servo is not None:
            return

        from gpiozero import AngularServo

        self._servo = AngularServo(
            self.pin,
            min_angle=0,
            max_angle=180,
            min_pulse_width=self.min_pulse_width,
            max_pulse_width=self.max_pulse_width,
        )
        self.move_to(self.close_angle)

    def teardown(self) -> None:
        if self._servo is not None:
            self.move_to(self.close_angle)
            self._servo.close()
            self._servo = None

    def move_to(self, angle: float) -> None:
        if not self.enabled:
            logger.debug("Servo (disabled): move to %.1f°", angle)
            return

        if self._servo is None:
            self.setup()

        assert self._servo is not None
        self._servo.angle = angle
        logger.debug("Servo moved to %.1f°", angle)


def create_servo_driver(
    pin: int | None = None,
    open_angle: float | None = None,
    close_angle: float | None = None,
    enabled: bool | None = None,
) -> ServoDriver:
    """Return a servo driver configured from env vars and constructor overrides."""
    return ServoDriver(
        pin=pin,
        open_angle=open_angle,
        close_angle=close_angle,
        enabled=enabled,
    )
