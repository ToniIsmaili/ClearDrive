import logging
import os
import time
from typing import Any

from cleardrive.core.config import (
    DEFAULT_SERVO_CLOSE_ANGLE,
    DEFAULT_SERVO_DETACH_AFTER_MOVE,
    DEFAULT_SERVO_ENABLED,
    DEFAULT_SERVO_GPIO_PIN,
    DEFAULT_SERVO_MAX_PULSE_WIDTH,
    DEFAULT_SERVO_MIN_PULSE_WIDTH,
    DEFAULT_SERVO_OPEN_ANGLE,
    DEFAULT_SERVO_SETTLE_SECONDS,
    env_bool,
    env_float,
    env_int,
)

logger = logging.getLogger(__name__)

_pin_factory_configured = False


def _configure_pin_factory() -> None:
    """Prefer lgpio hardware timing on Raspberry Pi (must run before creating devices)."""
    global _pin_factory_configured
    if _pin_factory_configured:
        return

    os.environ.setdefault("GPIOZERO_PIN_FACTORY", "lgpio")

    try:
        from gpiozero import Device
        from gpiozero.pins.lgpio import LGPIOFactory
    except ImportError:
        logger.warning(
            "lgpio not available; gpiozero falls back to software PWM (servo jitter, high CPU). "
            "In your venv run: pip install lgpio  "
            "Or on Raspberry Pi OS: sudo apt install -y swig liblgpio-dev python3-lgpio"
        )
        _pin_factory_configured = True
        return

    if not isinstance(Device.pin_factory, LGPIOFactory):
        Device.pin_factory = LGPIOFactory()

    _pin_factory_configured = True


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
        settle_seconds: float | None = None,
        detach_after_move: bool | None = None,
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
        self.settle_seconds = (
            settle_seconds
            if settle_seconds is not None
            else env_float("SERVO_SETTLE_SECONDS", DEFAULT_SERVO_SETTLE_SECONDS)
        )
        self.detach_after_move = (
            detach_after_move
            if detach_after_move is not None
            else env_bool("SERVO_DETACH_AFTER_MOVE", DEFAULT_SERVO_DETACH_AFTER_MOVE)
        )
        self._servo: Any | None = None

    def setup(self) -> None:
        """Validate config only; PWM is attached lazily on the first move."""
        if not self.enabled:
            logger.info("Servo driver disabled (SERVO_ENABLED=false)")
            return

        logger.info(
            "Servo driver ready on GPIO %s (PWM attaches only when the ramp moves)",
            self.pin,
        )

    def teardown(self) -> None:
        if self._servo is not None:
            self._apply_angle(self.close_angle)
            self._release_pwm()
            self._servo.close()
            self._servo = None

    def move_to(self, angle: float) -> None:
        if not self.enabled:
            logger.debug("Servo (disabled): move to %.1f°", angle)
            return

        self._apply_angle(angle)
        logger.debug("Servo moved to %.1f°", angle)

    def _ensure_servo(self) -> None:
        if self._servo is not None:
            return

        _configure_pin_factory()

        from gpiozero import AngularServo

        self._servo = AngularServo(
            self.pin,
            min_angle=0,
            max_angle=180,
            min_pulse_width=self.min_pulse_width,
            max_pulse_width=self.max_pulse_width,
        )

    def _apply_angle(self, angle: float) -> None:
        self._ensure_servo()
        assert self._servo is not None
        self._servo.angle = angle
        if self.settle_seconds > 0:
            time.sleep(self.settle_seconds)
        self._release_pwm()

    def _release_pwm(self) -> None:
        """Stop the PWM signal so the servo holds position without jitter or extra load."""
        if not self.detach_after_move or self._servo is None:
            return

        self._servo.detach()


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
