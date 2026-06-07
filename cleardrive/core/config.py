import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")

DEFAULT_PLATE_FORMAT = "FF 0000 XX"
DEFAULT_PLATE_PREFIX_VALUES = "SK,KU,ST,TE,VE"
DEFAULT_INFLUX_BUCKET = "Whitelist"
DEFAULT_INFLUX_WHITELIST_MEASUREMENT = "whitelist"
DEFAULT_INFLUX_WHITELIST_FIELD = "plate"
DEFAULT_WHITELIST_CACHE_TTL_SECONDS = 300
DEFAULT_AWS_REGION = "us-east-1"
DEFAULT_SERVO_GPIO_PIN = 18
DEFAULT_SERVO_OPEN_ANGLE = 90
DEFAULT_SERVO_CLOSE_ANGLE = 0
DEFAULT_SERVO_CLOSE_DELAY_SECONDS = 5
DEFAULT_SERVO_MIN_PULSE_WIDTH = 0.0005
DEFAULT_SERVO_MAX_PULSE_WIDTH = 0.0025
DEFAULT_SERVO_SETTLE_SECONDS = 0.4
DEFAULT_SERVO_DETACH_AFTER_MOVE = True
DEFAULT_SERVO_ENABLED = sys.platform == "linux"
DEFAULT_ENABLE_YOLO = False
# Headless mode frame pacing; ~4 FPS keeps Pi CPU/power use stable.
DEFAULT_HEADLESS_FRAME_INTERVAL_SECONDS = 0.25


def env_str(key: str, default: str) -> str:
    return os.getenv(key, default)


def env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None:
        return default
    return int(value)


def env_optional_str(key: str) -> str | None:
    value = os.getenv(key)
    if value is None or value.strip() == "":
        return None
    return value


def env_csv_list(key: str, default: str) -> list[str]:
    value = os.getenv(key, default)
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def env_float(key: str, default: float) -> float:
    value = os.getenv(key)
    if value is None:
        return default
    return float(value)
