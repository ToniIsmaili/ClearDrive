import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")

DEFAULT_PLATE_FORMAT = "FF 0000 XX"
DEFAULT_PLATE_PREFIX_VALUES = "SK,KU,ST,TE,VE"
DEFAULT_WHITELIST_PLATES = ""


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
