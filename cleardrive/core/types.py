from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ImageFrame:
    """Standard image output passed between pipeline modules."""

    data: np.ndarray
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def channels(self) -> int:
        return 1 if self.data.ndim == 2 else self.data.shape[2]
