from abc import ABC, abstractmethod
from typing import Any

from cleardrive.core.types import ImageFrame


class Module(ABC):
    """Base class for all pipeline modules."""

    name: str = "module"

    @abstractmethod
    def process(self, frame: ImageFrame | None = None) -> ImageFrame | None:
        """Run this module. Source modules ignore *frame*; downstream modules consume it."""

    def setup(self) -> None:
        """Optional one-time initialization before processing."""

    def teardown(self) -> None:
        """Optional cleanup after processing."""
