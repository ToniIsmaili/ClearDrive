import sys
from collections.abc import Iterator
from datetime import datetime, timezone

import cv2
import numpy as np

from cleardrive.core.module import Module
from cleardrive.core.types import ImageFrame


def _default_capture_backend() -> int | None:
    """DirectShow avoids long hangs when opening webcams on Windows."""
    if sys.platform == "win32":
        return cv2.CAP_DSHOW
    return None


class WebcamModule(Module):
    """Captures images from the system webcam for downstream pipeline modules."""

    name = "webcam"

    def __init__(
        self,
        device_id: int = 0,
        width: int | None = None,
        height: int | None = None,
        backend: int | None = None,
        warmup_frames: int = 3,
    ) -> None:
        self.device_id = device_id
        self.width = width
        self.height = height
        self.backend = backend if backend is not None else _default_capture_backend()
        self.warmup_frames = warmup_frames
        self._capture: cv2.VideoCapture | None = None

    def setup(self) -> None:
        if self._capture is not None:
            return

        if self.backend is not None:
            self._capture = cv2.VideoCapture(self.device_id, self.backend)
        else:
            self._capture = cv2.VideoCapture(self.device_id)

        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            raise RuntimeError(f"Could not open webcam at device index {self.device_id}")

        if self.width is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        for _ in range(self.warmup_frames):
            self._capture.read()

    def teardown(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def capture(self) -> ImageFrame:
        """Grab a single frame from the webcam."""
        self.setup()

        assert self._capture is not None
        success, frame = self._capture.read()
        if not success or frame is None:
            raise RuntimeError("Failed to read frame from webcam")

        return self._to_image_frame(frame)

    def process(self, frame: ImageFrame | None = None) -> ImageFrame:
        """Capture and return one frame (ignores any input)."""
        return self.capture()

    def stream(self) -> Iterator[ImageFrame]:
        """Yield frames continuously until the camera is closed."""
        self.setup()
        try:
            while True:
                yield self.capture()
        finally:
            self.teardown()

    def _to_image_frame(self, frame: np.ndarray) -> ImageFrame:
        actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if self._capture else frame.shape[1]
        actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self._capture else frame.shape[0]

        return ImageFrame(
            data=frame,
            timestamp=datetime.now(timezone.utc),
            source=self.name,
            metadata={
                "device_id": self.device_id,
                "width": actual_width,
                "height": actual_height,
            },
        )

    def __enter__(self) -> "WebcamModule":
        self.setup()
        return self

    def __exit__(self, *_: object) -> None:
        self.teardown()
