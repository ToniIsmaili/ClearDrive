from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.request import urlretrieve

import cv2
import numpy as np

from cleardrive.core.config import DEFAULT_ENABLE_YOLO, env_bool
from cleardrive.core.module import Module
from cleardrive.core.types import ImageFrame

if TYPE_CHECKING:
    from ultralytics import YOLO

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MODEL_PATH = _PROJECT_ROOT / "models" / "yolov8_license_plate.pt"
_DEFAULT_MODEL_URL = (
    "https://huggingface.co/orionwambert/yolov8-license-plate-detection/resolve/main/best.pt"
)

_MIN_PLATE_ASPECT = 1.5
_MAX_PLATE_ASPECT = 7.0
_MIN_PLATE_AREA_RATIO = 0.005
_MAX_PLATE_AREA_RATIO = 0.75


def _default_model_path() -> Path:
    return _DEFAULT_MODEL_PATH


class PlateDetectionModule(Module):
    """Detects license plates with YOLOv8 and returns a cropped plate region."""

    name = "plate_detection"

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence: float = 0.10,
        device: str | None = None,
        auto_download: bool = True,
        imgsz: int = 640,
        use_yolo: bool | None = None,
        use_contour_fallback: bool = True,
    ) -> None:
        self.model_path = Path(model_path) if model_path else _default_model_path()
        self.confidence = confidence
        self.device = device
        self.auto_download = auto_download
        self.imgsz = imgsz
        self.use_yolo = (
            use_yolo
            if use_yolo is not None
            else env_bool("CLEARDRIVE_ENABLE_YOLO", DEFAULT_ENABLE_YOLO)
        )
        self.use_contour_fallback = use_contour_fallback
        self._model: YOLO | None = None

    def setup(self) -> None:
        if not self.use_yolo or self._model is not None:
            return

        if not self.model_path.exists():
            if self.auto_download and self.model_path == _default_model_path():
                self._download_default_model()
            else:
                raise FileNotFoundError(
                    f"YOLOv8 model not found at {self.model_path}. "
                    "Provide a trained license-plate weights file via model_path=."
                )

        from ultralytics import YOLO

        self._model = YOLO(str(self.model_path))

    def teardown(self) -> None:
        self._model = None

    def process(self, frame: ImageFrame | None = None) -> ImageFrame | None:
        """Return a cropped license plate if one is found, otherwise None."""
        if frame is None:
            return None

        detection = self._detect_plate(frame.data)
        if detection is None:
            return None

        x1, y1, x2, y2, confidence, method = detection
        crop = frame.data[y1:y2, x1:x2].copy()

        return ImageFrame(
            data=crop,
            timestamp=datetime.now(timezone.utc),
            source=self.name,
            metadata={
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "confidence": confidence,
                "method": method,
                "input_source": frame.source,
            },
        )

    def detect(self, image: np.ndarray) -> np.ndarray | None:
        """Detect a license plate in a raw BGR image and return the crop, or None."""
        detection = self._detect_plate(image)
        if detection is None:
            return None

        x1, y1, x2, y2, _, _ = detection
        return image[y1:y2, x1:x2].copy()

    def _detect_plate(
        self, image: np.ndarray
    ) -> tuple[int, int, int, int, float, str] | None:
        if self.use_yolo:
            for variant in (image, self._enhance_for_detection(image)):
                detection = self._detect_with_yolo(variant)
                if detection is not None:
                    return (*detection, "yolo")

        if self.use_contour_fallback:
            detection = self._detect_with_contours(image)
            if detection is not None:
                return (*detection, "contour")

        return None

    def _detect_with_yolo(
        self, image: np.ndarray
    ) -> tuple[int, int, int, int, float] | None:
        self.setup()
        assert self._model is not None

        results = self._model.predict(
            source=image,
            conf=self.confidence,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )

        best_box: tuple[int, int, int, int, float] | None = None
        best_confidence = -1.0
        height, width = image.shape[:2]

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls.item())
                class_name = names[cls_id].lower()
                if not self._is_plate_class(class_name):
                    continue

                confidence = float(box.conf.item())
                if confidence <= best_confidence:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(width, int(x2))
                y2 = min(height, int(y2))

                if x2 <= x1 or y2 <= y1:
                    continue

                best_confidence = confidence
                best_box = (x1, y1, x2, y2, confidence)

        return best_box

    def _detect_with_contours(
        self, image: np.ndarray
    ) -> tuple[int, int, int, int, float] | None:
        height, width = image.shape[:2]
        image_area = height * width
        best_box: tuple[int, int, int, int, float] | None = None
        best_score = 0.0

        for edges in self._contour_edge_maps(image):
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < image_area * _MIN_PLATE_AREA_RATIO:
                    continue
                if area > image_area * _MAX_PLATE_AREA_RATIO:
                    continue

                x, y, box_w, box_h = cv2.boundingRect(contour)
                if box_h == 0:
                    continue

                aspect = box_w / float(box_h)
                if aspect < _MIN_PLATE_ASPECT or aspect > _MAX_PLATE_ASPECT:
                    continue

                rect_area = box_w * box_h
                extent = area / float(rect_area)
                if extent < 0.45:
                    continue

                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                rectangularity = 1.0 if len(approx) == 4 else 0.85

                aspect_target = 4.0
                aspect_score = 1.0 - min(abs(aspect - aspect_target) / aspect_target, 1.0)
                score = extent * rectangularity * (0.5 + 0.5 * aspect_score)
                if score <= best_score:
                    continue

                best_score = score
                best_box = (x, y, x + box_w, y + box_h, min(0.95, score))

        return best_box

    @staticmethod
    def _contour_edge_maps(image: np.ndarray) -> list[np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        maps: list[np.ndarray] = [cv2.Canny(gray, 30, 200)]

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        adaptive = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )
        maps.append(adaptive)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, blackhat = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        maps.append(blackhat)

        return maps

    @staticmethod
    def _enhance_for_detection(image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        enhanced = cv2.merge([l_channel, a_channel, b_channel])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _is_plate_class(class_name: str) -> bool:
        return "plate" in class_name or class_name in {"number_plate", "license"}

    def _download_default_model(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(_DEFAULT_MODEL_URL, self.model_path)

    def __enter__(self) -> "PlateDetectionModule":
        self.setup()
        return self

    def __exit__(self, *_: object) -> None:
        self.teardown()
