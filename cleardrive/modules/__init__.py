from cleardrive.modules.camera.webcam import WebcamModule
from cleardrive.modules.detection.plate import PlateDetectionModule
from cleardrive.modules.event.event import EventModule
from cleardrive.modules.recognition.plate_ocr import PlateOCRModule
from cleardrive.modules.whitelist.whitelist import WhiteListModule

__all__ = [
    "EventModule",
    "PlateDetectionModule",
    "PlateOCRModule",
    "WebcamModule",
    "WhiteListModule",
]
