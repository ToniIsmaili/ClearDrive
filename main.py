import argparse
import os
import signal
import time

import cv2

from cleardrive.core.config import (
    DEFAULT_HEADLESS_FRAME_INTERVAL_SECONDS,
    env_float,
)
from cleardrive.modules import (
    EventModule,
    PlateDetectionModule,
    PlateOCRModule,
    ServoModule,
    WebcamModule,
    WhiteListModule,
)


def _resolve_headless(cli_headless: bool | None) -> bool:
    """Return True when preview windows should be skipped."""
    if cli_headless is not None:
        return cli_headless

    env = os.getenv("CLEARDRIVE_HEADLESS", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False

    display = os.getenv("DISPLAY", "").strip()
    return display == ""


def main() -> None:
    """Capture webcam frames, detect plates, and optionally show a live preview."""
    parser = argparse.ArgumentParser(description="ClearDrive license plate pipeline")
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip OpenCV preview windows (default: on when DISPLAY is unset)",
    )
    args = parser.parse_args()
    headless = _resolve_headless(args.headless)
    frame_interval = env_float(
        "CLEARDRIVE_FRAME_INTERVAL_SECONDS",
        DEFAULT_HEADLESS_FRAME_INTERVAL_SECONDS if headless else 0.0,
    )

    print("Opening webcam and loading plate detector...", flush=True)

    stop = False

    def _request_stop(*_: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    with (
        WebcamModule(device_id=0) as camera,
        PlateDetectionModule() as detector,
        PlateOCRModule() as ocr,
        WhiteListModule() as whitelist,
        ServoModule() as servo,
        EventModule() as events,
    ):
        if headless:
            print("Running headless. Press Ctrl+C to quit.", flush=True)
            if frame_interval > 0:
                fps = 1.0 / frame_interval
                print(
                    f"Processing ~{fps:.1f} frame(s)/s "
                    f"(CLEARDRIVE_FRAME_INTERVAL_SECONDS={frame_interval}).",
                    flush=True,
                )
        else:
            print("Ready. Press 'q' in the preview window to quit.", flush=True)

        last_plate_text: str | None = None

        while not stop:
            frame = camera.process()
            plate = detector.process(frame)

            if plate is not None:
                x, y, w, h = plate.metadata["bbox"]
                confidence = plate.metadata["confidence"]
                method = plate.metadata.get("method", "yolo")
                ocr_result = ocr.process(plate)
                whitelist_result = whitelist.process(ocr_result) if ocr_result is not None else None
                servo_result = (
                    servo.process(whitelist_result) if whitelist_result is not None else None
                )
                event_result = events.process(servo_result) if servo_result is not None else None

                plate_text = (
                    event_result.metadata["plate"]
                    if event_result is not None
                    else None
                )
                is_whitelisted = (
                    event_result.metadata["whitelisted"]
                    if event_result is not None
                    else False
                )

                if plate_text is not None and plate_text != last_plate_text:
                    status = "whitelisted" if is_whitelisted else "not whitelisted"
                    print(f"Plate: {plate_text} ({status})", flush=True)
                    if event_result is not None:
                        if event_result.metadata.get("servo_opened"):
                            delay = event_result.metadata["servo_close_delay_seconds"]
                            print(f"  Ramp opened (closes in {delay}s)", flush=True)
                        elif event_result.metadata.get("servo_error"):
                            print(
                                f"  Servo failed: {event_result.metadata['servo_error']}",
                                flush=True,
                            )
                        if event_result.metadata.get("event_published"):
                            print(
                                f"  SNS event published (message id: "
                                f"{event_result.metadata['sns_message_id']})",
                                flush=True,
                            )
                        elif event_result.metadata.get("event_error"):
                            print(
                                f"  SNS publish failed: {event_result.metadata['event_error']}",
                                flush=True,
                            )
                    last_plate_text = plate_text

                if not headless:
                    if plate_text is not None:
                        label = f"{plate_text} ({'OK' if is_whitelisted else 'DENIED'})"
                        color = (0, 255, 0) if is_whitelisted else (0, 0, 255)
                    else:
                        label = f"{confidence:.0%} ({method})"
                        color = (0, 200, 255)

                    display = frame.data.copy()
                    cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(
                        display,
                        label,
                        (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )
                    ocr_view = ocr.preprocess(plate.data)
                    if ocr_view is not None:
                        cv2.imshow("ClearDrive - Plate", ocr_view)
                    cv2.imshow("ClearDrive - Webcam", display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            else:
                last_plate_text = None
                if not headless:
                    cv2.imshow("ClearDrive - Webcam", frame.data)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            if frame_interval > 0:
                time.sleep(frame_interval)

    if not headless:
        cv2.destroyAllWindows()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
