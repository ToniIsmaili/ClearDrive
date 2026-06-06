import cv2

from cleardrive.modules import PlateDetectionModule, PlateOCRModule, WebcamModule


def main() -> None:
    """Demo: capture webcam frames, detect plates, and display results until you press 'q'."""
    print("Opening webcam and loading plate detector...", flush=True)

    with (
        WebcamModule(device_id=0) as camera,
        PlateDetectionModule() as detector,
        PlateOCRModule() as ocr,
    ):
        print("Ready. Press 'q' in the preview window to quit.", flush=True)

        last_ocr_text: str | None = None

        while True:
            frame = camera.process()
            plate = detector.process(frame)

            display = frame.data.copy()
            if plate is not None:
                x, y, w, h = plate.metadata["bbox"]
                confidence = plate.metadata["confidence"]
                method = plate.metadata.get("method", "yolo")
                result = ocr.process(plate)
                plate_text = result.metadata["text"] if result is not None else None
                if plate_text is not None and plate_text != last_ocr_text:
                    print(f"OCR: {plate_text}", flush=True)
                    last_ocr_text = plate_text
                label = plate_text or f"{confidence:.0%} ({method})"
                color = (0, 255, 0) if plate_text else (0, 200, 255)
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
                cv2.imshow("ClearDrive - Plate", plate.data)
            else:
                last_ocr_text = None

            cv2.imshow("ClearDrive - Webcam", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
