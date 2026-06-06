import cv2

from cleardrive.modules import PlateDetectionModule, PlateOCRModule, WebcamModule, WhiteListModule


def main() -> None:
    """Demo: capture webcam frames, detect plates, and display results until you press 'q'."""
    print("Opening webcam and loading plate detector...", flush=True)

    with (
        WebcamModule(device_id=0) as camera,
        PlateDetectionModule() as detector,
        PlateOCRModule() as ocr,
        WhiteListModule() as whitelist,
    ):
        print("Ready. Press 'q' in the preview window to quit.", flush=True)

        last_plate_text: str | None = None

        while True:
            frame = camera.process()
            plate = detector.process(frame)

            display = frame.data.copy()
            if plate is not None:
                x, y, w, h = plate.metadata["bbox"]
                confidence = plate.metadata["confidence"]
                method = plate.metadata.get("method", "yolo")
                ocr_result = ocr.process(plate)
                whitelist_result = whitelist.process(ocr_result) if ocr_result is not None else None

                plate_text = (
                    whitelist_result.metadata["plate"]
                    if whitelist_result is not None
                    else None
                )
                is_whitelisted = (
                    whitelist_result.metadata["whitelisted"]
                    if whitelist_result is not None
                    else False
                )

                if plate_text is not None and plate_text != last_plate_text:
                    status = "whitelisted" if is_whitelisted else "not whitelisted"
                    print(f"Plate: {plate_text} ({status})", flush=True)
                    last_plate_text = plate_text

                if plate_text is not None:
                    label = f"{plate_text} ({'OK' if is_whitelisted else 'DENIED'})"
                    color = (0, 255, 0) if is_whitelisted else (0, 0, 255)
                else:
                    label = f"{confidence:.0%} ({method})"
                    color = (0, 200, 255)

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
                last_plate_text = None

            cv2.imshow("ClearDrive - Webcam", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
