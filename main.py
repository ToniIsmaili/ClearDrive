import cv2

from cleardrive.modules import PlateDetectionModule, WebcamModule


def main() -> None:
    """Demo: capture webcam frames, detect plates, and display results until you press 'q'."""
    print("Opening webcam and loading plate detector...", flush=True)

    with WebcamModule(device_id=0) as camera, PlateDetectionModule() as detector:
        print("Ready. Press 'q' in the preview window to quit.", flush=True)

        while True:
            frame = camera.process()
            plate = detector.process(frame)

            display = frame.data.copy()
            if plate is not None:
                x, y, w, h = plate.metadata["bbox"]
                confidence = plate.metadata["confidence"]
                method = plate.metadata.get("method", "yolo")
                label = f"{confidence:.0%} ({method})"
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    display,
                    label,
                    (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("ClearDrive - Plate", plate.data)

            cv2.imshow("ClearDrive - Webcam", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
