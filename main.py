import cv2

from cleardrive.modules.camera.webcam import WebcamModule


def main() -> None:
    """Demo: capture webcam frames and display them until you press 'q'."""
    print("Opening webcam...", flush=True)

    with WebcamModule(device_id=0) as camera:
        print("Webcam ready. Press 'q' in the preview window to quit.", flush=True)

        while True:
            frame = camera.capture()
            cv2.imshow("ClearDrive - Webcam", frame.data)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
