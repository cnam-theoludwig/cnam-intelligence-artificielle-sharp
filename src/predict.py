"""Real-time finger-count inference from a webcam stream."""

from ultralytics import YOLO

from src.config import (
    CLASS_TO_FINGER_COUNT,
    DEVICE,
    INFERENCE_CONFIDENCE,
    INFERENCE_IMAGE_SIZE,
    INFERENCE_IOU,
    MODEL_PATH,
)


def main() -> None:
    """Run the YOLO model on the webcam stream and print the finger count."""
    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=0,
        imgsz=INFERENCE_IMAGE_SIZE,
        conf=INFERENCE_CONFIDENCE,
        iou=INFERENCE_IOU,
        device=DEVICE,
        show=True,
        stream=True,
    )

    for result in results:
        boxes = result.boxes if result.boxes is not None else []
        total_fingers = sum(
            CLASS_TO_FINGER_COUNT[result.names[int(box.cls)]] for box in boxes
        )
        print(f"fingers: {total_fingers}")


if __name__ == "__main__":
    main()
