"""YOLO inference: decode-free prediction on a Pillow image."""

from dataclasses import dataclass
from functools import lru_cache

from PIL.Image import Image
from ultralytics import YOLO

from src.config import (
    CLASS_TO_FINGER_COUNT,
    DEVICE,
    INFERENCE_CONFIDENCE,
    INFERENCE_IMAGE_SIZE,
    INFERENCE_IOU,
    INFERENCE_MAX_DETECTIONS,
    MODEL_PATH,
)


@dataclass(frozen=True)
class Detection:
    """A single detected hand with its bounding box and predicted class."""

    x_minimum: float
    y_minimum: float
    x_maximum: float
    y_maximum: float
    class_name: str
    confidence: float


@dataclass(frozen=True)
class PredictionResult:
    """All detections for one frame and the total number of fingers shown."""

    detections: list[Detection]
    total_fingers: int


@lru_cache(maxsize=1)
def get_model() -> YOLO:
    """Return the YOLO model, loading it once and caching it for reuse."""
    return YOLO(MODEL_PATH)


def count_fingers(detections: list[Detection]) -> int:
    """Sum the finger counts of every detected hand."""
    return sum(CLASS_TO_FINGER_COUNT[detection.class_name] for detection in detections)


def predict_image(image: Image) -> PredictionResult:
    """Run the model on a single image and return detections with the total."""
    model = get_model()
    results = model.predict(
        source=image,
        imgsz=INFERENCE_IMAGE_SIZE,
        conf=INFERENCE_CONFIDENCE,
        iou=INFERENCE_IOU,
        max_det=INFERENCE_MAX_DETECTIONS,
        device=DEVICE,
        verbose=False,
    )
    result = results[0]
    boxes = result.boxes if result.boxes is not None else []

    detections: list[Detection] = []
    for box in boxes:
        x_minimum, y_minimum, x_maximum, y_maximum = box.xyxy[0].tolist()
        detections.append(
            Detection(
                x_minimum=x_minimum,
                y_minimum=y_minimum,
                x_maximum=x_maximum,
                y_maximum=y_maximum,
                class_name=result.names[int(box.cls)],
                confidence=float(box.conf),
            )
        )

    return PredictionResult(
        detections=detections,
        total_fingers=count_fingers(detections),
    )
