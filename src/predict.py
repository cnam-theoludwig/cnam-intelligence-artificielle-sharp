import torch
from ultralytics import YOLO

device = 0 if torch.cuda.is_available() else "cpu"

CLASS_TO_FINGER_COUNT = {
    "0_finger": 0,
    "1_finger": 1,
    "2_fingers": 2,
    "3_fingers": 3,
    "4_fingers": 4,
    "5_fingers": 5,
}

model = YOLO("sharp.pt")

results = model.predict(
    source=0,
    imgsz=960,
    conf=0.5,
    iou=0.45,
    max_det=4,
    device=device,
    show=True,
    stream=True,
)

for result in results:
    boxes = result.boxes if result.boxes is not None else []
    total_fingers = sum(
        CLASS_TO_FINGER_COUNT[result.names[int(box.cls)]] for box in boxes
    )
    print(f"fingers: {total_fingers}")
