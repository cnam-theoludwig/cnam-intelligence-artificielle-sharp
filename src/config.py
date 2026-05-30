"""Configuration constants and environment access for the training pipeline."""

import os
from dataclasses import dataclass

import torch
from dotenv import load_dotenv

CLASS_NAMES: list[str] = [
    "0_finger",
    "1_finger",
    "2_fingers",
    "3_fingers",
    "4_fingers",
    "5_fingers",
]

CLASS_TO_FINGER_COUNT: dict[str, int] = {
    class_name: finger_count for finger_count, class_name in enumerate(CLASS_NAMES)
}

DATASETS_DIRECTORY = "datasets"
RUNS_DIRECTORY = "runs"
DATASET_FORMAT = "yolo26"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

MODEL_PATH = "sharp.pt"
INFERENCE_IMAGE_SIZE = 960
INFERENCE_CONFIDENCE = 0.5
INFERENCE_IOU = 0.45

# Ultralytics device selector: a GPU index (0) or "cpu".
type Device = int | str
DEVICE: Device = 0 if torch.cuda.is_available() else "cpu"

SPLITS: tuple[str, str, str] = ("train", "valid", "test")

SPLIT_SEED = 42
TRAIN_SPLIT_RATIO = 0.6
VALIDATION_SPLIT_RATIO = 0.2

TRAINING_CONFIG: dict[str, str | int | float | bool] = {
    "model": "yolo26m.pt",
    "name": "yolo26m_960",
    "imgsz": 960,
    "batch": 4,
    "epochs": 300,
    "patience": 40,
    "close_mosaic": 15,
    "cache": "ram",
    "amp": True,
    "cos_lr": True,
    "warmup_epochs": 5,
    "dropout": 0.15,
    "weight_decay": 0.001,
    "degrees": 15,
    "translate": 0.15,
    "scale": 0.6,
    "shear": 5,
    "perspective": 0.001,
    "fliplr": 0.5,
    "flipud": 0.0,
    "hsv_h": 0.02,
    "hsv_s": 0.6,
    "hsv_v": 0.5,
    "mosaic": 1.0,
    "mixup": 0.2,
    "copy_paste": 0.2,
    "erasing": 0.4,
    "lr0": 0.0005,
    "lrf": 0.01,
    "optimizer": "AdamW",
}


@dataclass(frozen=True)
class RoboflowSettings:
    """Credentials and dataset coordinates for the Roboflow SDK."""

    api_key: str
    workspace: str
    project: str
    version: int


load_dotenv()

ROBOFLOW_SETTINGS = RoboflowSettings(
    api_key=os.environ.get("ROBOFLOW_API_KEY", ""),
    workspace=os.environ.get("ROBOFLOW_WORKSPACE", ""),
    project=os.environ.get("ROBOFLOW_PROJECT", ""),
    version=int(os.environ.get("ROBOFLOW_VERSION", "0")),
)
