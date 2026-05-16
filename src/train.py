import os
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO

load_dotenv()

ROBOFLOW_API_KEY = os.environ["ROBOFLOW_API_KEY"]
ROBOFLOW_WORKSPACE = os.environ["ROBOFLOW_WORKSPACE"]
ROBOFLOW_PROJECT = os.environ["ROBOFLOW_PROJECT"]
ROBOFLOW_VERSION = int(os.environ["ROBOFLOW_VERSION"])

DATASETS_DIR = "datasets"
RUNS_DIR = "runs"
DATASET_FORMAT = "yolo26"

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


def resolve_dataset_yaml(dataset_path: Path) -> Path:
    data_yaml_path = dataset_path / "data.yaml"
    if data_yaml_path.exists():
        return data_yaml_path

    config_yaml_path = dataset_path / "config.yaml"
    class_names = ["sharp"]
    dataset_config = {
        "path": str(dataset_path),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }
    with config_yaml_path.open("w") as config_file:
        yaml.safe_dump(dataset_config, config_file, sort_keys=False)
    return config_yaml_path


device = 0 if torch.cuda.is_available() else "cpu"

print("ROBOFLOW_VERSION:", ROBOFLOW_VERSION)
print("CUDA available:", torch.cuda.is_available())
print(
    "Hardware:",
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
)

roboflow_client = Roboflow(api_key=ROBOFLOW_API_KEY)
project = roboflow_client.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
version = project.version(ROBOFLOW_VERSION)

dataset = version.download(
    DATASET_FORMAT, location=f"{DATASETS_DIR}/{ROBOFLOW_PROJECT}"
)
runs_project_dir = Path(RUNS_DIR) / ROBOFLOW_PROJECT
yaml_path = resolve_dataset_yaml(Path(dataset.location))

print("\n==============================")
print("Training:", TRAINING_CONFIG["name"])
print("==============================")

model = YOLO(str(TRAINING_CONFIG["model"]))

train_results = model.train(
    data=str(yaml_path),
    device=device,
    plots=True,
    **{
        key: value
        for key, value in TRAINING_CONFIG.items()
        if key not in ("model", "name")
    },
    name=TRAINING_CONFIG["name"],
)

best_weights_path = Path(train_results.save_dir) / "weights" / "best.pt"
best_model = YOLO(str(best_weights_path))

metrics = best_model.val(
    data=str(yaml_path),
    split="test",
    device=device,
    plots=True,
    augment=True,
    iou=0.6,
    conf=0.001,
    project=str(Path(train_results.save_dir).parent),
    name=f"{TRAINING_CONFIG['name']}_test_eval",
    exist_ok=True,
)

test_images_dir = Path(dataset.location) / "test" / "images"
start_time = time.time()
predictions = best_model.predict(
    source=str(test_images_dir),
    device=device,
    imgsz=TRAINING_CONFIG["imgsz"],
    save=False,
    verbose=False,
)
elapsed = time.time() - start_time
frames_per_second = len(predictions) / elapsed if elapsed > 0 else 0.0

results_summary = [
    {
        "model": TRAINING_CONFIG["name"],
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
        "precision": metrics.box.mp,
        "recall": metrics.box.mr,
        "fps": frames_per_second,
    }
]

print("\n===== FINAL SUMMARY =====")
for result in results_summary:
    print(result)

summary_dataframe = pd.DataFrame(results_summary)
summary_csv_path = runs_project_dir / "summary_results.csv"
summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
summary_dataframe.to_csv(summary_csv_path, index=False)
print(summary_dataframe)
