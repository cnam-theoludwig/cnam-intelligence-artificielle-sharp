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

device = 0 if torch.cuda.is_available() else "cpu"

print("CUDA disponible :", torch.cuda.is_available())
print(
    "GPU :",
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Pas de GPU",
)

roboflow_client = Roboflow(api_key=ROBOFLOW_API_KEY)
project = roboflow_client.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
version = project.version(ROBOFLOW_VERSION)

DATASETS_DIR = "datasets"
RUNS_DIR = "runs"

dataset = version.download("yolov8", location=f"{DATASETS_DIR}/{ROBOFLOW_PROJECT}")
runs_project_dir = f"{RUNS_DIR}/{ROBOFLOW_PROJECT}"

dataset_path = Path(dataset.location)
data_yaml_path = dataset_path / "data.yaml"
config_yaml_path = dataset_path / "config.yaml"

if data_yaml_path.exists():
    yaml_path = data_yaml_path
else:
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

    yaml_path = config_yaml_path

training_configs: list[dict[str, str | int]] = [
    {"model": "yolo26n.pt", "batch": 8, "imgsz": 512, "name": "yolo26n_512"},
    {"model": "yolo26s.pt", "batch": 8, "imgsz": 640, "name": "yolo26s_640"},
    {"model": "yolo26m.pt", "batch": 4, "imgsz": 640, "name": "yolo26m_640"},
]

results_summary = []

for training_config in training_configs:
    print("\n==============================")
    print("Entraînement :", training_config["name"])
    print("==============================")

    model = YOLO(str(training_config["model"]))

    train_results = model.train(
        data=str(yaml_path),
        epochs=200,
        imgsz=512,
        batch=8,
        device=device,
        patience=40,
        close_mosaic=20,

        degrees=20,
        translate=0.15,
        scale=0.6,
        shear=5,
        perspective=0.0005,
        fliplr=0.5,
        hsv_h=0.02,
        hsv_s=0.5,
        hsv_v=0.4,
        mosaic=1.0,
        mixup=0.0,

        lr0=0.0005,
        lrf=0.01,
        optimizer="AdamW",
        plots=True
    )

    best_weights_path = Path(train_results.save_dir) / "weights" / "best.pt"
    best_model = YOLO(str(best_weights_path))

    metrics = best_model.val(
        data=str(yaml_path),
        split="test",
        device=device,
        plots=True,
        project=str(Path(train_results.save_dir).parent),
        name=f"{training_config['name']}_test_eval",
        exist_ok=True,
    )

    start_time = time.time()

    predictions = best_model.predict(
        source=f"{dataset.location}/test/images",
        device=device,
        imgsz=training_config["imgsz"],
        save=False,
        verbose=False,
    )

    end_time = time.time()

    frames_per_second = len(predictions) / (end_time - start_time)

    results_summary.append(
        {
            "model": training_config["name"],
            "mAP50": metrics.box.map50,
            "mAP50-95": metrics.box.map,
            "precision": metrics.box.mp,
            "recall": metrics.box.mr,
            "fps": frames_per_second,
        }
    )

print("\n===== RÉSUMÉ FINAL =====")
for result in results_summary:
    print(result)

summary_dataframe = pd.DataFrame(results_summary)
summary_csv_path = Path(runs_project_dir) / "summary_results.csv"
summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
summary_dataframe.to_csv(summary_csv_path, index=False)
print(summary_dataframe)
