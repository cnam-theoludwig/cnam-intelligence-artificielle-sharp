"""Model evaluation step: compute test metrics, measure FPS, save a CSV summary."""

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from src.config import IMAGE_EXTENSIONS, TRAINING_CONFIG, Device


@dataclass(frozen=True)
class EvaluationResult:
    """Test-set metrics and inference speed of a trained model."""

    model_name: str
    map50: float
    map50_95: float
    precision: float
    recall: float
    frames_per_second: float

    def as_row(self) -> dict[str, str | float]:
        """Return the result as a single CSV/DataFrame row."""
        return {
            "model": self.model_name,
            "mAP50": self.map50,
            "mAP50-95": self.map50_95,
            "precision": self.precision,
            "recall": self.recall,
            "fps": self.frames_per_second,
        }


def measure_frames_per_second(
    model: YOLO,
    test_images_directory: Path,
    device: Device,
    image_size: int,
) -> float:
    """Measure inference speed in frames per second on the test images.

    Predicts one image at a time (like the live webcam) after a warm-up pass.
    Returns 0.0 when no test image is found.
    """
    image_paths = sorted(
        path
        for path in test_images_directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        return 0.0

    model.predict(
        source=str(image_paths[0]),
        device=device,
        imgsz=image_size,
        save=False,
        verbose=False,
    )

    start_time = time.perf_counter()
    for image_path in image_paths:
        model.predict(
            source=str(image_path),
            device=device,
            imgsz=image_size,
            save=False,
            verbose=False,
        )
    elapsed_seconds = time.perf_counter() - start_time
    if elapsed_seconds <= 0:
        return 0.0
    return len(image_paths) / elapsed_seconds


def evaluate_model(
    best_weights_path: Path,
    dataset_yaml_path: Path,
    dataset_path: Path,
    device: Device,
    runs_project_directory: Path,
    training_config: dict[str, str | int | float | bool] = TRAINING_CONFIG,
) -> EvaluationResult:
    """Evaluate the trained model on the test split.

    Runs `model.val()`, measures inference FPS on the test images, writes a summary CSV and returns the evaluation result.
    """
    best_model = YOLO(str(best_weights_path))
    metrics = best_model.val(
        data=str(dataset_yaml_path),
        split="test",
        device=device,
        plots=True,
        iou=0.6,
        conf=0.001,
        project=str(runs_project_directory),
        name=f"{training_config['name']}_test_eval",
        exist_ok=True,
    )

    result = EvaluationResult(
        model_name=str(training_config["name"]),
        map50=metrics.box.map50,
        map50_95=metrics.box.map,
        precision=metrics.box.mp,
        recall=metrics.box.mr,
        frames_per_second=measure_frames_per_second(
            best_model,
            dataset_path / "test" / "images",
            device,
            int(training_config["imgsz"]),
        ),
    )

    summary_csv_path = runs_project_directory / "summary_results.csv"
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result.as_row()]).to_csv(summary_csv_path, index=False)
    return result
