"""Model training step: train the YOLO model on the prepared dataset."""

from pathlib import Path

from ultralytics import YOLO

from src.config import TRAINING_CONFIG, Device


def train_model(
    dataset_yaml_path: Path,
    device: Device,
    training_config: dict[str, str | int | float | bool] = TRAINING_CONFIG,
) -> Path:
    """Train the YOLO model and return the path to the best weights (best.pt)."""
    model = YOLO(str(training_config["model"]))
    training_results = model.train(
        data=str(dataset_yaml_path),
        device=device,
        plots=True,
        **{
            key: value
            for key, value in training_config.items()
            if key not in ("model", "name")
        },
        name=training_config["name"],
    )
    return Path(training_results.save_dir) / "weights" / "best.pt"
