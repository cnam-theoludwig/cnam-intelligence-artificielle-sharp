"""Training pipeline entry point.

Runs the ML pipeline steps from the project subject in order:
extraction, validation, preparation, training and evaluation.
"""

import datetime
import time
from pathlib import Path

from src.config import DEVICE, ROBOFLOW_SETTINGS, RUNS_DIRECTORY
from src.train.evaluation import evaluate_model
from src.train.extraction import extract_dataset
from src.train.preparation import prepare_dataset
from src.train.reporting import (
    report_config,
    report_evaluation,
    report_pipeline_end,
    report_pipeline_start,
    report_training_start,
)
from src.train.training import train_model
from src.train.validation import validate_dataset


def main() -> None:
    """Run the full finger-detection training pipeline end to end."""
    started_at = datetime.datetime.now().astimezone()
    start_time = time.perf_counter()
    report_pipeline_start(started_at)
    report_config(ROBOFLOW_SETTINGS)

    # 1. Extraction
    dataset_path = extract_dataset(ROBOFLOW_SETTINGS)

    # 2. Validation
    validate_dataset(dataset_path)

    # 3. Preparation
    dataset_yaml_path = prepare_dataset(dataset_path)

    # 4. Training
    report_training_start()
    best_weights_path = train_model(dataset_yaml_path, DEVICE)

    # 5. Evaluation
    runs_project_directory = Path(RUNS_DIRECTORY) / ROBOFLOW_SETTINGS.project
    result = evaluate_model(
        best_weights_path,
        dataset_yaml_path,
        dataset_path,
        DEVICE,
        runs_project_directory,
    )
    report_evaluation(result)

    elapsed_seconds = time.perf_counter() - start_time
    ended_at = datetime.datetime.now().astimezone()
    report_pipeline_end(started_at, ended_at, elapsed_seconds)


if __name__ == "__main__":
    main()
