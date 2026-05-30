"""Console reporting helpers for the training pipeline orchestrator."""

import datetime

import torch

from src.config import TRAINING_CONFIG, RoboflowSettings
from src.train.evaluation import EvaluationResult


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as ``HH:MM:SS``."""
    return str(datetime.timedelta(seconds=round(seconds)))


def report_pipeline_start(started_at: datetime.datetime) -> None:
    """Print the wall-clock time at which the pipeline started."""
    print("Started at:", started_at.isoformat(timespec="seconds"))


def report_pipeline_end(
    started_at: datetime.datetime,
    ended_at: datetime.datetime,
    elapsed_seconds: float,
) -> None:
    """Print the wall-clock end time and the total pipeline duration."""
    print("\nStarted at:", started_at.isoformat(timespec="seconds"))
    print("Ended at:", ended_at.isoformat(timespec="seconds"))
    print("Total duration:", format_duration(elapsed_seconds))


def report_config(settings: RoboflowSettings) -> None:
    """Print the dataset coordinates and the available training hardware."""
    cuda_available = torch.cuda.is_available()
    hardware = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    print("Roboflow project:", settings.project)
    print("Roboflow version:", settings.version)
    print("CUDA available:", cuda_available)
    print("Hardware:", hardware)


def report_training_start() -> None:
    """Print the training configuration."""
    print("Configuration:")
    for name, value in TRAINING_CONFIG.items():
        print(f"  {name}: {value}")


def report_evaluation(result: EvaluationResult) -> None:
    """Print the final evaluation summary."""
    print("\nFinal summary:")
    for name, value in result.as_row().items():
        print(f"  {name}: {value}")
