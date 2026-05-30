"""Console reporting helpers for the training pipeline orchestrator."""

import torch

from src.config import TRAINING_CONFIG, RoboflowSettings
from src.train.evaluation import EvaluationResult


def report_config(settings: RoboflowSettings) -> None:
    """Print the dataset coordinates and the available training hardware."""
    cuda_available = torch.cuda.is_available()
    hardware = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    print("Roboflow project:", settings.project)
    print("Roboflow version:", settings.version)
    print("CUDA available:", cuda_available)
    print("Hardware:", hardware)


def report_training_start(project: str) -> None:
    """Print the project name and the training configuration."""
    print("\nTraining:", project)
    print("Configuration:")
    for name, value in TRAINING_CONFIG.items():
        print(f"  {name}: {value}")


def report_evaluation(result: EvaluationResult) -> None:
    """Print the final evaluation summary."""
    print("\nFinal summary:")
    for name, value in result.as_row().items():
        print(f"  {name}: {value}")
