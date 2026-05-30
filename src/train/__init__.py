"""Structured YOLO finger-detection training pipeline."""

from src.config import (
    CLASS_NAMES,
    CLASS_TO_FINGER_COUNT,
    DEVICE,
    ROBOFLOW_SETTINGS,
    TRAINING_CONFIG,
    RoboflowSettings,
)
from src.train.evaluation import EvaluationResult, evaluate_model
from src.train.extraction import extract_dataset
from src.train.preparation import prepare_dataset
from src.train.training import train_model
from src.train.validation import validate_dataset

__all__ = [
    "CLASS_NAMES",
    "CLASS_TO_FINGER_COUNT",
    "DEVICE",
    "ROBOFLOW_SETTINGS",
    "TRAINING_CONFIG",
    "EvaluationResult",
    "RoboflowSettings",
    "evaluate_model",
    "extract_dataset",
    "prepare_dataset",
    "train_model",
    "validate_dataset",
]
