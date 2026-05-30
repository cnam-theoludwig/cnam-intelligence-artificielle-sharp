"""Dataset extraction step: download the annotated dataset from Roboflow."""

from pathlib import Path

from roboflow import Roboflow

from src.config import DATASET_FORMAT, DATASETS_DIRECTORY, RoboflowSettings


def extract_dataset(
    settings: RoboflowSettings,
    datasets_directory: str = DATASETS_DIRECTORY,
    dataset_format: str = DATASET_FORMAT,
) -> Path:
    """Download the annotated dataset (images and labels) from Roboflow.

    Returns the local path where the dataset was extracted.
    """
    roboflow_client = Roboflow(api_key=settings.api_key)
    project = roboflow_client.workspace(settings.workspace).project(settings.project)
    dataset_version = project.version(settings.version)
    dataset = dataset_version.download(
        dataset_format,
        location=f"{datasets_directory}/{settings.project}",
    )
    return Path(dataset.location)
