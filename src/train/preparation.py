"""Dataset preparation step: generate splits and the dataset YAML config."""

import random
import shutil
from pathlib import Path

import yaml

from src.config import (
    CLASS_NAMES,
    IMAGE_EXTENSIONS,
    SPLIT_SEED,
    SPLITS,
    TRAIN_SPLIT_RATIO,
    VALIDATION_SPLIT_RATIO,
)


def has_splits(dataset_path: Path) -> bool:
    """Return True if every train/valid/test split has a non-empty images dir."""
    return all(
        (images_directory := dataset_path / split / "images").is_dir()
        and any(images_directory.iterdir())
        for split in SPLITS
    )


def split_image_paths(
    image_paths: list[Path],
) -> dict[str, list[Path]]:
    """Split image paths into train/valid/test (60/20/20) with a fixed seed.

    The shuffle uses ``SPLIT_SEED`` so the partition is reproducible across runs.
    """
    shuffled = sorted(image_paths)
    random.Random(SPLIT_SEED).shuffle(shuffled)

    image_count = len(shuffled)
    train_end = round(image_count * TRAIN_SPLIT_RATIO)
    validation_end = train_end + round(image_count * VALIDATION_SPLIT_RATIO)
    return {
        "train": shuffled[:train_end],
        "valid": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }


def generate_splits(dataset_path: Path) -> None:
    """Generate train/valid/test splits from a flat images/labels pool.

    Reads every image under ``images/`` and copies each image and its matching ``labels/<name>.txt`` into the per-split directories expected by Ultralytics.
    """
    source_images_directory = dataset_path / "images"
    source_labels_directory = dataset_path / "labels"
    image_paths = [
        image_path
        for image_path in source_images_directory.iterdir()
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    for split, split_image_paths_for_split in split_image_paths(image_paths).items():
        split_images_directory = dataset_path / split / "images"
        split_labels_directory = dataset_path / split / "labels"
        split_images_directory.mkdir(parents=True, exist_ok=True)
        split_labels_directory.mkdir(parents=True, exist_ok=True)
        for image_path in split_image_paths_for_split:
            shutil.copy2(image_path, split_images_directory / image_path.name)
            label_path = source_labels_directory / f"{image_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, split_labels_directory / label_path.name)


def prepare_dataset(dataset_path: Path) -> Path:
    """Prepare the dataset and return its Ultralytics YAML config path.

    If a ``data.yaml`` already exists it is returned unchanged. Otherwise the train/valid/test splits are generated (60/20/20, fixed seed) when absent and a ``config.yaml`` is generated with the six finger-count classes.
    """
    data_yaml_path = dataset_path / "data.yaml"
    if data_yaml_path.exists():
        return data_yaml_path

    if not has_splits(dataset_path) and (dataset_path / "images").is_dir():
        generate_splits(dataset_path)

    config_yaml_path = dataset_path / "config.yaml"
    dataset_config = {
        "path": str(dataset_path),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    with config_yaml_path.open("w") as config_file:
        yaml.safe_dump(dataset_config, config_file, sort_keys=False)
    return config_yaml_path
