"""Tests for the dataset preparation step (split and YAML generation)."""

from pathlib import Path

import yaml

from src.config import CLASS_NAMES
from src.train.preparation import prepare_dataset, split_image_paths


def test_resolve_returns_existing_data_yaml(tmp_path: Path) -> None:
    """An existing data.yaml is returned and no config.yaml is generated."""
    # Arrange - Given
    data_yaml_path = tmp_path / "data.yaml"
    data_yaml_path.write_text("nc: 6\n")

    # Act - When
    resolved_path = prepare_dataset(tmp_path)

    # Assert - Then
    assert resolved_path == data_yaml_path
    assert not (tmp_path / "config.yaml").exists()


def test_generated_config_has_six_classes(tmp_path: Path) -> None:
    """The generated config.yaml declares the six finger-count classes."""
    # Arrange - Given
    expected_config_path = tmp_path / "config.yaml"

    # Act - When
    resolved_path = prepare_dataset(tmp_path)
    dataset_config = yaml.safe_load(resolved_path.read_text())

    # Assert - Then
    assert resolved_path == expected_config_path
    assert dataset_config["nc"] == 6
    assert dataset_config["names"] == CLASS_NAMES


def test_generated_config_paths(tmp_path: Path) -> None:
    """The generated config.yaml uses the expected split image directories."""
    # Arrange - Given
    expected_path = str(tmp_path)

    # Act - When
    resolved_path = prepare_dataset(tmp_path)
    dataset_config = yaml.safe_load(resolved_path.read_text())

    # Assert - Then
    assert dataset_config["path"] == expected_path
    assert dataset_config["train"] == "train/images"
    assert dataset_config["val"] == "valid/images"
    assert dataset_config["test"] == "test/images"


def test_split_image_paths_uses_60_20_20_ratios() -> None:
    """Ten image paths split into 6 train / 2 valid / 2 test."""
    # Arrange - Given
    image_paths = [Path(f"image_{index}.jpg") for index in range(10)]

    # Act - When
    splits = split_image_paths(image_paths)

    # Assert - Then
    assert len(splits["train"]) == 6
    assert len(splits["valid"]) == 2
    assert len(splits["test"]) == 2


def test_split_image_paths_is_deterministic() -> None:
    """The fixed seed makes the partition reproducible and non-overlapping."""
    # Arrange - Given
    image_paths = [Path(f"image_{index}.jpg") for index in range(10)]

    # Act - When
    first_splits = split_image_paths(image_paths)
    second_splits = split_image_paths(image_paths)

    # Assert - Then
    assert first_splits == second_splits
    all_split_paths = (
        first_splits["train"] + first_splits["valid"] + first_splits["test"]
    )
    assert sorted(all_split_paths) == sorted(image_paths)


def test_generate_splits_from_flat_pool(tmp_path: Path) -> None:
    """A flat images/labels pool is partitioned into train/valid/test dirs."""
    # Arrange - Given
    images_directory = tmp_path / "images"
    labels_directory = tmp_path / "labels"
    images_directory.mkdir()
    labels_directory.mkdir()
    for index in range(10):
        (images_directory / f"image_{index}.jpg").write_bytes(b"fake")
        (labels_directory / f"image_{index}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    # Act - When
    prepare_dataset(tmp_path)

    # Assert - Then
    train_images = list((tmp_path / "train" / "images").iterdir())
    valid_images = list((tmp_path / "valid" / "images").iterdir())
    test_images = list((tmp_path / "test" / "images").iterdir())
    assert len(train_images) == 6
    assert len(valid_images) == 2
    assert len(test_images) == 2
    assert len(list((tmp_path / "train" / "labels").iterdir())) == 6
