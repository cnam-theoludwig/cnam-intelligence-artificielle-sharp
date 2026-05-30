"""Tests for the dataset validation step."""

from pathlib import Path

import pytest
from PIL import Image

from src.train.validation import (
    is_image_corrupted,
    parse_yolo_annotation_line,
    validate_annotation_classes,
    validate_annotation_coordinates,
    validate_dataset,
    validate_label_file,
)


def test_parse_yolo_annotation_line() -> None:
    """A well-formed YOLO line parses into class index and four floats."""
    # Arrange - Given
    annotation_line = "3 0.5 0.5 0.2 0.2"

    # Act - When
    parsed = parse_yolo_annotation_line(annotation_line)

    # Assert - Then
    assert parsed == (3, 0.5, 0.5, 0.2, 0.2)


def test_validate_annotation_coordinates_accepts_valid() -> None:
    """Coordinates within [0, 1] are valid."""
    # Arrange - Given
    bounding_boxes = [(0.5, 0.5, 0.2, 0.2)]

    # Act - When
    result = validate_annotation_coordinates(bounding_boxes)

    # Assert - Then
    assert result is True


def test_validate_annotation_coordinates_rejects_above_one() -> None:
    """Coordinates greater than 1 are rejected."""
    # Arrange - Given
    bounding_boxes = [(1.2, 0.5, 0.2, 0.2)]

    # Act - When
    result = validate_annotation_coordinates(bounding_boxes)

    # Assert - Then
    assert result is False


def test_validate_annotation_coordinates_rejects_zero_size() -> None:
    """A box with zero width or height is rejected as degenerate."""
    # Arrange - Given
    bounding_boxes = [(0.5, 0.5, 0.0, 0.2)]

    # Act - When
    result = validate_annotation_coordinates(bounding_boxes)

    # Assert - Then
    assert result is False


def test_validate_annotation_coordinates_rejects_out_of_frame() -> None:
    """A box whose edge exceeds the frame is rejected."""
    # Arrange - Given
    bounding_boxes = [(0.9, 0.5, 0.5, 0.2)]

    # Act - When
    result = validate_annotation_coordinates(bounding_boxes)

    # Assert - Then
    assert result is False


def test_validate_annotation_classes_accepts_known() -> None:
    """Class indices within range are accepted."""
    # Arrange - Given
    class_indices = [0, 5]

    # Act - When
    result = validate_annotation_classes(class_indices, class_count=6)

    # Assert - Then
    assert result is True


def test_validate_annotation_classes_rejects_unknown() -> None:
    """A class index outside range is rejected."""
    # Arrange - Given
    class_indices = [6]

    # Act - When
    result = validate_annotation_classes(class_indices, class_count=6)

    # Assert - Then
    assert result is False


def test_validate_image_accepts_valid_png(tmp_path: Path) -> None:
    """A valid PNG is not flagged as corrupted."""
    # Arrange - Given
    image_path = tmp_path / "valid.png"
    Image.new("RGB", (2, 2)).save(image_path)

    # Act - When
    result = is_image_corrupted(image_path)

    # Assert - Then
    assert result is False


def test_validate_image_rejects_corrupt(tmp_path: Path) -> None:
    """A file with garbage bytes is flagged as corrupted."""
    # Arrange - Given
    image_path = tmp_path / "corrupt.png"
    image_path.write_bytes(b"not an image")

    # Act - When
    result = is_image_corrupted(image_path)

    # Assert - Then
    assert result is True


def test_validate_label_file_accepts_valid(tmp_path: Path) -> None:
    """A label file with valid coordinates and classes is accepted."""
    # Arrange - Given
    label_path = tmp_path / "valid.txt"
    label_path.write_text("0 0.5 0.5 0.2 0.2\n5 0.1 0.1 0.1 0.1\n")

    # Act - When
    result = validate_label_file(label_path, class_count=6)

    # Assert - Then
    assert result is True


def test_validate_label_file_rejects_bad_coords(tmp_path: Path) -> None:
    """A label file with a negative coordinate is rejected."""
    # Arrange - Given
    label_path = tmp_path / "bad.txt"
    label_path.write_text("0 -0.5 0.5 0.2 0.2\n")

    # Act - When
    result = validate_label_file(label_path, class_count=6)

    # Assert - Then
    assert result is False


def write_split(
    dataset_path: Path,
    split: str,
    image_name: str,
    label_content: str | None,
) -> Path:
    """Create one split with a single image and an optional label file."""
    images_directory = dataset_path / split / "images"
    labels_directory = dataset_path / split / "labels"
    images_directory.mkdir(parents=True)
    labels_directory.mkdir(parents=True)
    image_path = images_directory / image_name
    Image.new("RGB", (2, 2)).save(image_path)
    if label_content is not None:
        (labels_directory / f"{image_path.stem}.txt").write_text(label_content)
    return image_path


def test_validate_dataset_accepts_clean_dataset(tmp_path: Path) -> None:
    """A dataset with valid images and labels passes without raising."""
    # Arrange - Given
    write_split(tmp_path, "train", "image.png", "0 0.5 0.5 0.2 0.2\n")

    # Act - When / Assert - Then
    validate_dataset(tmp_path, class_count=6)


def test_validate_dataset_accepts_image_without_label(tmp_path: Path) -> None:
    """An image with no label file is a valid background image."""
    # Arrange - Given
    write_split(tmp_path, "train", "background.png", label_content=None)

    # Act - When / Assert - Then
    validate_dataset(tmp_path, class_count=6)


def test_validate_dataset_raises_on_corrupted_image(tmp_path: Path) -> None:
    """A corrupted image makes the dataset validation raise."""
    # Arrange - Given
    write_split(tmp_path, "train", "valid.png", "0 0.5 0.5 0.2 0.2\n")
    (tmp_path / "train" / "images" / "corrupt.png").write_bytes(b"not an image")

    # Act - When / Assert - Then
    with pytest.raises(ValueError, match="corrupted image"):
        validate_dataset(tmp_path, class_count=6)


def test_validate_dataset_raises_on_invalid_label(tmp_path: Path) -> None:
    """A label with out-of-frame coordinates makes validation raise."""
    # Arrange - Given
    write_split(tmp_path, "train", "image.png", "0 -0.5 0.5 0.2 0.2\n")

    # Act - When / Assert - Then
    with pytest.raises(ValueError, match="invalid label"):
        validate_dataset(tmp_path, class_count=6)
