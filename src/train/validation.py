"""Dataset validation step: verify image integrity and annotation validity."""

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from src.config import CLASS_NAMES, SPLITS


@dataclass
class ValidationReport:
    """Collected problems found while validating a dataset."""

    corrupted_images: list[Path] = field(default_factory=list)
    invalid_labels: list[Path] = field(default_factory=list)

    def is_valid(self) -> bool:
        """True when no corrupted image nor invalid label was found."""
        return not self.corrupted_images and not self.invalid_labels


def is_image_corrupted(image_path: Path) -> bool:
    """Return True if the image cannot be opened and verified by PIL."""
    try:
        with Image.open(image_path) as image:
            image.verify()
    except (OSError, SyntaxError, Image.DecompressionBombError):
        return True
    return False


def parse_yolo_annotation_line(
    annotation_line: str,
) -> tuple[int, float, float, float, float]:
    """Parse one YOLO label line into (class_index, x, y, width, height)."""
    parts = annotation_line.split()
    if len(parts) != 5:
        message = f"Invalid YOLO annotation line: {annotation_line!r}"
        raise ValueError(message)
    class_index = int(parts[0])
    x_center, y_center, width, height = (float(value) for value in parts[1:])
    return class_index, x_center, y_center, width, height


def validate_annotation_coordinates(
    bounding_boxes: list[tuple[float, float, float, float]],
) -> bool:
    """Check every box is normalized, non-degenerate and inside the frame.

    Centers and sizes must lie within [0, 1], width and height must be strictly positive, and the box (center plus half its size) must not exceed the frame.
    """
    for x_center, y_center, width, height in bounding_boxes:
        if any(value < 0 or value > 1 for value in (x_center, y_center, width, height)):
            return False
        if width <= 0 or height <= 0:
            return False
        if x_center - width / 2 < 0 or x_center + width / 2 > 1:
            return False
        if y_center - height / 2 < 0 or y_center + height / 2 > 1:
            return False
    return True


def validate_annotation_classes(
    class_indices: list[int],
    class_count: int = len(CLASS_NAMES),
) -> bool:
    """Check that every class index is within [0, class_count)."""
    return all(0 <= class_index < class_count for class_index in class_indices)


def validate_label_file(
    label_path: Path,
    class_count: int = len(CLASS_NAMES),
) -> bool:
    """Validate a YOLO label file: parseable lines, valid coordinates and classes.

    An empty file (an image with no objects) is considered valid.
    """
    annotation_lines = [
        line for line in label_path.read_text().splitlines() if line.strip()
    ]
    bounding_boxes: list[tuple[float, float, float, float]] = []
    class_indices: list[int] = []
    try:
        for annotation_line in annotation_lines:
            class_index, x_center, y_center, width, height = parse_yolo_annotation_line(
                annotation_line
            )
            class_indices.append(class_index)
            bounding_boxes.append((x_center, y_center, width, height))
    except ValueError:
        return False
    return validate_annotation_coordinates(
        bounding_boxes
    ) and validate_annotation_classes(class_indices, class_count)


def validate_split(
    images_directory: Path,
    labels_directory: Path,
    report: ValidationReport,
    class_count: int,
) -> None:
    """Validate one split's images and labels, recording problems in report."""
    for image_path in images_directory.iterdir():
        if not image_path.is_file():
            continue
        if is_image_corrupted(image_path):
            report.corrupted_images.append(image_path)
    if not labels_directory.is_dir():
        return
    for label_path in labels_directory.glob("*.txt"):
        if not validate_label_file(label_path, class_count):
            report.invalid_labels.append(label_path)


def validate_dataset(
    dataset_path: Path,
    class_count: int = len(CLASS_NAMES),
) -> None:
    """Validate every image and label across the train/valid/test splits.

    Returns nothing when the dataset is clean. Raises ValueError if any corrupted image or invalid label is found, so the pipeline halts before
    training. An image without a label file is treated as a valid background image, consistent with an empty label file.
    """
    report = ValidationReport()
    for split in SPLITS:
        images_directory = dataset_path / split / "images"
        labels_directory = dataset_path / split / "labels"
        if not images_directory.is_dir():
            continue
        validate_split(images_directory, labels_directory, report, class_count)
    if not report.is_valid():
        message = (
            f"Dataset validation failed: "
            f"{len(report.corrupted_images)} corrupted image(s), "
            f"{len(report.invalid_labels)} invalid label file(s)."
        )
        raise ValueError(message)
