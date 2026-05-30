"""Tests for the class names and the class-to-finger-count mapping."""

from src.config import CLASS_TO_FINGER_COUNT


def test_finger_count_mapping_matches_class_index() -> None:
    """Each class maps to a finger count equal to its position in the list."""
    # Arrange - Given
    expected_mapping = {
        "0_finger": 0,
        "1_finger": 1,
        "2_fingers": 2,
        "3_fingers": 3,
        "4_fingers": 4,
        "5_fingers": 5,
    }

    # Act - When
    actual_mapping = CLASS_TO_FINGER_COUNT

    # Assert - Then
    assert actual_mapping == expected_mapping


def test_finger_count_sum_example() -> None:
    """Summing the counts of detected classes gives the total finger count."""
    # Arrange - Given
    detected_classes = ["2_fingers", "3_fingers"]

    # Act - When
    total_fingers = sum(
        CLASS_TO_FINGER_COUNT[class_name] for class_name in detected_classes
    )

    # Assert - Then
    assert total_fingers == 5
