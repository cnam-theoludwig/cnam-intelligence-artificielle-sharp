"""Tests for the inference helpers and the prediction endpoint."""

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api import inference, server
from src.api.inference import Detection, PredictionResult, count_fingers


def test_count_fingers_sums_detections() -> None:
    """The total equals the sum of every detection's finger count."""
    # Arrange - Given
    detections = [
        Detection(0.0, 0.0, 1.0, 1.0, "2_fingers", 0.9),
        Detection(2.0, 2.0, 3.0, 3.0, "3_fingers", 0.8),
    ]

    # Act - When
    total_fingers = count_fingers(detections)

    # Assert - Then
    assert total_fingers == 5


def test_count_fingers_empty_is_zero() -> None:
    """No detections means no fingers."""
    # Arrange - Given
    detections: list[Detection] = []

    # Act - When
    total_fingers = count_fingers(detections)

    # Assert - Then
    assert total_fingers == 0


def test_predict_endpoint_returns_detections_and_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The endpoint returns the serialized detections and the finger total."""
    # Arrange - Given
    prediction = PredictionResult(
        detections=[Detection(1.0, 2.0, 3.0, 4.0, "4_fingers", 0.75)],
        total_fingers=4,
    )
    monkeypatch.setattr(
        inference, "predict_image", lambda _image: prediction, raising=True
    )
    monkeypatch.setattr(
        server, "predict_image", lambda _image: prediction, raising=True
    )
    image_buffer = BytesIO()
    Image.new("RGB", (8, 8)).save(image_buffer, format="JPEG")
    image_buffer.seek(0)
    client = TestClient(server.app)

    # Act - When
    response = client.post(
        "/predict",
        files={"file": ("frame.jpg", image_buffer, "image/jpeg")},
    )

    # Assert - Then
    assert response.status_code == 200
    body = response.json()
    assert body["total_fingers"] == 4
    assert body["detections"] == [
        {
            "x_minimum": 1.0,
            "y_minimum": 2.0,
            "x_maximum": 3.0,
            "y_maximum": 4.0,
            "class_name": "4_fingers",
            "confidence": 0.75,
        }
    ]
