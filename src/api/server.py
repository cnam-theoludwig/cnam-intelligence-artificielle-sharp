"""FastAPI application exposing the finger-count prediction endpoint."""

from dataclasses import asdict
from io import BytesIO
from typing import Any

from fastapi import FastAPI, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image

from src.api.inference import predict_image

app = FastAPI(title="SHARP", description="Smart Hand Automated Recognition Project.")


@app.post("/predict")
async def predict(file: UploadFile) -> dict[str, Any]:
    """Detect hands in the uploaded image and return boxes and finger total."""
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    result = predict_image(image)
    return {
        "detections": [asdict(detection) for detection in result.detections],
        "total_fingers": result.total_fingers,
    }


app.mount("/", StaticFiles(directory="web", html=True), name="web")
