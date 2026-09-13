import pytest
from fastapi.testclient import TestClient

from canary_vision.app import create_app
from canary_vision.model import MODEL_VERSION


class FakeClassifier:
    """Keep HTTP input tests fast and independent of model/network downloads."""

    def predict(self, image):
        assert image.mode == "RGB"
        return {
            "predictions": [
                {"label": "espresso", "score": 0.8},
                {"label": "cup", "score": 0.1},
                {"label": "coffee mug", "score": 0.05},
            ],
            "model_version": MODEL_VERSION,
            "inference_ms": 12.5,
        }


@pytest.fixture
def client():
    with TestClient(create_app(FakeClassifier)) as client:
        yield client
