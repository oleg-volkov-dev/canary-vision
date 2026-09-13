from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from canary_vision.app import create_app
from canary_vision.model import MODEL_VERSION, Classifier, model_path


def test_missing_weights_fail_with_actionable_message(tmp_path):
    with pytest.raises(RuntimeError, match="scripts.prepare_model"):
        Classifier(tmp_path / "missing.pth")


def test_corrupted_weights_fail_before_loading(tmp_path):
    weights = tmp_path / "bad.pth"
    weights.write_bytes(b"not model weights")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        Classifier(weights)


@pytest.mark.integration
@pytest.mark.skipif(not model_path().is_file(), reason="Run scripts.prepare_model first")
def test_real_model_api_and_preprocessing():
    # A real fixture protects the model/transform/label wiring; HTTP tests use a fake.
    image = Path("evaluation/images/coffee.png").read_bytes()
    with TestClient(create_app()) as client:
        assert client.get("/ready").json()["status"] == "ready"
        first = client.post("/predict", files={"file": ("coffee.png", image)}).json()
        second = client.post("/predict", files={"file": ("coffee.png", image)}).json()
    assert first["model_version"] == MODEL_VERSION
    assert first["predictions"][0]["label"] == "espresso"
    assert first["predictions"][0]["score"] > 0.5
    assert first["predictions"] == second["predictions"]
    assert first["request_id"] != second["request_id"]
