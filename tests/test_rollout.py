from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from canary_vision.app import create_app
from canary_vision.model import MODEL_VERSION, Classifier, model_path
from canary_vision.rollout import Rollout
from tests.conftest import FakeClassifier


def test_routing_manual_rollback_and_conflicts(monkeypatch):
    rollout = Rollout(FakeClassifier(), Path("unused"))
    image = Image.new("RGB", (32, 32))
    with pytest.raises(HTTPException):
        rollout.change("advance")
    assert rollout.change("bad")["candidate_percent"] == 10
    with pytest.raises(HTTPException):
        rollout.change("good")
    monkeypatch.setattr("canary_vision.rollout.randbelow", lambda _: 9)
    assert rollout.predict(image)["model_version"].endswith("-bad-v2")
    monkeypatch.setattr("canary_vision.rollout.randbelow", lambda _: 10)
    assert rollout.predict(image)["model_version"] == MODEL_VERSION
    assert rollout.change("rollback")["status"] == "rolled_back"
    assert rollout.predict(image)["model_version"] == MODEL_VERSION


def test_gate_error_withdraws_candidate(monkeypatch):
    rollout = Rollout(FakeClassifier(), Path("missing"))
    rollout.change("bad")
    with pytest.raises(HTTPException, match="Evaluation failed"):
        rollout.change("advance")
    assert rollout.snapshot()["status"] == "rolled_back"
    assert rollout.snapshot()["candidate_percent"] == 0


def test_concurrent_change_is_rejected_but_routing_remains_available(monkeypatch):
    entered, finish = Event(), Event()
    rollout = Rollout(FakeClassifier(), Path("unused"))
    rollout.change("good")

    def gate(*args):
        entered.set()
        assert finish.wait(5)
        return {"passed": True}

    monkeypatch.setattr("canary_vision.rollout.evaluate", gate)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(rollout.change, "advance")
        try:
            assert entered.wait(5)
            assert rollout.snapshot()["status"] == "evaluating"
            assert rollout.predict(Image.new("RGB", (32, 32)))["predictions"]
            with pytest.raises(HTTPException) as exc:
                rollout.change("advance")
            assert exc.value.status_code == 409
        finally:
            finish.set()
        assert future.result()["candidate_percent"] == 50


@pytest.mark.integration
@pytest.mark.skipif(not model_path().is_file(), reason="Run scripts.prepare_model first")
def test_real_rollout_promotion_and_bad_release_rollback(monkeypatch):
    image = Path("evaluation/images/coffee.png").read_bytes()
    with TestClient(create_app(Classifier)) as client:
        assert client.post("/rollout/advance").status_code == 409
        assert client.post("/rollout/unknown").status_code == 422
        assert client.post("/rollout/good").json()["candidate_percent"] == 10
        for percent in [50, 100, 0]:
            report = client.post("/rollout/advance").json()
            assert report["candidate_percent"] == percent
            assert report["checks"][-1]["passed"]
        assert report["status"] == "promoted"
        promoted = report["stable_version"]
        assert promoted.endswith("-good-v2")
        assert client.get("/ready").json()["model_version"] == promoted
        assert client.get("/model").json()["model_version"] == promoted
        assert client.post("/rollout/good").status_code == 409
        client.post("/rollout/bad")
        monkeypatch.setattr("canary_vision.rollout.randbelow", lambda _: 0)
        bad = client.post("/predict", files={"file": ("coffee.png", image)}).json()
        assert bad["model_version"].endswith("-bad-v2")
        assert bad["predictions"][0]["label"] != "espresso"
        report = client.post("/rollout/advance").json()
        assert report["status"] == "rolled_back"
        assert report["stable_version"] == promoted
        assert report["checks"][-1]["passed"] is False
        assert report["checks"][-1]["model_version"] == bad["model_version"]
        restored = client.post("/predict", files={"file": ("coffee.png", image)}).json()
        assert restored["model_version"] == promoted
        assert restored["predictions"][0]["label"] == "espresso"
        assert client.get("/rollout").json() == report


def wait_for_status(rollout, status):
    from time import monotonic, sleep

    deadline = monotonic() + 10
    while monotonic() < deadline:
        snapshot = rollout.snapshot()
        if snapshot["status"] == status:
            return snapshot
        sleep(0.01)
    pytest.fail(f"Expected {status}; got {rollout.snapshot()}")


@pytest.mark.parametrize("passed, outcome", [(True, "promoted"), (False, "rolled_back")])
def test_automatic_rollout_completes_without_browser(monkeypatch, passed, outcome):
    monkeypatch.setattr("canary_vision.rollout.evaluate", lambda *args: {"passed": passed})
    rollout = Rollout(FakeClassifier(), Path("unused"), stage_seconds=0.01)
    try:
        start = rollout.change("good" if passed else "bad", automatic=True)
        report = wait_for_status(rollout, outcome)
        assert report["candidate_percent"] == 0
        assert report["automatic"] is False
        assert report["last_candidate_version"] == start["candidate_version"]
        assert report["previous_version"] == MODEL_VERSION
        assert [c["candidate_percent"] for c in report["checks"]] == (
            [10, 50, 100] if passed else [10]
        )
        assert report["stable_version"] == (start["candidate_version"] if passed else MODEL_VERSION)
    finally:
        rollout.close()


def test_automatic_gate_error_restores_stable():
    rollout = Rollout(FakeClassifier(), Path("missing"), stage_seconds=0.01)
    try:
        rollout.change("bad", automatic=True)
        report = wait_for_status(rollout, "rolled_back")
        assert report["stable_version"] == MODEL_VERSION
        assert "could not complete" in report["reason"]
    finally:
        rollout.close()


def test_manual_stop_cancels_automatic_checks(monkeypatch):
    from time import sleep

    checks = []
    monkeypatch.setattr("canary_vision.rollout.evaluate", lambda *args: checks.append(1))
    rollout = Rollout(FakeClassifier(), Path("unused"), stage_seconds=0.05)
    try:
        rollout.change("good", automatic=True)
        with pytest.raises(HTTPException, match="already running"):
            rollout.change("advance")
        rollout.change("rollback")
        sleep(0.1)
        assert not checks
        assert rollout.snapshot()["status"] == "rolled_back"
    finally:
        rollout.close()


def test_invalid_model_upload_preserves_stable(client):
    from io import BytesIO

    import torch

    invalid_dict = BytesIO()
    torch.save({"wrong_key": torch.zeros(2)}, invalid_dict)
    nonfinite = BytesIO()
    torch.save({"weights": torch.tensor([float("nan")])}, nonfinite)
    for content in [b"", b"not weights", invalid_dict.getvalue(), nonfinite.getvalue()]:
        response = client.post("/rollout/upload", files={"file": ("custom.pth", content)})
        assert response.status_code == 422
        assert client.get("/rollout").json()["status"] == "idle"
    assert client.post("/rollout/upload").status_code == 422
    response = client.post(
        "/rollout/upload", content=b"x", headers={"content-length": str(33 * 1024 * 1024)}
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "model_too_large"


@pytest.mark.integration
@pytest.mark.skipif(not model_path().is_file(), reason="Run scripts.prepare_model first")
def test_uploaded_model_promotes_and_failed_preset_restores_it():
    import hashlib
    from io import BytesIO

    import torch

    weights = torch.load(model_path(), map_location="cpu", weights_only=True)
    weights["classifier.3.bias"] += 0.25  # Same ranking, different checkpoint checksum.
    buffer = BytesIO()
    torch.save(weights, buffer)
    content = buffer.getvalue()
    digest = hashlib.sha256(content).hexdigest()
    with TestClient(create_app(Classifier)) as client:
        rollout = client.app.state.classifier
        rollout._stage_seconds = 0.01
        response = client.post("/rollout/upload", files={"file": ("own-model.pth", content)})
        assert response.status_code == 200
        uploaded = response.json()["candidate_version"]
        assert "custom" in uploaded
        report = wait_for_status(rollout, "promoted")
        assert report["stable_version"] == uploaded
        assert all(check["passed"] for check in report["checks"])
        assert all(check["weights_sha256"] == digest for check in report["checks"])
        metadata = client.get("/model").json()
        assert metadata["weights_sha256"] == digest
        assert metadata["weights"] == "Uploaded state dict"
        assert client.post("/rollout/start/bad").status_code == 200
        report = wait_for_status(rollout, "rolled_back")
        assert report["stable_version"] == uploaded
        assert not report["checks"][0]["passed"]
