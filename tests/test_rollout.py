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
