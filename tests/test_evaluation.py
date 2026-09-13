import hashlib
import json
import sys
from io import BytesIO

import pytest
from PIL import Image

from scripts.evaluate import evaluate, main


@pytest.fixture
def dataset(tmp_path):
    image = BytesIO()
    Image.new("RGB", (32, 32)).save(image, "PNG")
    data = image.getvalue()
    (tmp_path / "image.png").write_bytes(data)
    manifest = {
        "name": "test",
        "min_top1_accuracy": 0.75,
        "min_top3_accuracy": 0.75,
        "samples": [
            {
                "id": str(i),
                "file": "image.png",
                "expected_labels": ["espresso"],
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for i in range(4)
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


class SequenceClassifier:
    def __init__(self, labels):
        self.labels = iter(labels)

    def predict(self, image):
        return {
            "predictions": [{"label": label, "score": 0.2} for label in next(self.labels)],
            "inference_ms": 1.0,
        }


def test_gate_distinguishes_top1_from_top3(dataset):
    path, _ = dataset
    classifier = SequenceClassifier(
        [
            ["espresso", "cup", "coffee mug"],
            ["espresso", "cup", "coffee mug"],
            ["cup", "espresso", "coffee mug"],
            ["cup", "coffee mug", "consomme"],
        ]
    )
    report = evaluate(path, classifier)
    assert report["top1_accuracy"] == 0.5
    assert report["top3_accuracy"] == 0.75
    assert report["passed"] is False
    assert report["sample_count"] == 4


def test_gate_passes_at_documented_threshold(dataset):
    path, _ = dataset
    classifier = SequenceClassifier(
        [
            ["espresso", "cup", "coffee mug"],
            ["espresso", "cup", "coffee mug"],
            ["espresso", "cup", "coffee mug"],
            ["cup", "coffee mug", "consomme"],
        ]
    )
    report = evaluate(path, classifier)
    assert report["top1_accuracy"] == 0.75
    assert report["top3_accuracy"] == 0.75
    assert report["passed"] is True


def test_changed_image_cannot_silently_pass(dataset):
    path, _ = dataset
    (path.parent / "image.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        evaluate(path, SequenceClassifier([]))


@pytest.mark.parametrize("invalid", ["not an ImageNet label", ""])
def test_invalid_ground_truth_is_rejected(dataset, invalid):
    path, manifest = dataset
    manifest["samples"][0]["expected_labels"] = [invalid]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Invalid ImageNet labels"):
        evaluate(path, SequenceClassifier([]))


def test_empty_dataset_is_rejected(dataset):
    path, manifest = dataset
    manifest["samples"] = []
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="cannot be empty"):
        evaluate(path, SequenceClassifier([]))


@pytest.mark.parametrize("passed", [True, False])
def test_command_records_report_and_fails_a_bad_release(dataset, monkeypatch, passed):
    path, _ = dataset
    output = path.parent / "report.json"
    labels = ["espresso", "cup", "coffee mug"] if passed else ["cup", "coffee mug", "consomme"]
    monkeypatch.setattr("scripts.evaluate.Classifier", lambda: SequenceClassifier([labels] * 4))
    monkeypatch.setattr(sys, "argv", ["evaluate", "--manifest", str(path), "--output", str(output)])
    if passed:
        main()
    else:
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
    assert json.loads(output.read_text())["passed"] is passed
