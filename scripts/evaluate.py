"""Run the fixed dataset regression gate and write a machine-readable report."""

import argparse
import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import torch
import torchvision

from canary_vision.images import decode_image
from canary_vision.model import MODEL_VERSION, WEIGHTS, WEIGHTS_SHA256, Classifier


def evaluate(manifest_path: Path, classifier) -> dict:
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    samples = manifest["samples"]
    if not samples:
        raise ValueError("The evaluation dataset cannot be empty.")
    if len({sample["id"] for sample in samples}) != len(samples):
        raise ValueError("Evaluation sample IDs must be unique.")
    thresholds = [manifest["min_top1_accuracy"], manifest["min_top3_accuracy"]]
    if not all(isinstance(value, (int, float)) and 0 <= value <= 1 for value in thresholds):
        raise ValueError("Accuracy thresholds must be between 0 and 1.")
    results = []
    for sample in samples:
        path = (manifest_path.parent / sample["file"]).resolve()
        if not path.is_relative_to(manifest_path.parent.resolve()):
            raise ValueError(f"Sample path escapes the dataset: {sample['id']}")
        expected = sample["expected_labels"]
        if not expected or not set(expected).issubset(WEIGHTS.meta["categories"]):
            raise ValueError(f"Invalid ImageNet labels for {sample['id']}: {expected}")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != sample["sha256"]:
            raise ValueError(f"Image checksum mismatch: {sample['id']}")
        image, _ = decode_image(data)
        with image:
            prediction = classifier.predict(image)
        labels = [p["label"] for p in prediction["predictions"]]
        results.append(
            {
                "id": sample["id"],
                "expected_labels": expected,
                "predictions": prediction["predictions"],
                "top1_correct": labels[0] in expected,
                "top3_correct": bool(set(labels[:3]) & set(expected)),
                "inference_ms": prediction["inference_ms"],
            }
        )
    count = len(results)
    top1 = sum(result["top1_correct"] for result in results) / count
    top3 = sum(result["top3_correct"] for result in results) / count
    return {
        "dataset": manifest["name"],
        "dataset_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "model_version": MODEL_VERSION,
        "weights_sha256": WEIGHTS_SHA256,
        "generated_at": datetime.now(UTC).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "device": "cpu",
            "platform": platform.platform(),
        },
        "sample_count": count,
        "top1_accuracy": top1,
        "top3_accuracy": top3,
        "min_top1_accuracy": thresholds[0],
        "min_top3_accuracy": thresholds[1],
        "passed": top1 >= thresholds[0] and top3 >= thresholds[1],
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("evaluation/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation.json"))
    args = parser.parse_args()
    report = evaluate(args.manifest, Classifier())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for sample in report["results"]:
        marker = "PASS" if sample["top1_correct"] else "MISS"
        print(f"{marker:4}  {sample['id']:10}  {sample['predictions'][0]['label']}")
    print(f"Top-1: {report['top1_accuracy']:.0%}; required: {report['min_top1_accuracy']:.0%}")
    print(f"Top-3: {report['top3_accuracy']:.0%}; required: {report['min_top3_accuracy']:.0%}")
    print(f"Report: {args.output}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
