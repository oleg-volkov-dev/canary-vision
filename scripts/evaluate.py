"""Run the fixed dataset regression gate and write a machine-readable report."""

import argparse
import json
from pathlib import Path

from canary_vision.evaluation import evaluate
from canary_vision.model import Classifier, Release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("evaluation/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation.json"))
    parser.add_argument("--release", choices=["stable", "good", "bad"], default="stable")
    args = parser.parse_args()
    report = evaluate(args.manifest, Release(Classifier(), args.release))
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
