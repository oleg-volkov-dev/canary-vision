"""One pinned MobileNet model, its preprocessing, and CPU-only inference."""

import hashlib
import logging
import os
from io import BytesIO
from pathlib import Path
from threading import Lock
from time import perf_counter

import torch
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

LOGGER = logging.getLogger(__name__)
WEIGHTS = MobileNet_V3_Small_Weights.IMAGENET1K_V1
MODEL_VERSION = "mobilenet-v3-small-imagenet1k-v1-047dcff4"
WEIGHTS_NAME = "mobilenet_v3_small-047dcff4.pth"
WEIGHTS_SHA256 = "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / ".cache" / "models" / WEIGHTS_NAME
MAX_MODEL_BYTES = 32 * 1024 * 1024


def model_path() -> Path:
    return Path(os.environ.get("CANARY_MODEL_PATH", str(DEFAULT_MODEL_PATH)))


class Classifier:
    """CPU classifier using checksum-verified local weights."""

    version = MODEL_VERSION
    weights_sha256 = WEIGHTS_SHA256
    weights_name = "IMAGENET1K_V1"

    def __init__(self, path: Path | None = None) -> None:
        path = path or model_path()
        if not path.is_file():
            raise RuntimeError(
                f"Model weights missing at {path}. Run: uv run python -m scripts.prepare_model"
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != WEIGHTS_SHA256:
            raise RuntimeError(f"Model weights checksum mismatch at {path}.")
        torch.set_num_threads(max(1, int(os.environ.get("CANARY_CPU_THREADS", "2"))))
        self._model = mobilenet_v3_small(weights=None).to("cpu")
        self._model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self._model.eval()
        self._transform = WEIGHTS.transforms()
        self._labels = WEIGHTS.meta["categories"]
        self._lock = Lock()
        # Warm up before reporting readiness.
        with torch.inference_mode():
            self._model(torch.zeros(1, 3, 224, 224))
        LOGGER.info("Loaded %s on CPU; weights sha256=%s", self.version, digest)

    @classmethod
    def from_upload(cls, data: bytes):
        """Load a tensor-only state dict for the same architecture and label order."""
        instance = cls.__new__(cls)
        instance.weights_sha256 = hashlib.sha256(data).hexdigest()
        instance.weights_name = "Uploaded state dict"
        instance.version = f"mobilenet-v3-small-custom-{instance.weights_sha256[:12]}"
        weights = torch.load(BytesIO(data), map_location="cpu", weights_only=True)
        if not isinstance(weights, dict) or not all(
            isinstance(value, torch.Tensor) and torch.isfinite(value).all()
            for value in weights.values()
        ):
            raise ValueError("Expected a state dict containing finite tensors only.")
        instance._model = mobilenet_v3_small(weights=None).to("cpu")
        instance._model.load_state_dict(weights, strict=True)
        instance._model.eval()
        instance._transform = WEIGHTS.transforms()
        instance._labels = WEIGHTS.meta["categories"]
        instance._lock = Lock()
        with torch.inference_mode():
            output = instance._model(torch.zeros(1, 3, 224, 224))
            if not torch.isfinite(output).all():
                raise ValueError("Model produces non-finite predictions.")
        return instance

    def predict(self, image: Image.Image) -> dict:
        # Queueing is excluded; preprocessing, forward pass, and scoring are included.
        with self._lock, torch.inference_mode():
            started = perf_counter()
            tensor = self._transform(image).unsqueeze(0).to("cpu")
            probabilities = self._model(tensor)[0].softmax(dim=0)
            scores, indices = probabilities.topk(3)
            predictions = [
                {"label": self._labels[index], "score": float(score)}
                for score, index in zip(scores.tolist(), indices.tolist(), strict=True)
            ]
            elapsed = round((perf_counter() - started) * 1000, 2)
        return {
            "predictions": predictions,
            "model_version": self.version,
            "inference_ms": elapsed,
        }


class Release:
    """A release sharing verified inference weights, with an optional label-map defect."""

    def __init__(self, classifier, name: str):
        if name not in {"stable", "good", "bad"}:
            raise ValueError(f"Unknown release: {name}")
        self.classifier = classifier
        self.name = name
        self.version = classifier.version + (f"-{name}-v2" if name != "stable" else "")
        self.weights_sha256 = getattr(classifier, "weights_sha256", WEIGHTS_SHA256)
        self.weights_name = getattr(classifier, "weights_name", "IMAGENET1K_V1")
        labels = WEIGHTS.meta["categories"]
        self._mapping = dict(zip(labels, labels[1:] + labels[:1], strict=True))

    def predict(self, image: Image.Image) -> dict:
        result = self.classifier.predict(image)
        predictions = result["predictions"]
        if self.name == "bad":
            predictions = [
                {**prediction, "label": self._mapping[prediction["label"]]}
                for prediction in predictions
            ]
        return {**result, "predictions": predictions, "model_version": self.version}
