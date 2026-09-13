"""One pinned MobileNet model, its preprocessing, and CPU-only inference."""

import hashlib
import logging
import os
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


def model_path() -> Path:
    return Path(os.environ.get("CANARY_MODEL_PATH", str(DEFAULT_MODEL_PATH)))


class Classifier:
    """Load verified weights once; never download anything during API startup."""

    version = MODEL_VERSION

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
        # Warm up before becoming ready; the first request is a normal inference.
        with torch.inference_mode():
            self._model(torch.zeros(1, 3, 224, 224))
        LOGGER.info("Loaded %s on CPU; weights sha256=%s", self.version, digest)

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
