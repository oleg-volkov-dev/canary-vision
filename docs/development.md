# Development

## Native setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python
3.12–3.13 on Apple Silicon macOS or x86-64/ARM64 Linux. Use Docker on other host
platforms. CI and the Docker build use uv 0.12.13.

Run from the repository root:

```sh
uv sync --locked
uv run --frozen python -m scripts.prepare_model
uv run --frozen uvicorn canary_vision.app:app --host 127.0.0.1 --port 8000
```

Open [localhost:8000](http://localhost:8000). To classify a file without HTTP:

```sh
uv run --frozen python -m scripts.classify evaluation/images/coffee.png
```

Weights are stored in `.cache/models/` and checked against the pinned SHA-256
checksum. Startup loads and warms the model once; missing or corrupt weights
prevent startup. Dependencies are locked in `uv.lock`, including platform-specific
CPU torchvision wheels. Docker base images are pinned by digest.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `CANARY_MODEL_PATH` | `.cache/models/mobilenet_v3_small-047dcff4.pth` | Override the location of the pinned weights. The default resolves from the repository root. |
| `CANARY_CPU_THREADS` | `2` | Number of PyTorch CPU threads. |

The Docker image stores weights under `/opt/models/`. Inference uses CPU and
serializes forward passes within each process.

## Input and response behavior

The API checks the decoded image format, applies EXIF orientation, and converts
images to RGB. Empty, corrupt, animated, and unsupported images are rejected.
The request body is limited before multipart parsing, including streamed uploads,
with a 64 KiB allowance for multipart overhead. Model uploads to `/rollout/upload`
have a separate 32 MiB limit; image uploads retain their 10 MiB limit. The parser may temporarily spool
larger uploads to disk; file handles are closed after processing.

Errors use `{"error": {"code": "...", "message": "..."}}`:

| Status | Cause |
| --- | --- |
| `400` | Invalid request headers or multipart data. |
| `413` | Upload or decoded pixel limit exceeded. |
| `415` | Unsupported format or animation. |
| `422` | Missing upload or invalid image. |
| `503` | Model or recorded evaluation unavailable. |
| `500` | Inference failure. |

Scores are softmax probabilities over 1,000 ImageNet classes. The top three
scores may add up to less than 100%, and are not calibrated certainty.
`inference_ms` excludes image upload/decoding, queueing, and network transport.

Model preprocessing follows the
[Torchvision weights definition](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html):
resize to 256 pixels, center-crop to 224 × 224, and normalize with the prescribed
ImageNet mean and standard deviation.

## Tests

```sh
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen pytest -q
uv run --frozen python -m scripts.evaluate --output artifacts/evaluation.json
```

API validation tests use a fake classifier. The model integration test uses the
included coffee image and skips if weights are absent. CI prepares weights before
testing so the integration test and evaluation run against the pretrained model.
Evaluation exits with status 1 below either accuracy threshold; reports include
per-image predictions and environment versions.

## Browser checks

Against a running service:

```sh
uv sync --locked --group browser
uv run --frozen --group browser playwright install chromium
uv run --frozen --group browser python -m scripts.browser_smoke
```

With an installed Chrome browser, omit the install command and pass
`--channel chrome` to the smoke script. Screenshots and the exported prediction
are saved under `artifacts/browser/`.

The check covers file selection, drag and drop, sample images, inference, JSON
export, input replacement, validation errors, API failures, dialogs, and mobile
overflow, automatic promotion and rollback, and custom model upload validation. [Mobile screenshot](mobile.png).

## Docker checks

Compose binds port 8000 to localhost and runs as user 10001 with a read-only
filesystem and temporary upload space. Stop with `Ctrl+C` and remove the
container with `docker compose down`.

To check startup, inference, and evaluation with networking disabled:

```sh
docker build -t canary-vision:local .
docker run -d --name canary-offline --network none --read-only --tmpfs /tmp:size=32m,mode=1777 canary-vision:local
docker exec canary-offline python -m scripts.smoke
docker exec canary-offline python -m scripts.evaluate --output /tmp/evaluation.json
docker rm -f canary-offline
```

## Good Luck !!
