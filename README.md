# CanaryVision

A small, working computer vision deployment lab. Upload an image, run pretrained
**MobileNetV3 Small on CPU**, and see its top three ImageNet labels, scores, model
version, and inference time. The same FastAPI service serves the responsive
HTML/CSS/JavaScript playground and the API.

![CanaryVision playground with real espresso predictions](docs/screenshot.png)

[View the mobile layout](docs/mobile.png).

## Run with Docker

Requires Docker with Compose. From a fresh clone:

```sh
docker compose up --build
```

Open **http://localhost:8000**. API documentation is at
**http://localhost:8000/docs**. Stop with `Ctrl+C`, then `docker compose down`.

The first build needs internet access to fetch the pinned base images, locked
dependencies, and approximately 10 MB of model weights. The finished image has
the weights baked in and requires no external network for startup or inference.
The service runs as user 10001 with a read-only filesystem and temporary upload
space. The port is bound to localhost.

## Run with Python and uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (the build
and CI use 0.12.13). Native development supports Python 3.12–3.13 on Apple Silicon
macOS or x86-64/ARM64 Linux. Use Docker on other host platforms.

```sh
uv sync --locked
uv run --frozen python -m scripts.prepare_model
uv run --frozen python -m scripts.classify evaluation/images/coffee.png
uv run --frozen uvicorn canary_vision.app:app --host 127.0.0.1 --port 8000
```

The model preparation step downloads and verifies the pinned full SHA-256
checksum into `.cache/models/`. API startup loads and warms the model once;
missing or corrupt weights stop startup with an actionable error. It never
silently downloads new weights or substitutes another model.

Set `CANARY_MODEL_PATH` to use another location for the same verified weights.
`CANARY_CPU_THREADS` defaults to `2`. Inference always uses CPU, even if a GPU is
available. Linux uses the official CPU wheel index, with explicit torchvision
pins for x86 and ARM wheel names. All transitive dependencies are locked in
`uv.lock`; Docker base images are pinned by digest.

## API

```sh
curl -F 'file=@evaluation/images/coffee.png' http://localhost:8000/predict
```

An actual response has this shape (scores and timing vary slightly by machine):

```json
{
  "predictions": [
    {"label": "espresso", "score": 0.9639945},
    {"label": "consomme", "score": 0.0152125},
    {"label": "chocolate sauce", "score": 0.0077917}
  ],
  "model_version": "mobilenet-v3-small-imagenet1k-v1-047dcff4",
  "inference_ms": 27.49,
  "image": {"width": 600, "height": 400, "format": "PNG"},
  "request_id": "875c2af9-b6c7-4385-9418-b6fe19b3a977"
}
```

| Endpoint | Purpose |
| --- | --- |
| `POST /predict` | A multipart upload in the `file` field; returns three predictions. |
| `GET /health` | Process liveness, independent of model readiness. |
| `GET /ready` | Returns 200 with model identity once ready, otherwise 503. |
| `GET /model` | Model, weights checksum, preprocessing dimensions, and input limits. |
| `GET /evaluation` | The bundled, recorded evaluation report; does not rerun evaluation. |
| `GET /samples` | Included sample images and their attribution. |
| `GET /docs` | Interactive API reference. |

Static JPEG, PNG, and WebP images are accepted up to **10 MiB** and **20 million
decoded pixels**. Validation checks the decoded format rather than trusting the
filename or MIME type, applies EXIF orientation, and converts grayscale/RGBA to
RGB. Animations, empty files, and corrupt images are rejected. The request body
is capped before multipart parsing, including requests without Content-Length.
Multipart overhead has a separate 64 KiB allowance.

Errors use `{"error": {"code": "...", "message": "..."}}`: 413 for limits, 415 for
unsupported images/animations, 422 for invalid images or missing uploads, 503
for an unready model, and 500 for inference failures. Uploads are not retained;
the multipart parser may temporarily spool larger uploads to disk, and handles
are closed after the request.

Scores are softmax probabilities over all 1,000 classes, not percentages
renormalized over only three results. The displayed scores need not add up to
100%. Timing includes preprocessing, the CPU forward pass, and scoring; it
excludes upload/decoding, request queueing, and transport. A single process uses
one model and serializes forward passes to bound CPU work.

## Checks and evaluation

```sh
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen pytest -q
uv run --frozen python -m scripts.evaluate --output artifacts/evaluation.json
```

Prepare weights first to run the real-model integration test; only that test
skips if the weights are absent. Other API/input tests use a fake classifier
without network access. CI prepares weights and runs the real test and
evaluation as mandatory checks.

The fixed dataset has **four** checked-in CC0/public domain photographs with
source URLs, authors, compatible human labels, and SHA-256 checksums. Evaluation
checks image integrity and label compatibility, writes per-image predictions,
and exits with status 1 if either top-1 or top-3 accuracy is below **75%**.

The [recorded baseline](evaluation/baseline.json) measured **75% top-1** and
**100% top-3**. The blurred clock is a top-1 miss (`bubble`); `wall clock` appears
third. The miss is preserved. This tiny dataset is a regression smoke test,
not evidence of real-world accuracy. See [dataset notes](evaluation/README.md)
for label choices and redistribution permissions.

For a browser check against a running app:

```sh
uv sync --locked --group browser
uv run --frozen --group browser playwright install chromium
uv run --frozen --group browser python -m scripts.browser_smoke
```

Use `--channel chrome` instead of installing Chromium if Chrome is already
installed. The script covers upload, real inference, samples, clearing inputs,
validation, API errors, export, model metadata, and mobile overflow. It saves
screenshots to `artifacts/browser/`. The page also supports drag and drop and
keyboard operation.

To prove the container serves without an external network:

```sh
docker build -t canary-vision:local .
docker run -d --name canary-offline --network none --read-only --tmpfs /tmp:size=32m,mode=1777 canary-vision:local
docker exec canary-offline python -m scripts.smoke
docker exec canary-offline python -m scripts.evaluate --output /tmp/evaluation.json
docker rm -f canary-offline
```

[GitHub Actions](.github/workflows/ci.yaml) runs lint/format checks, pytest,
evaluation, the real browser workflow, and an offline Docker smoke test and
evaluation on pushes and pull requests. Reports and screenshots are saved as
workflow artifacts.

## Initial scope

The first steps and their rationale are documented in [BUILD_PLAN.md](BUILD_PLAN.md).
The original [IDEA.md](IDEA.md) is preserved. The initial app has one pretrained
model, one API, local uploads, evaluation, and CI. Rollout automation and a
bad-release rollback demonstration come after this reproducible foundation.

The implementation follows the official
[Torchvision model and preprocessing contract](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html),
[FastAPI startup lifecycle](https://fastapi.tiangolo.com/advanced/events/), and
[uv's platform-aware CPU PyTorch configuration](https://docs.astral.sh/uv/guides/integration/pytorch/).
