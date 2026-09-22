# CanaryVision

CPU image classification with FastAPI and pretrained MobileNetV3 Small. Upload
an image through the web interface or API to get the top three ImageNet labels,
scores, model version, and inference time.

Python · uv · PyTorch/torchvision · FastAPI · Docker · GitHub Actions

![CanaryVision image classification interface](docs/screenshot.png)

## Quick start

With Docker and Compose installed:

```sh
docker compose up --build
```

Open [localhost:8000](http://localhost:8000) or the
[API reference](http://localhost:8000/docs). The first build downloads dependencies
and model weights; the finished container runs without external network access.

For native setup, configuration, and browser checks, see the
[development guide](docs/development.md).

## Canary rollout

Open **Canary rollout** in the sidebar (or **Manage canary rollout** below the
playground). Start a good or bad release, then use **Check and advance**:

- Candidate uploads receive 10%, then 50%, then 100% of traffic, selected randomly
  per request. Each prediction identifies the release that served it.
- Each advance evaluates the candidate directly against the fixed dataset. Both
  accuracy gates must pass; a final check at 100% promotes the candidate.
- A failed gate or evaluation error withdraws the candidate and restores all new
  requests to the previous stable release. Manual rollback is also available
  while a candidate is active. Requests already running may finish on that candidate.

The good release preserves baseline behavior. The deliberately bad release shifts
ImageNet labels by one position, reproducing a label-map packaging defect while
sharing the same verified weights. No extra weights or downloads are needed.

```sh
curl -X POST http://localhost:8000/rollout/bad
curl -X POST http://localhost:8000/rollout/advance
curl http://localhost:8000/rollout
```

The second command returns `rolled_back` with the failing evaluation report.
For a successful rollout, start `good` and advance three times. To evaluate the
bad release offline (expected exit code **1**):

```sh
uv run --frozen python -m scripts.evaluate --release bad --output artifacts/bad-evaluation.json
```

This is a local, single-process rollout lab. Run one Uvicorn worker; state and
reports are kept in memory and restart resets to the original stable release.
Controls are unauthenticated and intended for the loopback-bound local service.
Advancement is operator-triggered; rollback after a failed check is automatic.
The gate uses labeled samples, not accuracy inferred from user uploads. The
bundled `/evaluation` report continues to describe the original baseline.

## API

```sh
curl -F 'file=@evaluation/images/coffee.png' http://localhost:8000/predict
```

Example response; timing varies by machine:

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
| `POST /predict` | Classify an image uploaded in the multipart `file` field. |
| `GET /health` | Process liveness. |
| `GET /ready` | Model readiness and version. |
| `GET /model` | Model metadata and input limits. |
| `GET /evaluation` | Recorded evaluation report. |
| `GET /rollout` | Current traffic split, release versions, and gate reports. |
| `POST /rollout/{action}` | `good`, `bad`, `advance`, or `rollback`. |
| `GET /samples` | Sample images and attribution. |

Accepts static JPEG, PNG, and WebP images up to **10 MiB** and **20 megapixels**.
Uploads are not retained. Scores are probabilities over all 1,000 classes;
inference time covers preprocessing, the model forward pass, and scoring.

## Validation

After [native setup](docs/development.md#native-setup):

```sh
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen pytest -q
uv run --frozen python -m scripts.evaluate
```

[CI](.github/workflows/ci.yaml) runs linting, API and model tests, evaluation,
browser checks, and an offline Docker smoke test. Reports are saved as workflow
artifacts.

The [four-image baseline](evaluation/baseline.json) scores **75% top-1** and
**100% top-3**; both gates require at least **75%**. This is a regression smoke
test, not a representative accuracy benchmark. The blurred clock is a top-1
miss, with `wall clock` ranked third. [Dataset notes](evaluation/README.md) cover
labels, checksums, and image permissions.

The model loads once per process. Dependencies, Docker base images, and model
weights are pinned; the container runs as a non-root user.
