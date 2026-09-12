# CanaryVision

**Repository:** `canary-vision`

**Description:** A computer vision deployment lab, starting with a containerized inference API, model evaluation, and CI.

## Idea

Build a small image classification service to demonstrate practical MLOps and DevOps skills. Eventually, demonstrate a bad release being detected and rolled back. First, build a reliable, reproducible service.

Status: planning; features below are not implemented yet.

## First version

- Upload an image through a simple webpage.
- Return the top three labels and scores, model version, and inference time.
- Run locally on CPU in Docker.
- Test the API and check model accuracy against a small, fixed labeled dataset.
- Run checks automatically with GitHub Actions.

## Initial stack

- Python, uv, FastAPI, and Uvicorn.
- PyTorch/torchvision with pretrained MobileNetV3 Small; Pillow for images.
- Plain HTML/CSS/JavaScript served by the API.
- pytest, Ruff, Docker, and GitHub Actions.

## Build order

1. Load the model and classify one local image.
2. Add `POST /predict`, image validation, and health/readiness endpoints. Load the model once at startup.
3. Add the upload page and tests for valid images, invalid inputs, and prediction output.
4. Package in Docker with pinned dependencies and model weights available at startup.
5. Add a small evaluation dataset with compatible labels and redistribution permissions. Output an accuracy report and fail below a documented threshold. Treat this as a regression check, not proof of real-world accuracy.
6. Add CI for linting, tests, evaluation, and the Docker build. Write setup instructions and include a screenshot.

## Working rule

Implement and verify one step at a time. Keep the first release small: no Kubernetes, cloud hosting, monitoring stack, or rollout automation yet. Add those after the service works and can be reproduced from a fresh clone.
