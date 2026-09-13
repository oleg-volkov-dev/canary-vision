# Initial release plan

The core risk is reproducibility: the webpage is useful only if a fresh clone can
load the same model, return real predictions, and catch a broken release in CI.
The first release therefore follows the build order in `IDEA.md`.

1. **Verify inference.** Pin MobileNetV3 Small to `IMAGENET1K_V1`, use the weights'
   own ImageNet preprocessing, and classify a local image on CPU. Prepare and
   verify weights explicitly; serving must not depend on an internet connection.
2. **Define the service contract.** Load and warm the model once in FastAPI's
   lifespan. Expose `POST /predict`, `GET /health`, and `GET /ready`. Bound file
   size and decoded pixel count, check the actual image format, normalize EXIF
   orientation and color, and return structured errors.
3. **Build the upload workflow.** Serve plain HTML/CSS/JavaScript from the same
   API. Support file selection, drag and drop, and included sample images. Show
   image preview, the top three scores, model identity, and actual inference time.
   Cover input failures and the API response contract with pytest; exercise the
   page against the real model in a browser.
4. **Package it.** Lock Python dependencies with uv, use CPU PyTorch wheels in
   Linux, bake verified model weights into Docker, and serve as an unprivileged
   user. Prove startup and inference with networking disabled in the container.
5. **Add a fixed regression gate.** Check in four redistributable images, human
   labels compatible with ImageNet, attribution, and SHA-256 checksums. Report
   top-1 and top-3 accuracy; require at least 75% for each. This tiny collection
   detects obvious regressions; it cannot estimate real-world performance.
6. **Automate and document.** GitHub Actions runs Ruff, pytest, real-model
   evaluation, and Docker build/smoke checks. Document startup, API examples,
   validation limits, measured evaluation results, and a screenshot.

The initial release has one model and one API process. Kubernetes, cloud hosting,
monitoring infrastructure, traffic splitting, and automated rollback belong to
later releases once these checks are dependable.

Reference implementation details: [Torchvision's pinned model and transforms](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html),
[FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/), and
[uv's CPU PyTorch configuration](https://docs.astral.sh/uv/guides/integration/pytorch/).

## Verified initial version

All six steps are implemented. Local validation passed Ruff lint/format checks,
33 pytest cases (including real inference and a failing regression-gate command),
the real Chrome upload/drag-and-drop/export/error/mobile workflow, Compose
configuration validation, and actionlint validation of the GitHub Actions file.
The Docker image builds for Linux ARM64 and x86-64; offline startup and real
inference were exercised with the network disabled and a read-only filesystem.

The four-image baseline is 75% top-1 and 100% top-3. The motion-blurred clock
remains a documented top-1 miss, with the correct wall-clock label ranked third.
Docker evaluation reproduces the gate. GitHub Actions is configured to rerun all
required checks on push/PR; a hosted workflow has not been triggered from this
local implementation session.
