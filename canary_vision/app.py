"""Same-origin webpage and CPU inference API."""

import json
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Receive, Scope, Send

from canary_vision import __version__
from canary_vision.images import MAX_IMAGE_PIXELS, MAX_UPLOAD_BYTES, decode_image
from canary_vision.model import MODEL_VERSION, WEIGHTS, WEIGHTS_SHA256, Classifier

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)


class Prediction(BaseModel):
    label: str
    score: float = Field(ge=0, le=1)


class ImageInfo(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    format: str


class PredictionResponse(BaseModel):
    predictions: list[Prediction] = Field(min_length=3, max_length=3)
    model_version: str
    inference_ms: float = Field(ge=0)
    image: ImageInfo
    request_id: str


def error_response(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


class BodyLimitMiddleware:
    """Cap the body before multipart parsing, including streamed requests."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.limit = MAX_UPLOAD_BYTES + 64 * 1024  # Room for multipart headers.

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                length = int(content_length)
            except ValueError:
                await error_response(400, "invalid_length", "Invalid Content-Length.")(
                    scope, receive, send
                )
                return
            if length < 0:
                await error_response(400, "invalid_length", "Invalid Content-Length.")(
                    scope, receive, send
                )
                return
            if length > self.limit:
                await error_response(413, "image_too_large", "Images must be 10 MB or smaller.")(
                    scope, receive, send
                )
                return
        received = 0

        async def bounded_receive() -> dict:
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > self.limit:
                raise HTTPException(
                    413,
                    detail={
                        "code": "image_too_large",
                        "message": "Images must be 10 MB or smaller.",
                    },
                )
            return message

        await self.app(scope, bounded_receive, send)


def create_app(classifier_factory: Callable = Classifier) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.classifier = classifier_factory()
        yield
        app.state.classifier = None

    app = FastAPI(
        title="CanaryVision",
        description="CPU image classification with pinned MobileNetV3 Small weights.",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.classifier = None
    app.add_middleware(BodyLimitMiddleware)

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(request: Request, exc: StarletteHTTPException):
        if isinstance(exc.detail, dict):
            return error_response(exc.status_code, exc.detail["code"], exc.detail["message"])
        return error_response(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        return error_response(
            422, "invalid_request", "Send an image in the multipart field named 'file'."
        )

    @app.get("/health", tags=["Service"])
    def health():
        return {"status": "ok", "app_version": __version__}

    @app.get("/ready", tags=["Service"])
    def ready(request: Request):
        if request.app.state.classifier is None:
            return error_response(503, "model_not_ready", "The model is not ready yet.")
        return {"status": "ready", "model_version": MODEL_VERSION, "device": "cpu"}

    @app.get("/model", tags=["Service"])
    def model_info():
        return {
            "name": "MobileNetV3 Small",
            "model_version": MODEL_VERSION,
            "weights": "IMAGENET1K_V1",
            "weights_sha256": WEIGHTS_SHA256,
            "device": "cpu",
            "classes": len(WEIGHTS.meta["categories"]),
            "parameters": WEIGHTS.meta["num_params"],
            "input_size": [224, 224],
            "app_version": __version__,
            "max_upload_bytes": MAX_UPLOAD_BYTES,
            "max_image_pixels": MAX_IMAGE_PIXELS,
        }

    @app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
    def predict(
        request: Request,
        file: Annotated[UploadFile, File(description="Static JPEG, PNG, or WebP, up to 10 MB.")],
    ):
        classifier = request.app.state.classifier
        if classifier is None:
            raise HTTPException(
                503, detail={"code": "model_not_ready", "message": "The model is not ready yet."}
            )
        try:
            data = file.file.read(MAX_UPLOAD_BYTES + 1)
            image, info = decode_image(data)
            with image:
                try:
                    result = classifier.predict(image)
                except Exception as exc:
                    LOGGER.exception("Inference failed")
                    raise HTTPException(
                        500,
                        detail={
                            "code": "inference_failed",
                            "message": "Inference failed. Please try again.",
                        },
                    ) from exc
            return {**result, "image": info, "request_id": str(uuid4())}
        finally:
            file.file.close()

    @app.get("/samples", tags=["Playground"])
    def samples():
        manifest = json.loads((ROOT / "evaluation" / "manifest.json").read_text())
        return [
            {
                "id": sample["id"],
                "title": sample["title"],
                "url": f"/samples/{sample['id']}",
                "author": sample["author"],
                "license": sample["license"],
                "license_source": sample["license_source"],
            }
            for sample in manifest["samples"]
        ]

    @app.get("/samples/{sample_id}", include_in_schema=False)
    def sample_image(sample_id: str):
        manifest = json.loads((ROOT / "evaluation" / "manifest.json").read_text())
        sample = next((s for s in manifest["samples"] if s["id"] == sample_id), None)
        if sample is None:
            raise HTTPException(404, "Sample not found.")
        return FileResponse(ROOT / "evaluation" / sample["file"], media_type="image/png")

    @app.get("/evaluation", tags=["Service"])
    def evaluation():
        path = ROOT / "evaluation" / "baseline.json"
        if not path.is_file():
            return error_response(
                503, "evaluation_unavailable", "No recorded evaluation report is available."
            )
        return json.loads(path.read_text())

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(ROOT / "canary_vision" / "static" / "index.html")

    static = ROOT / "canary_vision" / "static"
    if static.is_dir():
        app.mount("/static", StaticFiles(directory=static), name="static")
    return app


app = create_app()
