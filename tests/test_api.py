from io import BytesIO
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from canary_vision.app import BodyLimitMiddleware, create_app
from canary_vision.images import MAX_UPLOAD_BYTES
from canary_vision.model import MODEL_VERSION
from tests.conftest import FakeClassifier


def image_bytes(mode="RGB", fmt="PNG", size=(64, 48), **kwargs):
    stream = BytesIO()
    Image.new(mode, size).save(stream, format=fmt, **kwargs)
    return stream.getvalue()


@pytest.mark.parametrize(
    ("mode", "fmt"), [("RGB", "JPEG"), ("RGBA", "PNG"), ("L", "PNG"), ("RGB", "WEBP")]
)
def test_predict_returns_valid_sorted_contract(client, mode, fmt):
    response = client.post(
        "/predict", files={"file": ("image", image_bytes(mode, fmt), "application/octet-stream")}
    )
    assert response.status_code == 200
    result = response.json()
    assert result["model_version"] == MODEL_VERSION
    assert result["inference_ms"] >= 0
    assert result["image"] == {"width": 64, "height": 48, "format": fmt}
    assert UUID(result["request_id"])
    predictions = result["predictions"]
    assert len(predictions) == 3
    scores = [p["score"] for p in predictions]
    assert scores == sorted(scores, reverse=True)
    assert all(0 <= score <= 1 for score in scores)
    assert all(isinstance(p["label"], str) and p["label"] for p in predictions)


def test_exif_orientation_is_applied(client):
    exif = Image.Exif()
    exif[274] = 6
    response = client.post(
        "/predict", files={"file": ("rotate.jpg", image_bytes(fmt="JPEG", exif=exif))}
    )
    assert response.json()["image"] == {"width": 48, "height": 64, "format": "JPEG"}


@pytest.mark.parametrize(
    ("data", "status", "code"),
    [
        (b"", 422, "empty_image"),
        (b"not an image", 422, "invalid_image"),
        (b"<svg></svg>", 422, "invalid_image"),
        (image_bytes(fmt="GIF"), 415, "unsupported_image"),
        (image_bytes()[:40], 422, "invalid_image"),
        (b"a" * (MAX_UPLOAD_BYTES + 1), 413, "image_too_large"),
    ],
)
def test_invalid_files_are_rejected(client, data, status, code):
    response = client.post("/predict", files={"file": ("fake.png", data, "image/png")})
    assert response.status_code == status
    assert response.json()["error"]["code"] == code


def test_decoded_pixel_limit(client, monkeypatch):
    monkeypatch.setattr("canary_vision.images.MAX_IMAGE_PIXELS", 100)
    response = client.post("/predict", files={"file": ("large.png", image_bytes())})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "too_many_pixels"


def test_animated_image_is_rejected(client):
    stream = BytesIO()
    Image.new("RGB", (32, 32), "red").save(
        stream,
        format="PNG",
        save_all=True,
        append_images=[Image.new("RGB", (32, 32), "blue")],
        duration=100,
    )
    response = client.post("/predict", files={"file": ("animated.png", stream.getvalue())})
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "animated_image"


def test_missing_file(client):
    response = client.post("/predict")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_health_and_readiness_are_separate():
    app = create_app(FakeClassifier)
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/ready").json()["status"] == "ready"
        app.state.classifier = None
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert (
            client.post("/predict", files={"file": ("image.png", image_bytes())}).status_code == 503
        )


def test_model_is_loaded_once():
    loads = []

    def factory():
        loads.append(True)
        return FakeClassifier()

    with TestClient(create_app(factory)) as client:
        for _ in range(3):
            assert (
                client.post("/predict", files={"file": ("image.png", image_bytes())}).status_code
                == 200
            )
    assert len(loads) == 1


def test_inference_failure_returns_safe_error():
    class BrokenClassifier:
        version = "broken"

        def predict(self, image):
            raise RuntimeError("private internals")

    with TestClient(create_app(BrokenClassifier)) as client:
        response = client.post("/predict", files={"file": ("image.png", image_bytes())})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "inference_failed"
    assert "private internals" not in response.text


def test_request_limit_before_parsing(client):
    response = client.post(
        "/predict", content=b"a", headers={"content-length": str(MAX_UPLOAD_BYTES + 65537)}
    )
    assert response.status_code == 413


def test_chunked_multipart_upload_is_bounded(client):
    def chunks():
        yield (
            b"--chunk-boundary\r\n"
            b'Content-Disposition: form-data; name="file"; filename="huge.png"\r\n'
            b"Content-Type: image/png\r\n\r\n"
        )
        for _ in range(11):
            yield b"x" * 1024 * 1024
        yield b"\r\n--chunk-boundary--\r\n"

    response = client.post(
        "/predict",
        content=chunks(),
        headers={"content-type": "multipart/form-data; boundary=chunk-boundary"},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "image_too_large"


def test_streamed_request_limit():
    # No Content-Length: ensure the actual stream is bounded before it reaches parsing.
    import asyncio

    observed = []

    async def app(scope, receive, send):
        await receive()

    async def receive():
        return {"type": "http.request", "body": b"x" * (MAX_UPLOAD_BYTES + 65537)}

    async def send(message):
        observed.append(message)

    middleware = BodyLimitMiddleware(app)
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        asyncio.run(middleware({"type": "http", "headers": []}, receive, send))
    assert exc.value.status_code == 413


def test_sample_allowlist(client):
    samples = client.get("/samples").json()
    assert len(samples) == 4
    assert client.get(samples[0]["url"]).headers["content-type"] == "image/png"
    assert client.get("/samples/not-a-sample").status_code == 404


def test_model_metadata(client):
    result = client.get("/model").json()
    assert result["device"] == "cpu"
    assert result["classes"] == 1000
    assert result["model_version"] == MODEL_VERSION
    assert result["max_upload_bytes"] == MAX_UPLOAD_BYTES


def test_frontend_assets_are_versioned_and_html_is_not_cached(client):
    import hashlib
    import re

    response = client.get("/")
    assert response.headers["cache-control"] == "no-store"
    assets = re.findall(r"/static/(?:app\.js|styles\.css)\?v=([a-f0-9]+)", response.text)
    assert len(assets) == 2 and assets[0] == assets[1]
    javascript = client.get(f"/static/app.js?v={assets[0]}")
    stylesheet = client.get(f"/static/styles.css?v={assets[1]}")
    assert javascript.status_code == stylesheet.status_code == 200
    assert assets[0] == hashlib.sha256(javascript.content + stylesheet.content).hexdigest()[:16]
    assert "__ASSET_VERSION__" not in response.text
