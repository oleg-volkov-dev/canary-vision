"""Verify health, readiness, webpage, and real inference using only the stdlib."""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--wait", type=float, default=30)
    args = parser.parse_args()
    deadline = time.monotonic() + args.wait
    while True:
        try:
            with urllib.request.urlopen(f"{args.url}/ready", timeout=2) as response:
                ready = json.load(response)
                assert ready["status"] == "ready"
                assert ready["device"] == "cpu"
                break
        except (urllib.error.URLError, TimeoutError):
            if time.monotonic() >= deadline:
                raise RuntimeError("Service did not become ready in time.") from None
            time.sleep(0.25)
    with urllib.request.urlopen(f"{args.url}/health", timeout=5) as response:
        assert json.load(response)["status"] == "ok"
    with urllib.request.urlopen(args.url, timeout=5) as response:
        assert b"See what your model sees" in response.read()
    image = Path("evaluation/images/coffee.png").read_bytes()
    boundary = "canary-smoke-boundary"
    body = (
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="coffee.png"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode()
        + image
        + f"\r\n--{boundary}--\r\n".encode()
    )
    request = urllib.request.Request(
        f"{args.url}/predict",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
    assert len(result["predictions"]) == 3
    assert result["predictions"][0]["label"] == "espresso"
    assert result["predictions"][0]["score"] > 0.5
    assert result["model_version"] == ready["model_version"]
    assert result["inference_ms"] >= 0
    print(json.dumps({"smoke": "passed", **result}, indent=2))


if __name__ == "__main__":
    main()
