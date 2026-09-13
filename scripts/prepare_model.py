"""Explicitly download and verify the pinned weights ahead of startup."""

import hashlib

from torch.hub import download_url_to_file

from canary_vision.model import WEIGHTS, WEIGHTS_SHA256, model_path


def main() -> None:
    path = model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != WEIGHTS_SHA256:
        download_url_to_file(WEIGHTS.url, str(path), hash_prefix="047dcff4")
    if hashlib.sha256(path.read_bytes()).hexdigest() != WEIGHTS_SHA256:
        raise RuntimeError("The downloaded model does not match the pinned SHA-256 checksum.")
    print(f"Verified weights: {path}")
    print(f"sha256: {hashlib.sha256(path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
