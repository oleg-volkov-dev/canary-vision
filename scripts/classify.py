"""Classify a local image before testing the HTTP layer."""

import argparse
import json

from PIL import Image, ImageOps

from canary_vision.model import Classifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    args = parser.parse_args()
    model = Classifier()
    with Image.open(args.image) as image:
        result = model.predict(ImageOps.exif_transpose(image).convert("RGB"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
