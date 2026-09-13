"""Bounded decoding of supported static images."""

import warnings
from io import BytesIO

from fastapi import HTTPException
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


def decode_image(data: bytes) -> tuple[Image.Image, dict]:
    if not data:
        raise HTTPException(
            422, detail={"code": "empty_image", "message": "Choose a non-empty image."}
        )
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413, detail={"code": "image_too_large", "message": "Images must be 10 MB or smaller."}
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                image_format = image.format
                if image_format not in SUPPORTED_FORMATS:
                    raise HTTPException(
                        415,
                        detail={
                            "code": "unsupported_image",
                            "message": "Use a JPEG, PNG, or WebP image.",
                        },
                    )
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise Image.DecompressionBombError
                if getattr(image, "n_frames", 1) != 1:
                    raise HTTPException(
                        415,
                        detail={
                            "code": "animated_image",
                            "message": "Use a static image; animations are not supported.",
                        },
                    )
                image.verify()
            with Image.open(BytesIO(data)) as image:
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                rgb.load()
                return rgb, {"width": rgb.width, "height": rgb.height, "format": image_format}
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(
            413,
            detail={
                "code": "too_many_pixels",
                "message": "Images must contain 20 megapixels or fewer.",
            },
        ) from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise HTTPException(
            422,
            detail={
                "code": "invalid_image",
                "message": "This file could not be decoded. Choose a valid image.",
            },
        ) from exc
