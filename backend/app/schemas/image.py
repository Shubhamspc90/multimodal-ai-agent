from typing import Literal

from pydantic import BaseModel, Field


class ImageInputResponse(BaseModel):
    """Represent metadata for an accepted image upload."""

    # Fixed response status for a successfully accepted image.
    status: Literal["accepted"] = "accepted"

    # Original filename sent by the client.
    filename: str

    # MIME content type declared by the client.
    content_type: str

    # Uploaded image size in bytes.
    size_bytes: int = Field(gt=0)

    # Real image format detected by Pillow.
    image_format: Literal["JPEG", "PNG", "WEBP"]

    # Image width in pixels.
    width: int = Field(gt=0)

    # Image height in pixels.
    height: int = Field(gt=0)
