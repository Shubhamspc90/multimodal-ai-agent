# BytesIO lets Pillow treat the uploaded image bytes like a file.
# Annotated attaches metadata to the UploadFile parameter.
# APIRouter groups related endpoints in a separate file.
# UploadFile represents an uploaded file.
# HTTPException sends an error response with an HTTP status code.
# Image and UnidentifiedImageError are Pillow classes used for image validation.
# ImageInputResponse is your custom response model.

from io import BytesIO
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from backend.app.schemas.image import ImageInputResponse

# Create an image API router with the "Images" tag for documentation.
router = APIRouter(tags=["Images"])

# Maximum allowed image file size: 5 MiB.
MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024

# Maximum allowed total image pixels to prevent memory-heavy images.
MAX_IMAGE_PIXELS = 20_000_000

# Map allowed content types to their expected Pillow image formats.
ALLOWED_CONTENT_TYPES = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}


@router.post(
    "/images",
    response_model=ImageInputResponse,
    status_code=status.HTTP_200_OK,
)
async def upload_image(
    file: Annotated[UploadFile, File(...)],
) -> ImageInputResponse:
    """Accept and validate an uploaded image."""

    # Get the expected image format from the uploaded file's content type.
    expected_format = ALLOWED_CONTENT_TYPES.get(file.content_type or "")

    # Reject unsupported image content types.
    if expected_format is None:
        await file.close()
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported image content type.",
        )

    try:
        # Read one extra byte to detect files exceeding the size limit.
        contents = await file.read(MAX_IMAGE_SIZE_BYTES + 1)
    finally:
        # Always close the uploaded file after reading it.
        await file.close()

    # Reject empty uploaded files.
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )

    # Reject images larger than the 5 MiB limit.
    if len(contents) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="Image exceeds the 5 MiB upload limit.",
        )

    try:
        # Open and inspect the uploaded image using Pillow.
        with Image.open(BytesIO(contents)) as image:
            # Detect the actual image format from the file contents.
            detected_format = image.format

            # Get the image width and height in pixels.
            width, height = image.size

            # Reject files whose real format does not match the declared type.
            if detected_format != expected_format:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail="File content does not match its content type.",
                )

            # Reject images that exceed the allowed pixel limit.
            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail="Image dimensions exceed the allowed pixel limit.",
                )

            # Decode the image to detect corrupted or truncated files.
            image.load()

    # Re-raise HTTP validation errors without changing them.
    except HTTPException:
        raise

    # Convert invalid or unreadable image errors into a 400 response.
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is not a valid image.",
        ) from exc

    # Return the validated image metadata.
    return ImageInputResponse(
        filename=file.filename or "upload",
        content_type=file.content_type or "",
        size_bytes=len(contents),
        image_format=detected_format,
        width=width,
        height=height,
    )
