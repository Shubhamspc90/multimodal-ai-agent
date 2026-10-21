from io import BytesIO
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from PIL import Image, UnidentifiedImageError

from backend.app.schemas.image_understanding import ImageUnderstandingResponse
from backend.app.services.image_service import ImageService
from backend.app.services.ollama_image_provider import ImageProviderError

"""Image upload validation and image-understanding API."""

router = APIRouter(tags=["Images"])

MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000

ALLOWED_CONTENT_TYPES = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}


def get_image_service() -> ImageService:
    """Create the service dependency for the current request."""
    return ImageService()


@router.post(
    "/images",
    response_model=ImageUnderstandingResponse,
)
async def understand_image(
    file: Annotated[UploadFile, File()],
    service: Annotated[ImageService, Depends(get_image_service)],
    prompt: Annotated[
        str,
        Form(description="Question or instruction about the uploaded image."),
    ] = "Describe this image in detail.",
) -> ImageUnderstandingResponse:
    """Validate an uploaded image and send it to the vision model."""
    expected_format = ALLOWED_CONTENT_TYPES.get(file.content_type or "")

    if expected_format is None:
        await file.close()
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported image content type. Use JPEG, PNG, or WEBP.",
        )

    if not prompt.strip():
        await file.close()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The image prompt must not be empty.",
        )

    try:
        contents = await file.read(MAX_IMAGE_SIZE_BYTES + 1)

        if not contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded image is empty.",
            )

        if len(contents) > MAX_IMAGE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail="Image exceeds the 5 MiB upload limit.",
            )

        # Inspect the real file contents instead of trusting the MIME type.
        with Image.open(BytesIO(contents)) as image:
            detected_format = image.format
            width, height = image.size

            if detected_format != expected_format:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail="File content does not match its content type.",
                )

            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Image dimensions exceed the allowed pixel limit.",
                )

            image.load()

    except HTTPException:
        raise
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
    finally:
        await file.close()

    try:
        # Inference occurs only after upload and image validation succeed.
        model_response = await service.process_image(
            image_bytes=contents,
            prompt=prompt,
        )
    except ImageProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The image-understanding provider failed to process the image.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return ImageUnderstandingResponse(
        filename=file.filename or "upload",
        content_type=file.content_type or "",
        size_bytes=len(contents),
        image_format=detected_format,
        width=width,
        height=height,
        response=model_response,
    )
