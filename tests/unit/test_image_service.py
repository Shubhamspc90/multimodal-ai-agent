"""Unit tests for image-understanding orchestration."""

from unittest.mock import AsyncMock

import pytest

from backend.app.services.image_service import ImageService


@pytest.mark.asyncio
async def test_service_returns_provider_response() -> None:
    provider = AsyncMock()
    provider.understand_image.return_value = "A red car."

    service = ImageService(provider=provider)

    result = await service.process_image(
        image_bytes=b"image-bytes",
        prompt="What is in this image?",
    )

    assert result == "A red car."
    provider.understand_image.assert_awaited_once_with(
        image_bytes=b"image-bytes",
        prompt="What is in this image?",
    )


@pytest.mark.asyncio
async def test_service_rejects_empty_image() -> None:
    provider = AsyncMock()
    service = ImageService(provider=provider)

    with pytest.raises(ValueError, match="Image content must not be empty"):
        await service.process_image(b"", "Describe this image.")

    provider.understand_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_service_rejects_blank_prompt() -> None:
    provider = AsyncMock()
    service = ImageService(provider=provider)

    with pytest.raises(ValueError, match="Image prompt must not be empty"):
        await service.process_image(b"image-bytes", "   ")

    provider.understand_image.assert_not_awaited()
