"""Unit tests for the Ollama image provider."""

import base64
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from backend.app.services.ollama_image_provider import (
    ImageProviderError,
    OllamaImageProvider,
)


@pytest.mark.asyncio
async def test_provider_sends_image_as_base64(monkeypatch) -> None:
    """The request contains the encoded image and the user's prompt."""
    provider = OllamaImageProvider()

    monkeypatch.setattr(
        "backend.app.services.ollama_image_provider.settings.ollama_api_key",
        MagicMock(get_secret_value=lambda: "test-key"),
    )

    response = MagicMock()
    response.json.return_value = {"message": {"content": "A cat sitting on a chair."}}
    response.raise_for_status.return_value = None

    client = AsyncMock()
    client.post.return_value = response

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            self.timeout = kwargs["timeout"]

        async def __aenter__(self):
            return client

        async def __aexit__(self, *args):
            return None

    monkeypatch.setattr(
        "backend.app.services.ollama_image_provider.httpx.AsyncClient",
        FakeAsyncClient,
    )

    image_bytes = b"fake-image-bytes"
    result = await provider.understand_image(
        image_bytes,
        "Describe the image.",
    )

    assert result == "A cat sitting on a chair."

    call_kwargs = client.post.call_args.kwargs
    assert call_kwargs["headers"]["Authorization"] == "Bearer test-key"

    message = call_kwargs["json"]["messages"][0]
    assert message["content"] == "Describe the image."
    assert message["images"] == [base64.b64encode(image_bytes).decode("ascii")]
    assert call_kwargs["json"]["stream"] is False


@pytest.mark.asyncio
async def test_provider_rejects_empty_image() -> None:
    provider = OllamaImageProvider()

    with pytest.raises(ValueError, match="must not be empty"):
        await provider.understand_image(b"", "Describe this image.")


@pytest.mark.asyncio
async def test_provider_rejects_empty_prompt() -> None:
    provider = OllamaImageProvider()

    with pytest.raises(ValueError, match="prompt must not be empty"):
        await provider.understand_image(b"image-bytes", "  ")


@pytest.mark.asyncio
async def test_provider_maps_http_errors_to_provider_error(monkeypatch) -> None:
    """Provider HTTP failures are normalized into a safe exception."""
    monkeypatch.setattr(
        "backend.app.services.ollama_image_provider.settings.ollama_api_key",
        MagicMock(get_secret_value=lambda: "test-key"),
    )

    request = httpx.Request("POST", "https://ollama.com/api/chat")
    response = httpx.Response(503, request=request)

    client = AsyncMock()
    client.post.side_effect = httpx.HTTPStatusError(
        "Unavailable",
        request=request,
        response=response,
    )

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return client

        async def __aexit__(self, *args):
            return None

    monkeypatch.setattr(
        "backend.app.services.ollama_image_provider.httpx.AsyncClient",
        FakeAsyncClient,
    )

    with pytest.raises(ImageProviderError, match="HTTP 503"):
        await OllamaImageProvider().understand_image(
            b"image-bytes",
            "Describe this image.",
        )
