"""Unit tests for the Ollama Cloud text provider."""

import json

import httpx
import pytest
from pydantic import SecretStr

from backend.app.core.config import Settings
from backend.app.services.ollama_provider import (
    OllamaCloudProvider,
    OllamaProviderError,
)


def make_settings() -> Settings:
    """Create isolated test settings without a real API key."""
    return Settings(
        ollama_api_key=SecretStr("test-key"),
        ollama_model="gemma4:31b",
    )


@pytest.mark.asyncio
async def test_generate_text_returns_model_content() -> None:
    """Return the assistant content from a valid cloud response."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        assert request.headers["Authorization"] == "Bearer test-key"

        payload = json.loads(request.content)
        assert payload["model"] == "gemma4:31b"
        assert payload["stream"] is False

        return httpx.Response(
            200,
            json={"message": {"content": "  Hello from Ollama!  "}},
        )

    provider = OllamaCloudProvider(
        make_settings(),
        transport=httpx.MockTransport(handler),
    )

    result = await provider.generate_text("Say hello.")

    assert result == "Hello from Ollama!"


@pytest.mark.asyncio
async def test_generate_text_rejects_empty_content() -> None:
    """Reject an empty assistant response."""

    transport = httpx.MockTransport(
        lambda _: httpx.Response(
            200,
            json={"message": {"content": "  "}},
        )
    )
    provider = OllamaCloudProvider(make_settings(), transport=transport)

    with pytest.raises(OllamaProviderError, match="empty response"):
        await provider.generate_text("Say hello.")


@pytest.mark.asyncio
async def test_generate_text_handles_http_errors() -> None:
    """Convert HTTP status failures to a safe provider error."""

    transport = httpx.MockTransport(
        lambda _: httpx.Response(401, json={"error": "Unauthorized"})
    )
    provider = OllamaCloudProvider(make_settings(), transport=transport)

    with pytest.raises(OllamaProviderError, match="HTTP 401"):
        await provider.generate_text("Say hello.")


@pytest.mark.asyncio
async def test_generate_text_handles_timeouts() -> None:
    """Convert network timeouts to a safe provider error."""

    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated timeout")

    provider = OllamaCloudProvider(
        make_settings(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(OllamaProviderError, match="timed out"):
        await provider.generate_text("Say hello.")


def test_provider_requires_api_key() -> None:
    """Reject missing cloud credentials before making a request."""

    settings = Settings(ollama_api_key=None)

    with pytest.raises(ValueError, match="OLLAMA_API_KEY"):
        OllamaCloudProvider(settings)
