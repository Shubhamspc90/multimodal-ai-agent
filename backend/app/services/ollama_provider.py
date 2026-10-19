"""Ollama Cloud implementation of the text-model provider."""

import httpx
from pydantic import SecretStr

from backend.app.core.config import Settings
from backend.app.services.model_provider import TextModelProvider


class OllamaProviderError(RuntimeError):
    """Represent a safe, user-independent Ollama provider failure."""


class OllamaCloudProvider(TextModelProvider):
    """Generate text using Ollama's hosted cloud API."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        # Fail clearly before sending requests when credentials are missing.
        if settings.ollama_api_key is None:
            raise ValueError("OLLAMA_API_KEY is not configured.")

        self._api_key: SecretStr = settings.ollama_api_key
        self._model = settings.ollama_model
        self._endpoint = f"{settings.ollama_base_url.rstrip('/')}/chat"
        self._timeout = settings.ollama_timeout_seconds

        # Inject a mock transport in tests without making network calls.
        self._transport = transport

    async def generate_text(self, message: str) -> str:
        """Send a message to Ollama Cloud and return the model response."""
        headers = {
            "Authorization": f"Bearer {self._api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": message}],
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                transport=self._transport,
            ) as client:
                response = await client.post(
                    self._endpoint,
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as exc:
            raise OllamaProviderError("Ollama Cloud request timed out.") from exc
        except httpx.HTTPStatusError as exc:
            # Do not expose response bodies, which may contain sensitive details.
            raise OllamaProviderError(
                f"Ollama Cloud returned HTTP {exc.response.status_code}."
            ) from exc
        except httpx.RequestError as exc:
            raise OllamaProviderError("Could not connect to Ollama Cloud.") from exc
        except ValueError as exc:
            raise OllamaProviderError("Ollama Cloud returned invalid JSON.") from exc

        try:
            content = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise OllamaProviderError(
                "Ollama Cloud returned an unexpected response format."
            ) from exc

        if not isinstance(content, str) or not content.strip():
            raise OllamaProviderError("Ollama Cloud returned an empty response.")

        return content.strip()
