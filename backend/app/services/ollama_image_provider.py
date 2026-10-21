"""Ollama Cloud provider for image-understanding requests."""

import base64

import httpx

from backend.app.core.config import settings


class ImageProviderError(RuntimeError):
    """Raised when image inference fails."""


class OllamaImageProvider:
    """Generate image descriptions using an Ollama vision model."""

    def __init__(self) -> None:
        self._model = settings.ollama_vision_model
        self._endpoint = f"{settings.ollama_base_url.rstrip('/')}/chat"
        self._timeout = settings.ollama_timeout_seconds

    async def understand_image(
        self,
        image_bytes: bytes,
        prompt: str,
    ) -> str:
        """Send a Base64 image and prompt to the configured vision model."""
        if not image_bytes:
            raise ValueError("Image content must not be empty.")

        if not prompt or not prompt.strip():
            raise ValueError("Image prompt must not be empty.")

        api_key = settings.ollama_api_key
        if api_key is None or not api_key.get_secret_value().strip():
            raise ImageProviderError("Ollama API credentials are not configured.")

        encoded_image = base64.b64encode(image_bytes).decode("ascii")

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt.strip(),
                    "images": [encoded_image],
                }
            ],
            "stream": False,
        }

        headers = {
            "Authorization": f"Bearer {api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    self._endpoint,
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as exc:
            raise ImageProviderError("The image model request timed out.") from exc
        except httpx.HTTPStatusError as exc:
            # Do not expose credentials or raw provider responses to API clients.
            raise ImageProviderError(
                f"The image model returned HTTP {exc.response.status_code}."
            ) from exc
        except httpx.RequestError as exc:
            raise ImageProviderError(
                "Could not connect to the image model provider."
            ) from exc
        except ValueError as exc:
            raise ImageProviderError(
                "The image model returned an invalid response."
            ) from exc

        message = data.get("message")
        result = message.get("content") if isinstance(message, dict) else None

        if not isinstance(result, str) or not result.strip():
            raise ImageProviderError("The image model returned an empty response.")

        return result.strip()
