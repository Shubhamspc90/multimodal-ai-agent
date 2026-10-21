"""Application service for image-understanding requests."""

from backend.app.services.ollama_image_provider import OllamaImageProvider


class ImageService:
    """Coordinate image understanding through the configured provider."""

    def __init__(self, provider: OllamaImageProvider | None = None) -> None:
        self._provider = provider or OllamaImageProvider()

    async def process_image(
        self,
        image_bytes: bytes,
        prompt: str,
    ) -> str:
        """Validate basic service inputs and request image inference."""
        if not image_bytes:
            raise ValueError("Image content must not be empty.")

        if not prompt or not prompt.strip():
            raise ValueError("Image prompt must not be empty.")

        return await self._provider.understand_image(
            image_bytes=image_bytes,
            prompt=prompt.strip(),
        )
