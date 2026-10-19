"""Core orchestration service for agent requests."""

from backend.app.schemas.agent import (
    AgentInputType,
    AgentRequest,
    AgentResponse,
)
from backend.app.services.model_provider import TextModelProvider
class AgentService:
    """Coordinate text requests independently of the model provider."""

    def __init__(self, text_provider: TextModelProvider) -> None:
        # Inject the provider so model implementations remain replaceable.
        self._text_provider = text_provider

    async def process_text(self, request: AgentRequest) -> AgentResponse:
        """Generate a response and normalize it into the public contract."""
        generated_text = await self._text_provider.generate_text(request.message)

        # Reject empty provider output instead of returning a false success.
        if not generated_text or not generated_text.strip():
            raise ValueError("The text model returned an empty response.")

        return AgentResponse(
            status="completed",
            input_type=AgentInputType.TEXT,
            response=generated_text.strip(),
            capability="text",
        )
