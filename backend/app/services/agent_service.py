"""Core orchestration service for agent requests."""

from backend.app.schemas.agent import (
    AgentInputType,
    AgentRequest,
    AgentResponse,
)
from backend.app.services.model_provider import TextModelProvider
from backend.app.services.tool_router import ToolName, ToolRouter


class AgentService:
    """Coordinate agent requests independently of model providers."""

    def __init__(
        self,
        text_provider: TextModelProvider,
        tool_router: ToolRouter | None = None,
    ) -> None:
        """Inject providers and routing dependencies."""
        self._text_provider = text_provider
        self._tool_router = tool_router or ToolRouter()

    @property
    def text_provider(self) -> TextModelProvider:
        """Expose the configured text provider for other services."""
        return self._text_provider

    async def process_text(
        self,
        request: AgentRequest,
        messages: list[dict[str, str]] | None = None,
    ) -> AgentResponse:
        """Route a text request and normalize its provider response."""
        decision = self._tool_router.route(AgentInputType.TEXT)

        if decision.tool != ToolName.TEXT_GENERATION:
            raise ValueError(f"Unsupported tool for text input: {decision.tool}")

        if messages is None:
            generated_text = await self._text_provider.generate_text(request.message)
        else:
            generated_text = await self._text_provider.generate_messages(messages)

        if not generated_text or not generated_text.strip():
            raise ValueError("The text model returned an empty response.")

        return AgentResponse(
            status="completed",
            input_type=decision.input_type,
            response=generated_text.strip(),
            capability="text",
        )
