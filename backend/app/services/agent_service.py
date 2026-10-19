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

    async def process_text(self, request: AgentRequest) -> AgentResponse:
        """Route a text request and normalize its provider response."""
        decision = self._tool_router.route(AgentInputType.TEXT)

        if decision.tool != ToolName.TEXT_GENERATION:
            raise ValueError(f"Unsupported tool for text input: {decision.tool}")

        generated_text = await self._text_provider.generate_text(request.message)

        if not generated_text or not generated_text.strip():
            raise ValueError("The text model returned an empty response.")

        return AgentResponse(
            status="completed",
            input_type=decision.input_type,
            response=generated_text.strip(),
            capability="text",
        )
