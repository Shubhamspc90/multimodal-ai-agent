"""Unit tests for the core agent orchestration service."""
import pytest

from backend.app.schemas.agent import AgentInputType, AgentRequest
from backend.app.services.agent_service import AgentService

class FakeTextProvider:
    """Provide deterministic responses without network access."""

    async def generate_text(self, message: str) -> str:
        # Echoing the input makes the test predictable and repeatable.
        return f"Processed: {message}"

class EmptyTextProvider:
    """Simulate a provider that returns no usable content."""

    async def generate_text(self, message: str) -> str:
        # Exercise the service's empty-output safeguard.
        return "   "

@pytest.mark.asyncio
async def test_process_text_returns_standardized_response() -> None:
    """Convert provider output into the agent response contract."""
    service = AgentService(text_provider=FakeTextProvider())

    response = await service.process_text(AgentRequest(message="Explain AI agents."))

    assert response.status == "completed"
    assert response.input_type == AgentInputType.TEXT
    assert response.response == "Processed: Explain AI agents."
    assert response.capability == "text"


@pytest.mark.asyncio
async def test_process_text_rejects_empty_provider_output() -> None:
    """Do not report successful completion when output is empty."""
    service = AgentService(text_provider=EmptyTextProvider())

    with pytest.raises(ValueError, match="empty response"):
        await service.process_text(AgentRequest(message="Hello"))
