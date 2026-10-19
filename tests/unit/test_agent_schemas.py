"""Unit tests for the agent request and response schemas."""

import pytest
from pydantic import ValidationError

from backend.app.schemas.agent import AgentInputType, AgentRequest, AgentResponse


def test_agent_request_accepts_valid_message() -> None:
    """Accept a valid message and normalize surrounding whitespace."""
    request = AgentRequest(message="  Explain AI agents.  ")

    assert request.message == "Explain AI agents."


def test_agent_request_rejects_whitespace_only_message() -> None:
    """Prevent empty instructions from reaching the agent."""
    with pytest.raises(ValidationError):
        AgentRequest(message="   ")


def test_agent_request_rejects_oversized_message() -> None:
    """Enforce the maximum supported text length."""
    with pytest.raises(ValidationError):
        AgentRequest(message="a" * 10_001)


def test_agent_request_rejects_unexpected_fields() -> None:
    """Reject fields that are not part of the public request contract."""
    with pytest.raises(ValidationError):
        AgentRequest(message="Hello", unexpected_field="value")


def test_agent_response_accepts_valid_data() -> None:
    """Build a response with the expected public contract."""
    response = AgentResponse(
        input_type=AgentInputType.TEXT,
        response="An AI agent can plan and use tools.",
        capability="text",
    )

    assert response.status == "completed"
    assert response.input_type == AgentInputType.TEXT
    assert response.capability == "text"


def test_agent_response_rejects_empty_response() -> None:
    """Prevent empty output from being represented as a completed response."""
    with pytest.raises(ValidationError):
        AgentResponse(
            input_type=AgentInputType.TEXT,
            response="",
            capability="text",
        )


def test_agent_response_rejects_unsupported_input_type() -> None:
    """Keep response modality values within the defined enum."""
    with pytest.raises(ValidationError):
        AgentResponse(
            input_type="video",  # type: ignore[arg-type]
            response="Processed.",
            capability="video",
        )
