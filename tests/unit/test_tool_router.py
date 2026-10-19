"""Unit tests for deterministic tool routing."""

import pytest

from backend.app.schemas.agent import AgentInputType
from backend.app.services.tool_router import ToolName, ToolRouter


@pytest.mark.parametrize(
    ("input_type", "expected_tool"),
    [
        (AgentInputType.TEXT, ToolName.TEXT_GENERATION),
        (AgentInputType.IMAGE, ToolName.IMAGE_UNDERSTANDING),
        (AgentInputType.AUDIO, ToolName.AUDIO_TRANSCRIPTION),
    ],
)
def test_router_selects_tool_for_supported_modality(
    input_type: AgentInputType,
    expected_tool: ToolName,
) -> None:
    """Select the expected tool for each supported input modality."""
    decision = ToolRouter().route(input_type)

    assert decision.input_type == input_type
    assert decision.tool == expected_tool


@pytest.mark.parametrize("input_type", ["video", None, 123])
def test_router_rejects_unsupported_modality(input_type: object) -> None:
    """Reject values that are not supported input modalities."""
    with pytest.raises(ValueError, match="Unsupported input modality"):
        ToolRouter().route(input_type)  # type: ignore[arg-type]


def test_routing_decision_is_immutable() -> None:
    """Prevent accidental mutation of a routing decision."""
    decision = ToolRouter().route(AgentInputType.TEXT)

    with pytest.raises(AttributeError):
        decision.tool = ToolName.IMAGE_UNDERSTANDING  # type: ignore[misc]
