"""Deterministic tool routing for multimodal agent requests."""

from dataclasses import dataclass
from enum import StrEnum

from backend.app.schemas.agent import AgentInputType


class ToolName(StrEnum):
    """Tool identifiers used by the agent routing layer."""

    TEXT_GENERATION = "text_generation"
    IMAGE_UNDERSTANDING = "image_understanding"
    AUDIO_TRANSCRIPTION = "audio_transcription"


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Immutable result describing the selected tool."""

    input_type: AgentInputType
    tool: ToolName


class ToolRouter:
    """Select a tool deterministically from the input modality."""

    def route(self, input_type: AgentInputType) -> RoutingDecision:
        """Return the tool associated with a supported input modality."""
        routes = {
            AgentInputType.TEXT: ToolName.TEXT_GENERATION,
            AgentInputType.IMAGE: ToolName.IMAGE_UNDERSTANDING,
            AgentInputType.AUDIO: ToolName.AUDIO_TRANSCRIPTION,
        }

        try:
            tool = routes[input_type]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Unsupported input modality: {input_type!r}") from exc

        return RoutingDecision(input_type=input_type, tool=tool)
