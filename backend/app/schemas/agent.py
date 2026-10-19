"""Public request and response schemas for the agent API."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AgentInputType(StrEnum):
    """Input modalities supported by the agent architecture."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"


class AgentRequest(BaseModel):
    """Request contract for text-based agent interactions."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(
        min_length=1,
        max_length=10_000,
        description="The user's text instruction.",
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        """Reject messages that contain only whitespace."""
        value = value.strip()
        if not value:
            raise ValueError("Message must not be empty or whitespace only.")
        return value


class AgentResponse(BaseModel):
    """Stable response contract for agent interactions."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(
        default="completed",
        description="Processing status.",
    )
    input_type: AgentInputType
    response: str = Field(
        min_length=1,
        description="The response returned to the user.",
    )
    capability: str = Field(
        min_length=1,
        description="The capability used to process the request.",
    )
