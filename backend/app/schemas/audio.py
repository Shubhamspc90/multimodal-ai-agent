from typing import Literal

from pydantic import BaseModel, Field


class AudioInputResponse(BaseModel):
    """Represent a successfully validated audio upload."""

    status: Literal["accepted"] = "accepted"
    filename: str
    content_type: str
    size_bytes: int = Field(gt=0)
    audio_format: Literal["WAV"]
    duration_seconds: float = Field(gt=0)
    sample_rate_hz: int = Field(gt=0)
    channels: int = Field(gt=0)
