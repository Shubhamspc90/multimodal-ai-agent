"""Response schema for completed image-understanding requests."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ImageUnderstandingResponse(BaseModel):
    """Describe the model result and validated image metadata."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["completed"] = "completed"
    filename: str
    content_type: str
    size_bytes: int = Field(gt=0)
    image_format: Literal["JPEG", "PNG", "WEBP"]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    response: str = Field(min_length=1)
    capability: Literal["image_understanding"] = "image_understanding"
