from typing import Annotated, Literal

from pydantic import BaseModel, StringConstraints


class TextInputRequest(BaseModel):
    """Validate a user's text input."""

    message: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            max_length=10_000,
        ),
    ]


class TextInputResponse(BaseModel):
    """Represent successful acceptance of a text input."""

    status: Literal["accepted"] = "accepted"
    received_message: str
