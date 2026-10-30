"""Request and response schemas for conversation history."""

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, StringConstraints


class ConversationCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_id: UUID


class ConversationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    client_id: UUID
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime


class ConversationDetail(ConversationSummary):
    messages: list[ConversationMessage]


class ChatMessageCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_id: UUID
    message: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            max_length=10_000,
        ),
    ]


class ChatMessagePairResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: UUID
    user_message: ConversationMessage
    assistant_message: ConversationMessage