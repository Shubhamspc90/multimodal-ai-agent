"""API routes for persistent chat conversations."""

import logging
from functools import lru_cache
from typing import Annotated
from uuid import UUID

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.app.api.routes.text import get_agent_service
from backend.app.schemas.agent import AgentRequest
from backend.app.schemas.chat_history import (
    ChatMessageCreateRequest,
    ChatMessagePairResponse,
    ConversationCreateRequest,
    ConversationDetail,
    ConversationSummary,
)
from backend.app.services.agent_service import AgentService
from backend.app.services.chat_history import PostgresChatHistoryStorage
from backend.app.services.ollama_provider import OllamaProviderError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["Conversations"])


@lru_cache
def get_chat_history_storage() -> PostgresChatHistoryStorage:
    """Return the shared chat history storage service."""
    return PostgresChatHistoryStorage()


def _database_unavailable(exc: Exception) -> HTTPException:
    logger.exception("Chat history database operation failed.")
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Chat history storage is temporarily unavailable.",
    )
    raise _database_unavailable(exc) from exc


@router.post(
    "",
    response_model=ConversationSummary,
    status_code=status.HTTP_201_CREATED,
)
def create_conversation(
    request: ConversationCreateRequest,
    storage: Annotated[
        PostgresChatHistoryStorage,
        Depends(get_chat_history_storage),
    ],
) -> dict:
    """Create an empty conversation."""
    try:
        return storage.create_conversation(request.client_id)
    except (psycopg.Error, ValueError) as exc:
        raise _database_unavailable(exc) from exc


@router.get("", response_model=list[ConversationSummary])
def list_conversations(
    client_id: Annotated[UUID, Query()],
    storage: Annotated[
        PostgresChatHistoryStorage,
        Depends(get_chat_history_storage),
    ],
) -> list[dict]:
    """List conversations belonging to the supplied client."""
    try:
        return storage.list_conversations(client_id)
    except (psycopg.Error, ValueError) as exc:
        raise _database_unavailable(exc) from exc


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetail,
)
def load_conversation(
    conversation_id: UUID,
    client_id: Annotated[UUID, Query()],
    storage: Annotated[
        PostgresChatHistoryStorage,
        Depends(get_chat_history_storage),
    ],
) -> dict:
    """Load a conversation and its messages."""
    try:
        conversation = storage.get_conversation(
            conversation_id,
            client_id,
        )
    except (psycopg.Error, ValueError) as exc:
        raise _database_unavailable(exc) from exc

    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )

    return conversation


@router.post(
    "/{conversation_id}/messages",
    response_model=ChatMessagePairResponse,
)
async def send_conversation_message(
    conversation_id: UUID,
    request: ChatMessageCreateRequest,
    storage: Annotated[
        PostgresChatHistoryStorage,
        Depends(get_chat_history_storage),
    ],
) -> dict:
    """Save a user message, generate an AI response, and save it."""
    try:
        user_message = storage.append_message(
            conversation_id=conversation_id,
            client_id=request.client_id,
            role="user",
            content=request.message,
        )
    except (psycopg.Error, ValueError) as exc:
        raise _database_unavailable(exc) from exc

    if user_message is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )

    try:
        service: AgentService = get_agent_service()
        agent_response = await service.process_text(
            AgentRequest(message=request.message)
        )
    except HTTPException:
        # The user message remains saved so it can be recovered.
        raise
    except OllamaProviderError as exc:
        logger.warning("The text generation provider failed.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The text generation provider is temporarily unavailable.",
        ) from exc
    except ValueError as exc:
        logger.warning("The text generation provider returned unusable output.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The text generation provider returned an unusable response.",
        ) from exc

    try:
        assistant_message = storage.append_message(
            conversation_id=conversation_id,
            client_id=request.client_id,
            role="assistant",
            content=agent_response.response,
        )
    except (psycopg.Error, ValueError) as exc:
        raise _database_unavailable(exc) from exc

    if assistant_message is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )

    return {
        "conversation_id": conversation_id,
        "user_message": user_message,
        "assistant_message": assistant_message,
    }
