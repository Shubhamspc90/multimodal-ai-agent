"""API route for AI-generated text responses."""

import logging

from fastapi import APIRouter, HTTPException, status

from backend.app.core.config import settings
from backend.app.schemas.agent import AgentRequest, AgentResponse
from backend.app.schemas.text import TextInputRequest
from backend.app.services.agent_service import AgentService
from backend.app.services.ollama_provider import (
    OllamaCloudProvider,
    OllamaProviderError,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Text"])


def get_agent_service() -> AgentService:
    """Build the text agent using the configured Ollama provider."""
    try:
        provider = OllamaCloudProvider(settings)
        return AgentService(text_provider=provider)
    except ValueError as exc:
        logger.warning("Text generation is not configured.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Text generation is not configured or available.",
        ) from exc


@router.post(
    "/text",
    response_model=AgentResponse,
    status_code=status.HTTP_200_OK,
)
async def submit_text(request: TextInputRequest) -> AgentResponse:
    """Generate an AI response for a user's text message."""
    service = get_agent_service()

    try:
        return await service.process_text(AgentRequest(message=request.message))
    except OllamaProviderError as exc:
        logger.warning("The text generation provider failed.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The text generation provider is temporarily unavailable.",
        ) from exc
    except ValueError as exc:
        logger.warning("Text generation returned an unusable response.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The text generation provider returned an unusable response.",
        ) from exc
