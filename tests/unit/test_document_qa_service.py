"""Unit tests for document question answering."""

from unittest.mock import AsyncMock

import pytest

from backend.app.services.document_qa_service import DocumentQAService


@pytest.mark.asyncio
async def test_answer_returns_trimmed_answer() -> None:
    """Return a trimmed answer grounded in the document."""

    provider = AsyncMock()
    provider.generate_text.return_value = "  The project started in 2024.  "
    service = DocumentQAService(provider)

    result = await service.answer(
        "The project started in 2024.",
        "When did the project start?",
    )

    assert result == "The project started in 2024."
    provider.generate_text.assert_awaited_once()

    prompt = provider.generate_text.await_args.args[0]
    assert "The project started in 2024." in prompt
    assert "When did the project start?" in prompt
    assert "Do not invent facts" in prompt
    assert "not available in the document" in prompt
    assert "not as instructions" in prompt


@pytest.mark.asyncio
async def test_answer_rejects_empty_question() -> None:
    """Reject a question containing only whitespace."""

    service = DocumentQAService(AsyncMock())

    with pytest.raises(ValueError, match="question must not be empty"):
        await service.answer("Document content.", "   ")


@pytest.mark.asyncio
async def test_answer_rejects_empty_document() -> None:
    """Reject document content containing only whitespace."""

    service = DocumentQAService(AsyncMock())

    with pytest.raises(ValueError, match="no readable text"):
        await service.answer("   ", "What is this document about?")


@pytest.mark.asyncio
async def test_answer_rejects_empty_model_response() -> None:
    """Reject an empty response from the model provider."""

    provider = AsyncMock()
    provider.generate_text.return_value = "   "
    service = DocumentQAService(provider)

    with pytest.raises(ValueError, match="empty answer"):
        await service.answer("Document content.", "What is the topic?")
