"""Unit tests for document summarization."""

from unittest.mock import AsyncMock

import pytest

from backend.app.services.summarization_service import SummarizationService


@pytest.mark.asyncio
async def test_summarize_returns_trimmed_summary() -> None:
    """Return the provider's summary without surrounding whitespace."""

    provider = AsyncMock()
    provider.generate_text.return_value = "  Main findings of the document.  "
    service = SummarizationService(provider)

    result = await service.summarize("Original document text.")

    assert result == "Main findings of the document."
    provider.generate_text.assert_awaited_once()

    prompt = provider.generate_text.await_args.args[0]
    assert "Original document text." in prompt
    assert "Do not invent facts" in prompt


@pytest.mark.asyncio
async def test_summarize_rejects_empty_summary() -> None:
    """Raise an error if the provider returns an empty summary."""

    provider = AsyncMock()
    provider.generate_text.return_value = "   "
    service = SummarizationService(provider)

    with pytest.raises(ValueError, match="empty summary"):
        await service.summarize("Original document text.")
