"""Tests for automatic long-term memory capture."""

from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from backend.app.services.long_term_memory_capture import LongTermMemoryCapture
from backend.app.services.long_term_memory_extractor import MemoryCandidate


@pytest.fixture
def dependencies():
    extractor = Mock()
    extractor.extract_memories = AsyncMock()

    embedding_service = Mock()
    memory_storage = Mock()

    capture = LongTermMemoryCapture(
        extractor=extractor,
        embedding_service=embedding_service,
        memory_storage=memory_storage,
    )

    return capture, extractor, embedding_service, memory_storage


@pytest.mark.asyncio
async def test_capture_memories_returns_empty_when_no_candidates(
    dependencies,
):
    capture, extractor, embedding_service, memory_storage = dependencies

    extractor.extract_memories.return_value = []

    result = await capture.capture_memories(
        user_id=uuid4(),
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert result == []
    embedding_service.embed_texts.assert_not_called()
    memory_storage.store_memory_if_new.assert_not_called()


@pytest.mark.asyncio
async def test_capture_memories_embeds_and_stores_candidates(
    dependencies,
):
    capture, extractor, embedding_service, memory_storage = dependencies

    user_id = uuid4()
    candidates = [
        MemoryCandidate(
            memory_text="I prefer Python",
            category="preference",
        ),
        MemoryCandidate(
            memory_text="I am developing a Multimodal AI Agent",
            category="project",
        ),
    ]
    embeddings = [[0.1] * 384, [0.2] * 384]
    stored_records = [
        {
            "id": uuid4(),
            "user_id": user_id,
            "memory_text": candidates[0].memory_text,
            "category": candidates[0].category,
        },
        {
            "id": uuid4(),
            "user_id": user_id,
            "memory_text": candidates[1].memory_text,
            "category": candidates[1].category,
        },
    ]

    extractor.extract_memories.return_value = candidates
    embedding_service.embed_texts.return_value = embeddings
    memory_storage.store_memory_if_new.side_effect = stored_records

    messages = [{"role": "user", "content": "I prefer Python."}]

    result = await capture.capture_memories(user_id, messages)

    assert result == stored_records
    extractor.extract_memories.assert_awaited_once_with(messages)
    embedding_service.embed_texts.assert_called_once_with(
        [candidate.memory_text for candidate in candidates]
    )

    assert memory_storage.store_memory_if_new.call_count == 2
    memory_storage.store_memory_if_new.assert_any_call(
        user_id,
        candidates[0].memory_text,
        candidates[0].category,
        embeddings[0],
    )
    memory_storage.store_memory_if_new.assert_any_call(
        user_id,
        candidates[1].memory_text,
        candidates[1].category,
        embeddings[1],
    )


@pytest.mark.asyncio
async def test_capture_memories_omits_existing_duplicates(dependencies):
    capture, extractor, embedding_service, memory_storage = dependencies

    user_id = uuid4()
    candidate = MemoryCandidate(
        memory_text="I prefer PostgreSQL",
        category="preference",
    )

    extractor.extract_memories.return_value = [candidate]
    embedding_service.embed_texts.return_value = [[0.1] * 384]
    memory_storage.store_memory_if_new.return_value = None

    result = await capture.capture_memories(
        user_id,
        [{"role": "user", "content": "I prefer PostgreSQL."}],
    )

    assert result == []
    memory_storage.store_memory_if_new.assert_called_once()


@pytest.mark.asyncio
async def test_capture_memories_rejects_embedding_count_mismatch(dependencies):
    capture, extractor, embedding_service, memory_storage = dependencies

    extractor.extract_memories.return_value = [
        MemoryCandidate(
            memory_text="I prefer Python",
            category="preference",
        ),
        MemoryCandidate(
            memory_text="I prefer PostgreSQL",
            category="preference",
        ),
    ]
    embedding_service.embed_texts.return_value = [[0.1] * 384]

    with pytest.raises(RuntimeError, match="different number of embeddings"):
        await capture.capture_memories(
            uuid4(),
            [{"role": "user", "content": "I prefer Python and PostgreSQL."}],
        )

    memory_storage.store_memory_if_new.assert_not_called()


@pytest.mark.asyncio
async def test_capture_memories_propagates_extraction_failure(dependencies):
    capture, extractor, embedding_service, memory_storage = dependencies

    extractor.extract_memories.side_effect = RuntimeError("Provider unavailable")

    with pytest.raises(RuntimeError, match="Provider unavailable"):
        await capture.capture_memories(
            uuid4(),
            [{"role": "user", "content": "I prefer Python."}],
        )

    embedding_service.embed_texts.assert_not_called()
    memory_storage.store_memory_if_new.assert_not_called()
