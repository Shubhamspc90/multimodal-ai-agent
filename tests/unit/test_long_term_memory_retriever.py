"""Unit tests for semantic long-term memory retrieval."""

from unittest.mock import Mock
from uuid import uuid4

import pytest

from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.long_term_memory_retriever import (
    LongTermMemoryRetriever,
)
from backend.app.services.long_term_memory_storage import LongTermMemoryStorage


def test_retrieve_memories_returns_only_results_within_distance_threshold():
    """Filter distant memories while preserving similarity ranking."""
    user_id = uuid4()
    embedding = [0.1] * 384

    embedding_service = Mock(spec=EmbeddingService)
    embedding_service.embed_text.return_value = embedding

    memory_storage = Mock(spec=LongTermMemoryStorage)
    memory_storage.search_memories.return_value = [
        {"memory_text": "Uses Python", "distance": 0.12},
        {"memory_text": "Works on an AI project", "distance": 0.35},
        {"memory_text": "Likes gardening", "distance": 0.72},
    ]

    retriever = LongTermMemoryRetriever(
        embedding_service=embedding_service,
        memory_storage=memory_storage,
        max_distance=0.35,
    )

    results = retriever.retrieve_memories(
        user_id=user_id,
        query="  Which programming language do I use?  ",
        limit=3,
    )

    assert [memory["memory_text"] for memory in results] == [
        "Uses Python",
        "Works on an AI project",
    ]

    embedding_service.embed_text.assert_called_once_with(
        "Which programming language do I use?"
    )
    memory_storage.search_memories.assert_called_once_with(
        user_id=user_id,
        query_embedding=embedding,
        limit=3,
    )


def test_retrieve_memories_includes_results_at_threshold():
    """Include a memory whose distance equals the configured threshold."""
    embedding_service = Mock(spec=EmbeddingService)
    memory_storage = Mock(spec=LongTermMemoryStorage)

    memory_storage.search_memories.return_value = [
        {"memory_text": "Relevant memory", "distance": 0.5},
    ]

    retriever = LongTermMemoryRetriever(
        embedding_service=embedding_service,
        memory_storage=memory_storage,
        max_distance=0.5,
    )

    results = retriever.retrieve_memories(
        user_id=uuid4(),
        query="Relevant query",
    )

    assert len(results) == 1
    assert results[0]["memory_text"] == "Relevant memory"


def test_retrieve_memories_returns_empty_list_when_no_results_match():
    """Return an empty list when every result exceeds the threshold."""
    embedding_service = Mock(spec=EmbeddingService)
    memory_storage = Mock(spec=LongTermMemoryStorage)

    memory_storage.search_memories.return_value = [
        {"memory_text": "Unrelated memory", "distance": 0.9},
    ]

    retriever = LongTermMemoryRetriever(
        embedding_service=embedding_service,
        memory_storage=memory_storage,
        max_distance=0.5,
    )

    results = retriever.retrieve_memories(
        user_id=uuid4(),
        query="Unrelated query",
    )

    assert results == []


def test_retrieve_memories_returns_empty_list_when_storage_returns_no_memories():
    """Handle users who do not have any stored memories."""
    embedding_service = Mock(spec=EmbeddingService)
    memory_storage = Mock(spec=LongTermMemoryStorage)
    memory_storage.search_memories.return_value = []

    retriever = LongTermMemoryRetriever(
        embedding_service=embedding_service,
        memory_storage=memory_storage,
    )

    results = retriever.retrieve_memories(
        user_id=uuid4(),
        query="What do I prefer?",
    )

    assert results == []


@pytest.mark.parametrize("query", ["", "   "])
def test_retrieve_memories_rejects_blank_query(query):
    """Reject empty queries before invoking either dependency."""
    embedding_service = Mock(spec=EmbeddingService)
    memory_storage = Mock(spec=LongTermMemoryStorage)

    retriever = LongTermMemoryRetriever(
        embedding_service=embedding_service,
        memory_storage=memory_storage,
    )

    with pytest.raises(ValueError, match="query must not be empty"):
        retriever.retrieve_memories(uuid4(), query)

    embedding_service.embed_text.assert_not_called()
    memory_storage.search_memories.assert_not_called()


def test_retrieve_memories_rejects_invalid_user_id():
    """Require a UUID to preserve the authenticated user's scope."""
    embedding_service = Mock(spec=EmbeddingService)
    memory_storage = Mock(spec=LongTermMemoryStorage)

    retriever = LongTermMemoryRetriever(
        embedding_service=embedding_service,
        memory_storage=memory_storage,
    )

    with pytest.raises(ValueError, match="user_id must be a UUID"):
        retriever.retrieve_memories("invalid-user-id", "Test query")

    embedding_service.embed_text.assert_not_called()
    memory_storage.search_memories.assert_not_called()


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_retrieve_memories_rejects_invalid_limit(limit):
    """Require a positive integer result limit."""
    embedding_service = Mock(spec=EmbeddingService)
    memory_storage = Mock(spec=LongTermMemoryStorage)

    retriever = LongTermMemoryRetriever(
        embedding_service=embedding_service,
        memory_storage=memory_storage,
    )

    with pytest.raises(ValueError, match="limit must be a positive integer"):
        retriever.retrieve_memories(uuid4(), "Test query", limit=limit)

    embedding_service.embed_text.assert_not_called()
    memory_storage.search_memories.assert_not_called()


@pytest.mark.parametrize(
    "max_distance",
    [True, -0.1, 2.1, float("nan"), float("inf"), "0.5"],
)
def test_retriever_rejects_invalid_max_distance(max_distance):
    """Reject invalid cosine-distance thresholds."""
    with pytest.raises(ValueError, match="max_distance must be a finite value"):
        LongTermMemoryRetriever(max_distance=max_distance)


@pytest.mark.parametrize("max_distance", [0, 2])
def test_retriever_accepts_distance_boundaries(max_distance):
    """Accept the valid minimum and maximum cosine distances."""
    retriever = LongTermMemoryRetriever(max_distance=max_distance)

    assert retriever._max_distance == float(max_distance)


def test_retriever_initializes_default_dependencies():
    """Create lazy default services when dependencies are not injected."""
    retriever = LongTermMemoryRetriever()

    assert isinstance(retriever._embedding_service, EmbeddingService)
    assert isinstance(retriever._memory_storage, LongTermMemoryStorage)
    assert retriever._embedding_service._model is None
