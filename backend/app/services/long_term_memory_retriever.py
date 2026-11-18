"""Semantic retrieval service for user-scoped long-term memories."""

import math
from uuid import UUID

from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.long_term_memory_storage import LongTermMemoryStorage


class LongTermMemoryRetriever:
    """Retrieve relevant long-term memories using semantic similarity."""

    DEFAULT_LIMIT = 5
    DEFAULT_MAX_DISTANCE = 0.5

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        memory_storage: LongTermMemoryStorage | None = None,
        max_distance: float = DEFAULT_MAX_DISTANCE,
    ) -> None:
        """Initialize the retriever with optional injectable dependencies."""
        if (
            isinstance(max_distance, bool)
            or not isinstance(max_distance, (int, float))
            or not math.isfinite(max_distance)
            or not 0 <= max_distance <= 2
        ):
            raise ValueError("max_distance must be a finite value between 0 and 2.")

        self._embedding_service = (
            embedding_service if embedding_service is not None else EmbeddingService()
        )
        self._memory_storage = (
            memory_storage if memory_storage is not None else LongTermMemoryStorage()
        )
        self._max_distance = float(max_distance)

    def retrieve_memories(
        self,
        user_id: UUID,
        query: str,
        limit: int = DEFAULT_LIMIT,
    ) -> list[dict]:
        """Return relevant memories belonging to the specified user."""
        if not isinstance(user_id, UUID):
            raise ValueError("user_id must be a UUID.")

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty.")

        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer.")

        query_embedding = self._embedding_service.embed_text(query.strip())

        memories = self._memory_storage.search_memories(
            user_id=user_id,
            query_embedding=query_embedding,
            limit=limit,
        )

        return [
            memory for memory in memories if memory["distance"] <= self._max_distance
        ]
