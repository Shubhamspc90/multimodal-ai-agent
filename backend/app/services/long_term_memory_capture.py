"""Automatic extraction and persistence of long-term memories."""

from typing import Any
from uuid import UUID

from fastapi.concurrency import run_in_threadpool

from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.long_term_memory_extractor import LongTermMemoryExtractor
from backend.app.services.long_term_memory_storage import LongTermMemoryStorage


class LongTermMemoryCapture:
    """Extract, embed, and persist useful memories for an authenticated user."""

    def __init__(
        self,
        extractor: LongTermMemoryExtractor,
        embedding_service: EmbeddingService,
        memory_storage: LongTermMemoryStorage,
    ) -> None:
        """Inject the extraction, embedding, and storage dependencies."""
        self._extractor = extractor
        self._embedding_service = embedding_service
        self._memory_storage = memory_storage

    async def capture_memories(
        self,
        user_id: UUID,
        messages: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        """Capture newly extracted memories without duplicating existing records."""
        candidates = await self._extractor.extract_memories(messages)

        if not candidates:
            return []

        memory_texts = [candidate.memory_text for candidate in candidates]

        embeddings = await run_in_threadpool(
            self._embedding_service.embed_texts,
            memory_texts,
        )

        if len(embeddings) != len(candidates):
            raise RuntimeError(
                "The embedding service returned a different number of "
                "embeddings than memory candidates."
            )

        stored_memories: list[dict[str, Any]] = []

        for index, candidate in enumerate(candidates):
            stored_memory = await run_in_threadpool(
                self._memory_storage.store_memory_if_new,
                user_id,
                candidate.memory_text,
                candidate.category,
                embeddings[index],
            )

            if stored_memory is not None:
                stored_memories.append(stored_memory)

        return stored_memories
