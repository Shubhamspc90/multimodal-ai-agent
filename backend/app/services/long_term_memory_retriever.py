"""Semantic retrieval service for user-scoped long-term memories."""

import logging
import math
import re
from typing import Any
from uuid import UUID

from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.long_term_memory_storage import LongTermMemoryStorage

logger = logging.getLogger(__name__)


class LongTermMemoryRetriever:
    """Retrieve relevant memories using single or multi-part semantic queries."""

    DEFAULT_LIMIT = 5
    DEFAULT_MAX_DISTANCE = 0.5

    _QUESTION_START_PATTERN = re.compile(
        r"^(?:what|which|who|where|when|why|how|"
        r"do|does|did|is|are|am|can|could|would|should|"
        r"will|have|has|was|were)\b",
        re.IGNORECASE,
    )

    _QUERY_SEPARATOR_PATTERN = re.compile(
        r"[,;\n]+|\s+and\s+(?="
        r"what|which|who|where|when|why|how|"
        r"do|does|did|is|are|am|can|could|would|should|"
        r"will|have|has|was|were"
        r"\b)",
        re.IGNORECASE,
    )

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

    @classmethod
    def _build_search_queries(cls, query: str) -> list[str]:
        """Build independent and alternative searches for question parts."""
        original_query = query.strip()
        segments = [
            segment.strip()
            for segment in cls._QUERY_SEPARATOR_PATTERN.split(original_query)
            if segment.strip()
        ]

        question_segments = [
            segment
            for segment in segments
            if cls._QUESTION_START_PATTERN.match(segment)
        ]

        if len(question_segments) < 2:
            return [original_query]

        queries = [original_query]
        seen = {original_query.casefold()}

        def add_query(candidate: str) -> None:
            normalized = " ".join(candidate.split()).strip()
            if not normalized:
                return

            key = normalized.casefold()
            if key not in seen:
                queries.append(normalized)
                seen.add(key)

        for segment in question_segments:
            normalized = " ".join(segment.split())
            add_query(normalized)

            if re.search(r"\bproject\b", normalized, re.IGNORECASE):
                add_query("What project am I currently developing?")
                add_query("What project am I working on?")

        return queries

    @staticmethod
    def _memory_identity(memory: dict[str, Any]) -> tuple[str, str]:
        """Identify a memory consistently across independent searches."""
        memory_id = memory.get("id")

        if memory_id is not None:
            return ("id", str(memory_id))

        memory_text = " ".join(memory["memory_text"].casefold().split())
        category = str(memory.get("category", ""))

        return ("text", f"{category}:{memory_text}")

    def retrieve_memories(
        self,
        user_id: UUID,
        query: str,
        limit: int = DEFAULT_LIMIT,
    ) -> list[dict[str, Any]]:
        """Return relevant, deduplicated memories belonging to one user."""
        if not isinstance(user_id, UUID):
            raise ValueError("user_id must be a UUID.")

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty.")

        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer.")

        search_queries = self._build_search_queries(query)
        best_memories: dict[tuple[str, str], dict[str, Any]] = {}

        logger.info(
            "LTM retrieval started: searches=%d, max_distance=%.3f, limit=%d",
            len(search_queries),
            self._max_distance,
            limit,
        )

        for search_index, search_query in enumerate(search_queries, start=1):
            logger.info(
                "LTM search %d/%d started",
                search_index,
                len(search_queries),
            )

            query_embedding = self._embedding_service.embed_text(search_query)

            memories = self._memory_storage.search_memories(
                user_id=user_id,
                query_embedding=query_embedding,
                limit=limit,
            )

            logger.info(
                "LTM search %d/%d returned %d database candidates",
                search_index,
                len(search_queries),
                len(memories),
            )

            for memory in memories:
                distance = float(memory["distance"])
                memory_id = memory.get("id")
                category = memory.get("category", "unknown")
                accepted = distance <= self._max_distance

                logger.info(
                    "LTM candidate: search=%d/%d, id=%s, "
                    "category=%s, distance=%.4f, accepted=%s",
                    search_index,
                    len(search_queries),
                    memory_id,
                    category,
                    distance,
                    accepted,
                )

                if not accepted:
                    continue

                candidate = {
                    **memory,
                    "distance": distance,
                }
                identity = self._memory_identity(candidate)
                existing = best_memories.get(identity)

                if existing is None or distance < existing["distance"]:
                    best_memories[identity] = candidate

        ranked_memories = sorted(
            best_memories.values(),
            key=lambda memory: memory["distance"],
        )

        final_memories = ranked_memories[:limit]

        logger.info(
            "LTM retrieval completed: accepted_unique=%d, final_count=%d",
            len(ranked_memories),
            len(final_memories),
        )

        for memory in final_memories:
            logger.info(
                "LTM final result: id=%s, category=%s, distance=%.4f",
                memory.get("id"),
                memory.get("category", "unknown"),
                memory["distance"],
            )

        return final_memories
