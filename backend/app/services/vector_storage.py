"""PostgreSQL storage service for document chunks and embeddings."""

from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from backend.app.core.config import Settings, settings


class PostgresVectorStorage:
    """Persist document chunks and their embeddings in PostgreSQL."""

    def __init__(self, config: Settings | None = None) -> None:
        self._config = config or settings

    @contextmanager
    def _connection(self):
        """Open a configured PostgreSQL connection."""
        password = self._config.postgres_password

        if password is None:
            raise ValueError(
                "POSTGRES_PASSWORD must be configured before using vector storage."
            )

        with psycopg.connect(
            host=self._config.postgres_host,
            port=self._config.postgres_port,
            dbname=self._config.postgres_db,
            user=self._config.postgres_user,
            password=password.get_secret_value(),
        ) as connection:
            register_vector(connection)
            yield connection

    def store_chunks(
        self,
        document_id: str,
        chunks: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Insert or update the chunks and embeddings for a document."""
        if not document_id.strip():
            raise ValueError("document_id must not be empty.")

        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding.")

        if any(not chunk.strip() for chunk in chunks):
            raise ValueError("Chunks must not be empty.")

        rows = []

        for index, (chunk, embedding) in enumerate(
            zip(chunks, embeddings, strict=True)
        ):
            if len(embedding) != 384:
                raise ValueError(
                    f"Embedding at index {index} must contain 384 dimensions."
                )

            vector = Vector(embedding)

            rows.append(
                (
                    document_id,
                    index,
                    chunk,
                    vector,
                    Jsonb(metadata or {}),
                )
            )

        if not rows:
            return 0

        query = """
            INSERT INTO document_chunks (
                document_id,
                chunk_index,
                chunk_text,
                embedding,
                metadata
            )
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (document_id, chunk_index)
            DO UPDATE SET
                chunk_text = EXCLUDED.chunk_text,
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata,
                updated_at = CURRENT_TIMESTAMP
        """

        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.executemany(query, rows)

        return len(rows)

    def search_similar_chunks(
        self,
        document_id: str,
        query_embedding: Sequence[float],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve the most relevant chunks belonging to one document."""
        if not document_id.strip():
            raise ValueError("document_id must not be empty.")

        if len(query_embedding) != 384:
            raise ValueError("Query embedding must contain 384 dimensions.")

        if limit <= 0:
            raise ValueError("limit must be greater than zero.")

        query_vector = Vector(query_embedding)

        query = """
            SELECT
                chunk_index,
                chunk_text,
                metadata,
                embedding <=> %s AS distance
            FROM document_chunks
            WHERE document_id = %s
            ORDER BY embedding <=> %s
            LIMIT %s
        """

        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    query,
                    (query_vector, document_id, query_vector, limit),
                )
                rows = cursor.fetchall()

        return [
            {
                "chunk_index": row[0],
                "chunk_text": row[1],
                "metadata": row[2],
                "distance": float(row[3]),
            }
            for row in rows
        ]
