"""PostgreSQL storage service for user-scoped long-term memories."""

import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import UUID, uuid4

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from backend.app.core.config import Settings, settings
from backend.app.services.auth_storage import PostgresUserStorage


class LongTermMemoryStorage:
    """Persist and retrieve long-term memories for authenticated users."""

    EMBEDDING_DIMENSIONS = 384
    CATEGORIES = frozenset({"preference", "project", "instruction", "fact"})

    def __init__(self, config: Settings | None = None) -> None:
        """Initialize storage without opening a database connection."""
        self._config = config or settings
        self._user_storage = PostgresUserStorage(self._config)
        self._schema_ready = False
        self._schema_lock = Lock()

    def _connection_parameters(self) -> dict[str, Any]:
        """Build connection parameters from application settings."""
        password = self._config.postgres_password

        if password is None:
            raise ValueError(
                "POSTGRES_PASSWORD must be configured before using "
                "long-term memory storage."
            )

        return {
            "host": self._config.postgres_host,
            "port": self._config.postgres_port,
            "dbname": self._config.postgres_db,
            "user": self._config.postgres_user,
            "password": password.get_secret_value(),
        }

    @contextmanager
    def _connection(self) -> Iterator[psycopg.Connection]:
        """Open a PostgreSQL connection with pgvector support."""
        with psycopg.connect(**self._connection_parameters()) as connection:
            register_vector(connection)
            yield connection

    def _ensure_schema(self) -> None:
        """Create the long-term memory schema once per storage instance."""
        if self._schema_ready:
            return

        with self._schema_lock:
            if self._schema_ready:
                return

            self._user_storage.ensure_schema()

            schema_path = Path(__file__).with_name("long_term_memory_schema.sql")
            schema_sql = schema_path.read_text(encoding="utf-8-sig")

            # Create the vector extension before registering its type.
            with psycopg.connect(**self._connection_parameters()) as connection:
                with connection.cursor() as cursor:
                    for statement in schema_sql.split(";"):
                        if statement.strip():
                            cursor.execute(statement)

            self._schema_ready = True

    @classmethod
    def _validate_memory_text(cls, memory_text: str) -> str:
        """Validate and normalize memory text."""
        if not isinstance(memory_text, str) or not memory_text.strip():
            raise ValueError("memory_text must not be empty.")

        return memory_text.strip()

    @classmethod
    def _validate_category(cls, category: str) -> str:
        """Ensure the category is supported by the database schema."""
        if not isinstance(category, str) or category not in cls.CATEGORIES:
            raise ValueError(
                "category must be one of: " + ", ".join(sorted(cls.CATEGORIES))
            )

        return category

    @classmethod
    def _validate_embedding(cls, embedding: Sequence[float]) -> Vector:
        """Validate and convert an embedding to a pgvector value."""
        try:
            values = tuple(float(value) for value in embedding)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Embedding values must be numeric.") from exc

        if len(values) != cls.EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Embedding must contain exactly {cls.EMBEDDING_DIMENSIONS} dimensions."
            )

        if not all(math.isfinite(value) for value in values):
            raise ValueError("Embedding values must all be finite.")

        return Vector(list(values))

    @staticmethod
    def _memory_dict(row: tuple[Any, ...]) -> dict[str, Any]:
        """Convert a database row into a memory dictionary."""
        return {
            "id": row[0],
            "user_id": row[1],
            "memory_text": row[2],
            "category": row[3],
            "created_at": row[4],
            "updated_at": row[5],
        }

    def store_memory(
        self,
        user_id: UUID,
        memory_text: str,
        category: str,
        embedding: Sequence[float],
    ) -> dict[str, Any]:
        """Store a new memory owned by the specified user."""
        normalized_text = self._validate_memory_text(memory_text)
        normalized_category = self._validate_category(category)
        memory_vector = self._validate_embedding(embedding)
        memory_id = uuid4()

        self._ensure_schema()

        query = """
            INSERT INTO long_term_memories (
                id,
                user_id,
                memory_text,
                category,
                embedding
            )
            VALUES (%s, %s, %s, %s, %s)
            RETURNING
                id,
                user_id,
                memory_text,
                category,
                created_at,
                updated_at
        """

        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    query,
                    (
                        memory_id,
                        user_id,
                        normalized_text,
                        normalized_category,
                        memory_vector,
                    ),
                )
                row = cursor.fetchone()

        if row is None:
            raise RuntimeError("The database did not return the stored memory.")

        return self._memory_dict(row)

    def store_memory_if_new(
        self,
        user_id: UUID,
        memory_text: str,
        category: str,
        embedding: Sequence[float],
    ) -> dict[str, Any] | None:
        """Store a memory only if an equivalent user-owned record does not exist."""
        normalized_text = self._validate_memory_text(memory_text)
        normalized_category = self._validate_category(category)
        memory_vector = self._validate_embedding(embedding)
        comparison_text = " ".join(normalized_text.lower().split())
        memory_id = uuid4()

        self._ensure_schema()

        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT pg_advisory_xact_lock(
                        hashtextextended(%s, 0)
                    )
                    """,
                    (str(user_id),),
                )

                cursor.execute(
                    """
                    SELECT id
                    FROM long_term_memories
                    WHERE user_id = %s
                      AND category = %s
                      AND lower(
                          regexp_replace(
                              btrim(memory_text),
                              '[[:space:]]+',
                              ' ',
                              'g'
                          )
                      ) = %s
                    LIMIT 1
                    """,
                    (user_id, normalized_category, comparison_text),
                )

                if cursor.fetchone() is not None:
                    return None

                cursor.execute(
                    """
                    INSERT INTO long_term_memories (
                        id,
                        user_id,
                        memory_text,
                        category,
                        embedding
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING
                        id,
                        user_id,
                        memory_text,
                        category,
                        created_at,
                        updated_at
                    """,
                    (
                        memory_id,
                        user_id,
                        normalized_text,
                        normalized_category,
                        memory_vector,
                    ),
                )

                row = cursor.fetchone()

        if row is None:
            raise RuntimeError("The database did not return the stored memory.")

        return self._memory_dict(row)

    def search_memories(
        self,
        user_id: UUID,
        query_embedding: Sequence[float],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve the most similar memories belonging to one user."""
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")

        query_vector = self._validate_embedding(query_embedding)
        self._ensure_schema()

        query = """
            SELECT
                id,
                user_id,
                memory_text,
                category,
                created_at,
                updated_at,
                embedding <=> %s AS distance
            FROM long_term_memories
            WHERE user_id = %s
            ORDER BY embedding <=> %s
            LIMIT %s
        """

        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    query,
                    (query_vector, user_id, query_vector, limit),
                )
                rows = cursor.fetchall()

        return [
            {
                **self._memory_dict(row[:6]),
                "distance": float(row[6]),
            }
            for row in rows
        ]

    def list_memories(
        self,
        user_id: UUID,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List the most recently created memories belonging to one user."""
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")

        self._ensure_schema()

        query = """
            SELECT
                id,
                user_id,
                memory_text,
                category,
                created_at,
                updated_at
            FROM long_term_memories
            WHERE user_id = %s
            ORDER BY created_at DESC, id
            LIMIT %s
        """

        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (user_id, limit))
                rows = cursor.fetchall()

        return [self._memory_dict(row) for row in rows]

    def update_memory(
        self,
        user_id: UUID,
        memory_id: UUID,
        memory_text: str,
        category: str,
        embedding: Sequence[float],
    ) -> dict[str, Any] | None:
        """Update a memory only when it belongs to the specified user."""
        normalized_text = self._validate_memory_text(memory_text)
        normalized_category = self._validate_category(category)
        memory_vector = self._validate_embedding(embedding)

        self._ensure_schema()

        query = """
            UPDATE long_term_memories
            SET
                memory_text = %s,
                category = %s,
                embedding = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s AND user_id = %s
            RETURNING
                id,
                user_id,
                memory_text,
                category,
                created_at,
                updated_at
        """

        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    query,
                    (
                        normalized_text,
                        normalized_category,
                        memory_vector,
                        memory_id,
                        user_id,
                    ),
                )
                row = cursor.fetchone()

        return self._memory_dict(row) if row else None

    def delete_memory(self, user_id: UUID, memory_id: UUID) -> bool:
        """Delete a memory only when it belongs to the specified user."""
        self._ensure_schema()

        query = """
            DELETE FROM long_term_memories
            WHERE id = %s AND user_id = %s
        """

        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (memory_id, user_id))
                deleted = cursor.rowcount > 0

        return deleted
