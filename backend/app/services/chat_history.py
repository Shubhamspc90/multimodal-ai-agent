"""PostgreSQL storage for chat conversations and messages."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from uuid import UUID, uuid4

import psycopg

from backend.app.core.config import Settings, settings


class PostgresChatHistoryStorage:
    """Persist conversations and messages using PostgreSQL."""

    def __init__(self, config: Settings | None = None) -> None:
        self._config = config or settings
        self._schema_ready = False
        self._schema_lock = Lock()

    @contextmanager
    def _connection(self) -> Iterator[psycopg.Connection]:
        """Open a configured PostgreSQL connection."""
        password = self._config.postgres_password

        if password is None:
            raise ValueError(
                "POSTGRES_PASSWORD must be configured before using chat history."
            )

        with psycopg.connect(
            host=self._config.postgres_host,
            port=self._config.postgres_port,
            dbname=self._config.postgres_db,
            user=self._config.postgres_user,
            password=password.get_secret_value(),
        ) as connection:
            yield connection

    def _ensure_schema(self) -> None:
        """Create the chat tables once per storage instance."""
        if self._schema_ready:
            return

        with self._schema_lock:
            if self._schema_ready:
                return

            schema_path = Path(__file__).with_name("chat_history_schema.sql")
            schema_sql = schema_path.read_text(encoding="utf-8")

            with self._connection() as connection:
                with connection.cursor() as cursor:
                    for statement in schema_sql.split(";"):
                        if statement.strip():
                            cursor.execute(statement)

            self._schema_ready = True

    def create_conversation(
        self,
        client_id: UUID,
    ) -> dict:
        """Create an empty conversation for a browser client."""
        self._ensure_schema()
        conversation_id = uuid4()

        query = """
            INSERT INTO conversations (id, client_id)
            VALUES (%s, %s)
            RETURNING id, client_id, title, created_at, updated_at
        """

        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (conversation_id, client_id))
                row = cursor.fetchone()

        return self._conversation_dict(row)

    def list_conversations(self, client_id: UUID) -> list[dict]:
        """List conversations belonging to one browser client."""
        self._ensure_schema()

        query = """
            SELECT id, client_id, title, created_at, updated_at
            FROM conversations
            WHERE client_id = %s
            ORDER BY updated_at DESC, id DESC
        """

        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (client_id,))
                rows = cursor.fetchall()

        return [self._conversation_dict(row) for row in rows]

    def get_conversation(
        self,
        conversation_id: UUID,
        client_id: UUID,
    ) -> dict | None:
        """Retrieve a conversation and its messages."""
        self._ensure_schema()

        conversation_query = """
            SELECT id, client_id, title, created_at, updated_at
            FROM conversations
            WHERE id = %s AND client_id = %s
        """

        messages_query = """
            SELECT id, role, content, created_at
            FROM chat_messages
            WHERE conversation_id = %s
            ORDER BY id ASC
        """

        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    conversation_query,
                    (conversation_id, client_id),
                )
                conversation = cursor.fetchone()

                if conversation is None:
                    return None

                cursor.execute(messages_query, (conversation_id,))
                messages = cursor.fetchall()

        result = self._conversation_dict(conversation)
        result["messages"] = [
            {
                "id": row[0],
                "role": row[1],
                "content": row[2],
                "created_at": row[3],
            }
            for row in messages
        ]

        return result

    def append_message(
        self,
        conversation_id: UUID,
        client_id: UUID,
        role: str,
        content: str,
    ) -> dict | None:
        """Append a message and update its conversation timestamp."""
        self._ensure_schema()

        if role not in {"user", "assistant"}:
            raise ValueError("Message role must be user or assistant.")

        content = content.strip()

        if not content:
            raise ValueError("Message content must not be empty.")

        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT title
                    FROM conversations
                    WHERE id = %s AND client_id = %s
                    FOR UPDATE
                    """,
                    (conversation_id, client_id),
                )
                row = cursor.fetchone()

                if row is None:
                    return None

                title = row[0]

                if role == "user" and title == "New conversation":
                    title = content[:77] + "..." if len(content) > 80 else content

                cursor.execute(
                    """
                    INSERT INTO chat_messages (
                        conversation_id,
                        role,
                        content
                    )
                    VALUES (%s, %s, %s)
                    RETURNING id, role, content, created_at
                    """,
                    (conversation_id, role, content),
                )
                message = cursor.fetchone()

                cursor.execute(
                    """
                    UPDATE conversations
                    SET title = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s AND client_id = %s
                    """,
                    (title, conversation_id, client_id),
                )

        return {
            "id": message[0],
            "role": message[1],
            "content": message[2],
            "created_at": message[3],
        }

    @staticmethod
    def _conversation_dict(row: tuple) -> dict:
        """Convert a database conversation row into a dictionary."""
        return {
            "id": row[0],
            "client_id": row[1],
            "title": row[2],
            "created_at": row[3],
            "updated_at": row[4],
        }
