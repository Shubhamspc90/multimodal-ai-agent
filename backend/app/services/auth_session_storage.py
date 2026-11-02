"""PostgreSQL storage for authenticated user sessions."""

import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from uuid import UUID, uuid4

import psycopg

from backend.app.core.config import Settings, settings

TOKEN_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class PostgresAuthSessionStorage:
    """Persist and validate authentication sessions using PostgreSQL."""

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
                "POSTGRES_PASSWORD must be configured before using "
                "authentication sessions."
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
        """Create session tables and indexes once per storage instance."""
        if self._schema_ready:
            return

        with self._schema_lock:
            if self._schema_ready:
                return

            schema_path = Path(__file__).with_name("auth_session_schema.sql")
            schema_sql = schema_path.read_text(encoding="utf-8")

            with self._connection() as connection:
                with connection.cursor() as cursor:
                    for statement in schema_sql.split(";"):
                        if statement.strip():
                            cursor.execute(statement)

            self._schema_ready = True

    @staticmethod
    def _validate_token_hash(token_hash: str) -> str:
        """Validate the SHA-256 hexadecimal session-token hash."""
        if not isinstance(token_hash, str) or not TOKEN_HASH_PATTERN.fullmatch(
            token_hash
        ):
            raise ValueError("Invalid session token hash.")

        return token_hash

    def create_session(
        self,
        user_id: UUID,
        token_hash: str,
        expires_at: datetime,
    ) -> dict:
        """Create a session using a hashed token."""
        token_hash = self._validate_token_hash(token_hash)

        if expires_at.tzinfo is None or expires_at.utcoffset() is None:
            raise ValueError("Session expiry must include a timezone.")

        if expires_at <= datetime.now(UTC):
            raise ValueError("Session expiry must be in the future.")

        self._ensure_schema()

        session_id = uuid4()

        query = """
            INSERT INTO auth_sessions (
                id,
                user_id,
                token_hash,
                expires_at
            )
            VALUES (%s, %s, %s, %s)
            RETURNING id, user_id, created_at, expires_at
        """

        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    query,
                    (session_id, user_id, token_hash, expires_at),
                )
                row = cursor.fetchone()

        if row is None:
            raise RuntimeError("Session creation did not return a database record.")

        return {
            "id": row[0],
            "user_id": row[1],
            "created_at": row[2],
            "expires_at": row[3],
        }

    def get_active_session(self, token_hash: str) -> dict | None:
        """Retrieve an active session and its associated user."""
        token_hash = self._validate_token_hash(token_hash)
        self._ensure_schema()

        query = """
            SELECT
                s.id,
                u.id,
                u.name,
                u.email,
                u.created_at,
                s.expires_at
            FROM auth_sessions AS s
            JOIN users AS u ON u.id = s.user_id
            WHERE s.token_hash = %s
              AND s.revoked_at IS NULL
              AND s.expires_at > CURRENT_TIMESTAMP
        """

        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (token_hash,))
                row = cursor.fetchone()

        if row is None:
            return None

        return {
            "session_id": row[0],
            "user": {
                "id": row[1],
                "name": row[2],
                "email": row[3],
                "created_at": row[4],
            },
            "expires_at": row[5],
        }

    def revoke_session(self, token_hash: str) -> bool:
        """Revoke a session using its hashed token."""
        token_hash = self._validate_token_hash(token_hash)
        self._ensure_schema()

        query = """
            UPDATE auth_sessions
            SET revoked_at = CURRENT_TIMESTAMP
            WHERE token_hash = %s
              AND revoked_at IS NULL
            RETURNING id
        """

        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (token_hash,))
                row = cursor.fetchone()

        return row is not None
