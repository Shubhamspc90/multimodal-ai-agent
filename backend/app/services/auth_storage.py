"""PostgreSQL storage for user accounts."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from uuid import UUID, uuid4

import psycopg

from backend.app.core.config import Settings, settings


class DuplicateEmailError(ValueError):
    """Raised when an account already exists for an email address."""


class PostgresUserStorage:
    """Persist user accounts using PostgreSQL."""

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
                "POSTGRES_PASSWORD must be configured before using user storage."
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
        """Create the user table and indexes once per storage instance."""
        if self._schema_ready:
            return

        with self._schema_lock:
            if self._schema_ready:
                return

            schema_path = Path(__file__).with_name("auth_schema.sql")
            schema_sql = schema_path.read_text(encoding="utf-8")

            with self._connection() as connection:
                with connection.cursor() as cursor:
                    for statement in schema_sql.split(";"):
                        if statement.strip():
                            cursor.execute(statement)

            self._schema_ready = True

    def ensure_schema(self) -> None:
        """Ensure the user table exists for dependent storage services."""
        self._ensure_schema()

    @staticmethod
    def _normalize_email(email: str) -> str:
        """Normalize an email address for consistent account lookup."""
        return email.strip().lower()

    def create_user(
        self,
        name: str,
        email: str,
        password_hash: str,
    ) -> dict:
        """Create a user account without storing or returning a plain password."""
        name = name.strip()
        email = self._normalize_email(email)
        password_hash = password_hash.strip()

        if not name:
            raise ValueError("Name must not be empty.")

        if not email:
            raise ValueError("Email must not be empty.")

        if not password_hash:
            raise ValueError("Password hash must not be empty.")

        self._ensure_schema()

        user_id = uuid4()

        query = """
            INSERT INTO users (id, name, email, password_hash)
            VALUES (%s, %s, %s, %s)
            RETURNING id, name, email, created_at
        """

        try:
            with self._connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        query,
                        (user_id, name, email, password_hash),
                    )
                    row = cursor.fetchone()
        except psycopg.errors.UniqueViolation as exc:
            raise DuplicateEmailError(
                "An account with this email already exists."
            ) from exc

        if row is None:
            raise RuntimeError("User creation did not return a database record.")

        return self._public_user_dict(row)

    def get_user_for_authentication(self, email: str) -> dict | None:
        """Retrieve account credentials for internal password verification."""
        normalized_email = self._normalize_email(email)

        if not normalized_email:
            raise ValueError("Email must not be empty.")

        self._ensure_schema()

        query = """
            SELECT id, name, email, password_hash, created_at
            FROM users
            WHERE LOWER(email) = %s
        """

        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (normalized_email,))
                row = cursor.fetchone()

        if row is None:
            return None

        return {
            "id": row[0],
            "name": row[1],
            "email": row[2],
            "password_hash": row[3],
            "created_at": row[4],
        }

    def get_user_by_id(self, user_id: UUID) -> dict | None:
        """Retrieve public account details without the password hash."""
        self._ensure_schema()

        query = """
            SELECT id, name, email, created_at
            FROM users
            WHERE id = %s
        """

        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (user_id,))
                row = cursor.fetchone()

        if row is None:
            return None

        return self._public_user_dict(row)

    def update_user_name(self, user_id: UUID, name: str) -> dict | None:
        """Update a user's display name and return public account details."""
        normalized_name = name.strip()

        if not normalized_name:
            raise ValueError("Name must not be empty.")

        if len(normalized_name) > 100:
            raise ValueError("Name must not exceed 100 characters.")

        self._ensure_schema()

        query = """
            UPDATE users
            SET name = %s
            WHERE id = %s
            RETURNING id, name, email, created_at
        """

        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (normalized_name, user_id))
                row = cursor.fetchone()

        if row is None:
            return None

        return self._public_user_dict(row)

    def get_user_credentials_by_id(self, user_id: UUID) -> dict | None:
        """Retrieve internal account credentials by user ID."""
        self._ensure_schema()

        query = """
            SELECT id, name, email, password_hash, created_at
            FROM users
            WHERE id = %s
        """

        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (user_id,))
                row = cursor.fetchone()

        if row is None:
            return None

        return {
            "id": row[0],
            "name": row[1],
            "email": row[2],
            "password_hash": row[3],
            "created_at": row[4],
        }

    def update_password_hash(
        self,
        user_id: UUID,
        password_hash: str,
    ) -> bool:
        """Update a user's password hash without exposing credentials."""
        normalized_hash = password_hash.strip()

        if not normalized_hash:
            raise ValueError("Password hash must not be empty.")

        self._ensure_schema()

        query = """
            UPDATE users
            SET password_hash = %s
            WHERE id = %s
            RETURNING id
        """

        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (normalized_hash, user_id))
                row = cursor.fetchone()

        return row is not None


    @staticmethod
    def _public_user_dict(row: tuple) -> dict:
        """Convert a database row into safe public account details."""
        return {
            "id": row[0],
            "name": row[1],
            "email": row[2],
            "created_at": row[3],
        }
