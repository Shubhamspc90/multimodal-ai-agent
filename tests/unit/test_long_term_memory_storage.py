"""Unit tests for user-scoped PostgreSQL long-term memory storage."""

from datetime import UTC, datetime
from unittest.mock import patch
from uuid import uuid4

import psycopg
import pytest
from pgvector import Vector

from backend.app.core.config import Settings
from backend.app.services.long_term_memory_storage import LongTermMemoryStorage


@pytest.fixture
def storage() -> LongTermMemoryStorage:
    """Create storage with test-only database settings."""
    config = Settings(
        _env_file=None,
        POSTGRES_PASSWORD="test-password",
    )
    return LongTermMemoryStorage(config)


@pytest.fixture
def database_mocks():
    """Mock PostgreSQL connections, vector registration, and cursors."""
    with (
        patch(
            "backend.app.services.long_term_memory_storage.psycopg.connect"
        ) as connect,
        patch(
            "backend.app.services.long_term_memory_storage.register_vector"
        ) as register,
    ):
        connection = connect.return_value.__enter__.return_value
        cursor = connection.cursor.return_value.__enter__.return_value

        yield connect, register, connection, cursor


def test_store_memory_inserts_user_owned_memory(storage, database_mocks):
    """Insert a memory with its owner, category, and embedding."""
    connect, register, connection, cursor = database_mocks
    user_id = uuid4()
    memory_id = uuid4()
    created_at = datetime.now(UTC)
    updated_at = created_at

    cursor.fetchone.return_value = (
        memory_id,
        user_id,
        "Prefers Python",
        "preference",
        created_at,
        updated_at,
    )

    with (
        patch.object(storage, "_ensure_schema") as ensure_schema,
        patch(
            "backend.app.services.long_term_memory_storage.uuid4",
            return_value=memory_id,
        ),
    ):
        result = storage.store_memory(
            user_id,
            "  Prefers Python  ",
            "preference",
            [0.1] * 384,
        )

    assert result["id"] == memory_id
    assert result["user_id"] == user_id
    assert result["memory_text"] == "Prefers Python"
    assert result["category"] == "preference"
    assert result["created_at"] == created_at

    ensure_schema.assert_called_once()
    connect.assert_called_once()
    register.assert_called_once_with(connection)

    query, params = cursor.execute.call_args.args
    assert "INSERT INTO long_term_memories" in query
    assert params[0] == memory_id
    assert params[1] == user_id
    assert params[2] == "Prefers Python"
    assert params[3] == "preference"
    assert isinstance(params[4], Vector)


def test_search_memories_returns_ranked_user_memories(storage, database_mocks):
    """Return matching memories with their similarity distances."""
    connect, register, connection, cursor = database_mocks
    user_id = uuid4()
    memory_id = uuid4()
    created_at = datetime.now(UTC)

    cursor.fetchall.return_value = [
        (
            memory_id,
            user_id,
            "Uses Python",
            "preference",
            created_at,
            created_at,
            0.12,
        )
    ]

    with patch.object(storage, "_ensure_schema"):
        results = storage.search_memories(user_id, [0.1] * 384, limit=3)

    assert len(results) == 1
    assert results[0]["memory_text"] == "Uses Python"
    assert results[0]["distance"] == pytest.approx(0.12)

    connect.assert_called_once()
    register.assert_called_once_with(connection)

    query, params = cursor.execute.call_args.args
    assert "embedding <=> %s" in query
    assert "WHERE user_id = %s" in query
    assert params[1] == user_id
    assert params[3] == 3


def test_list_memories_returns_only_user_owned_records(storage, database_mocks):
    """List memories using an owner-scoped query."""
    _, _, _, cursor = database_mocks
    user_id = uuid4()
    memory_id = uuid4()
    created_at = datetime.now(UTC)

    cursor.fetchall.return_value = [
        (
            memory_id,
            user_id,
            "Current project uses FastAPI",
            "project",
            created_at,
            created_at,
        )
    ]

    with patch.object(storage, "_ensure_schema"):
        results = storage.list_memories(user_id, limit=10)

    assert len(results) == 1
    assert results[0]["user_id"] == user_id
    assert results[0]["category"] == "project"

    query, params = cursor.execute.call_args.args
    assert "FROM long_term_memories" in query
    assert "WHERE user_id = %s" in query
    assert params == (user_id, 10)


def test_update_memory_requires_matching_owner(storage, database_mocks):
    """Update a memory using both its ID and owner ID."""
    _, _, _, cursor = database_mocks
    user_id = uuid4()
    memory_id = uuid4()
    created_at = datetime.now(UTC)

    cursor.fetchone.return_value = (
        memory_id,
        user_id,
        "Updated preference",
        "preference",
        created_at,
        created_at,
    )

    with patch.object(storage, "_ensure_schema"):
        result = storage.update_memory(
            user_id,
            memory_id,
            "Updated preference",
            "preference",
            [0.2] * 384,
        )

    assert result is not None
    assert result["memory_text"] == "Updated preference"

    query, params = cursor.execute.call_args.args
    assert "UPDATE long_term_memories" in query
    assert "WHERE id = %s AND user_id = %s" in query
    assert params[-2:] == (memory_id, user_id)


def test_update_memory_returns_none_when_not_owned(storage, database_mocks):
    """Return None when no memory matches the requested owner."""
    _, _, _, cursor = database_mocks
    cursor.fetchone.return_value = None

    with patch.object(storage, "_ensure_schema"):
        result = storage.update_memory(
            uuid4(),
            uuid4(),
            "Updated memory",
            "fact",
            [0.1] * 384,
        )

    assert result is None


def test_delete_memory_requires_matching_owner(storage, database_mocks):
    """Delete a memory using both its ID and owner ID."""
    _, _, _, cursor = database_mocks
    user_id = uuid4()
    memory_id = uuid4()
    cursor.rowcount = 1

    with patch.object(storage, "_ensure_schema"):
        result = storage.delete_memory(user_id, memory_id)

    assert result is True

    query, params = cursor.execute.call_args.args
    assert "DELETE FROM long_term_memories" in query
    assert "WHERE id = %s AND user_id = %s" in query
    assert params == (memory_id, user_id)


def test_delete_memory_returns_false_when_no_record_matches(storage, database_mocks):
    """Return False when the requested memory was not deleted."""
    _, _, _, cursor = database_mocks
    cursor.rowcount = 0

    with patch.object(storage, "_ensure_schema"):
        result = storage.delete_memory(uuid4(), uuid4())

    assert result is False


@pytest.mark.parametrize(
    ("memory_text", "category", "embedding", "message"),
    [
        (" ", "fact", [0.1] * 384, "memory_text"),
        ("Valid memory", "invalid", [0.1] * 384, "category"),
        ("Valid memory", "fact", [0.1] * 10, "384 dimensions"),
        ("Valid memory", "fact", [float("nan")] * 384, "finite"),
        ("Valid memory", "fact", [float("inf")] * 384, "finite"),
        ("Valid memory", "fact", ["invalid"] * 384, "numeric"),
        ("Valid memory", "fact", None, "numeric"),
    ],
)
def test_store_memory_rejects_invalid_arguments(
    storage,
    database_mocks,
    memory_text,
    category,
    embedding,
    message,
):
    """Reject invalid memory data before opening a database connection."""
    connect, *_ = database_mocks

    with (
        patch.object(storage, "_ensure_schema") as ensure_schema,
        pytest.raises(ValueError, match=message),
    ):
        storage.store_memory(
            uuid4(),
            memory_text,
            category,
            embedding,
        )

    connect.assert_not_called()
    ensure_schema.assert_not_called()


@pytest.mark.parametrize(
    ("embedding", "message"),
    [
        ([0.1] * 10, "384 dimensions"),
        ([float("nan")] * 384, "finite"),
        (["invalid"] * 384, "numeric"),
        (None, "numeric"),
    ],
)
def test_search_memories_rejects_invalid_embeddings(
    storage, database_mocks, embedding, message
):
    """Reject invalid query embeddings before connecting to PostgreSQL."""
    connect, *_ = database_mocks

    with pytest.raises(ValueError, match=message):
        storage.search_memories(uuid4(), embedding)

    connect.assert_not_called()


@pytest.mark.parametrize("limit", [0, -1])
def test_search_memories_rejects_invalid_limit(storage, database_mocks, limit):
    """Require a positive result limit."""
    connect, *_ = database_mocks

    with pytest.raises(ValueError, match="limit"):
        storage.search_memories(uuid4(), [0.1] * 384, limit)

    connect.assert_not_called()


def test_connection_registers_pgvector_using_configured_port(storage, database_mocks):
    """Use configured PostgreSQL settings and register pgvector."""
    connect, register, connection, _ = database_mocks

    with storage._connection() as actual_connection:
        assert actual_connection is connection

    connect.assert_called_once()
    assert connect.call_args.kwargs["port"] == storage._config.postgres_port
    assert connect.call_args.kwargs["port"] == 5433
    register.assert_called_once_with(connection)


def test_connection_requires_database_password(database_mocks):
    """Reject missing database credentials before connecting."""
    connect, *_ = database_mocks
    config = Settings(_env_file=None, POSTGRES_PASSWORD=None)
    storage = LongTermMemoryStorage(config)

    with pytest.raises(ValueError, match="POSTGRES_PASSWORD"):
        with storage._connection():
            pass

    connect.assert_not_called()


def test_schema_initialization_creates_extension_before_table(storage):
    """Create the vector extension before subsequent vector registration."""
    with (
        patch(
            "backend.app.services.long_term_memory_storage.psycopg.connect"
        ) as connect,
        patch(
            "backend.app.services.long_term_memory_storage.register_vector"
        ) as register,
        patch(
            "backend.app.services.long_term_memory_storage."
            "PostgresUserStorage.ensure_schema"
        ) as ensure_users,
    ):
        connection = connect.return_value.__enter__.return_value
        cursor = connection.cursor.return_value.__enter__.return_value

        storage._ensure_schema()
        storage._ensure_schema()

    ensure_users.assert_called_once()
    connect.assert_called_once()
    register.assert_not_called()

    executed_statements = [
        call.args[0].strip() for call in cursor.execute.call_args_list
    ]

    assert executed_statements[0].startswith("CREATE EXTENSION IF NOT EXISTS vector")
    assert any(
        statement.startswith("CREATE TABLE IF NOT EXISTS long_term_memories")
        for statement in executed_statements
    )
    assert storage._schema_ready is True


def test_schema_initialization_propagates_database_errors(storage):
    """Do not mark schema initialization complete after a database failure."""
    with (
        patch(
            "backend.app.services.long_term_memory_storage.psycopg.connect"
        ) as connect,
        patch(
            "backend.app.services.long_term_memory_storage."
            "PostgresUserStorage.ensure_schema"
        ),
    ):
        connect.side_effect = psycopg.OperationalError("connection unavailable")

        with pytest.raises(
            psycopg.OperationalError,
            match="connection unavailable",
        ):
            storage._ensure_schema()

    assert storage._schema_ready is False


def test_store_memory_if_new_inserts_when_no_duplicate_exists(
    storage: LongTermMemoryStorage,
    database_mocks,
) -> None:
    """Insert a memory after acquiring the lock and finding no duplicate."""
    _, _, _, cursor = database_mocks
    storage._schema_ready = True

    user_id = uuid4()
    memory_id = uuid4()
    memory_text = "I prefer Python"
    embedding = [0.1] * storage.EMBEDDING_DIMENSIONS

    cursor.fetchone.side_effect = [
        None,  # No existing duplicate.
        (
            memory_id,
            user_id,
            memory_text,
            "preference",
            None,
            None,
        ),
    ]

    result = storage.store_memory_if_new(
        user_id=user_id,
        memory_text=memory_text,
        category="preference",
        embedding=embedding,
    )

    assert result is not None
    assert result["id"] == memory_id
    assert result["user_id"] == user_id
    assert result["memory_text"] == memory_text
    assert result["category"] == "preference"

    assert cursor.execute.call_count == 3

    lock_sql = cursor.execute.call_args_list[0].args[0]
    duplicate_sql = cursor.execute.call_args_list[1].args[0]
    insert_sql = cursor.execute.call_args_list[2].args[0]

    assert "pg_advisory_xact_lock" in lock_sql
    assert "WHERE user_id = %s" in duplicate_sql
    assert "AND category = %s" in duplicate_sql
    assert "INSERT INTO long_term_memories" in insert_sql

    duplicate_params = cursor.execute.call_args_list[1].args[1]
    assert duplicate_params == (
        user_id,
        "preference",
        "i prefer python",
    )


def test_store_memory_if_new_skips_existing_duplicate(
    storage: LongTermMemoryStorage,
    database_mocks,
) -> None:
    """Do not insert a memory when an equivalent record already exists."""
    _, _, _, cursor = database_mocks
    storage._schema_ready = True

    user_id = uuid4()
    existing_memory_id = uuid4()
    embedding = [0.1] * storage.EMBEDDING_DIMENSIONS

    cursor.fetchone.return_value = (existing_memory_id,)

    result = storage.store_memory_if_new(
        user_id=user_id,
        memory_text="  I prefer   Python  ",
        category="preference",
        embedding=embedding,
    )

    assert result is None

    # Lock acquisition and duplicate lookup only; no INSERT.
    assert cursor.execute.call_count == 2

    duplicate_params = cursor.execute.call_args_list[1].args[1]
    assert duplicate_params == (
        user_id,
        "preference",
        "i prefer python",
    )


def test_store_memory_if_new_scopes_duplicate_check_to_user_and_category(
    storage: LongTermMemoryStorage,
    database_mocks,
) -> None:
    """Check duplicates using both the owning user and memory category."""
    _, _, _, cursor = database_mocks
    storage._schema_ready = True

    user_id = uuid4()
    memory_id = uuid4()
    embedding = [0.1] * storage.EMBEDDING_DIMENSIONS

    cursor.fetchone.side_effect = [
        None,
        (
            memory_id,
            user_id,
            "I am developing a Multimodal AI Agent",
            "project",
            None,
            None,
        ),
    ]

    storage.store_memory_if_new(
        user_id=user_id,
        memory_text="I am developing a Multimodal AI Agent",
        category="project",
        embedding=embedding,
    )

    duplicate_sql = cursor.execute.call_args_list[1].args[0]
    duplicate_params = cursor.execute.call_args_list[1].args[1]

    assert "WHERE user_id = %s" in duplicate_sql
    assert "AND category = %s" in duplicate_sql
    assert duplicate_params[0] == user_id
    assert duplicate_params[1] == "project"
