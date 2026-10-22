"""Unit tests for PostgreSQL document vector storage."""

from unittest.mock import patch

import psycopg
import pytest
from pgvector import Vector
from psycopg.types.json import Jsonb

from backend.app.core.config import Settings
from backend.app.services.vector_storage import PostgresVectorStorage


@pytest.fixture
def storage() -> PostgresVectorStorage:
    """Create storage with test-only database settings."""
    config = Settings(
        _env_file=None,
        POSTGRES_PASSWORD="test-password",
    )
    return PostgresVectorStorage(config)


@pytest.fixture
def database_mocks():
    """Mock the PostgreSQL connection and cursor."""
    with (
        patch("backend.app.services.vector_storage.psycopg.connect") as connect,
        patch("backend.app.services.vector_storage.register_vector") as register,
    ):
        connection = connect.return_value.__enter__.return_value
        cursor = connection.cursor.return_value.__enter__.return_value

        yield connect, register, connection, cursor


def test_store_chunks_inserts_vectors(storage, database_mocks):
    """Store each chunk with its corresponding embedding."""
    connect, register, connection, cursor = database_mocks
    chunks = ["First chunk", "Second chunk"]
    embeddings = [[0.1] * 384, [0.2] * 384]

    count = storage.store_chunks(
        document_id="document-1",
        chunks=chunks,
        embeddings=embeddings,
        metadata={"source": "test.txt"},
    )

    assert count == 2
    connect.assert_called_once()
    register.assert_called_once_with(connection)
    cursor.executemany.assert_called_once()

    query, rows = cursor.executemany.call_args.args
    assert "INSERT INTO document_chunks" in query
    assert "ON CONFLICT (document_id, chunk_index)" in query
    assert len(rows) == 2

    assert rows[0][0:3] == ("document-1", 0, "First chunk")
    assert isinstance(rows[0][3], Vector)
    assert len(embeddings[0]) == 384
    assert isinstance(rows[0][4], Jsonb)


def test_store_chunks_rejects_empty_document_id(storage, database_mocks):
    """Reject a blank document identifier."""
    connect, *_ = database_mocks

    with pytest.raises(ValueError, match="document_id"):
        storage.store_chunks(" ", ["chunk"], [[0.1] * 384])

    connect.assert_not_called()


def test_store_chunks_rejects_mismatched_lengths(storage, database_mocks):
    """Require one embedding per chunk."""
    connect, *_ = database_mocks

    with pytest.raises(ValueError, match="exactly one embedding"):
        storage.store_chunks("document-1", ["chunk"], [])

    connect.assert_not_called()


def test_store_chunks_rejects_blank_chunk(storage, database_mocks):
    """Reject empty chunk text."""
    connect, *_ = database_mocks

    with pytest.raises(ValueError, match="must not be empty"):
        storage.store_chunks("document-1", ["  "], [[0.1] * 384])

    connect.assert_not_called()


def test_store_chunks_rejects_wrong_embedding_dimension(storage, database_mocks):
    """Require embeddings to have exactly 384 dimensions."""
    connect, *_ = database_mocks

    with pytest.raises(ValueError, match="384 dimensions"):
        storage.store_chunks("document-1", ["chunk"], [[0.1] * 10])

    connect.assert_not_called()


def test_store_chunks_returns_zero_for_empty_input(storage, database_mocks):
    """Avoid opening a connection when there is nothing to store."""
    connect, *_ = database_mocks

    assert storage.store_chunks("document-1", [], []) == 0
    connect.assert_not_called()


def test_store_chunks_requires_database_password(database_mocks):
    """Fail clearly when the database password is not configured."""
    connect, *_ = database_mocks
    config = Settings(_env_file=None, POSTGRES_PASSWORD=None)
    storage = PostgresVectorStorage(config)

    with pytest.raises(ValueError, match="POSTGRES_PASSWORD"):
        storage.store_chunks("document-1", ["chunk"], [[0.1] * 384])

    connect.assert_not_called()


def test_store_chunks_propagates_connection_errors(storage, database_mocks):
    """Propagate connection failures to the caller."""
    connect, *_ = database_mocks
    connect.side_effect = psycopg.OperationalError("connection unavailable")

    with pytest.raises(psycopg.OperationalError, match="connection unavailable"):
        storage.store_chunks("document-1", ["chunk"], [[0.1] * 384])
