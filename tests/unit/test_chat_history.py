"""Unit tests for PostgreSQL chat history storage."""

from datetime import UTC, datetime
from unittest.mock import patch
from uuid import uuid4

import psycopg
import pytest

from backend.app.core.config import Settings
from backend.app.services.chat_history import PostgresChatHistoryStorage


@pytest.fixture
def storage() -> PostgresChatHistoryStorage:
    """Create storage with isolated test configuration."""
    config = Settings(
        _env_file=None,
        POSTGRES_PASSWORD="test-password",
    )
    instance = PostgresChatHistoryStorage(config)
    instance._schema_ready = True
    return instance


@pytest.fixture
def database_mocks():
    """Mock the PostgreSQL connection and cursor."""
    with patch("backend.app.services.chat_history.psycopg.connect") as connect:
        connection = connect.return_value.__enter__.return_value
        cursor = connection.cursor.return_value.__enter__.return_value

        yield connect, connection, cursor


def test_create_conversation(storage, database_mocks):
    client_id = uuid4()
    conversation_id = uuid4()
    timestamp = datetime.now(UTC)

    connect, _, cursor = database_mocks
    cursor.fetchone.return_value = (
        conversation_id,
        client_id,
        "New conversation",
        timestamp,
        timestamp,
    )

    result = storage.create_conversation(client_id)

    assert result["id"] == conversation_id
    assert result["client_id"] == client_id
    assert result["title"] == "New conversation"

    query, params = cursor.execute.call_args.args
    assert "INSERT INTO conversations" in query
    assert params[1] == client_id
    connect.assert_called_once()


def test_list_conversations(storage, database_mocks):
    client_id = uuid4()
    conversation_id = uuid4()
    timestamp = datetime.now(UTC)

    _, _, cursor = database_mocks
    cursor.fetchall.return_value = [
        (
            conversation_id,
            client_id,
            "AI research",
            timestamp,
            timestamp,
        )
    ]

    results = storage.list_conversations(client_id)

    assert len(results) == 1
    assert results[0]["title"] == "AI research"

    query, params = cursor.execute.call_args.args
    assert "WHERE client_id = %s" in query
    assert params == (client_id,)


def test_get_conversation_returns_messages(storage, database_mocks):
    client_id = uuid4()
    conversation_id = uuid4()
    timestamp = datetime.now(UTC)

    _, _, cursor = database_mocks
    cursor.fetchone.return_value = (
        conversation_id,
        client_id,
        "AI research",
        timestamp,
        timestamp,
    )
    cursor.fetchall.return_value = [
        (1, "user", "Explain AI.", timestamp),
        (2, "assistant", "AI is...", timestamp),
    ]

    result = storage.get_conversation(conversation_id, client_id)

    assert result is not None
    assert result["id"] == conversation_id
    assert len(result["messages"]) == 2
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][1]["content"] == "AI is..."


def test_get_conversation_returns_none_for_unknown_owner(
    storage,
    database_mocks,
):
    _, _, cursor = database_mocks
    cursor.fetchone.return_value = None

    result = storage.get_conversation(uuid4(), uuid4())

    assert result is None
    cursor.fetchall.assert_not_called()


def test_append_message_updates_title(storage, database_mocks):
    client_id = uuid4()
    conversation_id = uuid4()
    timestamp = datetime.now(UTC)

    _, _, cursor = database_mocks
    cursor.fetchone.side_effect = [
        ("New conversation",),
        (1, "user", "Explain PostgreSQL", timestamp),
    ]

    result = storage.append_message(
        conversation_id=conversation_id,
        client_id=client_id,
        role="user",
        content="Explain PostgreSQL",
    )

    assert result is not None
    assert result["id"] == 1
    assert result["content"] == "Explain PostgreSQL"

    calls = cursor.execute.call_args_list
    assert len(calls) == 3
    assert "FOR UPDATE" in calls[0].args[0]
    assert "INSERT INTO chat_messages" in calls[1].args[0]
    assert "UPDATE conversations" in calls[2].args[0]
    assert calls[2].args[1][0] == "Explain PostgreSQL"


def test_append_message_rejects_invalid_role(
    storage,
    database_mocks,
):
    connect, _, _ = database_mocks

    with pytest.raises(ValueError, match="role"):
        storage.append_message(
            uuid4(),
            uuid4(),
            "system",
            "Invalid role",
        )

    connect.assert_not_called()


def test_append_message_rejects_empty_content(
    storage,
    database_mocks,
):
    connect, _, _ = database_mocks

    with pytest.raises(ValueError, match="content"):
        storage.append_message(
            uuid4(),
            uuid4(),
            "user",
            "   ",
        )

    connect.assert_not_called()


def test_storage_requires_database_password(database_mocks):
    config = Settings(_env_file=None, POSTGRES_PASSWORD=None)
    storage = PostgresChatHistoryStorage(config)

    connect, _, _ = database_mocks

    with pytest.raises(ValueError, match="POSTGRES_PASSWORD"):
        storage.list_conversations(uuid4())

    connect.assert_not_called()


def test_storage_propagates_database_errors(
    storage,
    database_mocks,
):
    connect, _, _ = database_mocks
    connect.side_effect = psycopg.OperationalError("connection unavailable")

    with pytest.raises(
        psycopg.OperationalError,
        match="connection unavailable",
    ):
        storage.create_conversation(uuid4())
