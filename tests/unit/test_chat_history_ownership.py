"""Tests for authenticated conversation ownership."""

from datetime import UTC, datetime
from unittest.mock import patch
from uuid import uuid4

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


def test_create_authenticated_conversation_sets_user_id(
    storage,
    database_mocks,
    monkeypatch,
):
    client_id = uuid4()
    user_id = uuid4()
    conversation_id = uuid4()
    timestamp = datetime.now(UTC)

    monkeypatch.setattr(
        "backend.app.services.chat_history.uuid4",
        lambda: conversation_id,
    )

    _, _, cursor = database_mocks
    cursor.fetchone.return_value = (
        conversation_id,
        client_id,
        "New conversation",
        timestamp,
        timestamp,
    )

    result = storage.create_conversation(client_id, user_id)

    assert result["id"] == conversation_id

    query, params = cursor.execute.call_args.args

    normalized_query = " ".join(query.split())

    assert "INSERT INTO conversations (id, client_id, user_id)" in normalized_query
    assert params == (
        conversation_id,
        client_id,
        user_id,
    )


def test_list_authenticated_conversations_filters_by_user(
    storage,
    database_mocks,
):
    user_id = uuid4()
    conversation_id = uuid4()
    client_id = uuid4()
    timestamp = datetime.now(UTC)

    _, _, cursor = database_mocks
    cursor.fetchall.return_value = [
        (
            conversation_id,
            client_id,
            "Private conversation",
            timestamp,
            timestamp,
        )
    ]

    result = storage.list_conversations(
        client_id=uuid4(),
        user_id=user_id,
    )

    assert len(result) == 1
    assert result[0]["id"] == conversation_id

    query, params = cursor.execute.call_args.args

    assert "WHERE user_id = %s" in query
    assert params == (user_id,)


def test_list_legacy_conversations_excludes_claimed_records(
    storage,
    database_mocks,
):
    client_id = uuid4()

    _, _, cursor = database_mocks
    cursor.fetchall.return_value = []

    assert storage.list_conversations(client_id=client_id) == []

    query, params = cursor.execute.call_args.args

    assert "client_id = %s AND user_id IS NULL" in query
    assert params == (client_id,)


def test_get_authenticated_conversation_checks_user_ownership(
    storage,
    database_mocks,
):
    user_id = uuid4()
    conversation_id = uuid4()
    client_id = uuid4()
    timestamp = datetime.now(UTC)

    _, _, cursor = database_mocks
    cursor.fetchone.return_value = (
        conversation_id,
        client_id,
        "Private conversation",
        timestamp,
        timestamp,
    )
    cursor.fetchall.return_value = [(1, "user", "Hello", timestamp)]

    result = storage.get_conversation(
        conversation_id,
        client_id=uuid4(),
        user_id=user_id,
    )

    assert result is not None
    assert result["id"] == conversation_id
    assert len(result["messages"]) == 1

    query, params = cursor.execute.call_args_list[0].args

    assert "WHERE id = %s AND user_id = %s" in query
    assert params == (conversation_id, user_id)


def test_append_authenticated_message_checks_user_ownership(
    storage,
    database_mocks,
):
    user_id = uuid4()
    conversation_id = uuid4()
    client_id = uuid4()
    timestamp = datetime.now(UTC)

    _, _, cursor = database_mocks
    cursor.fetchone.side_effect = [
        ("New conversation",),
        (1, "user", "Hello", timestamp),
    ]

    result = storage.append_message(
        conversation_id=conversation_id,
        client_id=client_id,
        role="user",
        content="Hello",
        user_id=user_id,
    )

    assert result is not None
    assert result["content"] == "Hello"

    calls = cursor.execute.call_args_list

    assert "user_id = %s" in calls[0].args[0]
    assert calls[0].args[1] == (conversation_id, user_id)

    assert "INSERT INTO chat_messages" in calls[1].args[0]

    assert "user_id = %s" in calls[2].args[0]
    assert calls[2].args[1] == (
        "Hello",
        conversation_id,
        user_id,
    )


def test_link_legacy_conversations_claims_only_unowned_records(
    storage,
    database_mocks,
):
    client_id = uuid4()
    user_id = uuid4()

    _, _, cursor = database_mocks
    cursor.fetchall.return_value = [
        (uuid4(),),
        (uuid4(),),
    ]

    linked_count = storage.link_legacy_conversations(
        client_id,
        user_id,
    )

    assert linked_count == 2

    query, params = cursor.execute.call_args.args

    assert "SET user_id = %s" in query
    assert "client_id = %s" in query
    assert "user_id IS NULL" in query
    assert params == (user_id, client_id)


def test_conversation_storage_requires_an_owner(storage, database_mocks):
    connect, _, _ = database_mocks

    with pytest.raises(ValueError, match="required"):
        storage.list_conversations()

    with pytest.raises(ValueError, match="required"):
        storage.get_conversation(uuid4())

    with pytest.raises(ValueError, match="required"):
        storage.append_message(
            conversation_id=uuid4(),
            client_id=None,
            role="user",
            content="Hello",
        )

    connect.assert_not_called()
