"""Unit tests for PostgreSQL authentication session storage."""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch
from uuid import uuid4

import psycopg
import pytest

from backend.app.core.config import Settings
from backend.app.services.auth_session_storage import (
    PostgresAuthSessionStorage,
)


@pytest.fixture
def storage() -> PostgresAuthSessionStorage:
    """Create storage with isolated test configuration."""
    config = Settings(
        _env_file=None,
        POSTGRES_PASSWORD="test-password",
    )
    instance = PostgresAuthSessionStorage(config)
    instance._schema_ready = True
    return instance


@pytest.fixture
def database_mocks():
    """Mock the PostgreSQL connection and cursor."""
    with patch("backend.app.services.auth_session_storage.psycopg.connect") as connect:
        connection = connect.return_value.__enter__.return_value
        cursor = connection.cursor.return_value.__enter__.return_value

        yield connect, connection, cursor


@pytest.fixture
def valid_session_data():
    """Provide valid session creation data."""
    return {
        "user_id": uuid4(),
        "token_hash": "a" * 64,
        "csrf_token_hash": "b" * 64,
        "expires_at": datetime.now(UTC) + timedelta(hours=1),
    }


def test_create_session_stores_token_hash(
    storage,
    database_mocks,
    valid_session_data,
):
    session_id = uuid4()
    timestamp = datetime.now(UTC)
    _, _, cursor = database_mocks

    cursor.fetchone.return_value = (
        session_id,
        valid_session_data["user_id"],
        timestamp,
        valid_session_data["expires_at"],
    )

    with patch(
        "backend.app.services.auth_session_storage.uuid4",
        return_value=session_id,
    ):
        result = storage.create_session(**valid_session_data)

    assert result["id"] == session_id
    assert result["user_id"] == valid_session_data["user_id"]
    assert "token_hash" not in result

    query, params = cursor.execute.call_args.args

    assert "INSERT INTO auth_sessions" in query
    assert params[0] == session_id
    assert params[1] == valid_session_data["user_id"]
    assert params[2] == valid_session_data["token_hash"]
    assert params[3] == valid_session_data["csrf_token_hash"]


def test_get_active_session_returns_user(storage, database_mocks):
    session_id = uuid4()
    user_id = uuid4()
    timestamp = datetime.now(UTC)
    expires_at = timestamp + timedelta(hours=1)

    _, _, cursor = database_mocks

    cursor.fetchone.return_value = (
        session_id,
        user_id,
        "Shubham",
        "shubham@example.com",
        timestamp,
        expires_at,
        "b" * 64,
    )

    result = storage.get_active_session("a" * 64)

    assert result is not None
    assert result["session_id"] == session_id
    assert result["user"]["id"] == user_id
    assert result["user"]["email"] == "shubham@example.com"
    assert result["csrf_token_hash"] == "b" * 64
    assert "token_hash" not in result

    query, params = cursor.execute.call_args.args

    assert "revoked_at IS NULL" in query
    assert "expires_at > CURRENT_TIMESTAMP" in query
    assert params == ("a" * 64,)


def test_get_active_session_returns_none_when_missing(
    storage,
    database_mocks,
):
    _, _, cursor = database_mocks
    cursor.fetchone.return_value = None

    assert storage.get_active_session("a" * 64) is None


def test_revoke_session_returns_true_when_session_exists(
    storage,
    database_mocks,
):
    _, _, cursor = database_mocks
    cursor.fetchone.return_value = (uuid4(),)

    assert storage.revoke_session("a" * 64) is True

    query, params = cursor.execute.call_args.args

    assert "UPDATE auth_sessions" in query
    assert "revoked_at IS NULL" in query
    assert params == ("a" * 64,)


def test_revoke_session_returns_false_when_session_missing(
    storage,
    database_mocks,
):
    _, _, cursor = database_mocks
    cursor.fetchone.return_value = None

    assert storage.revoke_session("a" * 64) is False


@pytest.mark.parametrize(
    "invalid_hash",
    [
        "",
        "invalid-hash",
        "A" * 64,
        "a" * 63,
        "a" * 65,
    ],
)
def test_create_session_rejects_invalid_token_hash(
    storage,
    database_mocks,
    valid_session_data,
    invalid_hash,
):
    connect, _, _ = database_mocks

    with pytest.raises(ValueError, match="Invalid session token hash"):
        storage.create_session(
            user_id=valid_session_data["user_id"],
            token_hash=invalid_hash,
            csrf_token_hash=valid_session_data["csrf_token_hash"],
            expires_at=valid_session_data["expires_at"],
        )

    connect.assert_not_called()


def test_create_session_rejects_naive_expiry(
    storage,
    database_mocks,
    valid_session_data,
):
    connect, _, _ = database_mocks

    with pytest.raises(ValueError, match="timezone"):
        storage.create_session(
            user_id=valid_session_data["user_id"],
            token_hash=valid_session_data["token_hash"],
            csrf_token_hash=valid_session_data["csrf_token_hash"],
            expires_at=datetime.now(),
        )

    connect.assert_not_called()


def test_create_session_rejects_expired_timestamp(
    storage,
    database_mocks,
    valid_session_data,
):
    connect, _, _ = database_mocks

    with pytest.raises(ValueError, match="future"):
        storage.create_session(
            user_id=valid_session_data["user_id"],
            token_hash=valid_session_data["token_hash"],
            csrf_token_hash=valid_session_data["csrf_token_hash"],
            expires_at=datetime.now(UTC) - timedelta(minutes=1),
        )

    connect.assert_not_called()


def test_session_lookup_rejects_invalid_hash(storage, database_mocks):
    connect, _, _ = database_mocks

    with pytest.raises(ValueError, match="Invalid session token hash"):
        storage.get_active_session("invalid")

    connect.assert_not_called()


def test_revoke_session_rejects_invalid_hash(storage, database_mocks):
    connect, _, _ = database_mocks

    with pytest.raises(ValueError, match="Invalid session token hash"):
        storage.revoke_session("invalid")

    connect.assert_not_called()


def test_storage_requires_database_password(database_mocks):
    config = Settings(_env_file=None, POSTGRES_PASSWORD=None)
    storage = PostgresAuthSessionStorage(config)

    connect, _, _ = database_mocks

    with pytest.raises(ValueError, match="POSTGRES_PASSWORD"):
        storage.get_active_session("a" * 64)

    connect.assert_not_called()


def test_storage_propagates_database_errors(storage, database_mocks):
    connect, _, _ = database_mocks
    connect.side_effect = psycopg.OperationalError("connection unavailable")

    with pytest.raises(
        psycopg.OperationalError,
        match="connection unavailable",
    ):
        storage.get_active_session("a" * 64)


def test_schema_is_created_only_once(database_mocks):
    config = Settings(
        _env_file=None,
        POSTGRES_PASSWORD="test-password",
    )
    storage = PostgresAuthSessionStorage(config)

    connect, _, cursor = database_mocks

    storage._ensure_schema()
    storage._ensure_schema()

    assert storage._schema_ready is True
    connect.assert_called_once()
    assert cursor.execute.call_count == 4


@pytest.mark.parametrize(
    "invalid_hash", ["", "invalid-hash", "A" * 64, "a" * 63, "a" * 65]
)
def test_create_session_rejects_invalid_csrf_hash(
    storage,
    database_mocks,
    valid_session_data,
    invalid_hash,
):
    connect, _, _ = database_mocks

    with pytest.raises(ValueError, match="Invalid session token hash"):
        storage.create_session(
            user_id=valid_session_data["user_id"],
            token_hash=valid_session_data["token_hash"],
            csrf_token_hash=invalid_hash,
            expires_at=valid_session_data["expires_at"],
        )

    connect.assert_not_called()
