"""Unit tests for PostgreSQL user account storage."""

from datetime import UTC, datetime
from unittest.mock import patch
from uuid import uuid4

import psycopg
import pytest

from backend.app.core.config import Settings
from backend.app.services.auth_storage import (
    DuplicateEmailError,
    PostgresUserStorage,
)


@pytest.fixture
def storage() -> PostgresUserStorage:
    """Create storage with isolated test configuration."""
    config = Settings(
        _env_file=None,
        POSTGRES_PASSWORD="test-password",
    )
    instance = PostgresUserStorage(config)
    instance._schema_ready = True
    return instance


@pytest.fixture
def database_mocks():
    """Mock the PostgreSQL connection and cursor."""
    with patch("backend.app.services.auth_storage.psycopg.connect") as connect:
        connection = connect.return_value.__enter__.return_value
        cursor = connection.cursor.return_value.__enter__.return_value

        yield connect, connection, cursor


def test_create_user_normalizes_email_and_returns_public_data(
    storage,
    database_mocks,
):
    user_id = uuid4()
    timestamp = datetime.now(UTC)
    password_hash = "$argon2id$example-hash"

    _, _, cursor = database_mocks
    cursor.fetchone.return_value = (
        user_id,
        "Shubham Chauhan",
        "shubham@example.com",
        timestamp,
    )

    result = storage.create_user(
        name="  Shubham Chauhan  ",
        email="  Shubham@Example.com ",
        password_hash=password_hash,
    )

    assert result == {
        "id": user_id,
        "name": "Shubham Chauhan",
        "email": "shubham@example.com",
        "created_at": timestamp,
    }
    assert "password_hash" not in result

    query, params = cursor.execute.call_args.args

    assert "INSERT INTO users" in query
    assert params == (
        params[0],
        "Shubham Chauhan",
        "shubham@example.com",
        password_hash,
    )


def test_get_user_for_authentication_returns_password_hash(
    storage,
    database_mocks,
):
    user_id = uuid4()
    timestamp = datetime.now(UTC)
    password_hash = "$argon2id$example-hash"

    _, _, cursor = database_mocks
    cursor.fetchone.return_value = (
        user_id,
        "Shubham Chauhan",
        "shubham@example.com",
        password_hash,
        timestamp,
    )

    result = storage.get_user_for_authentication("Shubham@Example.com")

    assert result is not None
    assert result["id"] == user_id
    assert result["password_hash"] == password_hash

    query, params = cursor.execute.call_args.args

    assert "LOWER(email)" in query
    assert params == ("shubham@example.com",)


def test_get_user_for_authentication_returns_none_for_unknown_email(
    storage,
    database_mocks,
):
    _, _, cursor = database_mocks
    cursor.fetchone.return_value = None

    result = storage.get_user_for_authentication("unknown@example.com")

    assert result is None


def test_get_user_by_id_returns_public_data(storage, database_mocks):
    user_id = uuid4()
    timestamp = datetime.now(UTC)

    _, _, cursor = database_mocks
    cursor.fetchone.return_value = (
        user_id,
        "Shubham Chauhan",
        "shubham@example.com",
        timestamp,
    )

    result = storage.get_user_by_id(user_id)

    assert result is not None
    assert result["id"] == user_id
    assert result["email"] == "shubham@example.com"
    assert "password_hash" not in result

    query, params = cursor.execute.call_args.args

    assert "WHERE id = %s" in query
    assert params == (user_id,)


def test_get_user_by_id_returns_none_for_unknown_user(storage, database_mocks):
    _, _, cursor = database_mocks
    cursor.fetchone.return_value = None

    assert storage.get_user_by_id(uuid4()) is None


def test_create_user_raises_duplicate_email_error(storage, database_mocks):
    _, _, cursor = database_mocks
    cursor.execute.side_effect = psycopg.errors.UniqueViolation("duplicate email")

    with pytest.raises(DuplicateEmailError, match="already exists"):
        storage.create_user(
            "Shubham",
            "shubham@example.com",
            "$argon2id$example-hash",
        )


@pytest.mark.parametrize(
    ("name", "email", "password_hash", "message"),
    [
        (" ", "user@example.com", "hash", "Name"),
        ("Shubham", " ", "hash", "Email"),
        ("Shubham", "user@example.com", " ", "Password hash"),
    ],
)
def test_create_user_rejects_empty_values(
    storage,
    database_mocks,
    name,
    email,
    password_hash,
    message,
):
    connect, _, _ = database_mocks

    with pytest.raises(ValueError, match=message):
        storage.create_user(name, email, password_hash)

    connect.assert_not_called()


def test_storage_requires_database_password(database_mocks):
    config = Settings(_env_file=None, POSTGRES_PASSWORD=None)
    storage = PostgresUserStorage(config)

    connect, _, _ = database_mocks

    with pytest.raises(ValueError, match="POSTGRES_PASSWORD"):
        storage.get_user_by_id(uuid4())

    connect.assert_not_called()


def test_storage_propagates_database_errors(storage, database_mocks):
    connect, _, _ = database_mocks
    connect.side_effect = psycopg.OperationalError("connection unavailable")

    with pytest.raises(
        psycopg.OperationalError,
        match="connection unavailable",
    ):
        storage.get_user_by_id(uuid4())


def test_schema_is_created_only_once(database_mocks):
    config = Settings(
        _env_file=None,
        POSTGRES_PASSWORD="test-password",
    )
    storage = PostgresUserStorage(config)

    connect, _, cursor = database_mocks

    storage._ensure_schema()
    storage._ensure_schema()

    assert storage._schema_ready is True
    connect.assert_called_once()
    assert cursor.execute.call_count == 2


def test_create_user_initializes_schema_before_insert(database_mocks):
    config = Settings(
        _env_file=None,
        POSTGRES_PASSWORD="test-password",
    )
    storage = PostgresUserStorage(config)

    user_id = uuid4()
    timestamp = datetime.now(UTC)

    _, _, cursor = database_mocks
    cursor.fetchone.return_value = (
        user_id,
        "Shubham",
        "shubham@example.com",
        timestamp,
    )

    result = storage.create_user(
        "Shubham",
        "shubham@example.com",
        "$argon2id$example-hash",
    )

    assert result["id"] == user_id

    executed_queries = [call.args[0] for call in cursor.execute.call_args_list]

    assert any(
        "CREATE TABLE IF NOT EXISTS users" in query for query in executed_queries
    )
    assert any("CREATE UNIQUE INDEX" in query for query in executed_queries)
    assert any("INSERT INTO users" in query for query in executed_queries)


def test_update_user_name_normalizes_name_and_returns_public_data(
    storage,
    database_mocks,
):
    user_id = uuid4()
    timestamp = datetime.now(UTC)

    _, _, cursor = database_mocks
    cursor.fetchone.return_value = (
        user_id,
        "Updated Name",
        "shubham@example.com",
        timestamp,
    )

    result = storage.update_user_name(user_id, "  Updated Name  ")

    assert result == {
        "id": user_id,
        "name": "Updated Name",
        "email": "shubham@example.com",
        "created_at": timestamp,
    }
    assert "password_hash" not in result

    query, params = cursor.execute.call_args.args
    assert "UPDATE users" in query
    assert "SET name = %s" in query
    assert params == ("Updated Name", user_id)


@pytest.mark.parametrize("name", ["", "   ", "A" * 101])
def test_update_user_name_rejects_invalid_name(
    storage,
    database_mocks,
    name,
):
    connect, _, _ = database_mocks

    with pytest.raises(ValueError):
        storage.update_user_name(uuid4(), name)

    connect.assert_not_called()


def test_update_user_name_returns_none_for_unknown_user(
    storage,
    database_mocks,
):
    _, _, cursor = database_mocks
    cursor.fetchone.return_value = None

    assert storage.update_user_name(uuid4(), "Updated Name") is None


def test_get_user_credentials_by_id_returns_internal_credentials(
    storage,
    database_mocks,
):
    user_id = uuid4()
    timestamp = datetime.now(UTC)
    password_hash = "$argon2id$example-hash"

    _, _, cursor = database_mocks
    cursor.fetchone.return_value = (
        user_id,
        "Shubham Chauhan",
        "shubham@example.com",
        password_hash,
        timestamp,
    )

    result = storage.get_user_credentials_by_id(user_id)

    assert result == {
        "id": user_id,
        "name": "Shubham Chauhan",
        "email": "shubham@example.com",
        "password_hash": password_hash,
        "created_at": timestamp,
    }

    query, params = cursor.execute.call_args.args
    assert "password_hash" in query
    assert "WHERE id = %s" in query
    assert params == (user_id,)


def test_get_user_credentials_by_id_returns_none_for_unknown_user(
    storage,
    database_mocks,
):
    _, _, cursor = database_mocks
    cursor.fetchone.return_value = None

    assert storage.get_user_credentials_by_id(uuid4()) is None


def test_update_password_hash_returns_true_when_user_exists(
    storage,
    database_mocks,
):
    user_id = uuid4()
    password_hash = "$argon2id$new-example-hash"

    _, _, cursor = database_mocks
    cursor.fetchone.return_value = (user_id,)

    result = storage.update_password_hash(user_id, password_hash)

    assert result is True

    query, params = cursor.execute.call_args.args
    assert "UPDATE users" in query
    assert "SET password_hash = %s" in query
    assert params == (password_hash, user_id)


def test_update_password_hash_returns_false_for_unknown_user(
    storage,
    database_mocks,
):
    _, _, cursor = database_mocks
    cursor.fetchone.return_value = None

    assert (
        storage.update_password_hash(
            uuid4(),
            "$argon2id$new-example-hash",
        )
        is False
    )


def test_update_password_hash_rejects_empty_hash(storage, database_mocks):
    connect, _, _ = database_mocks

    with pytest.raises(ValueError, match="Password hash"):
        storage.update_password_hash(uuid4(), "   ")

    connect.assert_not_called()
