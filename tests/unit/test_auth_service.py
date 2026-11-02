"""Unit tests for user registration and authentication."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from backend.app.services.auth_service import (
    AuthenticationService,
    InvalidCredentialsError,
    PasswordPolicyError,
)
from backend.app.services.auth_storage import DuplicateEmailError


class FakeUserStorage:
    """Provide isolated in-memory user storage for tests."""

    def __init__(self) -> None:
        self.users: dict[str, dict] = {}

    def create_user(
        self,
        name: str,
        email: str,
        password_hash: str,
    ) -> dict:
        """Create an account or reject a duplicate email."""
        if email in self.users:
            raise DuplicateEmailError("An account with this email already exists.")

        user = {
            "id": uuid4(),
            "name": name,
            "email": email,
            "password_hash": password_hash,
            "created_at": datetime.now(UTC),
        }

        self.users[email] = user

        return {key: value for key, value in user.items() if key != "password_hash"}

    def get_user_for_authentication(self, email: str) -> dict | None:
        """Retrieve internal account data for password verification."""
        return self.users.get(email)


@pytest.fixture
def storage() -> FakeUserStorage:
    """Create isolated in-memory storage."""
    return FakeUserStorage()


@pytest.fixture
def auth_service(storage: FakeUserStorage) -> AuthenticationService:
    """Create the authentication service with fake storage."""
    return AuthenticationService(storage=storage)


def test_register_user_normalizes_name_and_email(auth_service, storage):
    result = auth_service.register_user(
        name="  Shubham Chauhan  ",
        email=" Shubham@Example.com ",
        password="SecurePassword123!",
    )

    assert result["name"] == "Shubham Chauhan"
    assert result["email"] == "shubham@example.com"
    assert result["id"]
    assert "password_hash" not in result

    stored_user = storage.users["shubham@example.com"]
    assert stored_user["password_hash"].startswith("$argon2id$")
    assert stored_user["password_hash"] != "SecurePassword123!"


def test_authenticate_user_returns_public_user(auth_service):
    registered = auth_service.register_user(
        "Shubham",
        "shubham@example.com",
        "SecurePassword123!",
    )

    authenticated = auth_service.authenticate_user(
        "SHUBHAM@example.com",
        "SecurePassword123!",
    )

    assert authenticated["id"] == registered["id"]
    assert authenticated["email"] == "shubham@example.com"
    assert "password_hash" not in authenticated


def test_authenticate_user_rejects_unknown_email(auth_service):
    with pytest.raises(InvalidCredentialsError, match="Invalid email or password"):
        auth_service.authenticate_user(
            "unknown@example.com",
            "SecurePassword123!",
        )


def test_authenticate_user_rejects_wrong_password(auth_service):
    auth_service.register_user(
        "Shubham",
        "shubham@example.com",
        "SecurePassword123!",
    )

    with pytest.raises(InvalidCredentialsError, match="Invalid email or password"):
        auth_service.authenticate_user(
            "shubham@example.com",
            "WrongPassword123!",
        )


def test_register_user_rejects_duplicate_email(auth_service):
    auth_service.register_user(
        "Shubham",
        "shubham@example.com",
        "SecurePassword123!",
    )

    with pytest.raises(DuplicateEmailError, match="already exists"):
        auth_service.register_user(
            "Another User",
            "SHUBHAM@example.com",
            "AnotherPassword123!",
        )


def test_register_user_rejects_empty_name(auth_service):
    with pytest.raises(ValueError, match="Name must not be empty"):
        auth_service.register_user(
            "   ",
            "shubham@example.com",
            "SecurePassword123!",
        )


def test_register_user_rejects_empty_email(auth_service):
    with pytest.raises(ValueError, match="Email must not be empty"):
        auth_service.register_user(
            "Shubham",
            "   ",
            "SecurePassword123!",
        )


@pytest.mark.parametrize(
    "password",
    [
        "",
        "short",
        "a" * 129,
    ],
)
def test_register_user_rejects_password_outside_policy(
    auth_service,
    password,
):
    with pytest.raises(PasswordPolicyError):
        auth_service.register_user(
            "Shubham",
            "shubham@example.com",
            password,
        )


def test_authenticate_user_rejects_empty_credentials(auth_service):
    with pytest.raises(InvalidCredentialsError):
        auth_service.authenticate_user("", "password")

    with pytest.raises(InvalidCredentialsError):
        auth_service.authenticate_user("user@example.com", "")
