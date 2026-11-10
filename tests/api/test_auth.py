"""API tests for user registration and authentication."""

import hashlib
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from backend.app.api.routes import auth as auth_route
from backend.app.main import app
from backend.app.services.auth_service import InvalidCredentialsError
from backend.app.services.auth_storage import DuplicateEmailError

COOKIE_NAME = auth_route.SESSION_COOKIE_NAME
CSRF_COOKIE_NAME = auth_route.CSRF_COOKIE_NAME
CSRF_HEADER_NAME = "X-CSRF-Token"


class FakeAuthenticationService:
    """Provide deterministic authentication behaviour."""

    def __init__(self) -> None:
        self.users: dict[str, dict] = {}
        self.passwords: dict[str, str] = {}

    def register_user(self, name: str, email: str, password: str) -> dict:
        email = email.strip().lower()

        if email in self.users:
            raise DuplicateEmailError("Account already exists.")

        user = {
            "id": uuid4(),
            "name": name.strip(),
            "email": email,
            "created_at": datetime.now(UTC),
        }

        self.users[email] = user
        self.passwords[email] = password

        return dict(user)

    def authenticate_user(self, email: str, password: str) -> dict:
        email = email.strip().lower()

        user = self.users.get(email)

        if user is None or self.passwords[email] != password:
            raise InvalidCredentialsError("Invalid credentials.")

        return dict(user)

    def update_user_name(self, user_id, name: str) -> dict | None:
        """Update a user's display name."""
        user = next(
            (
                item
                for item in self.users.values()
                if item["id"] == user_id
            ),
            None,
        )

        if user is None:
            return None

        user["name"] = name.strip()
        return dict(user)

    def change_password(
        self,
        user_id,
        current_password: str,
        new_password: str,
    ) -> None:
        """Change a user's password after verifying the current password."""
        user = next(
            (
                item
                for item in self.users.values()
                if item["id"] == user_id
            ),
            None,
        )

        if user is None:
            raise ValueError("User not found.")

        if self.passwords[user["email"]] != current_password:
            raise InvalidCredentialsError("Invalid credentials.")

        self.passwords[user["email"]] = new_password

class FakeAuthSessionStorage:
    """Provide in-memory session storage for API tests."""

    def __init__(self, authentication_service) -> None:
        self.authentication_service = authentication_service
        self.sessions: dict[str, dict] = {}

    def create_session(
        self,
        user_id,
        token_hash: str,
        csrf_token_hash: str,
        expires_at: datetime,
    ) -> dict:
        user = next(
            (
                item
                for item in self.authentication_service.users.values()
                if item["id"] == user_id
            ),
            None,
        )

        if user is None:
            raise ValueError("User not found.")

        session = {
            "id": uuid4(),
            "user_id": user_id,
            "user": dict(user),
            "expires_at": expires_at,
            "csrf_token_hash": csrf_token_hash,
            "revoked": False,
        }

        self.sessions[token_hash] = session

        return {
            "id": session["id"],
            "user_id": user_id,
            "expires_at": expires_at,
        }

    def get_active_session(self, token_hash: str) -> dict | None:
        session = self.sessions.get(token_hash)

        if session is None or session["revoked"]:
            return None

        if session["expires_at"] <= datetime.now(UTC):
            return None

        user = next(
            (
                item
                for item in self.authentication_service.users.values()
                if item["id"] == session["user_id"]
            ),
            None,
        )

        if user is None:
            return None

        return {
            "user": dict(user),
            "expires_at": session["expires_at"],
            "csrf_token_hash": session["csrf_token_hash"],
        }

    def revoke_session(self, token_hash: str) -> bool:
        session = self.sessions.get(token_hash)

        if session is None or session["revoked"]:
            return False

        session["revoked"] = True
        return True


class AuthenticationApiFixture:
    """Bundle test client and fake authentication dependencies."""

    def __init__(self) -> None:
        self.auth_service = FakeAuthenticationService()
        self.session_storage = FakeAuthSessionStorage(self.auth_service)


def override_auth_service():
    """Return the active fake authentication service."""
    return _api_fixture.auth_service


def override_session_storage():
    """Return the active fake session storage."""
    return _api_fixture.session_storage


_api_fixture = AuthenticationApiFixture()


@pytest.fixture
def api_client():
    """Create an isolated authentication API test client."""
    global _api_fixture

    _api_fixture = AuthenticationApiFixture()

    app.dependency_overrides[auth_route.get_authentication_service] = (
        override_auth_service
    )

    app.dependency_overrides[auth_route.get_auth_session_storage] = (
        override_session_storage
    )

    try:
        with TestClient(app) as client:
            yield client, _api_fixture
    finally:
        app.dependency_overrides.pop(
            auth_route.get_authentication_service,
            None,
        )
        app.dependency_overrides.pop(
            auth_route.get_auth_session_storage,
            None,
        )


def register_test_user(client: TestClient) -> None:
    """Create a test account."""
    response = client.post(
        "/api/v1/auth/register",
        json={
            "name": "Shubham Chauhan",
            "email": "shubham@example.com",
            "password": "SecurePassword123!",
        },
    )

    assert response.status_code == 201

def login_test_user(client: TestClient) -> str:
    """Register and log in a test user, returning the CSRF token."""
    register_test_user(client)

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "shubham@example.com",
            "password": "SecurePassword123!",
        },
    )

    assert response.status_code == 200

    csrf_token = client.cookies.get(CSRF_COOKIE_NAME)
    assert csrf_token

    return csrf_token


def test_register_user_returns_public_profile(api_client):
    client, _ = api_client

    response = client.post(
        "/api/v1/auth/register",
        json={
            "name": "Shubham",
            "email": "shubham@example.com",
            "password": "SecurePassword123!",
        },
    )

    assert response.status_code == 201
    assert response.json()["email"] == "shubham@example.com"
    assert "password" not in response.json()
    assert "password_hash" not in response.json()


def test_register_user_rejects_duplicate_email(api_client):
    client, _ = api_client
    register_test_user(client)

    response = client.post(
        "/api/v1/auth/register",
        json={
            "name": "Another User",
            "email": "SHUBHAM@example.com",
            "password": "AnotherPassword123!",
        },
    )

    assert response.status_code == 409


def test_register_user_rejects_invalid_email(api_client):
    client, _ = api_client

    response = client.post(
        "/api/v1/auth/register",
        json={
            "name": "Shubham",
            "email": "invalid-email",
            "password": "SecurePassword123!",
        },
    )

    assert response.status_code == 422


def test_login_creates_hashed_session_and_cookie(api_client):
    client, fixture = api_client
    register_test_user(client)

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "shubham@example.com",
            "password": "SecurePassword123!",
        },
    )

    assert response.status_code == 200
    assert response.json()["user"]["email"] == "shubham@example.com"

    raw_token = client.cookies.get(COOKIE_NAME)

    assert raw_token
    assert COOKIE_NAME in response.headers["set-cookie"]
    assert "httponly" in response.headers["set-cookie"].lower()
    assert "samesite=lax" in response.headers["set-cookie"].lower()

    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()

    assert token_hash in fixture.session_storage.sessions
    assert raw_token not in fixture.session_storage.sessions

    raw_csrf_token = client.cookies.get(CSRF_COOKIE_NAME)
    assert raw_csrf_token
    assert CSRF_COOKIE_NAME in response.headers["set-cookie"]
    assert fixture.session_storage.sessions[token_hash]["csrf_token_hash"] == (
        hashlib.sha256(raw_csrf_token.encode("utf-8")).hexdigest()
    )


def test_login_rejects_invalid_credentials(api_client):
    client, _ = api_client
    register_test_user(client)

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "shubham@example.com",
            "password": "WrongPassword123!",
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid email or password."


def test_me_requires_authenticated_session(api_client):
    client, _ = api_client

    response = client.get("/api/v1/auth/me")

    assert response.status_code == 401


def test_me_returns_logged_in_user(api_client):
    client, _ = api_client
    register_test_user(client)

    login_response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "shubham@example.com",
            "password": "SecurePassword123!",
        },
    )

    assert login_response.status_code == 200

    response = client.get("/api/v1/auth/me")

    assert response.status_code == 200
    assert response.json()["email"] == "shubham@example.com"
    assert "password_hash" not in response.json()


def test_logout_revokes_session_and_clears_cookie(api_client):
    client, fixture = api_client
    register_test_user(client)

    login_response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "shubham@example.com",
            "password": "SecurePassword123!",
        },
    )

    assert login_response.status_code == 200

    raw_token = client.cookies.get(COOKIE_NAME)
    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()

    csrf_token = client.cookies.get(CSRF_COOKIE_NAME)
    response = client.post(
        "/api/v1/auth/logout",
        headers={CSRF_HEADER_NAME: csrf_token},
    )

    assert response.status_code == 204
    assert fixture.session_storage.sessions[token_hash]["revoked"] is True

    me_response = client.get("/api/v1/auth/me")
    assert me_response.status_code == 401


def test_logout_rejects_invalid_csrf_token_without_revoking_session(api_client):
    client, fixture = api_client
    register_test_user(client)

    login_response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "shubham@example.com",
            "password": "SecurePassword123!",
        },
    )
    assert login_response.status_code == 200

    raw_token = client.cookies.get(COOKIE_NAME)
    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    response = client.post(
        "/api/v1/auth/logout",
        headers={CSRF_HEADER_NAME: "incorrect-token"},
    )

    assert response.status_code == 403
    assert fixture.session_storage.sessions[token_hash]["revoked"] is False


def test_logout_is_idempotent_without_cookie(api_client):
    client, _ = api_client

    response = client.post("/api/v1/auth/logout")

    assert response.status_code == 204

def test_update_profile_requires_authenticated_session(api_client):
    client, _ = api_client

    response = client.patch(
        "/api/v1/auth/me",
        json={"name": "Updated Name"},
    )

    assert response.status_code == 401


def test_update_profile_rejects_missing_csrf_token(api_client):
    client, _ = api_client
    login_test_user(client)

    response = client.patch(
        "/api/v1/auth/me",
        json={"name": "Updated Name"},
    )

    assert response.status_code == 403


def test_update_profile_rejects_invalid_csrf_token(api_client):
    client, _ = api_client
    login_test_user(client)

    response = client.patch(
        "/api/v1/auth/me",
        json={"name": "Updated Name"},
        headers={CSRF_HEADER_NAME: "incorrect-token"},
    )

    assert response.status_code == 403


def test_update_profile_with_valid_csrf_token(api_client):
    client, _ = api_client
    csrf_token = login_test_user(client)

    response = client.patch(
        "/api/v1/auth/me",
        json={"name": "Updated Shubham"},
        headers={CSRF_HEADER_NAME: csrf_token},
    )

    assert response.status_code == 200
    assert response.json()["name"] == "Updated Shubham"
    assert response.json()["email"] == "shubham@example.com"

    me_response = client.get("/api/v1/auth/me")

    assert me_response.status_code == 200
    assert me_response.json()["name"] == "Updated Shubham"


def test_change_password_requires_authenticated_session(api_client):
    client, _ = api_client

    response = client.post(
        "/api/v1/auth/change-password",
        json={
            "current_password": "SecurePassword123!",
            "new_password": "NewSecurePassword123!",
        },
    )

    assert response.status_code == 401


def test_change_password_rejects_missing_csrf_token(api_client):
    client, _ = api_client
    login_test_user(client)

    response = client.post(
        "/api/v1/auth/change-password",
        json={
            "current_password": "SecurePassword123!",
            "new_password": "NewSecurePassword123!",
        },
    )

    assert response.status_code == 403


def test_change_password_rejects_invalid_csrf_token(api_client):
    client, _ = api_client
    login_test_user(client)

    response = client.post(
        "/api/v1/auth/change-password",
        json={
            "current_password": "SecurePassword123!",
            "new_password": "NewSecurePassword123!",
        },
        headers={CSRF_HEADER_NAME: "incorrect-token"},
    )

    assert response.status_code == 403


def test_change_password_rejects_wrong_current_password(api_client):
    client, _ = api_client
    csrf_token = login_test_user(client)

    response = client.post(
        "/api/v1/auth/change-password",
        json={
            "current_password": "WrongPassword123!",
            "new_password": "NewSecurePassword123!",
        },
        headers={CSRF_HEADER_NAME: csrf_token},
    )

    assert response.status_code == 400


def test_change_password_with_valid_credentials(api_client):
    client, fixture = api_client
    csrf_token = login_test_user(client)

    response = client.post(
        "/api/v1/auth/change-password",
        json={
            "current_password": "SecurePassword123!",
            "new_password": "NewSecurePassword123!",
        },
        headers={CSRF_HEADER_NAME: csrf_token},
    )

    assert response.status_code == 204

    assert (
        fixture.auth_service.passwords["shubham@example.com"]
        == "NewSecurePassword123!"
    )