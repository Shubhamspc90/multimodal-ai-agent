"""Business logic for user registration and authentication."""

from uuid import UUID

from backend.app.core.security import hash_password, verify_password
from backend.app.services.auth_storage import (
    PostgresUserStorage,
)

MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128
MAX_NAME_LENGTH = 100
MAX_EMAIL_LENGTH = 320


class InvalidCredentialsError(ValueError):
    """Raised when the supplied login credentials are invalid."""


class PasswordPolicyError(ValueError):
    """Raised when a password does not meet account requirements."""


class UserNotFoundError(ValueError):
    """Raised when an account cannot be found."""


class AuthenticationService:
    """Handle registration and password-based authentication."""

    def __init__(
        self,
        storage: PostgresUserStorage | None = None,
    ) -> None:
        self._storage = storage or PostgresUserStorage()

    def register_user(
        self,
        name: str,
        email: str,
        password: str,
    ) -> dict:
        """Validate input, hash the password, and create an account."""
        name = name.strip()
        email = email.strip().lower()

        if not name:
            raise ValueError("Name must not be empty.")

        if len(name) > MAX_NAME_LENGTH:
            raise ValueError("Name must not exceed 100 characters.")

        if not email:
            raise ValueError("Email must not be empty.")

        if not 3 <= len(email) <= MAX_EMAIL_LENGTH:
            raise ValueError("Email must contain between 3 and 320 characters.")

        self._validate_password(password)

        password_hash = hash_password(password)

        # DuplicateEmailError is intentionally propagated.
        # The API layer will later translate it into HTTP 409.
        return self._storage.create_user(
            name=name,
            email=email,
            password_hash=password_hash,
        )

    def authenticate_user(self, email: str, password: str) -> dict:
        """Verify credentials and return public user information."""
        normalized_email = email.strip().lower()

        if not normalized_email or not password:
            raise InvalidCredentialsError("Invalid email or password.")

        user = self._storage.get_user_for_authentication(normalized_email)

        if user is None or not verify_password(
            password,
            user["password_hash"],
        ):
            raise InvalidCredentialsError("Invalid email or password.")

        # Never return the stored password hash to the caller.
        return {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
            "created_at": user["created_at"],
        }

    def update_user_name(self, user_id: UUID, name: str) -> dict:
        """Validate and update a user's display name."""
        normalized_name = name.strip()

        if not normalized_name:
            raise ValueError("Name must not be empty.")

        if len(normalized_name) > MAX_NAME_LENGTH:
            raise ValueError("Name must not exceed 100 characters.")

        user = self._storage.update_user_name(
            user_id,
            normalized_name,
        )

        if user is None:
            raise UserNotFoundError("User account was not found.")

        return user

    def change_password(
        self,
        user_id: UUID,
        current_password: str,
        new_password: str,
    ) -> None:
        """Verify the current password and securely update the password."""
        if not current_password:
            raise InvalidCredentialsError("Current password is incorrect.")

        self._validate_password(new_password)

        user = self._storage.get_user_credentials_by_id(user_id)

        if user is None:
            raise UserNotFoundError("User account was not found.")

        if not verify_password(
            current_password,
            user["password_hash"],
        ):
            raise InvalidCredentialsError("Current password is incorrect.")

        if verify_password(new_password, user["password_hash"]):
            raise PasswordPolicyError(
                "New password must be different from the current password."
            )

        new_password_hash = hash_password(new_password)

        updated = self._storage.update_password_hash(
            user_id,
            new_password_hash,
        )

        if not updated:
            raise UserNotFoundError("User account was not found.")

    @staticmethod
    def _validate_password(password: str) -> None:
        """Enforce password length requirements."""
        if (
            not password.strip()
            or not MIN_PASSWORD_LENGTH <= len(password) <= MAX_PASSWORD_LENGTH
        ):
            raise PasswordPolicyError(
                "Password must contain between 8 and 128 non-whitespace characters."
            )
