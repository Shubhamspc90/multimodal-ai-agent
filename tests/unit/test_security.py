"""Tests for authentication security utilities."""

import pytest

from backend.app.core.security import hash_password, verify_password


def test_hash_password_uses_argon2id() -> None:
    password_hash = hash_password("SecurePassword123!")

    assert password_hash.startswith("$argon2id$")
    assert password_hash != "SecurePassword123!"


def test_verify_password_accepts_correct_password() -> None:
    password_hash = hash_password("SecurePassword123!")

    assert verify_password("SecurePassword123!", password_hash)


def test_verify_password_rejects_incorrect_password() -> None:
    password_hash = hash_password("SecurePassword123!")

    assert not verify_password("WrongPassword123!", password_hash)


def test_verify_password_rejects_invalid_hash() -> None:
    assert not verify_password("SecurePassword123!", "invalid-hash")


def test_hash_password_rejects_empty_password() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        hash_password("")


def test_verify_password_rejects_empty_values() -> None:
    assert not verify_password("", "some-hash")
    assert not verify_password("password", "")
