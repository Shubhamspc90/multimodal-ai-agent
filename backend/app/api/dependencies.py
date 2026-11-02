"""Shared dependencies for authenticated API requests."""

import hashlib
import logging
import re
import secrets
from functools import lru_cache
from typing import Annotated

import psycopg
from fastapi import Cookie, Depends, Header, HTTPException, status

from backend.app.services.auth_session_storage import (
    PostgresAuthSessionStorage,
)

logger = logging.getLogger(__name__)

SESSION_COOKIE_NAME = "multimodal_ai_session"
CSRF_COOKIE_NAME = "multimodal_ai_csrf"
CSRF_HEADER_NAME = "X-CSRF-Token"

_TOKEN_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@lru_cache
def get_auth_session_storage() -> PostgresAuthSessionStorage:
    """Return the shared PostgreSQL session storage."""
    return PostgresAuthSessionStorage()


SessionStorageDependency = Annotated[
    PostgresAuthSessionStorage,
    Depends(get_auth_session_storage),
]

SessionCookie = Annotated[
    str | None,
    Cookie(alias=SESSION_COOKIE_NAME),
]

CsrfCookie = Annotated[
    str | None,
    Cookie(alias=CSRF_COOKIE_NAME),
]

CsrfHeader = Annotated[
    str | None,
    Header(alias=CSRF_HEADER_NAME),
]


def hash_session_token(token: str) -> str:
    """Hash a raw session token before database lookup."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def hash_csrf_token(token: str) -> str:
    """Hash a CSRF token before storing or comparing it."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def get_current_session(
    session_storage: SessionStorageDependency,
    session_token: SessionCookie = None,
) -> dict:
    """Return the active session associated with the browser cookie."""
    if not session_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
        )

    token_hash = hash_session_token(session_token)

    try:
        session = session_storage.get_active_session(token_hash)
    except (psycopg.Error, ValueError) as exc:
        logger.exception("Authentication session lookup failed.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service is temporarily unavailable.",
        ) from exc

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session is invalid or expired.",
        )

    return session


CurrentSession = Annotated[
    dict,
    Depends(get_current_session),
]


def get_current_user(
    current_session: CurrentSession,
) -> dict:
    """Return the authenticated user from the active session."""
    return current_session["user"]


CurrentUser = Annotated[
    dict,
    Depends(get_current_user),
]


def validate_csrf_token(
    csrf_cookie: str | None,
    csrf_header: str | None,
    expected_hash: str | None,
) -> None:
    """Validate the CSRF cookie, request header and stored session hash."""
    if (
        not isinstance(csrf_cookie, str)
        or not isinstance(csrf_header, str)
        or not csrf_cookie
        or not csrf_header
        or len(csrf_cookie) > 256
        or len(csrf_header) > 256
        or not csrf_cookie.isascii()
        or not csrf_header.isascii()
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF validation failed.",
        )

    if not secrets.compare_digest(csrf_cookie, csrf_header):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF validation failed.",
        )

    if not isinstance(expected_hash, str) or not _TOKEN_HASH_PATTERN.fullmatch(
        expected_hash
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF validation failed.",
        )

    supplied_hash = hash_csrf_token(csrf_header)

    if not secrets.compare_digest(supplied_hash, expected_hash):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF validation failed.",
        )


def require_csrf_token(
    current_session: CurrentSession,
    csrf_cookie: CsrfCookie = None,
    csrf_header: CsrfHeader = None,
) -> None:
    """Enforce CSRF validation for state-changing authenticated requests."""
    validate_csrf_token(
        csrf_cookie=csrf_cookie,
        csrf_header=csrf_header,
        expected_hash=current_session.get("csrf_token_hash"),
    )


CsrfProtection = Annotated[
    None,
    Depends(require_csrf_token),
]
