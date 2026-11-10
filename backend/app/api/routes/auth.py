"""API routes for user registration and authentication."""

import logging
import secrets
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Response, status

from backend.app.api.dependencies import (
    CSRF_COOKIE_NAME,
    SESSION_COOKIE_NAME,
    CsrfCookie,
    CsrfHeader,
    CsrfProtection,
    CurrentUser,
    SessionCookie,
    SessionStorageDependency,
    hash_csrf_token,
    hash_session_token,
    validate_csrf_token,
)
from backend.app.api.dependencies import (
    get_auth_session_storage as get_auth_session_storage,
)
from backend.app.core.config import settings
from backend.app.schemas.auth import (
    AuthUserResponse,
    ChangePasswordRequest,
    LoginRequest,
    LoginResponse,
    RegisterRequest,
    UpdateProfileRequest,
)
from backend.app.services.auth_service import (
    AuthenticationService,
    InvalidCredentialsError,
    PasswordPolicyError,
    UserNotFoundError,
)
from backend.app.services.auth_storage import DuplicateEmailError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])

SESSION_LIFETIME = timedelta(hours=24)
SESSION_MAX_AGE = int(SESSION_LIFETIME.total_seconds())


@lru_cache
def get_authentication_service() -> AuthenticationService:
    """Return the shared authentication service."""
    return AuthenticationService()


AuthServiceDependency = Annotated[
    AuthenticationService,
    Depends(get_authentication_service),
]


def _database_unavailable(exc: Exception) -> HTTPException:
    """Return a safe database availability error."""
    logger.exception("Authentication database operation failed.")
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Authentication service is temporarily unavailable.",
    )


def _secure_cookie() -> bool:
    """Enable secure cookies outside local development and testing."""
    environment = settings.app_environment.strip().lower()
    return environment not in {"development", "dev", "local", "test"}


def _set_auth_cookies(
    response: Response,
    session_token: str,
    csrf_token: str,
) -> None:
    """Set the HTTP-only session cookie and readable CSRF cookie."""
    secure = _secure_cookie()
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_token,
        max_age=SESSION_MAX_AGE,
        path="/",
        secure=secure,
        httponly=True,
        samesite="lax",
    )
    response.set_cookie(
        key=CSRF_COOKIE_NAME,
        value=csrf_token,
        max_age=SESSION_MAX_AGE,
        path="/",
        secure=secure,
        httponly=False,
        samesite="lax",
    )


def _delete_auth_cookies(response: Response) -> None:
    """Remove both authentication cookies."""
    secure = _secure_cookie()
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        secure=secure,
        httponly=True,
        samesite="lax",
    )
    response.delete_cookie(
        key=CSRF_COOKIE_NAME,
        path="/",
        secure=secure,
        httponly=False,
        samesite="lax",
    )


@router.post(
    "/register",
    response_model=AuthUserResponse,
    status_code=status.HTTP_201_CREATED,
)
def register_user(
    request: RegisterRequest,
    service: AuthServiceDependency,
) -> dict:
    """Register a new user account."""
    try:
        return service.register_user(
            name=request.name,
            email=str(request.email),
            password=request.password,
        )
    except DuplicateEmailError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        ) from exc
    except PasswordPolicyError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except (psycopg.Error, ValueError) as exc:
        raise _database_unavailable(exc) from exc


@router.post("/login", response_model=LoginResponse)
def login_user(
    request: LoginRequest,
    response: Response,
    service: AuthServiceDependency,
    session_storage: SessionStorageDependency,
) -> dict:
    """Authenticate the user and create a browser session."""
    try:
        user = service.authenticate_user(
            email=str(request.email),
            password=request.password,
        )
    except InvalidCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        ) from exc
    except (psycopg.Error, ValueError) as exc:
        raise _database_unavailable(exc) from exc

    raw_session_token = secrets.token_urlsafe(32)
    raw_csrf_token = secrets.token_urlsafe(32)
    expires_at = datetime.now(UTC) + SESSION_LIFETIME

    try:
        session_storage.create_session(
            user_id=user["id"],
            token_hash=hash_session_token(raw_session_token),
            csrf_token_hash=hash_csrf_token(raw_csrf_token),
            expires_at=expires_at,
        )
    except (psycopg.Error, ValueError) as exc:
        raise _database_unavailable(exc) from exc

    _set_auth_cookies(
        response=response,
        session_token=raw_session_token,
        csrf_token=raw_csrf_token,
    )
    return {"user": user, "expires_at": expires_at}


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout_user(
    response: Response,
    session_storage: SessionStorageDependency,
    session_token: SessionCookie = None,
    csrf_cookie: CsrfCookie = None,
    csrf_header: CsrfHeader = None,
) -> Response:
    """Revoke the current session and clear cookies; safe to call repeatedly."""
    if session_token:
        token_hash = hash_session_token(session_token)
        try:
            active_session = session_storage.get_active_session(token_hash)
            if active_session is not None:
                validate_csrf_token(
                    csrf_cookie=csrf_cookie,
                    csrf_header=csrf_header,
                    expected_hash=active_session.get("csrf_token_hash"),
                )
            session_storage.revoke_session(token_hash)
        except HTTPException:
            raise
        except (psycopg.Error, ValueError) as exc:
            raise _database_unavailable(exc) from exc

    _delete_auth_cookies(response)
    response.status_code = status.HTTP_204_NO_CONTENT
    return response


@router.get("/me", response_model=AuthUserResponse)
def get_current_user(current_user: CurrentUser) -> dict:
    """Return the currently authenticated user."""
    return current_user


@router.patch("/me", response_model=AuthUserResponse)
def update_current_user(
    request: UpdateProfileRequest,
    current_user: CurrentUser,
    service: AuthServiceDependency,
    csrf_protection: CsrfProtection,
) -> dict:
    """Update the authenticated user's profile."""
    try:
        return service.update_user_name(
            user_id=current_user["id"],
            name=request.name,
        )
    except UserNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except psycopg.Error as exc:
        raise _database_unavailable(exc) from exc


@router.post("/change-password", status_code=status.HTTP_204_NO_CONTENT)
def change_current_user_password(
    request: ChangePasswordRequest,
    current_user: CurrentUser,
    service: AuthServiceDependency,
    csrf_protection: CsrfProtection,
) -> Response:
    """Change the authenticated user's password."""
    try:
        service.change_password(
            user_id=current_user["id"],
            current_password=request.current_password,
            new_password=request.new_password,
        )
    except InvalidCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except PasswordPolicyError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except UserNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except psycopg.Error as exc:
        raise _database_unavailable(exc) from exc

    return Response(status_code=status.HTTP_204_NO_CONTENT)
