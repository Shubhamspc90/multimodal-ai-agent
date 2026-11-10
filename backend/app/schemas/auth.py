"""Request and response schemas for authentication APIs."""

from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, StringConstraints


class RegisterRequest(BaseModel):
    """Validate user registration data."""

    model_config = ConfigDict(extra="forbid")

    name: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            max_length=100,
        ),
    ]
    email: EmailStr
    password: Annotated[
        str,
        StringConstraints(min_length=8, max_length=128),
    ]


class LoginRequest(BaseModel):
    """Validate user login credentials."""

    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    password: Annotated[
        str,
        StringConstraints(min_length=1, max_length=128),
    ]


class UpdateProfileRequest(BaseModel):
    """Validate profile update data."""

    model_config = ConfigDict(extra="forbid")

    name: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            max_length=100,
        ),
    ]


class ChangePasswordRequest(BaseModel):
    """Validate password change data."""

    model_config = ConfigDict(extra="forbid")

    current_password: Annotated[
        str,
        StringConstraints(min_length=1, max_length=128),
    ]

    new_password: Annotated[
        str,
        StringConstraints(min_length=8, max_length=128),
    ]


class AuthUserResponse(BaseModel):
    """Expose safe public user information."""

    id: UUID
    name: str
    email: EmailStr
    created_at: datetime


class LoginResponse(BaseModel):
    """Represent a successful login response."""

    user: AuthUserResponse
    expires_at: datetime
