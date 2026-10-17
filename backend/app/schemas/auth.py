"""Authentication and user-management API contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from backend.app.db.enums import UserRole


def _email(value: str) -> str:
    normalized = value.strip().casefold()
    if (
        len(normalized) > 320
        or normalized.count("@") != 1
        or any(char.isspace() for char in normalized)
    ):
        raise ValueError("email address is invalid")
    local_part, domain = normalized.split("@", 1)
    if not local_part or not domain or local_part.startswith(".") or local_part.endswith("."):
        raise ValueError("email address is invalid")
    return normalized


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: SecretStr = Field(min_length=1, max_length=1024)

    _normalize_email = field_validator("email")(_email)


class RefreshRequest(BaseModel):
    refresh_token: SecretStr = Field(min_length=1, max_length=4096)


class LogoutRequest(BaseModel):
    refresh_token: SecretStr = Field(min_length=1, max_length=4096)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int = Field(gt=0)
    refresh_expires_in: int = Field(gt=0)


class AccessTokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int = Field(gt=0)


class UserCreate(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: SecretStr = Field(min_length=12, max_length=256)
    full_name: str | None = Field(default=None, max_length=200)
    role: UserRole = UserRole.FRAUD_ANALYST

    _normalize_email = field_validator("email")(_email)

    @field_validator("password")
    @classmethod
    def _strong_password(cls, value: SecretStr) -> SecretStr:
        raw = value.get_secret_value()
        if not 12 <= len(raw) <= 256:
            raise ValueError("password must contain between 12 and 256 characters")
        if not any(char.isalpha() for char in raw) or not any(char.isdigit() for char in raw):
            raise ValueError("password must contain letters and numbers")
        return value

    @field_validator("full_name")
    @classmethod
    def _clean_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: str
    full_name: str | None
    role: UserRole
    is_active: bool
    created_at: datetime
    updated_at: datetime
    last_login_at: datetime | None


class UserRoleUpdate(BaseModel):
    role: UserRole


class UserActivationUpdate(BaseModel):
    is_active: bool


class MessageResponse(BaseModel):
    message: str


__all__ = [
    "AccessTokenResponse",
    "LoginRequest",
    "LogoutRequest",
    "MessageResponse",
    "RefreshRequest",
    "TokenResponse",
    "UserActivationUpdate",
    "UserCreate",
    "UserRead",
    "UserRoleUpdate",
]
