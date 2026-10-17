"""Password hashing and typed JWT issuance/verification.

Refresh-token rows store only an HMAC of the JWT identifier.  A successful
refresh must revoke the current row and create a replacement in the same token
family; repository helpers enforce that rotation transactionally.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
from typing import Any, Literal, Mapping
from uuid import UUID, uuid4

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
import jwt

from backend.app.core.config import Settings, get_settings
from backend.app.db.enums import UserRole


TokenType = Literal["access", "refresh"]

_password_hasher = PasswordHasher(
    time_cost=3,
    memory_cost=65536,
    parallelism=4,
    hash_len=32,
    salt_len=16,
)


class AuthenticationError(ValueError):
    """Raised when credentials or a token cannot be authenticated."""


@dataclass(frozen=True)
class IssuedToken:
    token: str
    jti: str
    expires_at: datetime


@dataclass(frozen=True)
class IssuedTokenPair:
    access: IssuedToken
    refresh: IssuedToken
    family_id: UUID
    refresh_session_id: UUID


@dataclass(frozen=True)
class DecodedToken:
    subject: str
    role: UserRole
    token_type: TokenType
    jti: str
    issued_at: datetime
    expires_at: datetime
    family_id: UUID | None = None
    refresh_session_id: UUID | None = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def hash_password(password: str) -> str:
    """Hash a password using memory-hard Argon2id."""
    _validate_password_input(password)
    return _password_hasher.hash(password)


def verify_password(password: str, encoded_hash: str) -> bool:
    """Return ``False`` for mismatches and malformed stored hashes."""
    if not isinstance(password, str) or not isinstance(encoded_hash, str):
        return False
    try:
        return bool(_password_hasher.verify(encoded_hash, password))
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def password_needs_rehash(encoded_hash: str) -> bool:
    try:
        return _password_hasher.check_needs_rehash(encoded_hash)
    except (VerificationError, InvalidHashError):
        return True


def issue_access_token(
    subject: str | UUID,
    role: UserRole | str,
    *,
    settings: Settings | None = None,
    now: datetime | None = None,
    extra_claims: Mapping[str, Any] | None = None,
) -> IssuedToken:
    cfg = settings or get_settings()
    issued_at = _aware(now or utc_now())
    expires_at = issued_at + timedelta(minutes=cfg.access_token_minutes)
    return _issue_token(
        str(subject), _role(role), "access", issued_at, expires_at, cfg,
        extra_claims=extra_claims,
    )


def issue_refresh_token(
    subject: str | UUID,
    role: UserRole | str,
    *,
    family_id: UUID,
    refresh_session_id: UUID,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> IssuedToken:
    cfg = settings or get_settings()
    issued_at = _aware(now or utc_now())
    expires_at = issued_at + timedelta(days=cfg.refresh_token_days)
    return _issue_token(
        str(subject), _role(role), "refresh", issued_at, expires_at, cfg,
        extra_claims={"fid": str(family_id), "sid": str(refresh_session_id)},
    )


def issue_token_pair(
    subject: str | UUID,
    role: UserRole | str,
    *,
    family_id: UUID | None = None,
    refresh_session_id: UUID | None = None,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> IssuedTokenPair:
    """Issue access and refresh JWTs for a new or rotated session."""
    cfg = settings or get_settings()
    moment = _aware(now or utc_now())
    family = family_id or uuid4()
    session_id = refresh_session_id or uuid4()
    return IssuedTokenPair(
        access=issue_access_token(subject, role, settings=cfg, now=moment),
        refresh=issue_refresh_token(
            subject,
            role,
            family_id=family,
            refresh_session_id=session_id,
            settings=cfg,
            now=moment,
        ),
        family_id=family,
        refresh_session_id=session_id,
    )


def decode_token(
    token: str,
    *,
    expected_type: TokenType,
    settings: Settings | None = None,
) -> DecodedToken:
    """Validate signature, registered claims, type, role, and rotation IDs."""
    cfg = settings or get_settings()
    try:
        claims = jwt.decode(
            token,
            cfg.jwt_secret_key.get_secret_value(),
            algorithms=[cfg.jwt_algorithm],
            audience=cfg.jwt_audience,
            issuer=cfg.jwt_issuer,
            options={"require": ["sub", "type", "jti", "iat", "nbf", "exp", "role"]},
        )
    except jwt.PyJWTError as exc:
        raise AuthenticationError("token is invalid or expired") from exc

    token_type = claims.get("type")
    if token_type != expected_type:
        raise AuthenticationError("token type is not valid for this operation")
    subject = str(claims.get("sub", "")).strip()
    jti = str(claims.get("jti", "")).strip()
    if not subject or not jti:
        raise AuthenticationError("token subject or identifier is missing")
    try:
        role = UserRole(str(claims["role"]))
        issued_at = datetime.fromtimestamp(float(claims["iat"]), tz=timezone.utc)
        expires_at = datetime.fromtimestamp(float(claims["exp"]), tz=timezone.utc)
        family = UUID(str(claims["fid"])) if expected_type == "refresh" else None
        session_id = UUID(str(claims["sid"])) if expected_type == "refresh" else None
    except (KeyError, TypeError, ValueError) as exc:
        raise AuthenticationError("token claims are malformed") from exc
    return DecodedToken(
        subject=subject,
        role=role,
        token_type=expected_type,
        jti=jti,
        issued_at=issued_at,
        expires_at=expires_at,
        family_id=family,
        refresh_session_id=session_id,
    )


def hash_token_identifier(jti: str, *, settings: Settings | None = None) -> str:
    """Return a keyed digest suitable for storing a JWT identifier."""
    cfg = settings or get_settings()
    value = str(jti).strip()
    if not value:
        raise ValueError("token identifier cannot be empty")
    return hmac.new(
        cfg.jwt_secret_key.get_secret_value().encode("utf-8"),
        value.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _issue_token(
    subject: str,
    role: UserRole,
    token_type: TokenType,
    issued_at: datetime,
    expires_at: datetime,
    settings: Settings,
    *,
    extra_claims: Mapping[str, Any] | None = None,
) -> IssuedToken:
    if not subject.strip():
        raise ValueError("token subject cannot be empty")
    jti = uuid4().hex
    claims: dict[str, Any] = {
        "sub": subject,
        "role": role.value,
        "type": token_type,
        "jti": jti,
        "iat": issued_at,
        "nbf": issued_at,
        "exp": expires_at,
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
    }
    if extra_claims:
        reserved = {"sub", "role", "type", "jti", "iat", "nbf", "exp", "iss", "aud"}
        overlap = reserved & set(extra_claims)
        if overlap:
            raise ValueError(f"extra claims cannot replace registered claims: {sorted(overlap)}")
        claims.update(extra_claims)
    encoded = jwt.encode(
        claims,
        settings.jwt_secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )
    return IssuedToken(token=encoded, jti=jti, expires_at=expires_at)


def _role(value: UserRole | str) -> UserRole:
    try:
        return value if isinstance(value, UserRole) else UserRole(str(value))
    except ValueError as exc:
        raise ValueError("unknown user role") from exc


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("token timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


def _validate_password_input(password: str) -> None:
    if not isinstance(password, str):
        raise TypeError("password must be a string")
    if not 12 <= len(password) <= 1024:
        raise ValueError("password must contain between 12 and 1024 characters")


__all__ = [
    "AuthenticationError",
    "DecodedToken",
    "IssuedToken",
    "IssuedTokenPair",
    "decode_token",
    "hash_password",
    "hash_token_identifier",
    "issue_access_token",
    "issue_refresh_token",
    "issue_token_pair",
    "password_needs_rehash",
    "utc_now",
    "verify_password",
]
