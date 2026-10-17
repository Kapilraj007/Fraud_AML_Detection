from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest
from pydantic import ValidationError
from sqlalchemy.engine import make_url

from backend.app.core.config import Settings, normalize_database_url
from backend.app.core.security import (
    AuthenticationError,
    decode_token,
    hash_password,
    hash_token_identifier,
    issue_access_token,
    issue_token_pair,
    password_needs_rehash,
    utc_now,
    verify_password,
)
from backend.app.db.enums import UserRole
from backend.app.db.session import create_database_engine
from backend.app.schemas.auth import LoginRequest, RefreshRequest, UserCreate


SECRET = "security-test-secret-that-is-longer-than-thirty-two-characters"


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "environment": "test",
        "database_url": "sqlite+aiosqlite:///:memory:",
        "jwt_secret_key": SECRET,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_neon_urls_use_asyncpg_pooler_and_tls() -> None:
    runtime = make_url(
        normalize_database_url(
            "postgresql://user:password@ep-example.us-east-2.aws.neon.tech/neondb"
            "?sslmode=require&channel_binding=require"
        )
    )
    assert runtime.drivername == "postgresql+asyncpg"
    assert runtime.host == "ep-example-pooler.us-east-2.aws.neon.tech"
    assert runtime.query["ssl"] == "require"
    assert "sslmode" not in runtime.query
    assert "channel_binding" not in runtime.query

    migration = make_url(
        normalize_database_url(
            "postgresql://user:password@ep-example.us-east-2.aws.neon.tech/neondb"
            "?sslmode=verify-full",
            pooled=False,
        )
    )
    assert migration.host == "ep-example.us-east-2.aws.neon.tech"
    assert migration.query["ssl"] == "verify-full"

    with pytest.raises(ValueError, match="require TLS"):
        normalize_database_url(
            "postgresql://user:password@ep-example.us-east-2.aws.neon.tech/neondb"
            "?sslmode=disable"
        )


def test_production_configuration_is_neon_only_and_import_is_lazy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEBUG", "release")
    settings = _settings(
        environment="production",
        database_url=(
            "postgresql://user:password@ep-example.us-east-2.aws.neon.tech/neondb"
            "?sslmode=require"
        ),
        database_migration_url=(
            "postgresql://user:password@ep-example.us-east-2.aws.neon.tech/neondb"
            "?sslmode=require"
        ),
    )
    assert settings.debug is False
    assert "-pooler." in make_url(settings.database_url).host
    assert "-pooler." not in make_url(settings.migration_url).host

    # Engine construction must not make a network connection during imports or
    # application startup.  Disposing an unused pool is also connection-free.
    engine = create_database_engine(settings)
    assert engine.url.host == "ep-example-pooler.us-east-2.aws.neon.tech"
    asyncio.run(engine.dispose())

    with pytest.raises(ValidationError, match="production requires"):
        _settings(environment="production")


def test_password_hashing_and_hmac_identifier_storage() -> None:
    settings = _settings()
    encoded = hash_password("Correct-Horse-123")
    assert encoded != "Correct-Horse-123"
    assert encoded.startswith("$argon2id$")
    assert verify_password("Correct-Horse-123", encoded)
    assert not verify_password("incorrect-password", encoded)
    assert not verify_password("Correct-Horse-123", "not-a-hash")
    assert not password_needs_rehash(encoded)

    first = hash_token_identifier("token-jti", settings=settings)
    second = hash_token_identifier("token-jti", settings=settings)
    assert first == second
    assert len(first) == 64
    assert "token-jti" not in first


def test_typed_jwts_validate_claims_expiry_and_token_kind() -> None:
    settings = _settings()
    now = utc_now()
    pair = issue_token_pair(
        "00000000-0000-0000-0000-000000000001",
        UserRole.SENIOR_INVESTIGATOR,
        settings=settings,
        now=now,
    )
    access = decode_token(pair.access.token, expected_type="access", settings=settings)
    refresh = decode_token(pair.refresh.token, expected_type="refresh", settings=settings)
    assert access.role is UserRole.SENIOR_INVESTIGATOR
    assert refresh.family_id == pair.family_id
    assert refresh.refresh_session_id == pair.refresh_session_id

    with pytest.raises(AuthenticationError, match="token type"):
        decode_token(pair.access.token, expected_type="refresh", settings=settings)

    header, payload, signature = pair.access.token.split(".")
    changed = ("A" if signature[0] != "A" else "B") + signature[1:]
    with pytest.raises(AuthenticationError, match="invalid or expired"):
        decode_token(
            ".".join((header, payload, changed)),
            expected_type="access",
            settings=settings,
        )

    expired = issue_access_token(
        "00000000-0000-0000-0000-000000000001",
        UserRole.FRAUD_ANALYST,
        settings=settings,
        now=now - timedelta(hours=2),
    )
    with pytest.raises(AuthenticationError, match="invalid or expired"):
        decode_token(expired.token, expected_type="access", settings=settings)

    with pytest.raises(ValueError, match="registered claims"):
        issue_access_token(
            "subject",
            UserRole.ADMIN,
            settings=settings,
            extra_claims={"sub": "replacement"},
        )


def test_auth_schemas_reject_malformed_or_unbounded_credentials() -> None:
    request = LoginRequest(email=" Analyst@Example.COM ", password="x")
    assert request.email == "analyst@example.com"

    with pytest.raises(ValidationError):
        LoginRequest(email="bad address@example.com", password="x")
    with pytest.raises(ValidationError):
        RefreshRequest(refresh_token="")
    with pytest.raises(ValidationError):
        UserCreate(email="user@example.com", password="letters-only-password")

