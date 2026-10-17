"""Environment-backed configuration with Neon-safe URL handling.

Runtime PostgreSQL connections use SQLAlchemy's asyncpg dialect.  Neon URLs
are routed through the managed pooler and require TLS.  SQLite remains
available for isolated unit tests; production validation rejects it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL, make_url


Environment = Literal["development", "test", "production"]

_TEST_SECRET = "test-only-jwt-secret-change-before-production-0123456789"


def _is_neon_host(host: str | None) -> bool:
    return bool(host and (host.casefold() == "neon.tech" or host.casefold().endswith(".neon.tech")))


def normalize_database_url(value: str, *, pooled: bool = True) -> str:
    """Return an async SQLAlchemy URL with safe Neon defaults.

    ``postgres://`` and ordinary ``postgresql://`` URLs are converted to the
    asyncpg dialect.  Neon application URLs are directed to the provider's
    transaction pooler by adding ``-pooler`` to the endpoint label.  TLS is
    mandatory for every Neon connection.  Migration callers may pass
    ``pooled=False`` to preserve a direct Neon endpoint.
    """
    raw = str(value).strip()
    if not raw:
        raise ValueError("database URL cannot be empty")
    try:
        url = make_url(raw)
    except Exception as exc:  # SQLAlchemy exposes several parse exceptions.
        raise ValueError("database URL is invalid") from exc

    driver = url.drivername.casefold()
    if driver.startswith("sqlite"):
        if driver != "sqlite+aiosqlite":
            url = url.set(drivername="sqlite+aiosqlite")
        return url.render_as_string(hide_password=False)

    if not driver.startswith("postgres"):
        raise ValueError("database URL must use PostgreSQL or SQLite")

    host = url.host
    if _is_neon_host(host):
        labels = host.split(".") if host else []
        if pooled and labels and not labels[0].casefold().endswith("-pooler"):
            labels[0] = f"{labels[0]}-pooler"
            host = ".".join(labels)

    query = dict(url.query)
    ssl_mode = query.pop("sslmode", None)
    # Neon may include libpq's channel_binding option in copied connection
    # strings.  asyncpg does not implement that keyword and would otherwise
    # fail during connect; TLS is configured independently below.
    query.pop("channel_binding", None)
    if _is_neon_host(host):
        query["ssl"] = str(query.get("ssl") or ssl_mode or "require")
        if query["ssl"].casefold() in {"disable", "allow", "prefer"}:
            raise ValueError("Neon database connections must require TLS")
    elif ssl_mode is not None and "ssl" not in query:
        query["ssl"] = str(ssl_mode)

    url = url.set(
        drivername="postgresql+asyncpg",
        host=host,
        query=query,
    )
    return url.render_as_string(hide_password=False)


class Settings(BaseSettings):
    """Application configuration loaded from environment variables or `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Capacity-aware AML triage API"
    environment: Environment = "test"
    # Avoid consuming the generic DEBUG variable used by shells/tooling.  The
    # application-specific aliases remain explicit and unambiguous.
    debug: bool = Field(
        default=False,
        validation_alias=AliasChoices("APP_DEBUG", "AML_DEBUG"),
    )

    database_url: str = "sqlite+aiosqlite:///:memory:"
    database_migration_url: str | None = None
    database_echo: bool = False
    database_pool_size: int = Field(default=3, ge=1, le=10)
    database_max_overflow: int = Field(default=2, ge=0, le=10)
    database_pool_recycle_seconds: int = Field(default=300, ge=30, le=3600)

    jwt_secret_key: SecretStr = SecretStr(_TEST_SECRET)
    jwt_algorithm: Literal["HS256"] = "HS256"
    jwt_issuer: str = "capacity-aware-aml-triage"
    jwt_audience: str = "capacity-aware-aml-api"
    access_token_minutes: int = Field(default=15, ge=1, le=60)
    refresh_token_days: int = Field(default=7, ge=1, le=30)

    cors_origins: tuple[str, ...] = ("http://localhost:5173",)
    model_artifact_dir: Path = Path("artifacts/models")

    @field_validator("database_url")
    @classmethod
    def _normalise_runtime_url(cls, value: str) -> str:
        return normalize_database_url(value, pooled=True)

    @field_validator("database_migration_url")
    @classmethod
    def _normalise_migration_url(cls, value: str | None) -> str | None:
        if value is None or not str(value).strip():
            return None
        return normalize_database_url(value, pooled=False)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_origins(cls, value: object) -> object:
        if isinstance(value, str) and not value.lstrip().startswith("["):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        return value

    @model_validator(mode="after")
    def _validate_environment(self) -> "Settings":
        url = make_url(self.database_url)
        secret = self.jwt_secret_key.get_secret_value()
        if len(secret) < 32:
            raise ValueError("JWT_SECRET_KEY must contain at least 32 characters")
        if self.environment == "production":
            if not url.drivername.startswith("postgresql"):
                raise ValueError("production requires a PostgreSQL database URL")
            if not _is_neon_host(url.host):
                raise ValueError("production DATABASE_URL must point to Neon")
            if not (url.host or "").split(".", 1)[0].casefold().endswith("-pooler"):
                raise ValueError("production Neon connections must use the pooled endpoint")
            if str(url.query.get("ssl", "")).casefold() not in {
                "require", "verify-ca", "verify-full",
            }:
                raise ValueError("production Neon connections must require TLS")
            if secret == _TEST_SECRET:
                raise ValueError("production JWT_SECRET_KEY must not use the test default")
            if self.debug:
                raise ValueError("debug mode cannot be enabled in production")
            if "*" in self.cors_origins:
                raise ValueError("wildcard CORS is not allowed in production")
            migration_url = make_url(self.migration_url)
            if not migration_url.drivername.startswith("postgresql"):
                raise ValueError("production migrations require PostgreSQL")
            if not _is_neon_host(migration_url.host):
                raise ValueError("production migrations must target Neon")
            if str(migration_url.query.get("ssl", "")).casefold() not in {
                "require", "verify-ca", "verify-full",
            }:
                raise ValueError("production migration connections must require TLS")
        return self

    @property
    def migration_url(self) -> str:
        """URL used by Alembic, falling back to the runtime endpoint."""
        return self.database_migration_url or self.database_url

    @property
    def is_sqlite(self) -> bool:
        return make_url(self.database_url).drivername.startswith("sqlite")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide validated configuration."""
    return Settings()


__all__ = ["Environment", "Settings", "get_settings", "normalize_database_url"]
