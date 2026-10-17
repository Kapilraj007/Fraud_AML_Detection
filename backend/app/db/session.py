"""Async SQLAlchemy engine/session construction for Neon and test SQLite."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from backend.app.core.config import Settings, get_settings, normalize_database_url
from backend.app.db.base import Base


def create_database_engine(
    settings: Settings | None = None,
    *,
    database_url: str | None = None,
) -> AsyncEngine:
    """Create a small, pre-pinged async engine.

    Neon already supplies PgBouncer on pooled endpoints, so the local pool is
    intentionally small.  In-memory SQLite uses a StaticPool so every test
    session observes the same database.
    """
    cfg = settings or get_settings()
    url = cfg.database_url if database_url is None else normalize_database_url(database_url)
    if url.startswith("sqlite+aiosqlite"):
        options: dict[str, object] = {
            "echo": cfg.database_echo,
            "pool_pre_ping": True,
        }
        if ":memory:" in url:
            options["poolclass"] = StaticPool
        sqlite_engine = create_async_engine(url, **options)

        @event.listens_for(sqlite_engine.sync_engine, "connect")
        def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA busy_timeout=5000")
            finally:
                cursor.close()

        return sqlite_engine

    return create_async_engine(
        url,
        echo=cfg.database_echo,
        pool_pre_ping=True,
        pool_size=cfg.database_pool_size,
        max_overflow=cfg.database_max_overflow,
        pool_recycle=cfg.database_pool_recycle_seconds,
    )


def create_session_factory(
    database_engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=database_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


engine = create_database_engine()
AsyncSessionFactory = create_session_factory(engine)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding one transaction-neutral request session."""
    async with AsyncSessionFactory() as session:
        yield session


@asynccontextmanager
async def session_scope(
    factory: async_sessionmaker[AsyncSession] | None = None,
) -> AsyncIterator[AsyncSession]:
    """Commit on success and roll back on error for CLI/service operations."""
    chosen = factory or AsyncSessionFactory
    async with chosen() as session:
        try:
            yield session
            await session.commit()
        except BaseException:
            await session.rollback()
            raise


async def database_is_ready(session: AsyncSession) -> bool:
    try:
        await session.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


async def create_schema_for_tests(database_engine: AsyncEngine) -> None:
    """Create tables for isolated tests; production uses Alembic only."""
    async with database_engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def drop_schema_for_tests(database_engine: AsyncEngine) -> None:
    async with database_engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)


__all__ = [
    "AsyncSessionFactory",
    "create_database_engine",
    "create_schema_for_tests",
    "create_session_factory",
    "database_is_ready",
    "drop_schema_for_tests",
    "engine",
    "get_session",
    "session_scope",
]
