from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
from uuid import uuid4

from alembic import command
from alembic.config import Config
import pytest
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import configure_mappers

from backend.app.core.config import Settings, get_settings
from backend.app.core.security import hash_token_identifier, issue_token_pair
from backend.app.db.base import Base
from backend.app.db.enums import ModelStatus, UserRole
from backend.app.db.models import Case, ModelVersion, RefreshSession
from backend.app.db.repositories import (
    RefreshTokenReuseError,
    create_refresh_session,
    create_user,
    promote_model_version,
    register_model_candidate,
    rotate_refresh_session,
)
from backend.app.db.session import (
    create_database_engine,
    create_schema_for_tests,
    create_session_factory,
)


SECRET = "database-test-secret-that-is-longer-than-thirty-two-characters"
EXPECTED_TABLES = {
    "audit_events",
    "case_transactions",
    "cases",
    "dispositions",
    "drift_snapshots",
    "explanations",
    "model_versions",
    "predictions",
    "refresh_sessions",
    "retraining_requests",
    "transactions",
    "users",
}


def _settings(database_url: str = "sqlite+aiosqlite:///:memory:") -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        database_url=database_url,
        jwt_secret_key=SECRET,
    )


def test_all_mappers_relationships_and_schema_constraints_configure() -> None:
    configure_mappers()
    assert set(Base.metadata.tables) == EXPECTED_TABLES
    assert set(UserRole) == {
        UserRole.FRAUD_ANALYST,
        UserRole.SENIOR_INVESTIGATOR,
        UserRole.ADMIN,
    }

    expected_relationships = {
        "User": {"refresh_sessions", "assigned_cases"},
        "RefreshSession": {"user", "replacement"},
        "Transaction": {"predictions", "case_links"},
        "ModelVersion": {"predictions"},
        "Prediction": {"transaction", "model_version", "explanation"},
        "Explanation": {"prediction"},
        "Case": {"assigned_user", "transaction_links", "dispositions"},
        "CaseTransaction": {"case", "transaction", "prediction"},
        "Disposition": {"case"},
    }
    for mapper in Base.registry.mappers:
        if mapper.class_.__name__ in expected_relationships:
            assert set(mapper.relationships.keys()) == expected_relationships[mapper.class_.__name__]

    refresh_constraints = {constraint.name for constraint in RefreshSession.__table__.constraints}
    assert "uq_refresh_sessions_family_rotation" in refresh_constraints
    assert "ck_refresh_sessions_jti_hash_length" in refresh_constraints
    assert Base.metadata.tables["model_versions"].indexes


def test_sqlite_enforces_foreign_keys_checks_and_active_model_uniqueness() -> None:
    async def scenario() -> None:
        engine = create_database_engine(_settings())
        try:
            await create_schema_for_tests(engine)
            factory = create_session_factory(engine)
            async with engine.connect() as connection:
                assert await connection.run_sync(
                    lambda sync_connection: inspect(sync_connection).get_table_names()
                ) == sorted(EXPECTED_TABLES)

            async with factory() as session:
                session.add(
                    RefreshSession(
                        user_id=uuid4(),
                        family_id=uuid4(),
                        jti_hash="a" * 64,
                        rotation_counter=0,
                        expires_at=datetime.now(timezone.utc) + timedelta(days=1),
                    )
                )
                with pytest.raises(IntegrityError):
                    await session.flush()
                await session.rollback()

                session.add(
                    Case(
                        review_id="bad-risk",
                        status="open",
                        queue_rank=1,
                        alert_hour=datetime.now(timezone.utc),
                        risk_score=1.5,
                        study_arm="control",
                    )
                )
                with pytest.raises(IntegrityError):
                    await session.flush()
                await session.rollback()

                now = datetime.now(timezone.utc)
                common = {
                    "name": "primary",
                    "algorithm": "lightgbm",
                    "artifact_uri": "artifacts/model",
                    "artifact_sha256": "a" * 64,
                    "feature_schema": {},
                    "training_window_start": now - timedelta(days=2),
                    "training_window_end": now - timedelta(days=1),
                    "metrics": {},
                    "decision_threshold": 0.5,
                    "ensemble_weights": {},
                }
                first = await register_model_candidate(
                    session,
                    version="1",
                    **common,
                )
                second = await register_model_candidate(
                    session,
                    version="2",
                    **common,
                )
                await promote_model_version(session, model_version_id=first.id, now=now)
                await promote_model_version(session, model_version_id=second.id, now=now)
                await session.commit()

            async with factory() as session:
                active = list(
                    (
                        await session.scalars(
                            select(ModelVersion).where(
                                ModelVersion.status == ModelStatus.ACTIVE
                            )
                        )
                    ).all()
                )
                assert len(active) == 1
                assert active[0].version == "2"
        finally:
            await engine.dispose()

    asyncio.run(scenario())


def test_refresh_sessions_rotate_and_reuse_revokes_the_family() -> None:
    async def scenario() -> None:
        settings = _settings()
        engine = create_database_engine(settings)
        try:
            await create_schema_for_tests(engine)
            factory = create_session_factory(engine)
            now = datetime.now(timezone.utc)

            async with factory() as session:
                user = await create_user(
                    session,
                    email=" Analyst@Example.com ",
                    password_hash="test-password-hash",
                    role=UserRole.FRAUD_ANALYST,
                )
                original_pair = issue_token_pair(
                    user.id,
                    user.role,
                    settings=settings,
                    now=now,
                )
                original = await create_refresh_session(
                    session,
                    user_id=user.id,
                    jti=original_pair.refresh.jti,
                    expires_at=original_pair.refresh.expires_at,
                    family_id=original_pair.family_id,
                    session_id=original_pair.refresh_session_id,
                    settings=settings,
                )
                await session.commit()
                original_id = original.id
                family_id = original.family_id

            replacement_pair = issue_token_pair(
                user.id,
                user.role,
                family_id=family_id,
                settings=settings,
                now=now + timedelta(minutes=1),
            )
            async with factory() as session:
                replacement = await rotate_refresh_session(
                    session,
                    current_session_id=original_id,
                    presented_jti=original_pair.refresh.jti,
                    replacement_jti=replacement_pair.refresh.jti,
                    replacement_expires_at=replacement_pair.refresh.expires_at,
                    replacement_session_id=replacement_pair.refresh_session_id,
                    settings=settings,
                    now=now + timedelta(minutes=1),
                )
                await session.commit()
                replacement_id = replacement.id

            async with factory() as session:
                rows = list(
                    (
                        await session.scalars(
                            select(RefreshSession)
                            .where(RefreshSession.family_id == family_id)
                            .order_by(RefreshSession.rotation_counter)
                        )
                    ).all()
                )
                assert len(rows) == 2
                assert rows[0].revoked_at is not None
                assert rows[0].revoke_reason == "rotated"
                assert rows[0].replaced_by_id == replacement_id
                assert rows[1].rotation_counter == 1
                assert rows[1].revoked_at is None
                assert rows[0].jti_hash == hash_token_identifier(
                    original_pair.refresh.jti,
                    settings=settings,
                )
                assert original_pair.refresh.jti not in rows[0].jti_hash

            reuse_pair = issue_token_pair(
                user.id,
                user.role,
                family_id=family_id,
                settings=settings,
                now=now + timedelta(minutes=2),
            )
            async with factory() as session:
                with pytest.raises(RefreshTokenReuseError):
                    await rotate_refresh_session(
                        session,
                        current_session_id=original_id,
                        presented_jti=original_pair.refresh.jti,
                        replacement_jti=reuse_pair.refresh.jti,
                        replacement_expires_at=reuse_pair.refresh.expires_at,
                        replacement_session_id=reuse_pair.refresh_session_id,
                        settings=settings,
                        now=now + timedelta(minutes=2),
                    )
                # The API catches reuse inside its transaction and commits this
                # security mutation before returning an authentication error.
                await session.commit()

            async with factory() as session:
                replacement = await session.get(RefreshSession, replacement_id)
                assert replacement is not None
                assert replacement.revoked_at is not None
                assert replacement.revoke_reason == "refresh_token_reuse"
        finally:
            await engine.dispose()

    asyncio.run(scenario())


def test_initial_alembic_revision_upgrades_and_downgrades_sqlite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "migration.db"
    database_url = f"sqlite+aiosqlite:///{database_path.as_posix()}"
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.delenv("DATABASE_MIGRATION_URL", raising=False)
    get_settings.cache_clear()

    config = Config(str(Path(__file__).parents[1] / "alembic.ini"))
    command.upgrade(config, "head")
    with sqlite3.connect(database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert EXPECTED_TABLES <= tables
        assert connection.execute(
            "SELECT version_num FROM alembic_version"
        ).fetchone() == ("20261014_0002",)
        case_transaction_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(case_transactions)")
        }
        assert "prediction_id" in case_transaction_columns

    command.downgrade(config, "base")
    with sqlite3.connect(database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert not (EXPECTED_TABLES & tables)
    get_settings.cache_clear()
