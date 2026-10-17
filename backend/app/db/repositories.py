"""Small transactional repository helpers shared by API and CLI code."""

from __future__ import annotations

from datetime import datetime, timezone
import hmac
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping
from uuid import UUID, uuid4

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import Settings, get_settings
from backend.app.core.security import hash_token_identifier
from backend.app.db.enums import ModelStatus, UserRole
from backend.app.db.models import AuditEvent, ModelVersion, RefreshSession, User

if TYPE_CHECKING:
    from backend.app.ml.artifacts import VerifiedBundle


class RefreshSessionError(ValueError):
    """A presented refresh session cannot be used."""


class RefreshTokenReuseError(RefreshSessionError):
    """A revoked/replaced refresh token was presented again."""


def normalize_email(email: str) -> str:
    value = str(email).strip().casefold()
    if len(value) > 320 or value.count("@") != 1 or any(char.isspace() for char in value):
        raise ValueError("email address is invalid")
    local_part, domain = value.split("@", 1)
    if not local_part or not domain or local_part.startswith(".") or local_part.endswith("."):
        raise ValueError("email address is invalid")
    return value


async def get_user_by_email(session: AsyncSession, email: str) -> User | None:
    result = await session.execute(select(User).where(User.email == normalize_email(email)))
    return result.scalar_one_or_none()


async def get_user_by_id(session: AsyncSession, user_id: UUID) -> User | None:
    return await session.get(User, user_id)


async def create_user(
    session: AsyncSession,
    *,
    email: str,
    password_hash: str,
    role: UserRole = UserRole.FRAUD_ANALYST,
    full_name: str | None = None,
    is_active: bool = True,
) -> User:
    if not isinstance(password_hash, str) or not password_hash.strip():
        raise ValueError("password hash cannot be empty")
    cleaned_name = full_name.strip() if full_name else None
    if cleaned_name is not None and len(cleaned_name) > 200:
        raise ValueError("full name cannot exceed 200 characters")
    user = User(
        email=normalize_email(email),
        password_hash=password_hash,
        role=role,
        full_name=cleaned_name,
        is_active=is_active,
    )
    session.add(user)
    await session.flush()
    return user


async def create_refresh_session(
    session: AsyncSession,
    *,
    user_id: UUID,
    jti: str,
    expires_at: datetime,
    family_id: UUID | None = None,
    session_id: UUID | None = None,
    rotation_counter: int = 0,
    settings: Settings | None = None,
) -> RefreshSession:
    if rotation_counter < 0:
        raise ValueError("rotation counter cannot be negative")
    record = RefreshSession(
        id=session_id or uuid4(),
        user_id=user_id,
        family_id=family_id or uuid4(),
        jti_hash=hash_token_identifier(jti, settings=settings),
        expires_at=_require_aware(expires_at),
        rotation_counter=rotation_counter,
    )
    session.add(record)
    await session.flush()
    return record


async def rotate_refresh_session(
    session: AsyncSession,
    *,
    current_session_id: UUID,
    presented_jti: str,
    replacement_jti: str,
    replacement_expires_at: datetime,
    replacement_session_id: UUID | None = None,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> RefreshSession:
    """Atomically consume a refresh row and create its replacement.

    On reuse, every row in the token family is marked revoked.  Callers should
    commit that revocation before returning an authentication error.
    """
    cfg = settings or get_settings()
    moment = _require_aware(now or datetime.now(timezone.utc))
    replacement_expiration = _require_aware(replacement_expires_at)
    if replacement_expiration <= moment:
        raise ValueError("replacement refresh token must expire in the future")
    result = await session.execute(
        select(RefreshSession)
        .where(RefreshSession.id == current_session_id)
        .with_for_update()
    )
    current = result.scalar_one_or_none()
    if current is None:
        raise RefreshSessionError("refresh session does not exist")
    expected_hash = hash_token_identifier(presented_jti, settings=cfg)
    if not hmac.compare_digest(current.jti_hash, expected_hash):
        raise RefreshSessionError("refresh token does not match its session")
    if current.revoked_at is not None or current.replaced_by_id is not None:
        await revoke_refresh_family(
            session,
            family_id=current.family_id,
            reason="refresh_token_reuse",
            now=moment,
        )
        raise RefreshTokenReuseError("refresh token has already been consumed")
    if _database_utc(current.expires_at) <= moment:
        current.revoked_at = moment
        current.revoke_reason = "expired"
        await session.flush()
        raise RefreshSessionError("refresh token has expired")

    replacement = RefreshSession(
        id=replacement_session_id or uuid4(),
        user_id=current.user_id,
        family_id=current.family_id,
        jti_hash=hash_token_identifier(replacement_jti, settings=cfg),
        rotation_counter=current.rotation_counter + 1,
        expires_at=replacement_expiration,
    )
    session.add(replacement)
    await session.flush()
    current.revoked_at = moment
    current.revoke_reason = "rotated"
    current.replaced_by_id = replacement.id
    await session.flush()
    return replacement


async def revoke_refresh_family(
    session: AsyncSession,
    *,
    family_id: UUID,
    reason: str,
    now: datetime | None = None,
) -> int:
    cleaned_reason = str(reason).strip()
    if not cleaned_reason:
        raise ValueError("revocation reason cannot be empty")
    moment = _require_aware(now or datetime.now(timezone.utc))
    result = await session.execute(
        update(RefreshSession)
        .where(RefreshSession.family_id == family_id)
        .where(RefreshSession.revoked_at.is_(None))
        .values(revoked_at=moment, revoke_reason=cleaned_reason[:100])
    )
    await session.flush()
    return int(result.rowcount or 0)


async def get_active_model_version(session: AsyncSession) -> ModelVersion | None:
    result = await session.execute(
        select(ModelVersion)
        .where(ModelVersion.status == ModelStatus.ACTIVE)
        .order_by(ModelVersion.activated_at.desc(), ModelVersion.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def register_model_candidate(
    session: AsyncSession,
    *,
    name: str,
    version: str,
    algorithm: str,
    artifact_uri: str,
    artifact_sha256: str,
    feature_schema: Mapping[str, Any],
    training_window_start: datetime,
    training_window_end: datetime,
    metrics: Mapping[str, Any],
    decision_threshold: float,
    ensemble_weights: Mapping[str, Any] | None = None,
    created_by_id: UUID | None = None,
) -> ModelVersion:
    """Persist immutable metadata for a verified candidate model artifact."""
    fields = {
        "name": (name, 120),
        "version": (version, 80),
        "algorithm": (algorithm, 120),
        "artifact_uri": (artifact_uri, 1024),
    }
    cleaned: dict[str, str] = {}
    for field_name, (value, maximum) in fields.items():
        text_value = str(value).strip()
        if not text_value or len(text_value) > maximum:
            raise ValueError(
                f"{field_name} must contain between 1 and {maximum} characters"
            )
        cleaned[field_name] = text_value

    digest = str(artifact_sha256).strip().casefold()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("artifact_sha256 must be a 64-character hexadecimal digest")
    start = _require_aware(training_window_start)
    end = _require_aware(training_window_end)
    if end < start:
        raise ValueError("training window end cannot precede its start")
    threshold = float(decision_threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("decision threshold must be between zero and one")

    candidate = ModelVersion(
        **cleaned,
        artifact_sha256=digest,
        feature_schema=dict(feature_schema),
        training_window_start=start,
        training_window_end=end,
        metrics=dict(metrics),
        decision_threshold=threshold,
        ensemble_weights=dict(ensemble_weights or {"supervised": 1.0}),
        created_by_id=created_by_id,
        status=ModelStatus.CANDIDATE,
    )
    session.add(candidate)
    await session.flush()
    return candidate


async def register_verified_model_bundle(
    session: AsyncSession,
    *,
    bundle: "VerifiedBundle",
    name: str = "aml-primary",
    created_by_id: UUID | None = None,
    ensemble_weights: Mapping[str, Any] | None = None,
) -> ModelVersion:
    """Re-verify an on-disk bundle and register its candidate metadata.

    The held-out test period is deliberately excluded from the recorded
    training window.  The end is the start of the test split because model
    calibration and threshold selection use validation data.
    """
    from backend.app.ml.artifacts import MANIFEST_FILE, sha256_file, verify_model_bundle

    verified = verify_model_bundle(bundle.root)
    manifest = verified.manifest
    feature_schema = _read_verified_json(verified.path("feature_schema"))
    metrics = _read_verified_json(verified.path("metrics"))
    training = _read_verified_json(verified.path("training"))
    try:
        split_plan = training["split_plan"]
        start = _parse_artifact_timestamp(split_plan["minimum_event_time"])
        end = _parse_artifact_timestamp(split_plan["test_starts_at"])
        threshold = float(metrics["threshold_selection"]["threshold"])
        version = str(manifest["bundle_id"])
        algorithm = str(manifest["primary_model"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("verified bundle lacks model registry metadata") from exc

    return await register_model_candidate(
        session,
        name=name,
        version=version,
        algorithm=algorithm,
        artifact_uri=verified.root.as_uri(),
        artifact_sha256=sha256_file(verified.root / MANIFEST_FILE),
        feature_schema=feature_schema,
        training_window_start=start,
        training_window_end=end,
        metrics=metrics,
        decision_threshold=threshold,
        ensemble_weights=ensemble_weights,
        created_by_id=created_by_id,
    )


async def promote_model_version(
    session: AsyncSession,
    *,
    model_version_id: UUID,
    now: datetime | None = None,
) -> ModelVersion:
    moment = _require_aware(now or datetime.now(timezone.utc))
    result = await session.execute(
        select(ModelVersion).where(ModelVersion.id == model_version_id).with_for_update()
    )
    target = result.scalar_one_or_none()
    if target is None:
        raise LookupError("model version does not exist")
    await session.execute(
        update(ModelVersion)
        .where(ModelVersion.status == ModelStatus.ACTIVE)
        .where(ModelVersion.id != target.id)
        .values(status=ModelStatus.RETIRED)
    )
    target.status = ModelStatus.ACTIVE
    target.activated_at = moment
    await session.flush()
    return target


async def record_audit_event(
    session: AsyncSession,
    *,
    event_type: str,
    resource_type: str,
    actor_user_id: UUID | None = None,
    resource_id: str | UUID | None = None,
    request_id: str | None = None,
    ip_hash: str | None = None,
    safe_details: dict[str, Any] | None = None,
) -> AuditEvent:
    cleaned_event_type = event_type.strip()
    cleaned_resource_type = resource_type.strip()
    if not cleaned_event_type or len(cleaned_event_type) > 120:
        raise ValueError("audit event type must contain 1 to 120 characters")
    if not cleaned_resource_type or len(cleaned_resource_type) > 80:
        raise ValueError("audit resource type must contain 1 to 80 characters")
    event = AuditEvent(
        actor_user_id=actor_user_id,
        event_type=cleaned_event_type,
        resource_type=cleaned_resource_type,
        resource_id=str(resource_id) if resource_id is not None else None,
        request_id=request_id,
        ip_hash=ip_hash,
        safe_details=dict(safe_details or {}),
    )
    session.add(event)
    await session.flush()
    return event


def _require_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _database_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        # SQLite removes timezone information on round-trip; persisted values
        # are UTC by contract.
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _read_verified_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"verified bundle JSON is unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"verified bundle JSON must be an object: {path.name}")
    return value


def _parse_artifact_timestamp(value: object) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("bundle timestamp is invalid") from exc
    return _require_aware(parsed)


__all__ = [
    "RefreshSessionError",
    "RefreshTokenReuseError",
    "create_refresh_session",
    "create_user",
    "get_active_model_version",
    "get_user_by_email",
    "get_user_by_id",
    "normalize_email",
    "promote_model_version",
    "record_audit_event",
    "register_model_candidate",
    "register_verified_model_bundle",
    "revoke_refresh_family",
    "rotate_refresh_session",
]
