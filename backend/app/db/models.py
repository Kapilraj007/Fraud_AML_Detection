"""Operational persistence model for the AML analyst backend.

These tables deliberately contain no fraud/laundering labels, synthetic case
identifiers, or typology ground truth.  Research truth remains in the sealed
evaluation artifacts produced by the offline pipeline.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base, CreatedAtMixin, TimestampMixin, UUIDPrimaryKeyMixin
from backend.app.db.enums import (
    CaseStatus,
    DispositionAction,
    DriftStatus,
    ModelStatus,
    RetrainingStatus,
    UserRole,
)


JSON_VALUE = JSON().with_variant(JSONB(), "postgresql")


def _enum(enum_class: type, name: str) -> SAEnum:
    return SAEnum(
        enum_class,
        name=name,
        native_enum=False,
        create_constraint=True,
        validate_strings=True,
        values_callable=lambda members: [member.value for member in members],
    )


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True)
    full_name: Mapped[str | None] = mapped_column(String(200))
    password_hash: Mapped[str] = mapped_column(String(512), nullable=False)
    role: Mapped[UserRole] = mapped_column(
        _enum(UserRole, "user_role"), nullable=False, default=UserRole.FRAUD_ANALYST
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    refresh_sessions: Mapped[list["RefreshSession"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    assigned_cases: Mapped[list["Case"]] = relationship(
        back_populates="assigned_user", foreign_keys="Case.assigned_to_id"
    )

    __table_args__ = (
        CheckConstraint("email = lower(trim(email))", name="email_normalized"),
    )


class RefreshSession(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "refresh_sessions"

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    family_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False, index=True)
    jti_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    rotation_counter: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoke_reason: Mapped[str | None] = mapped_column(String(100))
    replaced_by_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("refresh_sessions.id", ondelete="SET NULL")
    )

    user: Mapped[User] = relationship(back_populates="refresh_sessions")
    replacement: Mapped["RefreshSession | None"] = relationship(
        remote_side="RefreshSession.id", foreign_keys=[replaced_by_id], post_update=True
    )

    __table_args__ = (
        CheckConstraint("rotation_counter >= 0", name="rotation_nonnegative"),
        CheckConstraint("length(jti_hash) = 64", name="jti_hash_length"),
        UniqueConstraint(
            "family_id", "rotation_counter",
            name="uq_refresh_sessions_family_rotation",
        ),
        Index("ix_refresh_sessions_family_revoked", "family_id", "revoked_at"),
    )


class Transaction(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "transactions"

    event_id: Mapped[str] = mapped_column(String(180), nullable=False, unique=True)
    dataset: Mapped[str] = mapped_column(String(64), nullable=False)
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    masked_source_account: Mapped[str | None] = mapped_column(String(160))
    masked_destination_account: Mapped[str | None] = mapped_column(String(160))
    transaction_type: Mapped[str | None] = mapped_column(String(80))
    currency: Mapped[str | None] = mapped_column(String(80))
    amount: Mapped[Decimal] = mapped_column(Numeric(28, 6), nullable=False)

    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="transaction", cascade="all, delete-orphan"
    )
    case_links: Mapped[list["CaseTransaction"]] = relationship(
        back_populates="transaction", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("amount >= 0", name="amount_nonnegative"),
        Index("ix_transactions_event_time", "event_time"),
    )


class ModelVersion(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "model_versions"

    name: Mapped[str] = mapped_column(String(120), nullable=False)
    version: Mapped[str] = mapped_column(String(80), nullable=False)
    algorithm: Mapped[str] = mapped_column(String(120), nullable=False)
    status: Mapped[ModelStatus] = mapped_column(
        _enum(ModelStatus, "model_status"), nullable=False, default=ModelStatus.CANDIDATE
    )
    artifact_uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    artifact_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    feature_schema: Mapped[dict[str, Any]] = mapped_column(
        JSON_VALUE, nullable=False, default=dict
    )
    training_window_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    training_window_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON_VALUE, nullable=False, default=dict)
    decision_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    ensemble_weights: Mapped[dict[str, Any]] = mapped_column(
        JSON_VALUE, nullable=False, default=dict
    )
    created_by_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    predictions: Mapped[list["Prediction"]] = relationship(back_populates="model_version")

    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_model_versions_name_version"),
        CheckConstraint(
            "decision_threshold >= 0 AND decision_threshold <= 1",
            name="decision_threshold_probability",
        ),
        CheckConstraint(
            "training_window_end >= training_window_start",
            name="training_window_ordered",
        ),
        Index("ix_model_versions_status", "status"),
        Index(
            "uq_model_versions_single_active",
            "status",
            unique=True,
            postgresql_where=text("status = 'active'"),
            sqlite_where=text("status = 'active'"),
        ),
    )


class Prediction(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "predictions"

    transaction_id: Mapped[UUID] = mapped_column(
        ForeignKey("transactions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_version_id: Mapped[UUID] = mapped_column(
        ForeignKey("model_versions.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    supervised_score: Mapped[float] = mapped_column(Float, nullable=False)
    anomaly_score: Mapped[float] = mapped_column(Float, nullable=False)
    combined_score: Mapped[float] = mapped_column(Float, nullable=False)
    decision_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_positive: Mapped[bool] = mapped_column(Boolean, nullable=False)
    uncertainty: Mapped[float] = mapped_column(Float, nullable=False)
    reliability_score: Mapped[float] = mapped_column(Float, nullable=False)
    reliability_status: Mapped[str] = mapped_column(String(40), nullable=False)
    explanation_gate_reason: Mapped[str | None] = mapped_column(String(240))
    drift_status: Mapped[DriftStatus] = mapped_column(
        _enum(DriftStatus, "prediction_drift_status"),
        nullable=False,
        default=DriftStatus.STABLE,
    )
    feature_schema_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    transaction: Mapped[Transaction] = relationship(back_populates="predictions")
    model_version: Mapped[ModelVersion] = relationship(back_populates="predictions")
    explanation: Mapped["Explanation | None"] = relationship(
        back_populates="prediction", uselist=False, cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint(
            "transaction_id", "model_version_id",
            name="uq_predictions_transaction_model",
        ),
        CheckConstraint(
            "supervised_score >= 0 AND supervised_score <= 1",
            name="supervised_score_probability",
        ),
        CheckConstraint(
            "anomaly_score >= 0 AND anomaly_score <= 1",
            name="anomaly_score_probability",
        ),
        CheckConstraint(
            "combined_score >= 0 AND combined_score <= 1",
            name="combined_score_probability",
        ),
        CheckConstraint(
            "decision_threshold >= 0 AND decision_threshold <= 1",
            name="prediction_threshold_probability",
        ),
        CheckConstraint(
            "uncertainty >= 0 AND uncertainty <= 1",
            name="uncertainty_probability",
        ),
        CheckConstraint(
            "reliability_score >= 0 AND reliability_score <= 1",
            name="reliability_probability",
        ),
    )


class Explanation(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "explanations"

    prediction_id: Mapped[UUID] = mapped_column(
        ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    top_features: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON_VALUE, nullable=False, default=list
    )
    narrative: Mapped[str | None] = mapped_column(Text)
    is_withheld: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    gate_reason: Mapped[str | None] = mapped_column(String(240))

    prediction: Mapped[Prediction] = relationship(back_populates="explanation")


class Case(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "cases"

    review_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    status: Mapped[CaseStatus] = mapped_column(
        _enum(CaseStatus, "case_status"), nullable=False, default=CaseStatus.OPEN
    )
    queue_rank: Mapped[int] = mapped_column(Integer, nullable=False)
    alert_hour: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    study_arm: Mapped[str] = mapped_column(String(80), nullable=False)
    assigned_to_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    assigned_user: Mapped[User | None] = relationship(
        back_populates="assigned_cases", foreign_keys=[assigned_to_id]
    )
    transaction_links: Mapped[list["CaseTransaction"]] = relationship(
        back_populates="case", cascade="all, delete-orphan"
    )
    dispositions: Mapped[list["Disposition"]] = relationship(
        back_populates="case", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("queue_rank >= 1", name="queue_rank_positive"),
        CheckConstraint("risk_score >= 0 AND risk_score <= 1", name="risk_score_probability"),
        Index("ix_cases_queue", "status", "alert_hour", "queue_rank"),
    )


class CaseTransaction(CreatedAtMixin, Base):
    __tablename__ = "case_transactions"

    case_id: Mapped[UUID] = mapped_column(
        ForeignKey("cases.id", ondelete="CASCADE"), primary_key=True
    )
    transaction_id: Mapped[UUID] = mapped_column(
        ForeignKey("transactions.id", ondelete="CASCADE"), primary_key=True
    )
    prediction_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("predictions.id", ondelete="RESTRICT"), nullable=True, index=True
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    case: Mapped[Case] = relationship(back_populates="transaction_links")
    transaction: Mapped[Transaction] = relationship(back_populates="case_links")
    prediction: Mapped[Prediction | None] = relationship(foreign_keys=[prediction_id])

    __table_args__ = (
        CheckConstraint("position >= 0", name="position_nonnegative"),
    )


class DriftSnapshot(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "drift_snapshots"

    model_version_id: Mapped[UUID] = mapped_column(
        ForeignKey("model_versions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[DriftStatus] = mapped_column(
        _enum(DriftStatus, "drift_status"), nullable=False
    )
    feature_metrics: Mapped[dict[str, Any]] = mapped_column(
        JSON_VALUE, nullable=False, default=dict
    )
    prediction_metrics: Mapped[dict[str, Any]] = mapped_column(
        JSON_VALUE, nullable=False, default=dict
    )
    adwin_detected: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (
        CheckConstraint("window_end >= window_start", name="drift_window_ordered"),
        Index("ix_drift_snapshots_model_window", "model_version_id", "window_end"),
    )


class Disposition(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "dispositions"

    case_id: Mapped[UUID] = mapped_column(
        ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    analyst_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    action: Mapped[DispositionAction] = mapped_column(
        _enum(DispositionAction, "disposition_action"), nullable=False
    )
    confidence: Mapped[float | None] = mapped_column(Float)
    notes: Mapped[str | None] = mapped_column(Text)
    model_agreement: Mapped[bool | None] = mapped_column(Boolean)
    is_override: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    decision_seconds: Mapped[int | None] = mapped_column(Integer)
    drift_snapshot_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("drift_snapshots.id", ondelete="SET NULL")
    )

    case: Mapped[Case] = relationship(back_populates="dispositions")

    __table_args__ = (
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="confidence_probability",
        ),
        CheckConstraint(
            "decision_seconds IS NULL OR decision_seconds >= 0",
            name="decision_seconds_nonnegative",
        ),
    )


class RetrainingRequest(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "retraining_requests"

    drift_snapshot_id: Mapped[UUID] = mapped_column(
        ForeignKey("drift_snapshots.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    requested_by_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    status: Mapped[RetrainingStatus] = mapped_column(
        _enum(RetrainingStatus, "retraining_status"),
        nullable=False,
        default=RetrainingStatus.OPEN,
    )
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    before_metrics: Mapped[dict[str, Any]] = mapped_column(
        JSON_VALUE, nullable=False, default=dict
    )
    after_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON_VALUE)


class AuditEvent(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "audit_events"

    actor_user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    event_type: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    resource_type: Mapped[str] = mapped_column(String(80), nullable=False)
    resource_id: Mapped[str | None] = mapped_column(String(180))
    request_id: Mapped[str | None] = mapped_column(String(120), index=True)
    ip_hash: Mapped[str | None] = mapped_column(String(64))
    safe_details: Mapped[dict[str, Any]] = mapped_column(
        JSON_VALUE, nullable=False, default=dict
    )

    __table_args__ = (
        Index("ix_audit_events_resource", "resource_type", "resource_id"),
    )


__all__ = [
    "AuditEvent",
    "Case",
    "CaseTransaction",
    "Disposition",
    "DriftSnapshot",
    "Explanation",
    "ModelVersion",
    "Prediction",
    "RefreshSession",
    "RetrainingRequest",
    "Transaction",
    "User",
]
