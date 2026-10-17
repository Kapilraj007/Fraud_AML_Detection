"""Create the Sprint 2 operational backend schema.

Revision ID: 20261013_0001
Revises: None
Create Date: 2026-10-13
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20261013_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _json_type() -> sa.JSON:
    return sa.JSON().with_variant(
        postgresql.JSONB(astext_type=sa.Text()),
        "postgresql",
    )


def _enum(*values: str, name: str) -> sa.Enum:
    return sa.Enum(
        *values,
        name=name,
        native_enum=False,
        create_constraint=True,
    )


def upgrade() -> None:
    op.create_table(
        "transactions",
        sa.Column("event_id", sa.String(length=180), nullable=False),
        sa.Column("dataset", sa.String(length=64), nullable=False),
        sa.Column("event_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("masked_source_account", sa.String(length=160), nullable=True),
        sa.Column("masked_destination_account", sa.String(length=160), nullable=True),
        sa.Column("transaction_type", sa.String(length=80), nullable=True),
        sa.Column("currency", sa.String(length=80), nullable=True),
        sa.Column("amount", sa.Numeric(precision=28, scale=6), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "amount >= 0",
            name=op.f("ck_transactions_amount_nonnegative"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_transactions")),
        sa.UniqueConstraint("event_id", name=op.f("uq_transactions_event_id")),
    )
    op.create_index(
        "ix_transactions_event_time",
        "transactions",
        ["event_time"],
        unique=False,
    )

    op.create_table(
        "users",
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("full_name", sa.String(length=200), nullable=True),
        sa.Column("password_hash", sa.String(length=512), nullable=False),
        sa.Column(
            "role",
            _enum(
                "fraud_analyst",
                "senior_investigator",
                "admin",
                name="user_role",
            ),
            nullable=False,
        ),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "email = lower(trim(email))",
            name=op.f("ck_users_email_normalized"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
        sa.UniqueConstraint("email", name=op.f("uq_users_email")),
    )

    op.create_table(
        "audit_events",
        sa.Column("actor_user_id", sa.Uuid(), nullable=True),
        sa.Column("event_type", sa.String(length=120), nullable=False),
        sa.Column("resource_type", sa.String(length=80), nullable=False),
        sa.Column("resource_id", sa.String(length=180), nullable=True),
        sa.Column("request_id", sa.String(length=120), nullable=True),
        sa.Column("ip_hash", sa.String(length=64), nullable=True),
        sa.Column("safe_details", _json_type(), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["actor_user_id"],
            ["users.id"],
            name=op.f("fk_audit_events_actor_user_id_users"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_audit_events")),
    )
    op.create_index(
        op.f("ix_audit_events_actor_user_id"),
        "audit_events",
        ["actor_user_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_audit_events_event_type"),
        "audit_events",
        ["event_type"],
        unique=False,
    )
    op.create_index(
        op.f("ix_audit_events_request_id"),
        "audit_events",
        ["request_id"],
        unique=False,
    )
    op.create_index(
        "ix_audit_events_resource",
        "audit_events",
        ["resource_type", "resource_id"],
        unique=False,
    )

    op.create_table(
        "cases",
        sa.Column("review_id", sa.String(length=64), nullable=False),
        sa.Column(
            "status",
            _enum(
                "open",
                "claimed",
                "in_review",
                "escalated",
                "dismissed",
                "more_info_requested",
                "closed",
                name="case_status",
            ),
            nullable=False,
        ),
        sa.Column("queue_rank", sa.Integer(), nullable=False),
        sa.Column("alert_hour", sa.DateTime(timezone=True), nullable=False),
        sa.Column("risk_score", sa.Float(), nullable=False),
        sa.Column("study_arm", sa.String(length=80), nullable=False),
        sa.Column("assigned_to_id", sa.Uuid(), nullable=True),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "queue_rank >= 1",
            name=op.f("ck_cases_queue_rank_positive"),
        ),
        sa.CheckConstraint(
            "risk_score >= 0 AND risk_score <= 1",
            name=op.f("ck_cases_risk_score_probability"),
        ),
        sa.ForeignKeyConstraint(
            ["assigned_to_id"],
            ["users.id"],
            name=op.f("fk_cases_assigned_to_id_users"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_cases")),
        sa.UniqueConstraint("review_id", name=op.f("uq_cases_review_id")),
    )
    op.create_index(
        op.f("ix_cases_assigned_to_id"),
        "cases",
        ["assigned_to_id"],
        unique=False,
    )
    op.create_index(
        "ix_cases_queue",
        "cases",
        ["status", "alert_hour", "queue_rank"],
        unique=False,
    )

    op.create_table(
        "model_versions",
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("version", sa.String(length=80), nullable=False),
        sa.Column("algorithm", sa.String(length=120), nullable=False),
        sa.Column(
            "status",
            _enum("candidate", "active", "retired", "failed", name="model_status"),
            nullable=False,
        ),
        sa.Column("artifact_uri", sa.String(length=1024), nullable=False),
        sa.Column("artifact_sha256", sa.String(length=64), nullable=False),
        sa.Column("feature_schema", _json_type(), nullable=False),
        sa.Column("training_window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("training_window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metrics", _json_type(), nullable=False),
        sa.Column("decision_threshold", sa.Float(), nullable=False),
        sa.Column("ensemble_weights", _json_type(), nullable=False),
        sa.Column("created_by_id", sa.Uuid(), nullable=True),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "decision_threshold >= 0 AND decision_threshold <= 1",
            name=op.f("ck_model_versions_decision_threshold_probability"),
        ),
        sa.CheckConstraint(
            "training_window_end >= training_window_start",
            name=op.f("ck_model_versions_training_window_ordered"),
        ),
        sa.ForeignKeyConstraint(
            ["created_by_id"],
            ["users.id"],
            name=op.f("fk_model_versions_created_by_id_users"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_model_versions")),
        sa.UniqueConstraint(
            "name",
            "version",
            name="uq_model_versions_name_version",
        ),
    )
    op.create_index(
        "ix_model_versions_status",
        "model_versions",
        ["status"],
        unique=False,
    )
    op.create_index(
        "uq_model_versions_single_active",
        "model_versions",
        ["status"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
        sqlite_where=sa.text("status = 'active'"),
    )

    op.create_table(
        "refresh_sessions",
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("family_id", sa.Uuid(), nullable=False),
        sa.Column("jti_hash", sa.String(length=64), nullable=False),
        sa.Column("rotation_counter", sa.Integer(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoke_reason", sa.String(length=100), nullable=True),
        sa.Column("replaced_by_id", sa.Uuid(), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "length(jti_hash) = 64",
            name=op.f("ck_refresh_sessions_jti_hash_length"),
        ),
        sa.CheckConstraint(
            "rotation_counter >= 0",
            name=op.f("ck_refresh_sessions_rotation_nonnegative"),
        ),
        sa.ForeignKeyConstraint(
            ["replaced_by_id"],
            ["refresh_sessions.id"],
            name=op.f("fk_refresh_sessions_replaced_by_id_refresh_sessions"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_refresh_sessions_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_refresh_sessions")),
        sa.UniqueConstraint("jti_hash", name=op.f("uq_refresh_sessions_jti_hash")),
        sa.UniqueConstraint(
            "family_id",
            "rotation_counter",
            name="uq_refresh_sessions_family_rotation",
        ),
    )
    op.create_index(
        op.f("ix_refresh_sessions_family_id"),
        "refresh_sessions",
        ["family_id"],
        unique=False,
    )
    op.create_index(
        "ix_refresh_sessions_family_revoked",
        "refresh_sessions",
        ["family_id", "revoked_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_refresh_sessions_user_id"),
        "refresh_sessions",
        ["user_id"],
        unique=False,
    )

    op.create_table(
        "case_transactions",
        sa.Column("case_id", sa.Uuid(), nullable=False),
        sa.Column("transaction_id", sa.Uuid(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "position >= 0",
            name=op.f("ck_case_transactions_position_nonnegative"),
        ),
        sa.ForeignKeyConstraint(
            ["case_id"],
            ["cases.id"],
            name=op.f("fk_case_transactions_case_id_cases"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["transaction_id"],
            ["transactions.id"],
            name=op.f("fk_case_transactions_transaction_id_transactions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "case_id",
            "transaction_id",
            name=op.f("pk_case_transactions"),
        ),
    )

    op.create_table(
        "drift_snapshots",
        sa.Column("model_version_id", sa.Uuid(), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "status",
            _enum("stable", "watch", "drift_detected", name="drift_status"),
            nullable=False,
        ),
        sa.Column("feature_metrics", _json_type(), nullable=False),
        sa.Column("prediction_metrics", _json_type(), nullable=False),
        sa.Column("adwin_detected", sa.Boolean(), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "window_end >= window_start",
            name=op.f("ck_drift_snapshots_drift_window_ordered"),
        ),
        sa.ForeignKeyConstraint(
            ["model_version_id"],
            ["model_versions.id"],
            name=op.f("fk_drift_snapshots_model_version_id_model_versions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_drift_snapshots")),
    )
    op.create_index(
        op.f("ix_drift_snapshots_model_version_id"),
        "drift_snapshots",
        ["model_version_id"],
        unique=False,
    )
    op.create_index(
        "ix_drift_snapshots_model_window",
        "drift_snapshots",
        ["model_version_id", "window_end"],
        unique=False,
    )

    op.create_table(
        "predictions",
        sa.Column("transaction_id", sa.Uuid(), nullable=False),
        sa.Column("model_version_id", sa.Uuid(), nullable=False),
        sa.Column("supervised_score", sa.Float(), nullable=False),
        sa.Column("anomaly_score", sa.Float(), nullable=False),
        sa.Column("combined_score", sa.Float(), nullable=False),
        sa.Column("decision_threshold", sa.Float(), nullable=False),
        sa.Column("predicted_positive", sa.Boolean(), nullable=False),
        sa.Column("uncertainty", sa.Float(), nullable=False),
        sa.Column("reliability_score", sa.Float(), nullable=False),
        sa.Column("reliability_status", sa.String(length=40), nullable=False),
        sa.Column("explanation_gate_reason", sa.String(length=240), nullable=True),
        sa.Column(
            "drift_status",
            _enum(
                "stable",
                "watch",
                "drift_detected",
                name="prediction_drift_status",
            ),
            nullable=False,
        ),
        sa.Column("feature_schema_hash", sa.String(length=64), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "anomaly_score >= 0 AND anomaly_score <= 1",
            name=op.f("ck_predictions_anomaly_score_probability"),
        ),
        sa.CheckConstraint(
            "combined_score >= 0 AND combined_score <= 1",
            name=op.f("ck_predictions_combined_score_probability"),
        ),
        sa.CheckConstraint(
            "decision_threshold >= 0 AND decision_threshold <= 1",
            name=op.f("ck_predictions_prediction_threshold_probability"),
        ),
        sa.CheckConstraint(
            "reliability_score >= 0 AND reliability_score <= 1",
            name=op.f("ck_predictions_reliability_probability"),
        ),
        sa.CheckConstraint(
            "supervised_score >= 0 AND supervised_score <= 1",
            name=op.f("ck_predictions_supervised_score_probability"),
        ),
        sa.CheckConstraint(
            "uncertainty >= 0 AND uncertainty <= 1",
            name=op.f("ck_predictions_uncertainty_probability"),
        ),
        sa.ForeignKeyConstraint(
            ["model_version_id"],
            ["model_versions.id"],
            name=op.f("fk_predictions_model_version_id_model_versions"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["transaction_id"],
            ["transactions.id"],
            name=op.f("fk_predictions_transaction_id_transactions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_predictions")),
        sa.UniqueConstraint(
            "transaction_id",
            "model_version_id",
            name="uq_predictions_transaction_model",
        ),
    )
    op.create_index(
        op.f("ix_predictions_model_version_id"),
        "predictions",
        ["model_version_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_predictions_transaction_id"),
        "predictions",
        ["transaction_id"],
        unique=False,
    )

    op.create_table(
        "dispositions",
        sa.Column("case_id", sa.Uuid(), nullable=False),
        sa.Column("analyst_id", sa.Uuid(), nullable=False),
        sa.Column(
            "action",
            _enum(
                "escalate",
                "dismiss",
                "request_more_info",
                name="disposition_action",
            ),
            nullable=False,
        ),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("model_agreement", sa.Boolean(), nullable=True),
        sa.Column("is_override", sa.Boolean(), nullable=False),
        sa.Column("decision_seconds", sa.Integer(), nullable=True),
        sa.Column("drift_snapshot_id", sa.Uuid(), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name=op.f("ck_dispositions_confidence_probability"),
        ),
        sa.CheckConstraint(
            "decision_seconds IS NULL OR decision_seconds >= 0",
            name=op.f("ck_dispositions_decision_seconds_nonnegative"),
        ),
        sa.ForeignKeyConstraint(
            ["analyst_id"],
            ["users.id"],
            name=op.f("fk_dispositions_analyst_id_users"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["case_id"],
            ["cases.id"],
            name=op.f("fk_dispositions_case_id_cases"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["drift_snapshot_id"],
            ["drift_snapshots.id"],
            name=op.f("fk_dispositions_drift_snapshot_id_drift_snapshots"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_dispositions")),
    )
    op.create_index(
        op.f("ix_dispositions_analyst_id"),
        "dispositions",
        ["analyst_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_dispositions_case_id"),
        "dispositions",
        ["case_id"],
        unique=False,
    )

    op.create_table(
        "explanations",
        sa.Column("prediction_id", sa.Uuid(), nullable=False),
        sa.Column("top_features", _json_type(), nullable=False),
        sa.Column("narrative", sa.Text(), nullable=True),
        sa.Column("is_withheld", sa.Boolean(), nullable=False),
        sa.Column("gate_reason", sa.String(length=240), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["prediction_id"],
            ["predictions.id"],
            name=op.f("fk_explanations_prediction_id_predictions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_explanations")),
        sa.UniqueConstraint(
            "prediction_id",
            name=op.f("uq_explanations_prediction_id"),
        ),
    )

    op.create_table(
        "retraining_requests",
        sa.Column("drift_snapshot_id", sa.Uuid(), nullable=False),
        sa.Column("requested_by_id", sa.Uuid(), nullable=False),
        sa.Column(
            "status",
            _enum(
                "open",
                "approved",
                "running",
                "completed",
                "rejected",
                "failed",
                name="retraining_status",
            ),
            nullable=False,
        ),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("before_metrics", _json_type(), nullable=False),
        sa.Column("after_metrics", _json_type(), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["drift_snapshot_id"],
            ["drift_snapshots.id"],
            name=op.f(
                "fk_retraining_requests_drift_snapshot_id_drift_snapshots"
            ),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["requested_by_id"],
            ["users.id"],
            name=op.f("fk_retraining_requests_requested_by_id_users"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_retraining_requests")),
    )
    op.create_index(
        op.f("ix_retraining_requests_drift_snapshot_id"),
        "retraining_requests",
        ["drift_snapshot_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_retraining_requests_drift_snapshot_id"),
        table_name="retraining_requests",
    )
    op.drop_table("retraining_requests")
    op.drop_table("explanations")
    op.drop_index(op.f("ix_dispositions_case_id"), table_name="dispositions")
    op.drop_index(op.f("ix_dispositions_analyst_id"), table_name="dispositions")
    op.drop_table("dispositions")
    op.drop_index(op.f("ix_predictions_transaction_id"), table_name="predictions")
    op.drop_index(op.f("ix_predictions_model_version_id"), table_name="predictions")
    op.drop_table("predictions")
    op.drop_index("ix_drift_snapshots_model_window", table_name="drift_snapshots")
    op.drop_index(
        op.f("ix_drift_snapshots_model_version_id"),
        table_name="drift_snapshots",
    )
    op.drop_table("drift_snapshots")
    op.drop_table("case_transactions")
    op.drop_index(op.f("ix_refresh_sessions_user_id"), table_name="refresh_sessions")
    op.drop_index("ix_refresh_sessions_family_revoked", table_name="refresh_sessions")
    op.drop_index(op.f("ix_refresh_sessions_family_id"), table_name="refresh_sessions")
    op.drop_table("refresh_sessions")
    op.drop_index("uq_model_versions_single_active", table_name="model_versions")
    op.drop_index("ix_model_versions_status", table_name="model_versions")
    op.drop_table("model_versions")
    op.drop_index("ix_cases_queue", table_name="cases")
    op.drop_index(op.f("ix_cases_assigned_to_id"), table_name="cases")
    op.drop_table("cases")
    op.drop_index("ix_audit_events_resource", table_name="audit_events")
    op.drop_index(op.f("ix_audit_events_request_id"), table_name="audit_events")
    op.drop_index(op.f("ix_audit_events_event_type"), table_name="audit_events")
    op.drop_index(op.f("ix_audit_events_actor_user_id"), table_name="audit_events")
    op.drop_table("audit_events")
    op.drop_table("users")
    op.drop_index("ix_transactions_event_time", table_name="transactions")
    op.drop_table("transactions")
