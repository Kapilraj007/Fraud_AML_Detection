"""Pin each case transaction to its materialization-time prediction.

Revision ID: 20261014_0002
Revises: 20261013_0001
Create Date: 2026-10-14
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20261014_0002"
down_revision: str | None = "20261013_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "case_transactions",
        sa.Column("prediction_id", sa.Uuid(), nullable=True),
    )

    # Best-effort backfill for queues created before the explicit link existed.
    # Prefer a score-identical prediction that existed when the case was created;
    # otherwise retain the latest prediction available at that point in time.
    links = sa.table(
        "case_transactions",
        sa.column("case_id", sa.Uuid()),
        sa.column("transaction_id", sa.Uuid()),
        sa.column("prediction_id", sa.Uuid()),
    )
    cases = sa.table(
        "cases",
        sa.column("id", sa.Uuid()),
        sa.column("risk_score", sa.Float()),
        sa.column("created_at", sa.DateTime(timezone=True)),
    )
    predictions = sa.table(
        "predictions",
        sa.column("id", sa.Uuid()),
        sa.column("transaction_id", sa.Uuid()),
        sa.column("combined_score", sa.Float()),
        sa.column("created_at", sa.DateTime(timezone=True)),
    )
    originating_prediction = (
        sa.select(predictions.c.id)
        .select_from(predictions, cases)
        .where(
            cases.c.id == links.c.case_id,
            predictions.c.transaction_id == links.c.transaction_id,
            predictions.c.created_at <= cases.c.created_at,
        )
        .order_by(
            sa.case(
                (predictions.c.combined_score == cases.c.risk_score, 0),
                else_=1,
            ),
            predictions.c.created_at.desc(),
            predictions.c.id.asc(),
        )
        .limit(1)
        .correlate(links)
        .scalar_subquery()
    )
    op.execute(links.update().values(prediction_id=originating_prediction))

    op.create_index(
        "ix_case_transactions_prediction_id",
        "case_transactions",
        ["prediction_id"],
        unique=False,
    )
    with op.batch_alter_table("case_transactions") as batch_op:
        batch_op.create_foreign_key(
            "fk_case_transactions_prediction_id_predictions",
            "predictions",
            ["prediction_id"],
            ["id"],
            ondelete="RESTRICT",
        )


def downgrade() -> None:
    with op.batch_alter_table("case_transactions") as batch_op:
        batch_op.drop_constraint(
            "fk_case_transactions_prediction_id_predictions",
            type_="foreignkey",
        )
    op.drop_index(
        "ix_case_transactions_prediction_id",
        table_name="case_transactions",
    )
    op.drop_column("case_transactions", "prediction_id")
