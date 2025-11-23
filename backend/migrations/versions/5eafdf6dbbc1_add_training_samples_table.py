"""add_training_samples_table

Revision ID: 5eafdf6dbbc1
Revises: d97cc188fa45
Create Date: 2025-11-21 00:16:07.665953

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "5eafdf6dbbc1"
down_revision: Union[str, None] = "d97cc188fa45"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create training_samples table
    op.create_table(
        "training_samples",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column("incident_id", sa.Integer(), nullable=True),
        sa.Column("ml_prediction", sa.String(length=100), nullable=False),
        sa.Column("ml_confidence", sa.Float(), nullable=False),
        sa.Column("council_verdict", sa.String(length=50), nullable=False),
        sa.Column("correct_label", sa.String(length=100), nullable=False),
        sa.Column("was_override", sa.Boolean(), nullable=True),
        sa.Column("features_stored", sa.Boolean(), nullable=True),
        sa.Column("features_path", sa.String(length=512), nullable=True),
        sa.Column("used_for_training", sa.Boolean(), nullable=True),
        sa.Column("training_run_id", sa.String(length=100), nullable=True),
        sa.ForeignKeyConstraint(["incident_id"], ["incidents.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes
    op.create_index(
        op.f("ix_training_samples_id"), "training_samples", ["id"], unique=False
    )
    op.create_index(
        op.f("ix_training_samples_created_at"),
        "training_samples",
        ["created_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_training_samples_incident_id"),
        "training_samples",
        ["incident_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_training_samples_council_verdict"),
        "training_samples",
        ["council_verdict"],
        unique=False,
    )
    op.create_index(
        op.f("ix_training_samples_correct_label"),
        "training_samples",
        ["correct_label"],
        unique=False,
    )
    op.create_index(
        op.f("ix_training_samples_was_override"),
        "training_samples",
        ["was_override"],
        unique=False,
    )
    op.create_index(
        op.f("ix_training_samples_used_for_training"),
        "training_samples",
        ["used_for_training"],
        unique=False,
    )
    op.create_index(
        "ix_training_samples_unused",
        "training_samples",
        ["used_for_training", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    # Drop indexes
    op.drop_index("ix_training_samples_unused", table_name="training_samples")
    op.drop_index(
        op.f("ix_training_samples_used_for_training"), table_name="training_samples"
    )
    op.drop_index(
        op.f("ix_training_samples_was_override"), table_name="training_samples"
    )
    op.drop_index(
        op.f("ix_training_samples_correct_label"), table_name="training_samples"
    )
    op.drop_index(
        op.f("ix_training_samples_council_verdict"), table_name="training_samples"
    )
    op.drop_index(
        op.f("ix_training_samples_incident_id"), table_name="training_samples"
    )
    op.drop_index(op.f("ix_training_samples_created_at"), table_name="training_samples")
    op.drop_index(op.f("ix_training_samples_id"), table_name="training_samples")

    # Drop table
    op.drop_table("training_samples")
