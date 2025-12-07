"""add_investigation_results_table

Revision ID: dcb3af6e7502
Revises: 191220d80f32
Create Date: 2025-11-29 15:39:45.889188

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "dcb3af6e7502"
down_revision: Union[str, None] = "191220d80f32"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create investigation_results table
    op.create_table(
        "investigation_results",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column("investigation_id", sa.String(length=64), nullable=True),
        sa.Column("incident_id", sa.Integer(), nullable=True),
        sa.Column("tool_name", sa.String(length=128), nullable=True),
        sa.Column("tool_category", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("execution_time_ms", sa.Integer(), nullable=True),
        sa.Column("parameters", sa.JSON(), nullable=True),
        sa.Column("results", sa.JSON(), nullable=True),
        sa.Column("findings_count", sa.Integer(), nullable=True),
        sa.Column("iocs_discovered", sa.JSON(), nullable=True),
        sa.Column("severity", sa.String(length=16), nullable=True),
        sa.Column("confidence_score", sa.Float(), nullable=True),
        sa.Column("triggered_by", sa.String(length=128), nullable=True),
        sa.Column("triggered_by_user_id", sa.Integer(), nullable=True),
        sa.Column("auto_triggered", sa.Boolean(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=True),
        sa.Column("exported", sa.Boolean(), nullable=True),
        sa.Column("export_formats", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(
            ["incident_id"],
            ["incidents.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for performance
    op.create_index(
        op.f("ix_investigation_results_created_at"),
        "investigation_results",
        ["created_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_investigation_results_incident_id"),
        "investigation_results",
        ["incident_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_investigation_results_investigation_id"),
        "investigation_results",
        ["investigation_id"],
        unique=True,
    )
    op.create_index(
        op.f("ix_investigation_results_status"),
        "investigation_results",
        ["status"],
        unique=False,
    )
    op.create_index(
        op.f("ix_investigation_results_tool_name"),
        "investigation_results",
        ["tool_name"],
        unique=False,
    )


def downgrade() -> None:
    # Drop indexes first
    op.drop_index(
        op.f("ix_investigation_results_tool_name"), table_name="investigation_results"
    )
    op.drop_index(
        op.f("ix_investigation_results_status"), table_name="investigation_results"
    )
    op.drop_index(
        op.f("ix_investigation_results_investigation_id"),
        table_name="investigation_results",
    )
    op.drop_index(
        op.f("ix_investigation_results_incident_id"), table_name="investigation_results"
    )
    op.drop_index(
        op.f("ix_investigation_results_created_at"), table_name="investigation_results"
    )

    # Drop table
    op.drop_table("investigation_results")
