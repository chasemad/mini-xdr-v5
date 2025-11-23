"""add_council_fields_to_incidents

Revision ID: d97cc188fa45
Revises: 99d70952c5da
Create Date: 2025-11-20 23:36:56.696735

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d97cc188fa45"
down_revision: Union[str, None] = "99d70952c5da"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add ML confidence field (should have been in earlier migration)
    op.add_column("incidents", sa.Column("ml_confidence", sa.Float(), nullable=True))

    # Add Council of Models fields
    op.add_column(
        "incidents", sa.Column("council_verdict", sa.String(length=50), nullable=True)
    )
    op.add_column("incidents", sa.Column("council_reasoning", sa.Text(), nullable=True))
    op.add_column(
        "incidents", sa.Column("council_confidence", sa.Float(), nullable=True)
    )
    op.add_column("incidents", sa.Column("routing_path", sa.JSON(), nullable=True))
    op.add_column("incidents", sa.Column("api_calls_made", sa.JSON(), nullable=True))
    op.add_column(
        "incidents", sa.Column("processing_time_ms", sa.Float(), nullable=True)
    )
    op.add_column("incidents", sa.Column("gemini_analysis", sa.Text(), nullable=True))
    op.add_column("incidents", sa.Column("grok_intel", sa.Text(), nullable=True))
    op.add_column(
        "incidents", sa.Column("openai_remediation", sa.Text(), nullable=True)
    )


def downgrade() -> None:
    # Remove Council of Models fields
    op.drop_column("incidents", "openai_remediation")
    op.drop_column("incidents", "grok_intel")
    op.drop_column("incidents", "gemini_analysis")
    op.drop_column("incidents", "processing_time_ms")
    op.drop_column("incidents", "api_calls_made")
    op.drop_column("incidents", "routing_path")
    op.drop_column("incidents", "council_confidence")
    op.drop_column("incidents", "council_reasoning")
    op.drop_column("incidents", "council_verdict")

    # Remove ML confidence field
    op.drop_column("incidents", "ml_confidence")
