"""add_event_protocol_and_src_port_fields

Revision ID: d9ea5c5841db
Revises: 5eafdf6dbbc1
Create Date: 2025-11-21 15:18:48.622803

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d9ea5c5841db"
down_revision: Union[str, None] = "5eafdf6dbbc1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add src_port and protocol fields to events table
    op.add_column("events", sa.Column("src_port", sa.Integer(), nullable=True))
    op.add_column("events", sa.Column("protocol", sa.String(length=32), nullable=True))


def downgrade() -> None:
    # Remove src_port and protocol fields from events table
    op.drop_column("events", "protocol")
    op.drop_column("events", "src_port")
