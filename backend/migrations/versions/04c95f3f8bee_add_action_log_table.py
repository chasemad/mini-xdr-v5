"""add_action_log_table

Revision ID: 04c95f3f8bee
Revises: c65b5eaef6b2
Create Date: 2025-10-05 23:11:00.619029

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '04c95f3f8bee'
down_revision: Union[str, None] = 'c65b5eaef6b2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create action_logs table
    op.create_table(
        'action_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('action_id', sa.String(), nullable=False),
        sa.Column('agent_id', sa.String(), nullable=False),
        sa.Column('agent_type', sa.String(), nullable=True),
        sa.Column('action_name', sa.String(), nullable=False),
        sa.Column('incident_id', sa.Integer(), nullable=True),
        sa.Column('params', sa.JSON(), nullable=False),
        sa.Column('result', sa.JSON(), nullable=True),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('rollback_id', sa.String(), nullable=True),
        sa.Column('rollback_data', sa.JSON(), nullable=True),
        sa.Column('rollback_executed', sa.Boolean(), nullable=True, default=False),
        sa.Column('rollback_timestamp', sa.DateTime(timezone=True), nullable=True),
        sa.Column('rollback_result', sa.JSON(), nullable=True),
        sa.Column('executed_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.ForeignKeyConstraint(['incident_id'], ['incidents.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes
    op.create_index('ix_action_logs_id', 'action_logs', ['id'])
    op.create_index('ix_action_logs_action_id', 'action_logs', ['action_id'], unique=True)
    op.create_index('ix_action_logs_agent_id', 'action_logs', ['agent_id'])
    op.create_index('ix_action_logs_agent_type', 'action_logs', ['agent_type'])
    op.create_index('ix_action_logs_action_name', 'action_logs', ['action_name'])
    op.create_index('ix_action_logs_incident_id', 'action_logs', ['incident_id'])
    op.create_index('ix_action_logs_rollback_id', 'action_logs', ['rollback_id'], unique=True)
    op.create_index('ix_action_logs_executed_at', 'action_logs', ['executed_at'])


def downgrade() -> None:
    # Drop indexes
    op.drop_index('ix_action_logs_executed_at', 'action_logs')
    op.drop_index('ix_action_logs_rollback_id', 'action_logs')
    op.drop_index('ix_action_logs_incident_id', 'action_logs')
    op.drop_index('ix_action_logs_action_name', 'action_logs')
    op.drop_index('ix_action_logs_agent_type', 'action_logs')
    op.drop_index('ix_action_logs_agent_id', 'action_logs')
    op.drop_index('ix_action_logs_action_id', 'action_logs')
    op.drop_index('ix_action_logs_id', 'action_logs')
    
    # Drop table
    op.drop_table('action_logs')
