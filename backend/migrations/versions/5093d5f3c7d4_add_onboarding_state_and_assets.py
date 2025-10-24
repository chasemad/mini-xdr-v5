"""add_onboarding_state_and_assets

Revision ID: 5093d5f3c7d4
Revises: 8976084bce10
Create Date: 2025-10-09 21:47:08.988248

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5093d5f3c7d4'
down_revision: Union[str, None] = '8976084bce10'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create new discovered_assets table
    op.create_table('discovered_assets',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('organization_id', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
    sa.Column('ip', sa.String(length=64), nullable=False),
    sa.Column('hostname', sa.String(length=255), nullable=True),
    sa.Column('mac_address', sa.String(length=17), nullable=True),
    sa.Column('os_type', sa.String(length=64), nullable=True),
    sa.Column('os_role', sa.String(length=128), nullable=True),
    sa.Column('classification', sa.String(length=64), nullable=True),
    sa.Column('classification_confidence', sa.Float(), nullable=True),
    sa.Column('open_ports', sa.JSON(), nullable=True),
    sa.Column('services', sa.JSON(), nullable=True),
    sa.Column('deployment_profile', sa.JSON(), nullable=True),
    sa.Column('agent_compatible', sa.Boolean(), nullable=True),
    sa.Column('deployment_priority', sa.String(length=16), nullable=True),
    sa.Column('discovered_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
    sa.Column('last_seen', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
    sa.Column('scan_id', sa.String(length=64), nullable=True),
    sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_discovered_assets_created_at'), 'discovered_assets', ['created_at'], unique=False)
    op.create_index(op.f('ix_discovered_assets_ip'), 'discovered_assets', ['ip'], unique=False)
    op.create_index('ix_discovered_assets_org_ip', 'discovered_assets', ['organization_id', 'ip'], unique=False)
    op.create_index(op.f('ix_discovered_assets_organization_id'), 'discovered_assets', ['organization_id'], unique=False)
    op.create_index(op.f('ix_discovered_assets_scan_id'), 'discovered_assets', ['scan_id'], unique=False)
    
    # Create new agent_enrollments table
    op.create_table('agent_enrollments',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('organization_id', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
    sa.Column('agent_token', sa.String(length=128), nullable=False),
    sa.Column('agent_id', sa.String(length=64), nullable=True),
    sa.Column('hostname', sa.String(length=255), nullable=True),
    sa.Column('platform', sa.String(length=64), nullable=True),
    sa.Column('ip_address', sa.String(length=64), nullable=True),
    sa.Column('status', sa.String(length=20), nullable=True),
    sa.Column('first_checkin', sa.DateTime(timezone=True), nullable=True),
    sa.Column('last_heartbeat', sa.DateTime(timezone=True), nullable=True),
    sa.Column('agent_metadata', sa.JSON(), nullable=True),
    sa.Column('enrollment_source', sa.String(length=64), nullable=True),
    sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('revoked_reason', sa.String(length=255), nullable=True),
    sa.Column('discovered_asset_id', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['discovered_asset_id'], ['discovered_assets.id'], ),
    sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_agent_enrollments_agent_id'), 'agent_enrollments', ['agent_id'], unique=True)
    op.create_index(op.f('ix_agent_enrollments_agent_token'), 'agent_enrollments', ['agent_token'], unique=True)
    op.create_index(op.f('ix_agent_enrollments_created_at'), 'agent_enrollments', ['created_at'], unique=False)
    op.create_index('ix_agent_enrollments_org_status', 'agent_enrollments', ['organization_id', 'status'], unique=False)
    op.create_index(op.f('ix_agent_enrollments_organization_id'), 'agent_enrollments', ['organization_id'], unique=False)
    op.create_index(op.f('ix_agent_enrollments_status'), 'agent_enrollments', ['status'], unique=False)
    
    # Add onboarding columns to organizations table
    op.add_column('organizations', sa.Column('onboarding_status', sa.String(length=20), nullable=True))
    op.add_column('organizations', sa.Column('onboarding_step', sa.String(length=50), nullable=True))
    op.add_column('organizations', sa.Column('onboarding_data', sa.JSON(), nullable=True))
    op.add_column('organizations', sa.Column('onboarding_completed_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('organizations', sa.Column('first_login_completed', sa.Boolean(), nullable=True))
    op.create_index(op.f('ix_organizations_onboarding_status'), 'organizations', ['onboarding_status'], unique=False)


def downgrade() -> None:
    # Remove onboarding columns from organizations
    op.drop_index(op.f('ix_organizations_onboarding_status'), table_name='organizations')
    op.drop_column('organizations', 'first_login_completed')
    op.drop_column('organizations', 'onboarding_completed_at')
    op.drop_column('organizations', 'onboarding_data')
    op.drop_column('organizations', 'onboarding_step')
    op.drop_column('organizations', 'onboarding_status')
    
    # Drop agent_enrollments table
    op.drop_index(op.f('ix_agent_enrollments_status'), table_name='agent_enrollments')
    op.drop_index(op.f('ix_agent_enrollments_organization_id'), table_name='agent_enrollments')
    op.drop_index('ix_agent_enrollments_org_status', table_name='agent_enrollments')
    op.drop_index(op.f('ix_agent_enrollments_created_at'), table_name='agent_enrollments')
    op.drop_index(op.f('ix_agent_enrollments_agent_token'), table_name='agent_enrollments')
    op.drop_index(op.f('ix_agent_enrollments_agent_id'), table_name='agent_enrollments')
    op.drop_table('agent_enrollments')
    
    # Drop discovered_assets table
    op.drop_index(op.f('ix_discovered_assets_scan_id'), table_name='discovered_assets')
    op.drop_index(op.f('ix_discovered_assets_organization_id'), table_name='discovered_assets')
    op.drop_index('ix_discovered_assets_org_ip', table_name='discovered_assets')
    op.drop_index(op.f('ix_discovered_assets_ip'), table_name='discovered_assets')
    op.drop_index(op.f('ix_discovered_assets_created_at'), table_name='discovered_assets')
    op.drop_table('discovered_assets')
