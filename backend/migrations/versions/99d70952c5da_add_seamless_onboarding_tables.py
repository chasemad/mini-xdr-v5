"""add_seamless_onboarding_tables

Revision ID: 99d70952c5da
Revises: 5093d5f3c7d4
Create Date: 2025-10-24 01:55:14.833812

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "99d70952c5da"
down_revision: Union[str, None] = "5093d5f3c7d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add new columns to organizations table
    op.add_column(
        "organizations",
        sa.Column(
            "onboarding_flow_version",
            sa.String(length=20),
            nullable=True,
            server_default="seamless",
        ),
    )
    op.add_column(
        "organizations",
        sa.Column(
            "auto_discovery_enabled",
            sa.Boolean(),
            nullable=True,
            server_default=sa.text("true"),
        ),
    )
    op.add_column(
        "organizations",
        sa.Column(
            "integration_settings", sa.JSON(), nullable=True, server_default="{}"
        ),
    )

    # Create integration_credentials table
    op.create_table(
        "integration_credentials",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("organization_id", sa.Integer(), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("credential_type", sa.String(length=50), nullable=False),
        sa.Column("credential_data", sa.JSON(), nullable=False),
        sa.Column(
            "status", sa.String(length=20), nullable=True, server_default="active"
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=True,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=True,
            server_default=sa.text("now()"),
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["organization_id"], ["organizations.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "organization_id",
            "provider",
            name="uq_integration_credentials_org_provider",
        ),
    )

    # Create cloud_assets table
    op.create_table(
        "cloud_assets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("organization_id", sa.Integer(), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("asset_type", sa.String(length=100), nullable=False),
        sa.Column("asset_id", sa.String(length=255), nullable=False),
        sa.Column("asset_data", sa.JSON(), nullable=False),
        sa.Column("region", sa.String(length=50), nullable=True),
        sa.Column(
            "discovered_at",
            sa.DateTime(timezone=True),
            nullable=True,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            nullable=True,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "agent_deployed",
            sa.Boolean(),
            nullable=True,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "agent_status",
            sa.String(length=20),
            nullable=True,
            server_default="pending",
        ),
        sa.Column("tags", sa.JSON(), nullable=True, server_default="{}"),
        sa.ForeignKeyConstraint(
            ["organization_id"], ["organizations.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "organization_id",
            "provider",
            "asset_id",
            name="uq_cloud_assets_org_provider_asset",
        ),
    )

    # Create indexes for performance
    op.create_index(
        "idx_integration_credentials_org_provider",
        "integration_credentials",
        ["organization_id", "provider"],
    )
    op.create_index(
        "idx_cloud_assets_org_provider", "cloud_assets", ["organization_id", "provider"]
    )
    op.create_index("idx_cloud_assets_type", "cloud_assets", ["asset_type"])
    op.create_index("idx_cloud_assets_status", "cloud_assets", ["agent_status"])


def downgrade() -> None:
    # Drop indexes
    op.drop_index("idx_cloud_assets_status", table_name="cloud_assets")
    op.drop_index("idx_cloud_assets_type", table_name="cloud_assets")
    op.drop_index("idx_cloud_assets_org_provider", table_name="cloud_assets")
    op.drop_index(
        "idx_integration_credentials_org_provider", table_name="integration_credentials"
    )

    # Drop tables
    op.drop_table("cloud_assets")
    op.drop_table("integration_credentials")

    # Drop columns from organizations
    op.drop_column("organizations", "integration_settings")
    op.drop_column("organizations", "auto_discovery_enabled")
    op.drop_column("organizations", "onboarding_flow_version")
