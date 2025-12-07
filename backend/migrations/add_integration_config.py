"""
Database migration to add IntegrationConfig table
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio

from app.database_models import Base, IntegrationConfig
from app.db import AsyncSessionLocal, engine
from sqlalchemy import text


async def create_integration_config_table():
    """Create the integration_configs table"""
    async with engine.begin() as conn:
        # Create table
        await conn.run_sync(Base.metadata.create_all)
        print("✅ Created integration_configs table")


async def seed_default_integrations():
    """Seed some default integration configurations for testing"""
    async with AsyncSessionLocal() as session:
        # Check if any integrations exist
        result = await session.execute(text("SELECT COUNT(*) FROM integration_configs"))
        count = result.scalar()

        if count > 0:
            print(f"⚠️  Integrations already exist ({count} records), skipping seed")
            return

        default_integrations = [
            # Firewalls
            IntegrationConfig(
                vendor="palo_alto",
                vendor_display_name="Palo Alto Networks",
                category="firewall",
                enabled=True,
                priority=1,
                capabilities=["block_ip", "block_domain", "allow_ip", "allow_domain"],
                config={"api_endpoint": "https://firewall.example.com"},
                health_status="unknown",
            ),
            IntegrationConfig(
                vendor="cisco",
                vendor_display_name="Cisco Firewall",
                category="firewall",
                enabled=True,
                priority=2,
                capabilities=["block_ip", "block_domain"],
                config={"api_endpoint": "https://cisco-fw.example.com"},
                health_status="unknown",
            ),
            # EDR
            IntegrationConfig(
                vendor="crowdstrike",
                vendor_display_name="CrowdStrike Falcon",
                category="edr",
                enabled=True,
                priority=1,
                capabilities=[
                    "isolate_host",
                    "kill_process",
                    "get_processes",
                    "quarantine_file",
                ],
                config={"api_endpoint": "https://api.crowdstrike.com"},
                health_status="unknown",
            ),
            IntegrationConfig(
                vendor="sentinelone",
                vendor_display_name="SentinelOne",
                category="edr",
                enabled=True,
                priority=2,
                capabilities=["isolate_host", "remediate_threat", "rollback_threat"],
                config={"api_endpoint": "https://sentinelone.example.com"},
                health_status="unknown",
            ),
            # IAM
            IntegrationConfig(
                vendor="okta",
                vendor_display_name="Okta",
                category="iam",
                enabled=True,
                priority=1,
                capabilities=[
                    "disable_account",
                    "revoke_sessions",
                    "reset_password",
                    "force_mfa",
                ],
                config={"api_endpoint": "https://example.okta.com"},
                health_status="unknown",
            ),
            IntegrationConfig(
                vendor="azure_ad",
                vendor_display_name="Azure Active Directory",
                category="iam",
                enabled=True,
                priority=2,
                capabilities=["disable_account", "revoke_tokens", "reset_password"],
                config={"api_endpoint": "https://graph.microsoft.com"},
                health_status="unknown",
            ),
            # Email Security
            IntegrationConfig(
                vendor="office365",
                vendor_display_name="Microsoft Office 365",
                category="email",
                enabled=True,
                priority=1,
                capabilities=["quarantine_email", "delete_email", "get_email_details"],
                config={"api_endpoint": "https://graph.microsoft.com"},
                health_status="unknown",
            ),
            IntegrationConfig(
                vendor="proofpoint",
                vendor_display_name="Proofpoint",
                category="email",
                enabled=True,
                priority=2,
                capabilities=["quarantine_email", "block_sender"],
                config={"api_endpoint": "https://proofpoint.example.com"},
                health_status="unknown",
            ),
            # SIEM
            IntegrationConfig(
                vendor="splunk",
                vendor_display_name="Splunk",
                category="siem",
                enabled=True,
                priority=1,
                capabilities=["query_logs", "create_alert", "run_search"],
                config={"api_endpoint": "https://splunk.example.com:8089"},
                health_status="unknown",
            ),
            # Threat Intel
            IntegrationConfig(
                vendor="virustotal",
                vendor_display_name="VirusTotal",
                category="threat_intel",
                enabled=True,
                priority=1,
                capabilities=["lookup_ip", "lookup_domain", "lookup_hash"],
                config={"api_endpoint": "https://www.virustotal.com/api/v3"},
                health_status="unknown",
            ),
        ]

        session.add_all(default_integrations)
        await session.commit()
        print(f"✅ Seeded {len(default_integrations)} default integrations")


async def main():
    print("🔧 Running IntegrationConfig migration...")

    try:
        await create_integration_config_table()
        await seed_default_integrations()
        print("\n✅ Migration completed successfully!")
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
