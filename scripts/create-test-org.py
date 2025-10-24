#!/usr/bin/env python3
"""
Create test organization for seamless onboarding testing

Run this script from the backend pod:
  kubectl exec -it deployment/mini-xdr-backend -n mini-xdr -- python scripts/create-test-org.py
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.db import get_async_session_local
from app.models import Organization, User
from app.auth import get_password_hash


async def create_test_organization():
    """Create test organization with seamless onboarding enabled"""

    print("=" * 70)
    print("Creating Test Organization for Seamless Onboarding")
    print("=" * 70)
    print()

    async for db in get_async_session_local():
        # Check if test org already exists
        from sqlalchemy import select
        stmt = select(Organization).where(Organization.slug == "test-org")
        result = await db.execute(stmt)
        existing_org = result.scalars().first()

        if existing_org:
            print(f"⚠️  Test organization already exists:")
            print(f"   ID: {existing_org.id}")
            print(f"   Name: {existing_org.name}")
            print(f"   Slug: {existing_org.slug}")
            print(f"   Onboarding Version: {existing_org.onboarding_flow_version}")
            print()

            # Check if user exists
            stmt = select(User).where(
                User.organization_id == existing_org.id,
                User.email == "test@minixdr.com"
            )
            result = await db.execute(stmt)
            existing_user = result.scalars().first()

            if existing_user:
                print(f"✅ Test user already exists:")
                print(f"   Email: test@minixdr.com")
                print(f"   Password: TestPassword123!")
                print()
                return

        else:
            # Create test organization
            test_org = Organization(
                name="Test Organization",
                slug="test-org",
                onboarding_flow_version="seamless",
                auto_discovery_enabled=True,
                integration_settings={
                    "agent_public_base_url": "http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com"
                },
                onboarding_status="not_started",
                status="active",
                max_users=10,
                max_log_sources=50
            )

            db.add(test_org)
            await db.commit()
            await db.refresh(test_org)

            print(f"✅ Created test organization:")
            print(f"   ID: {test_org.id}")
            print(f"   Name: {test_org.name}")
            print(f"   Slug: {test_org.slug}")
            print(f"   Onboarding Version: {test_org.onboarding_flow_version}")
            print(f"   Auto Discovery: {test_org.auto_discovery_enabled}")
            print(f"   Agent Public URL: {test_org.integration_settings.get('agent_public_base_url')}")
            print()

            existing_org = test_org

        # Create test user
        test_user = User(
            organization_id=existing_org.id,
            email="test@minixdr.com",
            hashed_password=get_password_hash("TestPassword123!"),
            full_name="Test User",
            role="admin",
            is_active=True
        )

        db.add(test_user)
        await db.commit()
        await db.refresh(test_user)

        print(f"✅ Created test user:")
        print(f"   ID: {test_user.id}")
        print(f"   Email: {test_user.email}")
        print(f"   Password: TestPassword123!")
        print(f"   Role: {test_user.role}")
        print()

        print("=" * 70)
        print("Test Organization Setup Complete!")
        print("=" * 70)
        print()
        print("You can now login with:")
        print(f"  Email: test@minixdr.com")
        print(f"  Password: TestPassword123!")
        print()

        break


if __name__ == "__main__":
    try:
        asyncio.run(create_test_organization())
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error creating test organization: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
