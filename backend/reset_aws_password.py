#!/usr/bin/env python3
"""Script to reset password for AWS RDS database"""

import asyncio
import os
import sys
import bcrypt

# Add the app directory to the path so we can import the backend modules
sys.path.append('/app')

from app.db import get_db
from app.models import User
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

async def reset_password():
    email = "admin@example.com"
    new_password = "demo-tpot-api-key"

    print("🔄 Connecting to database...")

    try:
        async for db in get_db():
            # Check if user exists
            stmt = select(User).where(User.email == email)
            result = await db.execute(stmt)
            user = result.scalar_one_or_none()

            if not user:
                print(f"❌ User not found with email: {email}")
                return

            print("📊 Current user status:")
            print(f"   Email: {user.email}")
            print(f"   Active: {user.is_active}")
            print(f"   Role: {user.role}")
            print(f"   Failed attempts: {user.failed_login_attempts}")
            print(f"   Locked until: {user.locked_until}")
            print(f"   Organization ID: {user.organization_id}")

            # Hash the new password
            salt = bcrypt.gensalt(rounds=12)
            hashed = bcrypt.hashpw(new_password.encode(), salt).decode()

            # Update user password and reset login attempts
            user.hashed_password = hashed
            user.failed_login_attempts = 0
            user.locked_until = None
            user.is_active = True

            await db.commit()

            print("\n✅ Password reset and account unlocked successfully!")
            print(f"📧 Email: {email}")
            print(f"🔑 Password: {new_password}")
            print("\n🌐 Login at: http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com/login")
            break

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(reset_password())
