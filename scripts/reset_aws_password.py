#!/usr/bin/env python3
"""Script to reset password for AWS RDS database"""

import asyncio
import os
import bcrypt
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

async def reset_password():
    # AWS RDS connection details
    db_url = "postgresql://user:password@mini-xdr-postgres.ccnkck4wij3r.us-east-1.rds.amazonaws.com:5432/xdrdb"
    email = "chasemadrian@protonmail.com"
    new_password = "SecurePass123!"

    print("🔄 Connecting to AWS RDS database...")

    # Create async engine
    engine = create_async_engine(db_url, echo=False)

    try:
        async with engine.begin() as conn:
            # Hash the password
            salt = bcrypt.gensalt(rounds=12)
            hashed = bcrypt.hashpw(new_password.encode(), salt).decode()

            # Update password in database
            query = text("""
                UPDATE users
                SET hashed_password = :hashed,
                    failed_login_attempts = 0,
                    locked_until = NULL,
                    updated_at = NOW()
                WHERE email = :email
            """)

            result = await conn.execute(query, {"hashed": hashed, "email": email})

            if result.rowcount > 0:
                print("✅ Password reset successfully!"                print(f"📧 Email: {email}")
                print(f"🔑 New Password: {new_password}")
                print("
🌐 Login at: http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com/login"            else:
                print("❌ User not found with email:", email)

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(reset_password())
