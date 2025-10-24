#!/usr/bin/env python3
"""Quick script to reset password for local testing"""

import asyncio
import sqlite3

import bcrypt


async def reset_password():
    db_path = "/Users/chasemad/Desktop/mini-xdr/backend/xdr.db"
    email = "chasemadrian@protonmail.com"
    new_password = "test123"  # Simple password for local testing

    # Hash the password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(new_password.encode(), salt).decode()

    # Update in database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE users SET hashed_password = ? WHERE email = ?", (hashed, email)
    )
    conn.commit()
    conn.close()

    print(f"✅ Password reset successfully for {email}")
    print(f"📧 Email: {email}")
    print(f"🔑 Password: {new_password}")
    print(f"\n🌐 Login at: http://localhost:3001/login")


if __name__ == "__main__":
    asyncio.run(reset_password())
