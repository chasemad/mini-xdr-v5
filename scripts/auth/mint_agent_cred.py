#!/usr/bin/env python3
"""Utility script to mint per-device credentials for HMAC authentication."""
import asyncio
import hashlib
import secrets
import sys
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # Go up to mini-xdr root
BACKEND_PATH = BASE_DIR / 'backend'
if str(BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PATH))

from app.db import AsyncSessionLocal, init_db
from app.models import AgentCredential

DEFAULT_TTL_DAYS = 90


def generate_secret() -> tuple[str, str]:
    secret = secrets.token_urlsafe(32)
    hmac_key = hashlib.sha256(secret.encode("utf-8")).hexdigest()
    return secret, hmac_key


async def store_credential(device_id: str, public_id: str, hmac_key: str, ttl_days: int) -> None:
    expires_at = datetime.now(timezone.utc) + timedelta(days=ttl_days) if ttl_days else None
    async with AsyncSessionLocal() as session:
        credential = AgentCredential(
            device_id=device_id,
            public_id=public_id,
            secret_hash=hmac_key,
            created_at=datetime.now(timezone.utc),
            expires_at=expires_at,
        )
        session.add(credential)
        await session.commit()


async def main(ttl_days: int = DEFAULT_TTL_DAYS) -> None:
    await init_db()
    device_id = str(uuid.uuid4())
    public_id = str(uuid.uuid4())
    secret, hmac_key = generate_secret()
    await store_credential(device_id, public_id, hmac_key, ttl_days)

    print("New Mini-XDR agent credential minted:\n")
    print(f" Device ID : {device_id}")
    print(f" Public ID : {public_id}")
    print(f" Secret    : {secret}")
    print(f" HMAC Key  : {hmac_key}")
    print("\nUse the HMAC key when signing requests. Store the raw secret securely; it is shown once.")


if __name__ == "__main__":
    ttl = DEFAULT_TTL_DAYS
    if len(sys.argv) > 1:
        try:
            ttl = int(sys.argv[1])
        except ValueError:
            print("TTL must be an integer number of days", file=sys.stderr)
            sys.exit(1)
    asyncio.run(main(ttl))
