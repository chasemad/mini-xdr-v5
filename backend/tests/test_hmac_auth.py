#!/usr/bin/env python3
import hashlib
import hmac
import json
import uuid
from datetime import datetime, timezone

import requests

# Credentials from database
MINI_XDR_API = "http://localhost:8000"
DEVICE_ID = "ffb56f4f-b0c8-4258-8922-0f976e536a7b"
HMAC_KEY = "678aae7bdf9e61cbb5fd059f0c774baf6d3143495cd091b2759265fe15c0beb3"
API_KEY = "demo-minixdr-api-key"


def build_hmac_headers(method, path, body):
    """Build HMAC authentication headers for Mini-XDR"""
    timestamp = str(int(datetime.now(timezone.utc).timestamp()))
    nonce = str(uuid.uuid4())

    # Build canonical message
    canonical_message = "|".join([method.upper(), path, body, timestamp, nonce])

    # Generate HMAC signature
    signature = hmac.new(
        HMAC_KEY.encode("utf-8"), canonical_message.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    return {
        "X-Device-ID": DEVICE_ID,
        "X-TS": timestamp,
        "X-Nonce": nonce,
        "X-Signature": signature,
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }


def test_hmac_ingestion():
    # Create test payload
    payload = {
        "source_type": "cowrie",
        "hostname": "test-honeypot",
        "events": [
            {
                "eventid": "cowrie.login.failed",
                "src_ip": "13.220.211.0",
                "dst_ip": "10.0.1.100",
                "dst_port": 2222,
                "message": "HMAC Test - Brute force attack detected",
                "raw": {
                    "username": "admin",
                    "password": "test123",
                    "session": "hmac_test_session",
                },
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            }
        ],
    }

    body = json.dumps(payload)
    headers = build_hmac_headers("POST", "/ingest/multi", body)

    print("🔐 Testing HMAC Authentication...")
    print(f"Device ID: {DEVICE_ID}")
    print(f"Headers: {list(headers.keys())}")

    try:
        response = requests.post(
            f"{MINI_XDR_API}/ingest/multi", data=body, headers=headers, timeout=10
        )

        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Processed: {result.get('processed', 0)} events")
            print(f"Response: {result}")
        else:
            print(f"❌ Error: {response.text}")

    except Exception as e:
        print(f"❌ Request failed: {e}")


if __name__ == "__main__":
    test_hmac_ingestion()
