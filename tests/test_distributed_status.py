#!/usr/bin/env python3
"""
Test distributed MCP server status with authentication
"""

import hashlib
import hmac
import json
import uuid
from datetime import datetime, timezone
import requests


def build_canonical_message(method: str, path: str, body: str, timestamp: str, nonce: str) -> str:
    return "|".join([method.upper(), path, body, timestamp, nonce])


def compute_signature(secret: str, canonical_message: str) -> str:
    return hmac.new(secret.encode("utf-8"), canonical_message.encode("utf-8"), hashlib.sha256).hexdigest()


def make_authenticated_request(base_url: str, path: str, device_id: str, secret_hash: str, method: str = "GET"):
    """Make an authenticated request using HMAC"""

    body = ""
    timestamp = str(int(datetime.now(timezone.utc).timestamp()))
    nonce = str(uuid.uuid4())

    canonical_message = build_canonical_message(method, path, body, timestamp, nonce)
    signature = compute_signature(secret_hash, canonical_message)

    headers = {
        "Content-Type": "application/json",
        "X-Device-ID": device_id,
        "X-TS": timestamp,
        "X-Nonce": nonce,
        "X-Signature": signature
    }

    url = f"{base_url}{path}"
    print(f"Testing {method} {path}")

    try:
        response = requests.get(url, headers=headers, timeout=30)
        print(f"Status: {response.status_code}")
        try:
            result = response.json()
            print(json.dumps(result, indent=2))
            return response, result
        except:
            print(f"Response: {response.text}")
            return response, response.text

    except Exception as e:
        print(f"Request failed: {e}")
        return None, None


def test_distributed_status():
    """Test distributed MCP server status"""

    base_url = "http://54.91.233.149:8000"
    device_id = "ffb56f4f-b0c8-4258-8922-0f976e536a7b"
    secret_hash = "678aae7bdf9e61cbb5fd059f0c774baf6d3143495cd091b2759265fe15c0beb3"

    print("=== Testing Distributed MCP Server Status ===\n")

    # Test distributed status
    make_authenticated_request(
        base_url=base_url,
        path="/api/distributed/status",
        device_id=device_id,
        secret_hash=secret_hash,
        method="GET"
    )


if __name__ == "__main__":
    test_distributed_status()