#!/usr/bin/env python3
"""
Test HMAC authentication for Mini-XDR agents
"""

import hashlib
import hmac
import json
import time
import uuid
import requests
from datetime import datetime, timezone


def build_canonical_message(method: str, path: str, body: str, timestamp: str, nonce: str) -> str:
    return "|".join([method.upper(), path, body, timestamp, nonce])


def compute_signature(secret: str, canonical_message: str) -> str:
    return hmac.new(secret.encode("utf-8"), canonical_message.encode("utf-8"), hashlib.sha256).hexdigest()


def make_authenticated_request(base_url: str, path: str, device_id: str, secret_hash: str, method: str = "GET", data: dict = None):
    """Make an authenticated request using HMAC"""

    # Prepare request data
    body = json.dumps(data) if data else ""
    timestamp = str(int(datetime.now(timezone.utc).timestamp()))
    nonce = str(uuid.uuid4())

    # Build canonical message and signature
    canonical_message = build_canonical_message(method, path, body, timestamp, nonce)
    signature = compute_signature(secret_hash, canonical_message)

    # Headers - using the correct header names expected by middleware
    headers = {
        "Content-Type": "application/json",
        "X-Device-ID": device_id,
        "X-TS": timestamp,
        "X-Nonce": nonce,
        "X-Signature": signature
    }

    url = f"{base_url}{path}"

    print(f"Making {method} request to {url}")
    print(f"Device ID: {device_id}")
    print(f"Timestamp: {timestamp}")
    print(f"Nonce: {nonce}")
    print(f"Canonical message: {canonical_message}")
    print(f"Signature: {signature}")
    print()

    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, timeout=30)
        elif method.upper() == "POST":
            response = requests.post(url, headers=headers, data=body, timeout=30)
        else:
            raise ValueError(f"Unsupported method: {method}")

        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        return response

    except Exception as e:
        print(f"Request failed: {e}")
        return None


def test_agent_authentication():
    """Test agent authentication with the Mini-XDR backend"""

    # Configuration
    base_url = "http://54.91.233.149:8000"  # New instance IP
    device_id = "ffb56f4f-b0c8-4258-8922-0f976e536a7b"  # From database
    secret_hash = "678aae7bdf9e61cbb5fd059f0c774baf6d3143495cd091b2759265fe15c0beb3"  # From database

    print("=== Testing HMAC Authentication ===")
    print()

    # Test 1: Health check (should work without auth)
    print("1. Testing health endpoint (no auth required):")
    response = requests.get(f"{base_url}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

    # Test 2: ML Status endpoint
    print("2. Testing ML status endpoint:")
    response = make_authenticated_request(
        base_url=base_url,
        path="/api/ml/status",
        device_id=device_id,
        secret_hash=secret_hash,
        method="GET"
    )
    print()

    # Test 3: Orchestrator status
    print("3. Testing orchestrator status endpoint:")
    response = make_authenticated_request(
        base_url=base_url,
        path="/api/orchestrator/status",
        device_id=device_id,
        secret_hash=secret_hash,
        method="GET"
    )
    print()

    # Test 4: Test ingestion endpoint (Cowrie)
    print("4. Testing Cowrie ingestion:")
    test_data = {
        "source": "test_agent",
        "eventid": "cowrie.session.connect",
        "ts": datetime.now(timezone.utc).isoformat(),
        "src_ip": "192.168.1.100",
        "dst_port": 22,
        "message": "Test SSH connection attempt"
    }

    response = make_authenticated_request(
        base_url=base_url,
        path="/ingest/cowrie",
        device_id=device_id,
        secret_hash=secret_hash,
        method="POST",
        data=test_data
    )


if __name__ == "__main__":
    test_agent_authentication()