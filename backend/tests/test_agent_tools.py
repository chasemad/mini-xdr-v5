#!/usr/bin/env python3
"""
Test agent tool execution capabilities for Mini-XDR
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


def make_authenticated_request(base_url: str, path: str, device_id: str, secret_hash: str, method: str = "GET", data: dict = None):
    """Make an authenticated request using HMAC"""

    body = json.dumps(data) if data else ""
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
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, timeout=30)
        elif method.upper() == "POST":
            response = requests.post(url, headers=headers, data=body, timeout=30)
        else:
            raise ValueError(f"Unsupported method: {method}")

        print(f"Status: {response.status_code}")
        try:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            return response, result
        except:
            print(f"Response: {response.text}")
            return response, response.text

    except Exception as e:
        print(f"Request failed: {e}")
        return None, None


def test_agent_tools():
    """Test agent tool execution capabilities"""

    base_url = "http://54.91.233.149:8000"
    device_id = "ffb56f4f-b0c8-4258-8922-0f976e536a7b"
    secret_hash = "678aae7bdf9e61cbb5fd059f0c774baf6d3143495cd091b2759265fe15c0beb3"

    print("=== Testing Agent Tool Execution Capabilities ===\n")

    # Test 1: Attribution Agent Tool
    print("1. Testing Attribution Agent:")
    attribution_task = {
        "query": "investigate source attribution",
        "incident_id": 1,
        "priority": "high",
        "context": {
            "src_ip": "192.168.1.100",
            "attack_type": "ssh_brute_force"
        }
    }

    response, result = make_authenticated_request(
        base_url=base_url,
        path="/api/agents/orchestrate",
        device_id=device_id,
        secret_hash=secret_hash,
        method="POST",
        data=attribution_task
    )
    print()

    # Test 2: Containment Agent Tools
    print("2. Testing Containment Agent:")
    containment_task = {
        "query": "initiate containment for suspicious IP",
        "incident_id": 2,
        "priority": "critical",
        "context": {
            "src_ip": "192.168.1.100",
            "containment_type": "isolate_host",
            "evidence": "ssh_brute_force_detected"
        }
    }

    response, result = make_authenticated_request(
        base_url=base_url,
        path="/api/agents/orchestrate",
        device_id=device_id,
        secret_hash=secret_hash,
        method="POST",
        data=containment_task
    )
    print()

    # Test 3: Forensics Agent Tools
    print("3. Testing Forensics Agent:")
    forensics_task = {
        "query": "collect forensic evidence for incident",
        "incident_id": 3,
        "priority": "high",
        "context": {
            "src_ip": "192.168.1.100",
            "evidence_types": ["network_logs", "system_logs", "process_info"],
            "time_range": "last_hour"
        }
    }

    response, result = make_authenticated_request(
        base_url=base_url,
        path="/api/agents/orchestrate",
        device_id=device_id,
        secret_hash=secret_hash,
        method="POST",
        data=forensics_task
    )
    print()

    # Test 4: Deception Agent Tools
    print("4. Testing Deception Agent:")
    deception_task = {
        "query": "activate deception countermeasures",
        "incident_id": 4,
        "priority": "medium",
        "context": {
            "src_ip": "192.168.1.100",
            "deception_type": "honeypot_redirect",
            "threat_level": "high"
        }
    }

    response, result = make_authenticated_request(
        base_url=base_url,
        path="/api/agents/orchestrate",
        device_id=device_id,
        secret_hash=secret_hash,
        method="POST",
        data=deception_task
    )
    print()

    # Test 5: Multi-Agent Coordination
    print("5. Testing Multi-Agent Coordination:")
    coordination_task = {
        "query": "coordinate full incident response",
        "incident_id": 5,
        "priority": "critical",
        "context": {
            "src_ip": "192.168.1.100",
            "attack_vector": "ssh_brute_force",
            "coordination_mode": "parallel",
            "required_agents": ["attribution", "containment", "forensics", "deception"]
        }
    }

    response, result = make_authenticated_request(
        base_url=base_url,
        path="/api/agents/orchestrate",
        device_id=device_id,
        secret_hash=secret_hash,
        method="POST",
        data=coordination_task
    )
    print()

    print("=== Agent Tool Testing Complete ===")


if __name__ == "__main__":
    test_agent_tools()