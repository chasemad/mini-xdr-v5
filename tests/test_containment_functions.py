#!/usr/bin/env python3
"""
Test containment and response functions for Mini-XDR
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


def test_containment_functions():
    """Test containment and response functions"""

    base_url = "http://54.91.233.149:8000"
    device_id = "ffb56f4f-b0c8-4258-8922-0f976e536a7b"
    secret_hash = "678aae7bdf9e61cbb5fd059f0c774baf6d3143495cd091b2759265fe15c0beb3"

    print("=== Testing Containment and Response Functions ===\n")

    # Test 1: Distributed Tool Execution
    print("1. Testing Distributed Tool Execution:")
    tool_request = {
        "tool_name": "containment",
        "parameters": {
            "action": "isolate_host",
            "target_ip": "192.168.1.100",
            "reason": "ssh_brute_force_detected"
        }
    }

    response, result = make_authenticated_request(
        base_url=base_url,
        path="/api/distributed/execute-tool",
        device_id=device_id,
        secret_hash=secret_hash,
        method="POST",
        data=tool_request
    )
    print()

    # Test 2: Workflow Creation for Incident Response
    print("2. Testing Workflow Creation:")
    workflow_request = {
        "name": "incident_response_containment",
        "incident_id": 1,
        "priority": "critical",
        "steps": [
            {
                "step_id": "isolate_host",
                "agent": "containment",
                "action": "isolate_host",
                "parameters": {
                    "target_ip": "192.168.1.100"
                }
            },
            {
                "step_id": "collect_evidence",
                "agent": "forensics",
                "action": "collect_logs",
                "parameters": {
                    "target_ip": "192.168.1.100",
                    "time_range": "last_hour"
                }
            }
        ]
    }

    response, result = make_authenticated_request(
        base_url=base_url,
        path="/api/orchestrator/workflows",
        device_id=device_id,
        secret_hash=secret_hash,
        method="POST",
        data=workflow_request
    )
    print()

    # Test 3: NLP Threat Analysis
    print("3. Testing NLP Threat Analysis:")
    nlp_request = {
        "query": "urgent containment needed for SSH brute force attack from 192.168.1.100",
        "context": {
            "incident_id": 1,
            "severity": "critical"
        }
    }

    response, result = make_authenticated_request(
        base_url=base_url,
        path="/api/nlp/threat-analysis",
        device_id=device_id,
        secret_hash=secret_hash,
        method="POST",
        data=nlp_request
    )
    print()

    # Test 4: Distributed System Status
    print("4. Testing Distributed System Status:")
    response, result = make_authenticated_request(
        base_url=base_url,
        path="/api/distributed/status",
        device_id=device_id,
        secret_hash=secret_hash,
        method="GET"
    )
    print()

    # Test 5: Agent Containment Query
    print("5. Testing Agent Containment Query:")
    containment_query = {
        "query": "activate emergency containment protocol for critical threat from 192.168.1.100",
        "incident_id": 1,
        "priority": "critical",
        "context": {
            "threat_type": "ssh_brute_force",
            "containment_actions": ["isolate_host", "block_traffic", "reset_passwords"],
            "impact_assessment": "high"
        }
    }

    response, result = make_authenticated_request(
        base_url=base_url,
        path="/api/agents/orchestrate",
        device_id=device_id,
        secret_hash=secret_hash,
        method="POST",
        data=containment_query
    )
    print()

    # Test 6: ML Explanation for Incident
    print("6. Testing ML Explanation for Incident:")
    response, result = make_authenticated_request(
        base_url=base_url,
        path="/api/ml/explain/1",
        device_id=device_id,
        secret_hash=secret_hash,
        method="POST",
        data={"context": {"user_question": "Why is this incident considered critical?"}}
    )
    print()

    print("=== Containment and Response Function Testing Complete ===")


if __name__ == "__main__":
    test_containment_functions()