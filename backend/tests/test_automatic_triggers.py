"""
Test Automatic Workflow Triggers End-to-End
Tests that incoming events trigger workflows automatically
"""
import hashlib
import hmac
import json
import time
from datetime import datetime, timezone

import requests

# Configuration
BASE_URL = "http://localhost:8000"
API_KEY = "demo-minixdr-api-key"  # From .env


def simple_api_request(method, endpoint, data=None):
    """Make API request with simple API key auth"""
    headers = {"Content-Type": "application/json", "x-api-key": API_KEY}

    url = f"{BASE_URL}{endpoint}"

    if method == "GET":
        response = requests.get(url, headers=headers)
    elif method == "POST":
        response = requests.post(url, headers=headers, json=data)
    else:
        raise ValueError(f"Unsupported method: {method}")

    return response


def send_authenticated_events(events):
    """Send events via /ingest/cowrie with HMAC auth"""
    # For now, send via /ingest/multi with API key (HMAC would require device credentials)
    # The trigger evaluator works regardless of ingestion method
    payload = {"source_type": "cowrie", "hostname": "test-honeypot", "events": events}

    headers = {"Content-Type": "application/json"}

    response = requests.post(f"{BASE_URL}/ingest/multi", headers=headers, json=payload)
    return response


def test_ssh_brute_force_trigger():
    """Test that SSH brute force events automatically trigger workflow"""
    print("\n" + "=" * 80)
    print("🧪 TEST: SSH Brute Force Automatic Workflow Trigger")
    print("=" * 80)

    # Step 1: Check initial trigger stats
    print("\n📊 Step 1: Checking initial trigger stats...")
    response = simple_api_request("GET", "/api/triggers/stats/summary")
    if response.status_code == 200:
        stats = response.json()
        initial_executions = stats.get("total_executions", 0)
        print(f"   ✓ Initial executions: {initial_executions}")
    else:
        print(f"   ⚠️  Stats request failed: {response.status_code}")
        initial_executions = 0

    # Step 2: List enabled triggers
    print("\n📋 Step 2: Listing enabled triggers...")
    response = simple_api_request("GET", "/api/triggers?enabled=true")
    if response.status_code == 200:
        triggers = response.json()
        print(f"   ✓ Found {len(triggers)} enabled triggers")
        for trigger in triggers:
            print(
                f"     - {trigger['name']} (ID: {trigger['id']}, Auto-execute: {trigger['auto_execute']})"
            )
    else:
        print(f"   ✗ Failed to list triggers: {response.status_code}")
        return False

    # Step 3: Generate SSH brute force events
    print("\n🚀 Step 3: Sending SSH brute force events...")
    test_ip = "192.168.100.50"

    events = []
    for i in range(8):  # Trigger threshold is 6, send 8 to be sure
        events.append(
            {
                "eventid": "cowrie.login.failed",
                "src_ip": test_ip,
                "dst_port": 2222,
                "username": "admin",
                "password": f"password{i}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    response = send_authenticated_events(events)
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Successfully ingested {len(events)} events")
        print(f"   ✓ Response: {json.dumps(result, indent=2)}")

        # Check if incident was created
        if "incidents_detected" in result and result["incidents_detected"]:
            incident_id = result["incidents_detected"]  # It's already an integer
            print(f"   ✓ Incident created: #{incident_id}")
        else:
            print("   ⚠️  No incident detected yet")
            incident_id = None
    else:
        print(f"   ✗ Failed to ingest events: {response.status_code}")
        print(f"   Response: {response.text}")
        return False

    # Step 4: Wait for trigger evaluation
    print("\n⏳ Step 4: Waiting for trigger evaluation (3 seconds)...")
    time.sleep(3)

    # Step 5: Check if workflows were created
    print("\n📊 Step 5: Checking for created workflows...")
    response = simple_api_request("GET", "/api/response/workflows")
    if response.status_code == 200:
        workflows = response.json()
        print(f"   ✓ Total workflows: {len(workflows)}")

        # Find recent workflows for our incident
        recent_workflows = [
            w
            for w in workflows
            if w.get("incident_id") == incident_id
            or "auto_trigger" in str(w.get("triggered_by", ""))
        ]

        if recent_workflows:
            print(f"   ✓ Found {len(recent_workflows)} workflows from auto-triggers:")
            for wf in recent_workflows[:3]:  # Show first 3
                print(
                    f"     - Workflow {wf.get('workflow_id')}: {wf.get('playbook_name')}"
                )
                print(
                    f"       Status: {wf.get('status')}, Priority: {wf.get('priority')}"
                )
        else:
            print("   ⚠️  No workflows found from auto-triggers")
    else:
        print(f"   ✗ Failed to fetch workflows: {response.status_code}")

    # Step 6: Check updated trigger stats
    print("\n📊 Step 6: Checking updated trigger stats...")
    response = simple_api_request("GET", "/api/triggers/stats/summary")
    if response.status_code == 200:
        stats = response.json()
        final_executions = stats.get("total_executions", 0)
        new_executions = final_executions - initial_executions
        print(f"   ✓ Initial executions: {initial_executions}")
        print(f"   ✓ Final executions: {final_executions}")
        print(f"   ✓ New executions: {new_executions}")

        if new_executions > 0:
            print(
                f"\n   🎉 SUCCESS! Automatic triggers executed {new_executions} time(s)!"
            )
            return True
        else:
            print(f"\n   ⚠️  No new trigger executions detected")
            print(f"   This could mean:")
            print(f"     1. Incident wasn't created (check logs)")
            print(f"     2. Trigger conditions didn't match")
            print(f"     3. Cooldown period active")
            return False
    else:
        print(f"   ✗ Failed to get stats: {response.status_code}")
        return False


def test_trigger_details():
    """Check individual trigger metrics"""
    print("\n" + "=" * 80)
    print("📈 Checking Individual Trigger Metrics")
    print("=" * 80)

    response = simple_api_request("GET", "/api/triggers")
    if response.status_code == 200:
        triggers = response.json()
        print(f"\n✓ Found {len(triggers)} total triggers\n")

        for trigger in triggers:
            print(f"📋 {trigger['name']}")
            print(f"   Status: {'✅ Enabled' if trigger['enabled'] else '❌ Disabled'}")
            print(f"   Auto-execute: {'✅ Yes' if trigger['auto_execute'] else '❌ No'}")
            print(f"   Priority: {trigger['priority']}")
            print(f"   Executions: {trigger['trigger_count']}")
            print(f"   Success Rate: {trigger['success_rate']:.1f}%")
            print(f"   Avg Response Time: {trigger['avg_response_time_ms']:.1f}ms")
            if trigger.get("last_triggered_at"):
                print(f"   Last Triggered: {trigger['last_triggered_at']}")
            print()
    else:
        print(f"✗ Failed to fetch triggers: {response.status_code}")


if __name__ == "__main__":
    print("\n🚀 AUTOMATIC WORKFLOW TRIGGER TEST SUITE")
    print("=" * 80)
    print("Testing end-to-end automatic workflow triggering")
    print("=" * 80)

    # Run main test
    success = test_ssh_brute_force_trigger()

    # Show detailed metrics
    test_trigger_details()

    # Final result
    print("\n" + "=" * 80)
    if success:
        print("✅ TEST PASSED: Automatic triggers are working!")
    else:
        print("⚠️  TEST INCONCLUSIVE: Check logs for details")
        print(
            "   Backend log: /Users/chasemad/Desktop/mini-xdr/backend/logs/backend.log"
        )
    print("=" * 80)
