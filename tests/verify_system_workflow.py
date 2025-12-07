import json
import time
import urllib.request

BASE_URL = "http://localhost:8000"
API_KEY = "test_key"


def make_request(endpoint, method="GET", data=None):
    url = f"{BASE_URL}{endpoint}"
    headers = {"Content-Type": "application/json", "x-api-key": API_KEY}

    try:
        if data:
            data_bytes = json.dumps(data).encode("utf-8")
            req = urllib.request.Request(
                url, data=data_bytes, headers=headers, method=method
            )
        else:
            req = urllib.request.Request(url, headers=headers, method=method)

        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                return json.loads(response.read().decode())
            else:
                print(f"❌ {method} {endpoint} Failed: {response.status}")
                return None
    except urllib.error.HTTPError as e:
        print(f"❌ {method} {endpoint} Failed: {e.code} - {e.read().decode()}")
    except Exception as e:
        print(f"❌ {method} {endpoint} Error: {e}")
    return None


def verify_system_workflow():
    print("Starting System Workflow Verification...")

    # 1. Save System Workflow with Trigger
    print("Testing Save System Workflow...")
    data = {
        "name": "Auto Block System Rule",
        "incident_id": None,
        "graph": {
            "nodes": [{"id": "1", "type": "triggerNode", "data": {"label": "Start"}}],
            "edges": [],
        },
        "trigger_config": {
            "event_type": "cowrie.login.failed",
            "threshold": 5,
            "window_seconds": 60,
        },
    }

    result = make_request("/api/workflows/save", "POST", data)
    if result and result.get("success"):
        print(f"✅ System Workflow Saved: {result}")

        # Check if trigger was created
        print("Checking if trigger was created...")
        triggers = make_request("/api/triggers/")
        if triggers:
            found = False
            for t in triggers:
                if t.get("playbook_name") == data["name"]:
                    print(f"✅ Trigger Found: {t['name']}")
                    found = True
                    break
            if not found:
                print("❌ Trigger NOT found (Server code might not be updated)")
        else:
            print("❌ Failed to list triggers")

    else:
        print("❌ Failed to save system workflow")
        return

    # 2. Run System Workflow (Ad-hoc test)
    print("Testing Run System Workflow (Ad-hoc) - Sending None...")
    run_data = {"incident_id": None, "graph": data["graph"]}
    result = make_request("/api/workflows/run", "POST", run_data)
    if result and result.get("success"):
        print(f"✅ System Workflow Started (None): {result}")
    else:
        print("❌ Failed to run system workflow (None)")

    # Note: Omitting incident_id might fail if Pydantic strictness varies, but None should work.


if __name__ == "__main__":
    verify_system_workflow()
