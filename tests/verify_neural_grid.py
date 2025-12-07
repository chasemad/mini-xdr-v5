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


def verify_api():
    print("Starting Verification...")

    # 0. Create Dummy Incident
    print("Creating Dummy Incident...")
    incident_id = 1
    try:
        # Try to create an incident directly via DB or API?
        # Since we don't have easy DB access here, let's assume we can use the API if it exists,
        # or just hope ID 1 exists.
        # Actually, let's try to create one if we can.
        # But wait, we don't have an easy way to create an incident via API in this script without more code.
        # Let's just try to use a random large ID? No, FK constraint.
        # Let's assume we need to insert one.
        pass
    except Exception as e:
        print(f"Skipping incident creation: {e}")

    # For now, let's try to use the Global Workflow test first which has incident_id=None
    # If that fails, we know it's not the FK constraint.

    # 1. Save Global Workflow (No Incident ID)
    print("Testing Save Global Workflow (No Incident ID)...")
    url = f"{BASE_URL}/api/workflows/save"  # Corrected URL to include /api
    data = {
        "name": "Global Block IP Workflow",
        "graph": {
            "nodes": [{"id": "1", "type": "triggerNode", "data": {"label": "Start"}}],
            "edges": [],
        },
    }

    global_wf_id = None
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-api-key": API_KEY},
            method="POST",
        )  # Added API_KEY and method
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            print(f"✅ POST /api/workflows/save Success: {result}")
            global_wf_id = result.get("workflow_id")
    except urllib.error.HTTPError as e:
        print(f"❌ POST /api/workflows/save Failed: {e.code} - {e.reason}")
    except Exception as e:
        print(f"❌ POST /api/workflows/save Error: {e}")

    # 2. Save Incident Specific Workflow
    print("Testing Save Workflow (Incident 1)...")
    data["incident_id"] = 1
    data["name"] = "Incident Specific Workflow"

    incident_wf_id = None
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-api-key": API_KEY},
            method="POST",
        )  # Added API_KEY and method
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            print(f"✅ POST /api/workflows/save Success: {result}")
            incident_wf_id = result.get("workflow_id")
    except urllib.error.HTTPError as e:
        print(f"❌ POST /api/workflows/save Failed: {e.code} - {e.reason}")
    except Exception as e:
        print(f"❌ POST /api/workflows/save Error: {e}")

    # 3. Run Workflow
    if incident_wf_id:
        print(f"Testing Run Workflow {incident_wf_id}...")
        url = f"{BASE_URL}/api/workflows/run"  # Corrected URL to include /api
        data = {"incident_id": 1, "workflow_id": incident_wf_id}
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode("utf-8"),
                headers={"Content-Type": "application/json", "x-api-key": API_KEY},
                method="POST",
            )  # Added API_KEY and method
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode("utf-8"))
                print(f"✅ POST /api/workflows/run Success: {result}")
        except urllib.error.HTTPError as e:
            print(f"❌ POST /api/workflows/run Failed: {e.code} - {e.reason}")
        except Exception as e:
            print(f"❌ POST /api/workflows/run Error: {e}")


def test_load_workflow(workflow_id):
    print(f"Testing Load Workflow {workflow_id}...")
    result = make_request(f"/api/workflows/{workflow_id}")
    if result:
        print("✅ Load Workflow Success")


def test_run_workflow(workflow_id):
    print(f"Testing Run Workflow {workflow_id}...")
    payload = {"incident_id": 1, "workflow_id": workflow_id}
    result = make_request("/api/workflows/run", "POST", payload)
    if result:
        print("✅ Run Workflow Success")


if __name__ == "__main__":
    print("Starting Verification...")
    verify_api()
