#!/usr/bin/env python3
"""
End-to-End Test for Chat → Workflow Integration
Tests the complete flow from chat to workflow creation and investigation
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import httpx
from datetime import datetime
import json
from typing import Dict, Any, List
import time

BASE_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:3000"
API_KEY = "demo-minixdr-api-key"  # Default API key

class Color:
    """ANSI color codes"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    """Print colored header"""
    print(f"\n{Color.BOLD}{Color.CYAN}{'='*80}{Color.END}")
    print(f"{Color.BOLD}{Color.CYAN}{text.center(80)}{Color.END}")
    print(f"{Color.BOLD}{Color.CYAN}{'='*80}{Color.END}\n")

def print_success(text: str):
    """Print success message"""
    print(f"{Color.GREEN}✅ {text}{Color.END}")

def print_error(text: str):
    """Print error message"""
    print(f"{Color.RED}❌ {text}{Color.END}")

def print_info(text: str):
    """Print info message"""
    print(f"{Color.BLUE}ℹ️  {text}{Color.END}")

def print_test(text: str):
    """Print test name"""
    print(f"\n{Color.BOLD}{Color.MAGENTA}🧪 TEST: {text}{Color.END}")

async def check_service_health(client: httpx.AsyncClient) -> bool:
    """Check if backend is healthy"""
    try:
        response = await client.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print_success("Backend service is healthy")
            return True
        else:
            print_error(f"Backend health check failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Backend is not running: {e}")
        return False

async def get_incidents(client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    """Get all incidents"""
    try:
        response = await client.get(f"{BASE_URL}/incidents")
        if response.status_code == 200:
            incidents = response.json()
            print_success(f"Retrieved {len(incidents)} incidents")
            return incidents
        else:
            print_error(f"Failed to get incidents: {response.status_code}")
            return []
    except Exception as e:
        print_error(f"Error getting incidents: {e}")
        return []

async def test_workflow_creation_from_chat(client: httpx.AsyncClient, incident_id: int) -> Dict[str, Any]:
    """Test 1: Create workflow from chat with action keywords"""
    print_test(f"Workflow Creation from Chat (Incident #{incident_id})")
    
    test_queries = [
        {
            "query": f"Block IP from incident {incident_id} and isolate the host",
            "expected_actions": ["block_ip", "isolate_host"],
            "description": "Block IP + Isolate Host"
        },
        {
            "query": f"Alert security team about incident {incident_id}",
            "expected_actions": ["alert_security_analysts"],
            "description": "Alert Security Team"
        },
        {
            "query": f"Reset passwords and enable MFA for compromised accounts in incident {incident_id}",
            "expected_actions": ["reset_passwords", "enforce_mfa"],
            "description": "Identity Protection"
        }
    ]
    
    results = []
    
    for test in test_queries:
        print_info(f"\nTesting: {test['description']}")
        print_info(f"Query: '{test['query']}'")
        
        try:
            response = await client.post(
                f"{BASE_URL}/api/agents/orchestrate",
                json={
                    "query": test["query"],
                    "incident_id": incident_id,
                    "context": {
                        "test": True,
                        "source": "e2e_test"
                    }
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("workflow_created"):
                    print_success(f"✅ Workflow created: {data.get('workflow_id')}")
                    print_info(f"   - Workflow DB ID: {data.get('workflow_db_id')}")
                    print_info(f"   - Approval required: {data.get('approval_required', False)}")
                    print_info(f"   - Analysis type: {data.get('analysis_type')}")
                    
                    results.append({
                        "test": test["description"],
                        "success": True,
                        "workflow_id": data.get("workflow_id"),
                        "workflow_db_id": data.get("workflow_db_id"),
                        "message": data.get("message", "")[:200]
                    })
                else:
                    print_info(f"ℹ️  No workflow created (might not have recognized action keywords)")
                    print_info(f"   Response: {data.get('message', '')[:200]}")
                    results.append({
                        "test": test["description"],
                        "success": False,
                        "reason": "No workflow created"
                    })
            else:
                print_error(f"Request failed: {response.status_code}")
                results.append({
                    "test": test["description"],
                    "success": False,
                    "reason": f"HTTP {response.status_code}"
                })
                
        except Exception as e:
            print_error(f"Exception: {e}")
            results.append({
                "test": test["description"],
                "success": False,
                "reason": str(e)
            })
    
    return {
        "test_name": "Workflow Creation from Chat",
        "results": results,
        "passed": sum(1 for r in results if r.get("success")) > 0
    }

async def test_investigation_trigger(client: httpx.AsyncClient, incident_id: int) -> Dict[str, Any]:
    """Test 2: Trigger investigation from chat"""
    print_test(f"Investigation Trigger from Chat (Incident #{incident_id})")
    
    test_queries = [
        "Investigate this attack pattern and check for similar incidents",
        "Analyze the events and look for anomalies",
        "Deep dive into the forensics of this incident"
    ]
    
    results = []
    
    for query in test_queries:
        print_info(f"\nQuery: '{query}'")
        
        try:
            response = await client.post(
                f"{BASE_URL}/api/agents/orchestrate",
                json={
                    "query": query,
                    "incident_id": incident_id,
                    "context": {
                        "test": True,
                        "source": "e2e_test"
                    }
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("investigation_started"):
                    print_success(f"✅ Investigation started: {data.get('case_id')}")
                    print_info(f"   - Evidence count: {data.get('evidence_count')}")
                    print_info(f"   - Analysis type: {data.get('analysis_type')}")
                    
                    results.append({
                        "query": query[:50],
                        "success": True,
                        "case_id": data.get("case_id"),
                        "evidence_count": data.get("evidence_count")
                    })
                else:
                    print_info(f"ℹ️  Regular response (investigation not triggered)")
                    results.append({
                        "query": query[:50],
                        "success": False,
                        "reason": "Investigation not triggered"
                    })
            else:
                print_error(f"Request failed: {response.status_code}")
                results.append({
                    "query": query[:50],
                    "success": False,
                    "reason": f"HTTP {response.status_code}"
                })
                
        except Exception as e:
            print_error(f"Exception: {e}")
            results.append({
                "query": query[:50],
                "success": False,
                "reason": str(e)
            })
    
    return {
        "test_name": "Investigation Trigger",
        "results": results,
        "passed": sum(1 for r in results if r.get("success")) > 0
    }

async def test_workflow_sync(client: httpx.AsyncClient, incident_id: int) -> Dict[str, Any]:
    """Test 3: Verify workflow appears in incident data"""
    print_test(f"Workflow Sync Verification (Incident #{incident_id})")
    
    try:
        # First, create a workflow via chat
        print_info("Creating workflow via chat...")
        create_response = await client.post(
            f"{BASE_URL}/api/agents/orchestrate",
            json={
                "query": f"Block the IP address for incident {incident_id}",
                "incident_id": incident_id,
                "context": {"test": True}
            }
        )
        
        if create_response.status_code != 200:
            return {
                "test_name": "Workflow Sync",
                "passed": False,
                "reason": "Failed to create workflow"
            }
        
        data = create_response.json()
        
        if not data.get("workflow_created"):
            return {
                "test_name": "Workflow Sync",
                "passed": False,
                "reason": "Workflow not created from chat"
            }
        
        workflow_db_id = data.get("workflow_db_id")
        print_success(f"Workflow created: DB ID {workflow_db_id}")
        
        # Wait a bit for sync
        await asyncio.sleep(1)
        
        # Fetch incident workflows
        print_info("Fetching workflows for incident...")
        workflows_response = await client.get(f"{BASE_URL}/api/response/workflows")
        
        if workflows_response.status_code == 200:
            workflows = workflows_response.json()
            
            # Find our workflow
            our_workflow = None
            for workflow in workflows:
                if workflow.get("id") == workflow_db_id:
                    our_workflow = workflow
                    break
            
            if our_workflow:
                print_success(f"✅ Workflow found in incident workflows!")
                print_info(f"   - ID: {our_workflow.get('id')}")
                print_info(f"   - Name: {our_workflow.get('playbook_name')}")
                print_info(f"   - Status: {our_workflow.get('status')}")
                print_info(f"   - Steps: {our_workflow.get('total_steps')}")
                
                return {
                    "test_name": "Workflow Sync",
                    "passed": True,
                    "workflow": our_workflow
                }
            else:
                print_error(f"Workflow {workflow_db_id} not found in workflows list")
                return {
                    "test_name": "Workflow Sync",
                    "passed": False,
                    "reason": "Workflow not found in list"
                }
        else:
            print_error(f"Failed to fetch workflows: {workflows_response.status_code}")
            return {
                "test_name": "Workflow Sync",
                "passed": False,
                "reason": f"HTTP {workflows_response.status_code}"
            }
            
    except Exception as e:
        print_error(f"Exception: {e}")
        return {
            "test_name": "Workflow Sync",
            "passed": False,
            "reason": str(e)
        }

async def test_different_attack_types(client: httpx.AsyncClient) -> Dict[str, Any]:
    """Test 4: Test with different attack types"""
    print_test("Different Attack Types & Response Actions")
    
    attack_scenarios = [
        {
            "type": "SSH Brute Force",
            "query": "Block this SSH brute force attack and alert the team",
            "expected_keywords": ["block", "alert"]
        },
        {
            "type": "DDoS Attack",
            "query": "Deploy firewall rules to mitigate DDoS and capture network traffic",
            "expected_keywords": ["firewall", "capture"]
        },
        {
            "type": "Malware Detection",
            "query": "Isolate infected host, terminate suspicious processes, and run forensics",
            "expected_keywords": ["isolate", "terminate", "forensics"]
        },
        {
            "type": "Data Exfiltration",
            "query": "Block IP, revoke user sessions, encrypt sensitive data, and investigate",
            "expected_keywords": ["block", "revoke", "encrypt", "investigate"]
        }
    ]
    
    results = []
    
    # Get available incidents
    incidents = await get_incidents(client)
    if not incidents:
        return {
            "test_name": "Attack Type Testing",
            "passed": False,
            "reason": "No incidents available"
        }
    
    for i, scenario in enumerate(attack_scenarios):
        incident_id = incidents[i % len(incidents)]["id"]
        
        print_info(f"\nScenario: {scenario['type']}")
        print_info(f"Query: '{scenario['query']}'")
        
        try:
            response = await client.post(
                f"{BASE_URL}/api/agents/orchestrate",
                json={
                    "query": scenario["query"],
                    "incident_id": incident_id,
                    "context": {
                        "attack_type": scenario["type"],
                        "test": True
                    }
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                
                workflow_created = data.get("workflow_created", False)
                investigation_started = data.get("investigation_started", False)
                
                print_info(f"   - Workflow created: {workflow_created}")
                print_info(f"   - Investigation started: {investigation_started}")
                
                results.append({
                    "attack_type": scenario["type"],
                    "workflow_created": workflow_created,
                    "investigation_started": investigation_started,
                    "success": workflow_created or investigation_started
                })
                
                if workflow_created:
                    print_success(f"✅ Workflow: {data.get('workflow_id')}")
                if investigation_started:
                    print_success(f"✅ Investigation: {data.get('case_id')}")
            else:
                print_error(f"Request failed: {response.status_code}")
                results.append({
                    "attack_type": scenario["type"],
                    "success": False,
                    "error": f"HTTP {response.status_code}"
                })
                
        except Exception as e:
            print_error(f"Exception: {e}")
            results.append({
                "attack_type": scenario["type"],
                "success": False,
                "error": str(e)
            })
    
    return {
        "test_name": "Attack Type Testing",
        "results": results,
        "passed": sum(1 for r in results if r.get("success")) > 0
    }

async def run_all_tests():
    """Run all end-to-end tests"""
    print_header("END-TO-END INTEGRATION TEST SUITE")
    print_info(f"Testing: Chat → Workflow Integration")
    print_info(f"Backend: {BASE_URL}")
    print_info(f"Frontend: {FRONTEND_URL}")
    print_info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    async with httpx.AsyncClient(
        timeout=30.0,
        headers={"x-api-key": API_KEY, "Content-Type": "application/json"}
    ) as client:
        # Check service health
        print_header("SERVICE HEALTH CHECK")
        if not await check_service_health(client):
            print_error("Backend is not running! Please start the backend service.")
            return
        
        # Get incidents
        print_header("INCIDENT AVAILABILITY")
        incidents = await get_incidents(client)
        
        if not incidents:
            print_error("No incidents available for testing!")
            return
        
        # Use first incident for most tests
        test_incident_id = incidents[0]["id"]
        print_info(f"Using incident #{test_incident_id} for testing")
        print_info(f"Incident IP: {incidents[0].get('src_ip')}")
        print_info(f"Incident Reason: {incidents[0].get('reason')}")
        
        # Run tests
        test_results = []
        
        # Test 1: Workflow creation
        result1 = await test_workflow_creation_from_chat(client, test_incident_id)
        test_results.append(result1)
        
        # Test 2: Investigation trigger
        result2 = await test_investigation_trigger(client, test_incident_id)
        test_results.append(result2)
        
        # Test 3: Workflow sync
        result3 = await test_workflow_sync(client, test_incident_id)
        test_results.append(result3)
        
        # Test 4: Different attack types
        result4 = await test_different_attack_types(client)
        test_results.append(result4)
        
        # Summary
        print_header("TEST SUMMARY")
        
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results if r.get("passed"))
        
        for result in test_results:
            status = "✅ PASS" if result.get("passed") else "❌ FAIL"
            print(f"{status} - {result['test_name']}")
        
        print(f"\n{Color.BOLD}Total: {passed_tests}/{total_tests} tests passed{Color.END}")
        
        if passed_tests == total_tests:
            print_success(f"\n🎉 ALL TESTS PASSED! Integration is working correctly!")
        elif passed_tests > 0:
            print_info(f"\n⚠️  PARTIAL SUCCESS: {passed_tests}/{total_tests} tests passed")
        else:
            print_error(f"\n❌ ALL TESTS FAILED! Check the implementation")
        
        # Write results to file
        results_file = Path(__file__).parent / "e2e_test_results.json"
        with open(results_file, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "test_results": test_results
            }, f, indent=2)
        
        print_info(f"\nDetailed results saved to: {results_file}")

if __name__ == "__main__":
    asyncio.run(run_all_tests())

