"""
End-to-End Workflow Integration Tests
Tests the complete flow: NLP → API → Database → UI Display
"""
import asyncio
import httpx
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.app.db import AsyncSessionLocal
from backend.app.models import WorkflowTrigger, NLPWorkflowSuggestion, Incident
from sqlalchemy import select
import json


API_BASE = "http://localhost:8000"
API_KEY = "dev-key-12345"


class E2EWorkflowTester:
    """End-to-end workflow integration tester"""

    def __init__(self):
        self.db = None
        self.results = {
            "api_tests": [],
            "database_tests": [],
            "integration_tests": []
        }

    async def setup(self):
        """Setup test environment"""
        self.db = AsyncSessionLocal()
        print("✓ Database connection established")

    async def teardown(self):
        """Cleanup"""
        if self.db:
            await self.db.close()

    async def test_nlp_parse_api(self):
        """Test NLP parsing via API endpoint"""
        print("\n" + "="*80)
        print("TEST 1: NLP Parsing API Endpoint")
        print("="*80)

        test_prompts = [
            "Block IP 192.168.1.100 and isolate the host",
            "Investigate malware infection and alert SOC",
            "Whenever brute force is detected, automatically block the IP",
            "Show me incident statistics for last week",
        ]

        async with httpx.AsyncClient() as client:
            for i, prompt in enumerate(test_prompts, 1):
                print(f"\n[Test 1.{i}] Prompt: \"{prompt}\"")

                try:
                    response = await client.post(
                        f"{API_BASE}/api/nlp-suggestions/parse",
                        json={"prompt": prompt},
                        headers={"X-API-Key": API_KEY},
                        timeout=30.0
                    )

                    if response.status_code == 200:
                        data = response.json()
                        print(f"  ✓ API call successful")
                        print(f"    - Suggestion ID: {data['id']}")
                        print(f"    - Request Type: {data['request_type']}")
                        print(f"    - Priority: {data['priority']}")
                        print(f"    - Confidence: {data['confidence']:.2f}")
                        print(f"    - Actions: {len(data['workflow_steps'])} steps")

                        self.results["api_tests"].append({
                            "test": f"parse_nlp_{i}",
                            "passed": True,
                            "suggestion_id": data['id']
                        })
                    else:
                        print(f"  ✗ API call failed: {response.status_code}")
                        self.results["api_tests"].append({
                            "test": f"parse_nlp_{i}",
                            "passed": False,
                            "error": f"HTTP {response.status_code}"
                        })

                except Exception as e:
                    print(f"  ✗ Error: {str(e)}")
                    self.results["api_tests"].append({
                        "test": f"parse_nlp_{i}",
                        "passed": False,
                        "error": str(e)
                    })

    async def test_suggestion_storage(self):
        """Test that suggestions are properly stored in database"""
        print("\n" + "="*80)
        print("TEST 2: Database Suggestion Storage")
        print("="*80)

        try:
            # Query all suggestions
            result = await self.db.execute(select(NLPWorkflowSuggestion))
            suggestions = result.scalars().all()

            print(f"\n  Found {len(suggestions)} suggestions in database")

            for i, suggestion in enumerate(suggestions[:5], 1):  # Show first 5
                print(f"\n  Suggestion #{i}:")
                print(f"    - ID: {suggestion.id}")
                print(f"    - Prompt: \"{suggestion.prompt[:60]}...\"")
                print(f"    - Type: {suggestion.request_type}")
                print(f"    - Status: {suggestion.status}")
                print(f"    - Steps: {len(suggestion.workflow_steps)}")

            self.results["database_tests"].append({
                "test": "suggestion_storage",
                "passed": len(suggestions) > 0,
                "count": len(suggestions)
            })

            return suggestions

        except Exception as e:
            print(f"  ✗ Database query failed: {str(e)}")
            self.results["database_tests"].append({
                "test": "suggestion_storage",
                "passed": False,
                "error": str(e)
            })
            return []

    async def test_trigger_creation_api(self, suggestion_id: int):
        """Test trigger creation from suggestion via API"""
        print("\n" + "="*80)
        print(f"TEST 3: Trigger Creation from Suggestion #{suggestion_id}")
        print("="*80)

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{API_BASE}/api/nlp-suggestions/{suggestion_id}/approve",
                    json={
                        "trigger_name": f"E2E_Test_Trigger_{suggestion_id}",
                        "trigger_description": "Auto-generated for E2E testing",
                        "auto_execute": False,
                        "category": "e2e_test",
                        "owner": "test_system"
                    },
                    headers={"X-API-Key": API_KEY},
                    timeout=30.0
                )

                if response.status_code == 200:
                    data = response.json()
                    print(f"  ✓ Trigger created successfully")
                    print(f"    - Trigger ID: {data['trigger_id']}")
                    print(f"    - Message: {data['message']}")

                    self.results["integration_tests"].append({
                        "test": f"trigger_creation_{suggestion_id}",
                        "passed": True,
                        "trigger_id": data['trigger_id']
                    })

                    return data['trigger_id']
                else:
                    print(f"  ✗ API call failed: {response.status_code}")
                    print(f"    - Response: {response.text}")
                    self.results["integration_tests"].append({
                        "test": f"trigger_creation_{suggestion_id}",
                        "passed": False,
                        "error": f"HTTP {response.status_code}"
                    })
                    return None

            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
                self.results["integration_tests"].append({
                    "test": f"trigger_creation_{suggestion_id}",
                    "passed": False,
                    "error": str(e)
                })
                return None

    async def test_trigger_list_api(self):
        """Test retrieving trigger list via API"""
        print("\n" + "="*80)
        print("TEST 4: Trigger List API")
        print("="*80)

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{API_BASE}/api/triggers/",
                    headers={"X-API-Key": API_KEY},
                    timeout=30.0
                )

                if response.status_code == 200:
                    triggers = response.json()
                    print(f"  ✓ Retrieved {len(triggers)} triggers")

                    for i, trigger in enumerate(triggers[:5], 1):
                        print(f"\n  Trigger #{i}:")
                        print(f"    - ID: {trigger['id']}")
                        print(f"    - Name: {trigger['name']}")
                        print(f"    - Source: {trigger['source']}")
                        print(f"    - Status: {trigger['status']}")
                        print(f"    - Request Type: {trigger.get('request_type', 'N/A')}")
                        if trigger.get('source_prompt'):
                            print(f"    - Original Prompt: \"{trigger['source_prompt'][:60]}...\"")

                    self.results["integration_tests"].append({
                        "test": "trigger_list",
                        "passed": True,
                        "count": len(triggers)
                    })

                    return triggers
                else:
                    print(f"  ✗ API call failed: {response.status_code}")
                    self.results["integration_tests"].append({
                        "test": "trigger_list",
                        "passed": False,
                        "error": f"HTTP {response.status_code}"
                    })
                    return []

            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
                self.results["integration_tests"].append({
                    "test": "trigger_list",
                    "passed": False,
                    "error": str(e)
                })
                return []

    async def test_trigger_simulation(self, trigger_id: int):
        """Test trigger simulation endpoint"""
        print("\n" + "="*80)
        print(f"TEST 5: Trigger Simulation (ID: {trigger_id})")
        print("="*80)

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{API_BASE}/api/triggers/{trigger_id}/simulate",
                    json={},
                    headers={"X-API-Key": API_KEY},
                    timeout=30.0
                )

                if response.status_code == 200:
                    data = response.json()
                    print(f"  ✓ Simulation successful")
                    print(f"    - Would Execute: {data['would_execute']}")
                    print(f"    - Workflow Steps: {len(data['workflow_steps'])}")
                    print(f"    - Estimated Duration: {data['estimated_duration_seconds']}s")
                    print(f"    - Safety Checks:")
                    for check, value in data['safety_checks'].items():
                        print(f"      • {check}: {value}")

                    self.results["integration_tests"].append({
                        "test": f"simulation_{trigger_id}",
                        "passed": True
                    })
                else:
                    print(f"  ✗ API call failed: {response.status_code}")
                    self.results["integration_tests"].append({
                        "test": f"simulation_{trigger_id}",
                        "passed": False,
                        "error": f"HTTP {response.status_code}"
                    })

            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
                self.results["integration_tests"].append({
                    "test": f"simulation_{trigger_id}",
                    "passed": False,
                    "error": str(e)
                })

    async def test_bulk_operations(self, trigger_ids: list):
        """Test bulk trigger operations"""
        print("\n" + "="*80)
        print("TEST 6: Bulk Trigger Operations")
        print("="*80)

        async with httpx.AsyncClient() as client:
            operations = ["pause", "resume"]

            for op in operations:
                print(f"\n  Testing bulk {op}...")
                try:
                    response = await client.post(
                        f"{API_BASE}/api/triggers/bulk/{op}",
                        json=trigger_ids[:2],  # Test with first 2 triggers
                        headers={"X-API-Key": API_KEY},
                        timeout=30.0
                    )

                    if response.status_code == 200:
                        data = response.json()
                        count_key = f"{op}d_count" if op == "pause" else f"{op}d_count"
                        print(f"    ✓ Bulk {op} successful: {data.get(count_key, data)} triggers")
                        self.results["integration_tests"].append({
                            "test": f"bulk_{op}",
                            "passed": True
                        })
                    else:
                        print(f"    ✗ Bulk {op} failed: {response.status_code}")
                        self.results["integration_tests"].append({
                            "test": f"bulk_{op}",
                            "passed": False,
                            "error": f"HTTP {response.status_code}"
                        })

                except Exception as e:
                    print(f"    ✗ Error: {str(e)}")
                    self.results["integration_tests"].append({
                        "test": f"bulk_{op}",
                        "passed": False,
                        "error": str(e)
                    })

    async def test_stats_api(self):
        """Test statistics endpoints"""
        print("\n" + "="*80)
        print("TEST 7: Statistics APIs")
        print("="*80)

        async with httpx.AsyncClient() as client:
            endpoints = [
                ("/api/triggers/stats/summary", "Trigger Stats"),
                ("/api/nlp-suggestions/stats", "Suggestion Stats")
            ]

            for endpoint, name in endpoints:
                print(f"\n  Testing {name}...")
                try:
                    response = await client.get(
                        f"{API_BASE}{endpoint}",
                        headers={"X-API-Key": API_KEY},
                        timeout=30.0
                    )

                    if response.status_code == 200:
                        data = response.json()
                        print(f"    ✓ {name} retrieved successfully")
                        for key, value in data.items():
                            if isinstance(value, dict):
                                print(f"      • {key}:")
                                for k, v in value.items():
                                    print(f"          - {k}: {v}")
                            else:
                                print(f"      • {key}: {value}")

                        self.results["integration_tests"].append({
                            "test": f"stats_{name}",
                            "passed": True
                        })
                    else:
                        print(f"    ✗ {name} failed: {response.status_code}")
                        self.results["integration_tests"].append({
                            "test": f"stats_{name}",
                            "passed": False,
                            "error": f"HTTP {response.status_code}"
                        })

                except Exception as e:
                    print(f"    ✗ Error: {str(e)}")
                    self.results["integration_tests"].append({
                        "test": f"stats_{name}",
                        "passed": False,
                        "error": str(e)
                    })

    async def run_all_tests(self):
        """Run all E2E tests"""
        print("="*80)
        print("END-TO-END WORKFLOW INTEGRATION TESTS")
        print("="*80)
        print("\nThis test suite validates:")
        print("  ✓ NLP parsing via API")
        print("  ✓ Suggestion storage in database")
        print("  ✓ Trigger creation from suggestions")
        print("  ✓ Trigger retrieval and display")
        print("  ✓ Bulk operations")
        print("  ✓ Statistics endpoints")
        print()

        await self.setup()

        # Run tests
        await self.test_nlp_parse_api()
        suggestions = await self.test_suggestion_storage()

        if suggestions:
            # Test trigger creation with first suggestion
            trigger_id = await self.test_trigger_creation_api(suggestions[0].id)

        # Test trigger list
        triggers = await self.test_trigger_list_api()

        if triggers:
            # Test simulation
            await self.test_trigger_simulation(triggers[0]['id'])

            # Test bulk operations
            trigger_ids = [t['id'] for t in triggers]
            if trigger_ids:
                await self.test_bulk_operations(trigger_ids)

        # Test stats
        await self.test_stats_api()

        await self.teardown()

        # Print final results
        print("\n" + "="*80)
        print("E2E TEST RESULTS SUMMARY")
        print("="*80)

        api_passed = sum(1 for t in self.results["api_tests"] if t.get("passed"))
        api_total = len(self.results["api_tests"])
        print(f"\nAPI Tests: {api_passed}/{api_total} passed")

        db_passed = sum(1 for t in self.results["database_tests"] if t.get("passed"))
        db_total = len(self.results["database_tests"])
        print(f"Database Tests: {db_passed}/{db_total} passed")

        int_passed = sum(1 for t in self.results["integration_tests"] if t.get("passed"))
        int_total = len(self.results["integration_tests"])
        print(f"Integration Tests: {int_passed}/{int_total} passed")

        total_passed = api_passed + db_passed + int_passed
        total_tests = api_total + db_total + int_total
        print(f"\nOVERALL: {total_passed}/{total_tests} passed ({total_passed/total_tests*100:.1f}%)")

        # Check for failures
        failed_tests = []
        for category in ["api_tests", "database_tests", "integration_tests"]:
            for test in self.results[category]:
                if not test.get("passed"):
                    failed_tests.append(f"{category}: {test['test']}")

        if failed_tests:
            print(f"\n⚠ {len(failed_tests)} Failed Tests:")
            for test in failed_tests:
                print(f"  • {test}")
        else:
            print("\n✓ ALL TESTS PASSED!")

        print("\n" + "="*80)

        return total_passed == total_tests


async def main():
    """Main test runner"""
    print("\nSTARTING E2E INTEGRATION TESTS")
    print("Make sure the backend is running on http://localhost:8000\n")

    tester = E2EWorkflowTester()

    try:
        all_passed = await tester.run_all_tests()
        exit_code = 0 if all_passed else 1
    except Exception as e:
        print(f"\n✗ CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
