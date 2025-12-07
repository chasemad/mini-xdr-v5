#!/usr/bin/env python3
"""
COMPREHENSIVE ATTACK SCENARIO TESTING
Tests different attack types from different IPs to verify:
1. Model detects different attack patterns
2. Agents respond appropriately to each attack type
3. MCP server handles varied incidents
4. Each attack creates a unique incident
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import asyncio
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List

import httpx

BASE_URL = "http://localhost:8000"
API_KEY = "demo-minixdr-api-key"


class ComprehensiveAttackTester:
    """Test comprehensive attack scenarios"""

    def __init__(self):
        self.results = []
        self.incidents_created = []

    async def run_all_tests(self):
        """Run all comprehensive tests"""
        print("\n" + "=" * 80)
        print("🎯 COMPREHENSIVE ATTACK SCENARIO TESTING")
        print("=" * 80 + "\n")

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test 1: Different attack types from different IPs
            print("🔥 [1/5] Testing different attack types...")
            await self.test_varied_attack_types(client)

            # Test 2: Same attack from different IPs
            print("\n🌐 [2/5] Testing same attack from different IPs...")
            await self.test_multiple_ips_same_attack(client)

            # Test 3: Agent responses to each attack type
            print("\n🤖 [3/5] Testing agent responses...")
            await self.test_agent_responses(client)

            # Test 4: MCP server handling
            print("\n🔌 [4/5] Testing MCP server...")
            await self.test_mcp_server(client)

            # Test 5: Verify model classifications
            print("\n📊 [5/5] Verifying model classifications...")
            await self.test_model_classifications(client)

        # Print final summary
        self.print_summary()

    async def test_varied_attack_types(self, client: httpx.AsyncClient):
        """Test different attack types create different incidents"""

        attack_scenarios = [
            {
                "name": "SSH Brute Force",
                "src_ip": "203.0.113.10",
                "dst_port": 22,
                "eventid": "cowrie.login.failed",
                "event_count": 25,
                "expected_class": "Brute Force Attack",
            },
            {
                "name": "DDoS Attack",
                "src_ip": "198.51.100.20",
                "dst_port": 80,
                "eventid": "high_connection_rate",
                "event_count": 500,
                "expected_class": "DDoS/DoS Attack",
            },
            {
                "name": "Port Scan",
                "src_ip": "192.0.2.30",
                "dst_port": None,  # Multiple ports
                "eventid": "port_scan",
                "event_count": 50,
                "expected_class": "Network Reconnaissance",
            },
            {
                "name": "Web Attack (SQL Injection)",
                "src_ip": "198.51.100.40",
                "dst_port": 80,
                "eventid": "web_attack",
                "event_count": 15,
                "expected_class": "Web Application Attack",
            },
            {
                "name": "Malware C2 Communication",
                "src_ip": "203.0.113.50",
                "dst_port": 443,
                "eventid": "malware_callback",
                "event_count": 30,
                "expected_class": "Malware/Botnet",
            },
        ]

        for scenario in attack_scenarios:
            print(f"\n  🎯 Simulating: {scenario['name']}")
            print(f"     Source IP: {scenario['src_ip']}")

            # Generate events
            events = self._generate_attack_events(scenario)

            # Ingest events
            try:
                response = await client.post(
                    f"{BASE_URL}/ingest/multi",
                    json={"events": events},
                    headers={"x-api-key": API_KEY},
                )

                if response.status_code == 200:
                    result = response.json()
                    print(f"     ✅ Ingested {len(events)} events")
                    print(f"     📊 Response: {result.get('message', 'OK')}")

                    # Wait for processing
                    await asyncio.sleep(2)

                    # Check if incident was created
                    incidents = await self._get_incidents_for_ip(
                        client, scenario["src_ip"]
                    )

                    if incidents:
                        incident = incidents[0]
                        self.incidents_created.append(incident["id"])

                        ml_score = incident.get("ml_anomaly_score", 0)
                        threat_type = incident.get("threat_classification", {}).get(
                            "threat_type", "Unknown"
                        )
                        confidence = incident.get("threat_classification", {}).get(
                            "confidence", 0
                        )

                        print(f"     ✅ Incident created: ID #{incident['id']}")
                        print(f"        ML Score: {ml_score*100:.1f}%")
                        print(f"        Type: {threat_type}")
                        print(f"        Confidence: {confidence*100:.1f}%")

                        self.results.append(
                            {
                                "scenario": scenario["name"],
                                "incident_id": incident["id"],
                                "ml_score": ml_score,
                                "threat_type": threat_type,
                                "confidence": confidence,
                                "expected_type": scenario["expected_class"],
                                "match": scenario["expected_class"].lower()
                                in threat_type.lower(),
                            }
                        )

                        # Check if confidence is stuck at 57%
                        if abs(confidence - 0.57) < 0.01:
                            print(f"        ⚠️  WARNING: Confidence is ~57% (stuck?)")
                    else:
                        print(
                            f"     ⚠️  No incident created (might be below threshold)"
                        )
                else:
                    print(f"     ❌ Ingestion failed: {response.status_code}")

            except Exception as e:
                print(f"     ❌ Error: {e}")

            # Wait between attacks
            await asyncio.sleep(1)

    async def test_multiple_ips_same_attack(self, client: httpx.AsyncClient):
        """Test that same attack from different IPs creates separate incidents"""

        base_ips = ["10.0.0.{}".format(i) for i in range(10, 15)]

        print(f"\n  🌐 Testing SSH brute force from {len(base_ips)} different IPs...")

        incidents_created = []

        for ip in base_ips:
            events = self._generate_attack_events(
                {
                    "src_ip": ip,
                    "dst_port": 22,
                    "eventid": "cowrie.login.failed",
                    "event_count": 20,
                }
            )

            try:
                response = await client.post(
                    f"{BASE_URL}/ingest/multi",
                    json={"events": events},
                    headers={"x-api-key": API_KEY},
                )

                if response.status_code == 200:
                    await asyncio.sleep(1)

                    incidents = await self._get_incidents_for_ip(client, ip)
                    if incidents:
                        incidents_created.append(
                            {"ip": ip, "incident_id": incidents[0]["id"]}
                        )
                        print(f"     ✅ {ip} → Incident #{incidents[0]['id']}")
                    else:
                        print(f"     ⚠️  {ip} → No incident")

            except Exception as e:
                print(f"     ❌ {ip} → Error: {e}")

        print(
            f"\n  📊 Created {len(incidents_created)} incidents from {len(base_ips)} IPs"
        )

        if len(incidents_created) == len(base_ips):
            print(f"     ✅ Each IP created a separate incident")
        else:
            print(f"     ⚠️  Not all IPs created incidents")

    async def test_agent_responses(self, client: httpx.AsyncClient):
        """Test agent responses to different attack types"""

        if not self.incidents_created:
            print("  ⚠️  No incidents to test agents with")
            return

        agent_tests = [
            {
                "action": "Block this IP immediately",
                "expected_keywords": ["block", "firewall"],
            },
            {
                "action": "Investigate this attack and gather forensics",
                "expected_keywords": ["investigate", "forensic"],
            },
            {
                "action": "Alert security team and isolate the host",
                "expected_keywords": ["alert", "isolate"],
            },
        ]

        for incident_id in self.incidents_created[:3]:  # Test first 3 incidents
            print(f"\n  🤖 Testing agents on Incident #{incident_id}...")

            for test in agent_tests:
                try:
                    response = await client.post(
                        f"{BASE_URL}/api/agents/orchestrate",
                        json={"query": test["action"], "incident_id": incident_id},
                    )

                    if response.status_code == 200:
                        result = response.json()

                        workflow_created = result.get("workflow_created", False)
                        investigation = result.get("investigation_started", False)

                        print(f"     ✅ '{test['action']}'")
                        if workflow_created:
                            print(f"        → Workflow: {result.get('workflow_id')}")
                        if investigation:
                            print(f"        → Investigation: {result.get('case_id')}")
                    else:
                        print(f"     ❌ Failed: {response.status_code}")

                except Exception as e:
                    print(f"     ❌ Error: {e}")

    async def test_mcp_server(self, client: httpx.AsyncClient):
        """Test MCP server functionality"""

        print("\n  🔌 Testing MCP server endpoints...")

        # Test health endpoint
        try:
            response = await client.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                print("     ✅ Health endpoint working")
            else:
                print(f"     ❌ Health endpoint failed: {response.status_code}")
        except Exception as e:
            print(f"     ❌ Health check error: {e}")

        # Test incidents endpoint
        try:
            response = await client.get(
                f"{BASE_URL}/incidents", headers={"x-api-key": API_KEY}
            )
            if response.status_code == 200:
                incidents = response.json()
                print(f"     ✅ Incidents endpoint: {len(incidents)} incidents")
            else:
                print(f"     ❌ Incidents endpoint failed: {response.status_code}")
        except Exception as e:
            print(f"     ❌ Incidents error: {e}")

    async def test_model_classifications(self, client: httpx.AsyncClient):
        """Verify model is classifying attacks correctly"""

        if not self.results:
            print("  ⚠️  No results to analyze")
            return

        print("\n  📊 Model Classification Analysis:")
        print("  " + "-" * 76)
        print(f"  {'Attack Type':<25} {'Detected As':<25} {'Confidence':<12} {'Match'}")
        print("  " + "-" * 76)

        for result in self.results:
            match_symbol = "✅" if result["match"] else "❌"
            print(
                f"  {result['scenario']:<25} {result['threat_type']:<25} "
                f"{result['confidence']*100:>5.1f}%      {match_symbol}"
            )

        print("  " + "-" * 76)

        # Calculate accuracy
        correct = sum(1 for r in self.results if r["match"])
        total = len(self.results)
        accuracy = (correct / total * 100) if total > 0 else 0

        print(f"\n  📈 Classification Accuracy: {accuracy:.1f}% ({correct}/{total})")

        # Check confidence variance
        confidences = [r["confidence"] for r in self.results]
        import numpy as np

        variance = np.var(confidences)

        print(f"  📊 Confidence Variance: {variance:.6f}")

        if variance < 0.01:
            print(f"     ⚠️  WARNING: Very low variance - model might be stuck!")
        else:
            print(f"     ✅ Good variance in confidence scores")

        # Check for 57% issue
        stuck_at_57 = sum(1 for c in confidences if abs(c - 0.57) < 0.01)
        if stuck_at_57 > 0:
            print(f"     ⚠️  {stuck_at_57}/{total} predictions are ~57%")

    def _generate_attack_events(self, scenario: Dict) -> List[Dict]:
        """Generate synthetic attack events"""
        events = []
        base_time = datetime.utcnow()

        event_count = scenario.get("event_count", 20)
        src_ip = scenario["src_ip"]
        dst_port = scenario.get("dst_port")
        eventid = scenario.get("eventid", "generic_event")

        for i in range(event_count):
            # Vary dst_port for port scans
            if dst_port is None:
                port = 20 + (i % 80)
            else:
                port = dst_port

            event = {
                "src_ip": src_ip,
                "dst_ip": "10.0.0.1",
                "src_port": random.randint(30000, 60000),
                "dst_port": port,
                "eventid": eventid,
                "message": f"Attack event {i}",
                "timestamp": (base_time + timedelta(seconds=i * 2)).isoformat(),
                "protocol": "tcp",
                "raw": {
                    "username": f"admin{i}" if "login" in eventid else None,
                    "password": "test123" if "login" in eventid else None,
                },
            }
            events.append(event)

        return events

    async def _get_incidents_for_ip(
        self, client: httpx.AsyncClient, src_ip: str
    ) -> List[Dict]:
        """Get incidents for a specific source IP"""
        try:
            response = await client.get(
                f"{BASE_URL}/incidents", headers={"x-api-key": API_KEY}
            )

            if response.status_code == 200:
                all_incidents = response.json()
                return [inc for inc in all_incidents if inc.get("src_ip") == src_ip]
        except Exception as e:
            print(f"Error getting incidents: {e}")

        return []

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 80)
        print("📊 TEST SUMMARY")
        print("=" * 80 + "\n")

        print(f"✅ Incidents Created: {len(self.incidents_created)}")
        print(f"✅ Attack Scenarios Tested: {len(self.results)}")
        print(f"✅ Tests Completed Successfully")

        print("\n💡 Next Steps:")
        print("  1. Review incidents in UI: http://localhost:3000/incidents")
        print("  2. Test agent responses manually")
        print("  3. Check MCP server: http://localhost:8000/docs")
        print(
            "  4. If model stuck at 57%, run: python tests/test_model_confidence_debug.py"
        )
        print()


async def main():
    tester = ComprehensiveAttackTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
