#!/usr/bin/env python3
"""
Trigger Real AI Agent Analysis on Test Incidents
Simulates the full AI/ML analysis pipeline as if these were real detections
"""

import asyncio
import json
import sys
from datetime import datetime

import httpx

# Backend API configuration
API_BASE = (
    "http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com"
)
API_KEY = "demo-minixdr-api-key"

# Test incidents to analyze
INCIDENT_IDS = [1, 2, 3, 4, 5]


async def trigger_ai_analysis(incident_id: int):
    """Trigger AI analysis for an incident"""
    print(f"\n🤖 Triggering AI agent orchestration for Incident #{incident_id}...")

    url = f"{API_BASE}/api/agents/orchestrate"
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                url,
                headers=headers,
                json={
                    "incident_id": incident_id,
                    "agents": ["attribution", "containment", "forensics", "deception"],
                    "collaborative": True,
                },
            )

            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ Orchestration complete for Incident #{incident_id}")
                print(f"     Agents involved: {len(data.get('agent_results', {}))}")
                print(
                    f"     Consensus decision: {data.get('consensus_decision', {}).get('action', 'N/A')}"
                )
                return data
            else:
                print(f"  ⚠️  Analysis returned status {response.status_code}")
                print(f"     Response: {response.text[:200]}")
                return None

        except Exception as e:
            print(f"  ❌ Failed to analyze Incident #{incident_id}: {e}")
            return None


async def trigger_soc_actions(incident_id: int):
    """Trigger SOC analyst actions to demonstrate the workflow"""
    print(f"\n⚡ Triggering SOC actions for Incident #{incident_id}...")

    # Different actions for different incidents
    actions_map = {
        1: ["honeypot-profile-attacker", "deep-dive"],  # SSH brute force
        2: ["deep-dive", "alert-analysts"],  # SQL injection
        3: ["create-case", "deep-dive"],  # Ransomware (already contained)
        4: [],  # Port scan (dismissed, no action needed)
        5: ["honeypot-profile-attacker", "alert-analysts"],  # Credential stuffing
    }

    actions = actions_map.get(incident_id, [])

    async with httpx.AsyncClient(timeout=60.0) as client:
        for action in actions:
            try:
                url = f"{API_BASE}/api/incidents/{incident_id}/actions/{action}"
                headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

                response = await client.post(url, headers=headers, json={})

                if response.status_code == 200:
                    data = response.json()
                    print(f"  ✅ Action '{action}' executed")
                    if "message" in data:
                        print(f"     {data['message']}")
                else:
                    print(f"  ⚠️  Action '{action}' returned {response.status_code}")

            except Exception as e:
                print(f"  ⚠️  Action '{action}' failed: {e}")

            # Small delay between actions
            await asyncio.sleep(1)


async def verify_incident_accessible(incident_id: int):
    """Verify incident is accessible via API"""
    url = f"{API_BASE}/api/incidents/{incident_id}"
    headers = {"X-API-Key": API_KEY}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers)
            if response.status_code == 200:
                return True
            else:
                print(
                    f"  ⚠️  Incident #{incident_id} returned status {response.status_code}"
                )
                return False
        except Exception as e:
            print(f"  ❌ Failed to access incident #{incident_id}: {e}")
            return False


async def main():
    print("=" * 70)
    print("🤖 TRIGGER AI AGENT ANALYSIS ON TEST INCIDENTS")
    print("=" * 70)
    print(f"\n📍 API: {API_BASE}")
    print(f"🎯 Incidents: {INCIDENT_IDS}")
    print("")

    # First verify all incidents are accessible
    print("\n📊 Verifying incidents...")
    accessible = []
    for inc_id in INCIDENT_IDS:
        if await verify_incident_accessible(inc_id):
            print(f"  ✅ Incident #{inc_id} accessible")
            accessible.append(inc_id)
        else:
            print(f"  ❌ Incident #{inc_id} NOT accessible")

    if not accessible:
        print("\n❌ No incidents accessible. Exiting.")
        return

    print(f"\n✅ Found {len(accessible)} accessible incidents")

    # Trigger AI orchestration for each incident
    print("\n" + "=" * 70)
    print("🤖 TRIGGERING AI AGENT ORCHESTRATION")
    print("=" * 70)

    results = []
    for inc_id in accessible:
        result = await trigger_ai_analysis(inc_id)
        if result:
            results.append((inc_id, result))
        await asyncio.sleep(2)  # Delay between analyses

    # Trigger SOC actions
    print("\n" + "=" * 70)
    print("⚡ EXECUTING SOC ANALYST ACTIONS")
    print("=" * 70)

    for inc_id in accessible:
        await trigger_soc_actions(inc_id)
        await asyncio.sleep(2)

    # Summary
    print("\n" + "=" * 70)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"\n✅ Analyzed {len(results)} incidents")
    print(f"⚡ Triggered SOC actions for {len(accessible)} incidents")
    print("\n🌐 View results in the UI:")
    print(f"  {API_BASE}")
    print("\n💡 The incidents now have:")
    print("  • AI agent analysis")
    print("  • Orchestrator recommendations")
    print("  • SOC analyst actions")
    print("  • Complete action timelines")
    print("")


if __name__ == "__main__":
    asyncio.run(main())
