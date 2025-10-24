#!/usr/bin/env python3
"""
Comprehensive Workflow and Agent Action Testing Script
Tests all workflows, agent actions, and automated responses for Azure T-Pot honeypot
"""

import asyncio
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import subprocess

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

try:
    import requests
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy import select
    from app.models import WorkflowTrigger, Incident, Event, Action
    from app.config import settings
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the mini-xdr directory with venv activated")
    sys.exit(1)

# Color codes
GREEN = '\033[0;32m'
BLUE = '\033[0;34m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
CYAN = '\033[0;36m'
MAGENTA = '\033[0;35m'
NC = '\033[0m'
BOLD = '\033[1m'

# Azure honeypot configuration
HONEYPOT_IP = "74.235.242.205"
HONEYPOT_USER = "azureuser"
HONEYPOT_SSH_KEY = "~/.ssh/mini-xdr-tpot-azure"
HONEYPOT_SSH_PORT = 64295
BACKEND_URL = "http://localhost:8000"

# Test results
test_results = {
    "timestamp": datetime.utcnow().isoformat(),
    "workflows_tested": [],
    "actions_tested": [],
    "agents_tested": [],
    "summary": {
        "total_tests": 0,
        "passed": 0,
        "failed": 0,
        "warnings": 0
    }
}


def log(msg, color=BLUE):
    print(f"{color}[INFO]{NC} {msg}")


def success(msg):
    print(f"{GREEN}✅ {msg}{NC}")
    test_results["summary"]["passed"] += 1


def warning(msg):
    print(f"{YELLOW}⚠️  {msg}{NC}")
    test_results["summary"]["warnings"] += 1


def error(msg):
    print(f"{RED}❌ {msg}{NC}")
    test_results["summary"]["failed"] += 1


def section(title):
    print(f"\n{BOLD}{CYAN}{'='*80}{NC}")
    print(f"{BOLD}{CYAN}{title.center(80)}{NC}")
    print(f"{BOLD}{CYAN}{'='*80}{NC}\n")


class WorkflowTester:
    """Test all workflows and agent actions"""
    
    def __init__(self):
        self.session = None
        self.test_results = []
        
    async def initialize(self):
        """Initialize database connection"""
        engine = create_async_engine(settings.database_url, echo=False)
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        self.session = async_session()
        
    async def cleanup(self):
        """Cleanup database connection"""
        if self.session:
            await self.session.close()
    
    async def test_workflow_triggers(self):
        """Test all workflow triggers"""
        section("TESTING WORKFLOW TRIGGERS")
        
        # Get all enabled workflow triggers
        result = await self.session.execute(
            select(WorkflowTrigger).where(WorkflowTrigger.enabled == True)
        )
        triggers = result.scalars().all()
        
        log(f"Found {len(triggers)} enabled workflow triggers")
        
        for trigger in triggers:
            test_results["summary"]["total_tests"] += 1
            
            try:
                # Test trigger configuration
                log(f"Testing trigger: {trigger.name}")
                
                # Verify trigger has required fields
                checks = {
                    "has_name": bool(trigger.name),
                    "has_conditions": bool(trigger.conditions),
                    "has_workflow_steps": bool(trigger.workflow_steps),
                    "enabled": trigger.enabled,
                }
                
                if all(checks.values()):
                    success(f"Trigger '{trigger.name}' - Configuration valid")
                    test_results["workflows_tested"].append({
                        "name": trigger.name,
                        "status": "pass",
                        "auto_execute": trigger.auto_execute,
                        "priority": trigger.priority,
                        "conditions": trigger.conditions,
                        "workflow_steps": len(trigger.workflow_steps)
                    })
                else:
                    failed_checks = [k for k, v in checks.items() if not v]
                    error(f"Trigger '{trigger.name}' - Missing: {', '.join(failed_checks)}")
                    test_results["workflows_tested"].append({
                        "name": trigger.name,
                        "status": "fail",
                        "reason": f"Missing: {', '.join(failed_checks)}"
                    })
                    
            except Exception as e:
                error(f"Error testing trigger '{trigger.name}': {e}")
                test_results["workflows_tested"].append({
                    "name": trigger.name,
                    "status": "error",
                    "reason": str(e)
                })
    
    async def test_agent_actions(self):
        """Test individual agent actions"""
        section("TESTING AGENT ACTIONS")
        
        # Define test actions for each agent
        agent_tests = {
            "containment": [
                {"action": "block_ip", "description": "Block IP address"},
                {"action": "isolate_host", "description": "Isolate compromised host"},
                {"action": "deploy_firewall", "description": "Deploy firewall rules"},
            ],
            "forensics": [
                {"action": "collect_evidence", "description": "Collect forensic evidence"},
                {"action": "analyze_malware", "description": "Analyze malware samples"},
                {"action": "capture_traffic", "description": "Capture network traffic"},
            ],
            "attribution": [
                {"action": "profile_threat_actor", "description": "Profile threat actor"},
                {"action": "identify_campaign", "description": "Identify attack campaign"},
                {"action": "track_c2", "description": "Track C2 infrastructure"},
            ],
            "threat_hunting": [
                {"action": "hunt_similar_attacks", "description": "Hunt for similar attacks"},
                {"action": "analyze_patterns", "description": "Analyze behavioral patterns"},
                {"action": "proactive_search", "description": "Proactive threat search"},
            ],
            "deception": [
                {"action": "deploy_honeypot", "description": "Deploy decoy honeypot"},
                {"action": "track_attacker", "description": "Track attacker behavior"},
            ]
        }
        
        for agent_name, actions in agent_tests.items():
            log(f"\nTesting {agent_name.upper()} agent actions...")
            
            for action_test in actions:
                test_results["summary"]["total_tests"] += 1
                
                try:
                    # Test action availability via API
                    response = requests.post(
                        f"{BACKEND_URL}/api/agents/orchestrate",
                        json={
                            "query": f"Test {action_test['description']}",
                            "agent_type": agent_name,
                            "context": {"test_mode": True}
                        },
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        success(f"{agent_name}: {action_test['description']} - Available")
                        test_results["actions_tested"].append({
                            "agent": agent_name,
                            "action": action_test["action"],
                            "status": "pass",
                            "description": action_test["description"]
                        })
                    else:
                        warning(f"{agent_name}: {action_test['description']} - Response: {response.status_code}")
                        test_results["actions_tested"].append({
                            "agent": agent_name,
                            "action": action_test["action"],
                            "status": "warning",
                            "reason": f"HTTP {response.status_code}"
                        })
                        
                except Exception as e:
                    error(f"{agent_name}: {action_test['description']} - Error: {e}")
                    test_results["actions_tested"].append({
                        "agent": agent_name,
                        "action": action_test["action"],
                        "status": "error",
                        "reason": str(e)
                    })
    
    async def test_containment_on_honeypot(self):
        """Test containment actions on Azure honeypot"""
        section("TESTING AZURE HONEYPOT CONTAINMENT")
        
        test_results["summary"]["total_tests"] += 1
        
        # Test SSH connection
        log("Testing SSH connection to Azure T-Pot...")
        try:
            ssh_cmd = [
                "ssh",
                "-i", HONEYPOT_SSH_KEY.replace("~", str(Path.home())),
                "-p", str(HONEYPOT_SSH_PORT),
                "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=10",
                f"{HONEYPOT_USER}@{HONEYPOT_IP}",
                "echo 'SSH connection successful'"
            ]
            
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                success("SSH connection to Azure T-Pot successful")
                test_results["agents_tested"].append({
                    "component": "azure_ssh_access",
                    "status": "pass",
                    "details": "SSH connection verified"
                })
            else:
                error(f"SSH connection failed: {result.stderr}")
                test_results["agents_tested"].append({
                    "component": "azure_ssh_access",
                    "status": "fail",
                    "reason": result.stderr
                })
                return
                
        except Exception as e:
            error(f"SSH test error: {e}")
            test_results["agents_tested"].append({
                "component": "azure_ssh_access",
                "status": "error",
                "reason": str(e)
            })
            return
        
        # Test iptables access (read-only)
        test_results["summary"]["total_tests"] += 1
        log("Testing iptables access on Azure T-Pot...")
        try:
            iptables_cmd = [
                "ssh",
                "-i", HONEYPOT_SSH_KEY.replace("~", str(Path.home())),
                "-p", str(HONEYPOT_SSH_PORT),
                "-o", "StrictHostKeyChecking=no",
                f"{HONEYPOT_USER}@{HONEYPOT_IP}",
                "sudo iptables -L INPUT -n | head -10"
            ]
            
            result = subprocess.run(iptables_cmd, capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                success("Iptables access verified on Azure T-Pot")
                test_results["agents_tested"].append({
                    "component": "iptables_access",
                    "status": "pass",
                    "details": "Can read iptables rules"
                })
            else:
                warning(f"Iptables access limited: {result.stderr}")
                test_results["agents_tested"].append({
                    "component": "iptables_access",
                    "status": "warning",
                    "reason": "Limited access"
                })
                
        except Exception as e:
            error(f"Iptables test error: {e}")
            test_results["agents_tested"].append({
                "component": "iptables_access",
                "status": "error",
                "reason": str(e)
            })
    
    async def test_incident_workflow_execution(self):
        """Test workflow execution on real incidents"""
        section("TESTING INCIDENT WORKFLOW EXECUTION")
        
        # Get recent incidents
        result = await self.session.execute(
            select(Incident).order_by(Incident.created_at.desc()).limit(5)
        )
        incidents = result.scalars().all()
        
        if not incidents:
            warning("No recent incidents found to test workflow execution")
            return
        
        log(f"Testing workflow execution on {len(incidents)} recent incidents...")
        
        for incident in incidents:
            test_results["summary"]["total_tests"] += 1
            
            try:
                # Check if any workflows were triggered
                result = await self.session.execute(
                    select(Action).where(Action.incident_id == incident.id)
                )
                actions = result.scalars().all()
                
                if actions:
                    success(f"Incident #{incident.id}: {len(actions)} actions executed")
                    
                    for action in actions:
                        log(f"  - Action: {action.action_type} (Status: {action.status})")
                else:
                    warning(f"Incident #{incident.id}: No actions executed")
                    
            except Exception as e:
                error(f"Error checking incident #{incident.id}: {e}")
    
    async def test_real_attack_simulation(self):
        """Simulate a real attack to test end-to-end workflow"""
        section("TESTING END-TO-END ATTACK SIMULATION")
        
        test_results["summary"]["total_tests"] += 1
        
        log("Simulating SSH brute force attack...")
        
        try:
            # Generate fake SSH brute force events
            test_ip = "192.168.100.99"
            events = []
            
            for i in range(6):  # Trigger threshold
                event_data = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "eventid": "cowrie.login.failed",
                    "src_ip": test_ip,
                    "username": f"admin{i}",
                    "password": f"password{i}",
                    "message": f"Failed password for admin{i}",
                    "source": "honeypot"
                }
                events.append(event_data)
            
            # Send events to backend
            response = requests.post(
                f"{BACKEND_URL}/ingest/multi",
                json={"events": events, "source": "test_simulation"},
                timeout=10
            )
            
            if response.status_code == 200:
                log("Events ingested successfully")
                
                # Wait for processing
                await asyncio.sleep(2)
                
                # Check if incident was created
                result = await self.session.execute(
                    select(Incident).where(Incident.src_ip == test_ip).order_by(Incident.created_at.desc())
                )
                incident = result.scalars().first()
                
                if incident:
                    success(f"Incident created: #{incident.id}")
                    
                    # Check if workflows were triggered
                    result = await self.session.execute(
                        select(Action).where(Action.incident_id == incident.id)
                    )
                    actions = result.scalars().all()
                    
                    if actions:
                        success(f"Workflows executed: {len(actions)} actions")
                        for action in actions:
                            log(f"  - {action.action_type}: {action.status}")
                    else:
                        warning("No workflows were automatically triggered")
                else:
                    warning("No incident was created from simulated attack")
            else:
                error(f"Event ingestion failed: {response.status_code}")
                
        except Exception as e:
            error(f"Attack simulation error: {e}")
    
    async def run_all_tests(self):
        """Run all tests"""
        try:
            await self.initialize()
            
            await self.test_workflow_triggers()
            await self.test_agent_actions()
            await self.test_containment_on_honeypot()
            await self.test_incident_workflow_execution()
            await self.test_real_attack_simulation()
            
        finally:
            await self.cleanup()


async def main():
    """Main test execution"""
    section("COMPREHENSIVE WORKFLOW AND AGENT ACTION TESTING")
    
    log(f"Backend URL: {BACKEND_URL}")
    log(f"Azure Honeypot: {HONEYPOT_IP}:{HONEYPOT_SSH_PORT}")
    log(f"Test Time: {datetime.utcnow().isoformat()}")
    
    # Check backend health
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            success("Backend is healthy")
        else:
            error("Backend health check failed")
            return 1
    except Exception as e:
        error(f"Cannot connect to backend: {e}")
        return 1
    
    # Run all tests
    tester = WorkflowTester()
    await tester.run_all_tests()
    
    # Print summary
    section("TEST SUMMARY")
    
    print(f"{BOLD}Total Tests:{NC} {test_results['summary']['total_tests']}")
    print(f"{GREEN}Passed:{NC} {test_results['summary']['passed']}")
    print(f"{YELLOW}Warnings:{NC} {test_results['summary']['warnings']}")
    print(f"{RED}Failed:{NC} {test_results['summary']['failed']}")
    
    # Calculate pass rate
    if test_results['summary']['total_tests'] > 0:
        pass_rate = (test_results['summary']['passed'] / test_results['summary']['total_tests']) * 100
        print(f"\n{BOLD}Pass Rate:{NC} {pass_rate:.1f}%")
    
    # Save detailed results
    output_file = f"workflow_test_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    success(f"Detailed results saved to: {output_file}")
    
    print(f"\n{CYAN}{'='*80}{NC}\n")
    
    return 0 if test_results['summary']['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))


