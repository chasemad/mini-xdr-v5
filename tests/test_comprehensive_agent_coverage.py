#!/usr/bin/env python3
"""
Comprehensive Agent & Attack Type Coverage Test
Tests ALL agents with ALL attack types from AWS honeypot
"""

import asyncio
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import httpx
from datetime import datetime
import json
from typing import Dict, Any, List

BASE_URL = "http://localhost:8000"
API_KEY = "demo-minixdr-api-key"

# Comprehensive attack type to agent/action mapping
ATTACK_SCENARIOS = {
    # 1. SSH Brute Force Attacks
    "ssh_brute_force": {
        "attack_type": "SSH Brute Force",
        "honeypot_events": ["cowrie.login.failed", "cowrie.login.success"],
        "chat_commands": [
            "Block this SSH brute force attack",
            "Investigate the brute force pattern",
            "Hunt for similar brute force attacks",
            "Analyze the attacker's behavior"
        ],
        "agents": ["containment", "forensics", "threat_hunting", "attribution"],
        "expected_workflows": ["block_ip", "isolate_host", "alert_security_analysts"],
        "expected_investigations": ["forensic_investigation", "pattern_correlation"]
    },
    
    # 2. DDoS/DoS Attacks
    "ddos_attack": {
        "attack_type": "DDoS/DoS",
        "honeypot_events": ["high_volume", "syn_flood", "udp_flood"],
        "chat_commands": [
            "Deploy firewall rules to mitigate this DDoS",
            "Capture network traffic during this attack",
            "Investigate the DDoS attack pattern",
            "Block all attacking IPs"
        ],
        "agents": ["containment", "forensics", "deception"],
        "expected_workflows": ["deploy_firewall_rules", "capture_network_traffic", "block_ip"],
        "expected_investigations": ["network_analysis", "traffic_pattern_analysis"]
    },
    
    # 3. Malware/Botnet
    "malware_botnet": {
        "attack_type": "Malware/Botnet",
        "honeypot_events": ["cowrie.session.file_download", "cowrie.command.input"],
        "chat_commands": [
            "Isolate infected systems and quarantine the malware",
            "Investigate the malware behavior and analyze the payload",
            "Hunt for similar malware across the network",
            "Capture forensic evidence and analyze the binary"
        ],
        "agents": ["containment", "forensics", "threat_hunting", "attribution"],
        "expected_workflows": ["isolate_host", "capture_network_traffic", "hunt_similar_attacks"],
        "expected_investigations": ["forensic_investigation", "malware_analysis", "binary_analysis"]
    },
    
    # 4. Web Application Attacks
    "web_attacks": {
        "attack_type": "Web Attack (SQL Injection/XSS)",
        "honeypot_events": ["http.request", "web.attack"],
        "chat_commands": [
            "Deploy WAF rules to block this SQL injection",
            "Investigate the web attack pattern",
            "Block the attacking IP and analyze the payload",
            "Check database integrity after this attack"
        ],
        "agents": ["containment", "forensics", "threat_hunting"],
        "expected_workflows": ["deploy_waf_rules", "block_ip", "check_database_integrity"],
        "expected_investigations": ["forensic_investigation", "sql_injection_analysis"]
    },
    
    # 5. Advanced Persistent Threat (APT)
    "apt_attack": {
        "attack_type": "Advanced Persistent Threat",
        "honeypot_events": ["multi_stage_attack", "lateral_movement"],
        "chat_commands": [
            "Investigate this APT activity and track the threat actor",
            "Hunt for lateral movement indicators",
            "Isolate affected systems and analyze the attack chain",
            "Capture all evidence and perform deep forensics"
        ],
        "agents": ["attribution", "forensics", "threat_hunting", "containment"],
        "expected_workflows": ["isolate_host", "capture_network_traffic", "hunt_similar_attacks"],
        "expected_investigations": ["forensic_investigation", "threat_actor_attribution", "campaign_tracking"]
    },
    
    # 6. Credential Stuffing
    "credential_stuffing": {
        "attack_type": "Credential Stuffing",
        "honeypot_events": ["cowrie.login.failed", "credential_reuse"],
        "chat_commands": [
            "Reset passwords for compromised accounts",
            "Block the credential stuffing attack",
            "Investigate the credential list source",
            "Enable MFA for affected accounts"
        ],
        "agents": ["containment", "forensics", "threat_hunting"],
        "expected_workflows": ["reset_passwords", "block_ip", "enforce_mfa", "revoke_user_sessions"],
        "expected_investigations": ["credential_analysis", "breach_investigation"]
    },
    
    # 7. Lateral Movement
    "lateral_movement": {
        "attack_type": "Lateral Movement",
        "honeypot_events": ["multi_host_scanning", "credential_reuse"],
        "chat_commands": [
            "Investigate lateral movement across the network",
            "Isolate compromised hosts to prevent spread",
            "Hunt for similar movement patterns",
            "Analyze the attacker's pivot strategy"
        ],
        "agents": ["threat_hunting", "forensics", "containment", "attribution"],
        "expected_workflows": ["isolate_host", "block_ip", "hunt_similar_attacks"],
        "expected_investigations": ["lateral_movement_analysis", "pivot_investigation"]
    },
    
    # 8. Data Exfiltration
    "data_exfiltration": {
        "attack_type": "Data Exfiltration",
        "honeypot_events": ["large_downloads", "database_queries"],
        "chat_commands": [
            "Block IP and encrypt sensitive data immediately",
            "Investigate data exfiltration patterns",
            "Capture network traffic and analyze data flow",
            "Enable DLP and backup critical data"
        ],
        "agents": ["containment", "forensics", "threat_hunting"],
        "expected_workflows": ["block_ip", "encrypt_sensitive_data", "enable_dlp", "backup_critical_data"],
        "expected_investigations": ["data_flow_analysis", "exfiltration_investigation"]
    },
    
    # 9. Reconnaissance/Scanning
    "reconnaissance": {
        "attack_type": "Network Reconnaissance",
        "honeypot_events": ["port_scanning", "service_enumeration"],
        "chat_commands": [
            "Investigate this reconnaissance activity",
            "Deploy deception services to track the attacker",
            "Block scanning IPs and analyze the pattern",
            "Hunt for similar reconnaissance across the network"
        ],
        "agents": ["deception", "threat_hunting", "forensics", "containment"],
        "expected_workflows": ["block_ip", "deploy_honeypot", "hunt_similar_attacks"],
        "expected_investigations": ["reconnaissance_analysis", "scanning_pattern_investigation"]
    },
    
    # 10. Command & Control (C2)
    "command_control": {
        "attack_type": "Command & Control",
        "honeypot_events": ["beaconing", "encrypted_channels"],
        "chat_commands": [
            "Investigate C2 communication and identify the server",
            "Block C2 traffic and isolate infected hosts",
            "Analyze the C2 protocol and track the campaign",
            "Hunt for other systems communicating with this C2"
        ],
        "agents": ["forensics", "attribution", "threat_hunting", "containment"],
        "expected_workflows": ["block_ip", "isolate_host", "capture_network_traffic"],
        "expected_investigations": ["c2_analysis", "campaign_tracking", "beacon_analysis"]
    },
    
    # 11. Password Spray
    "password_spray": {
        "attack_type": "Password Spray Attack",
        "honeypot_events": ["distributed_login_attempts"],
        "chat_commands": [
            "Block this password spray attack",
            "Reset passwords and enforce MFA",
            "Investigate the spray pattern and target accounts",
            "Hunt for distributed attack sources"
        ],
        "agents": ["containment", "threat_hunting", "forensics"],
        "expected_workflows": ["block_ip", "reset_passwords", "enforce_mfa"],
        "expected_investigations": ["spray_pattern_analysis", "distributed_attack_investigation"]
    },
    
    # 12. Insider Threat
    "insider_threat": {
        "attack_type": "Insider Threat",
        "honeypot_events": ["unusual_access", "privilege_escalation"],
        "chat_commands": [
            "Investigate this insider threat activity",
            "Revoke user sessions and disable the account",
            "Analyze access patterns and data accessed",
            "Track user behavior and identify anomalies"
        ],
        "agents": ["forensics", "threat_hunting", "containment"],
        "expected_workflows": ["disable_user_account", "revoke_user_sessions", "capture_network_traffic"],
        "expected_investigations": ["insider_threat_investigation", "behavior_analysis", "access_pattern_analysis"]
    }
}

class Color:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    print(f"\n{Color.BOLD}{Color.CYAN}{'='*80}{Color.END}")
    print(f"{Color.BOLD}{Color.CYAN}{text.center(80)}{Color.END}")
    print(f"{Color.BOLD}{Color.CYAN}{'='*80}{Color.END}\n")

def print_success(text: str):
    print(f"{Color.GREEN}✅ {text}{Color.END}")

def print_error(text: str):
    print(f"{Color.RED}❌ {text}{Color.END}")

def print_info(text: str):
    print(f"{Color.BLUE}ℹ️  {text}{Color.END}")

def print_section(text: str):
    print(f"\n{Color.BOLD}{Color.MAGENTA}🔍 {text}{Color.END}")

async def test_attack_scenario(client: httpx.AsyncClient, scenario_name: str, scenario: Dict[str, Any], incident_id: int):
    """Test a specific attack scenario with all its commands"""
    
    print_section(f"Testing: {scenario['attack_type']}")
    print_info(f"Scenario: {scenario_name}")
    print_info(f"Expected Agents: {', '.join(scenario['agents'])}")
    
    results = {
        "scenario": scenario_name,
        "attack_type": scenario['attack_type'],
        "commands_tested": [],
        "workflows_created": [],
        "investigations_started": [],
        "agents_triggered": [],
        "success": False
    }
    
    for cmd_idx, command in enumerate(scenario['chat_commands']):
        print_info(f"\n  Command {cmd_idx + 1}: '{command}'")
        
        try:
            response = await client.post(
                f"{BASE_URL}/api/agents/orchestrate",
                json={
                    "query": command,
                    "incident_id": incident_id,
                    "context": {
                        "attack_type": scenario['attack_type'],
                        "test": True,
                        "comprehensive_test": True
                    }
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Track command results
                cmd_result = {
                    "command": command,
                    "workflow_created": data.get("workflow_created", False),
                    "investigation_started": data.get("investigation_started", False),
                    "analysis_type": data.get("analysis_type", "unknown")
                }
                results["commands_tested"].append(cmd_result)
                
                # Track workflows
                if data.get("workflow_created"):
                    workflow_id = data.get("workflow_id")
                    results["workflows_created"].append(workflow_id)
                    print_success(f"    Workflow created: {workflow_id}")
                
                # Track investigations
                if data.get("investigation_started"):
                    case_id = data.get("case_id")
                    results["investigations_started"].append(case_id)
                    print_success(f"    Investigation started: {case_id}")
                
                # Regular response
                if not data.get("workflow_created") and not data.get("investigation_started"):
                    print_info(f"    Response type: {data.get('analysis_type', 'contextual_chat')}")
                
            else:
                print_error(f"    HTTP {response.status_code}")
                cmd_result = {
                    "command": command,
                    "error": f"HTTP {response.status_code}"
                }
                results["commands_tested"].append(cmd_result)
                
        except Exception as e:
            print_error(f"    Exception: {e}")
            results["commands_tested"].append({
                "command": command,
                "error": str(e)
            })
    
    # Determine success
    results["success"] = (
        len(results["workflows_created"]) > 0 or 
        len(results["investigations_started"]) > 0
    )
    
    if results["success"]:
        print_success(f"\n✅ Scenario '{scenario_name}' PASSED")
        print_info(f"   Workflows: {len(results['workflows_created'])}, Investigations: {len(results['investigations_started'])}")
    else:
        print_error(f"\n❌ Scenario '{scenario_name}' FAILED - No workflows or investigations created")
    
    return results

async def run_comprehensive_tests():
    """Run comprehensive tests for all attack scenarios"""
    
    print_header("COMPREHENSIVE AGENT & ATTACK TYPE COVERAGE TEST")
    print_info(f"Testing {len(ATTACK_SCENARIOS)} attack scenarios")
    print_info(f"Backend: {BASE_URL}")
    print_info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    async with httpx.AsyncClient(
        timeout=60.0,
        headers={"x-api-key": API_KEY, "Content-Type": "application/json"}
    ) as client:
        
        # Check backend health
        print_header("SERVICE HEALTH CHECK")
        try:
            response = await client.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                print_success("Backend is healthy")
            else:
                print_error(f"Backend health check failed: {response.status_code}")
                return
        except Exception as e:
            print_error(f"Backend is not running: {e}")
            return
        
        # Get incidents
        print_header("INCIDENT AVAILABILITY")
        try:
            response = await client.get(f"{BASE_URL}/incidents")
            if response.status_code == 200:
                incidents = response.json()
                print_success(f"Retrieved {len(incidents)} incidents")
                
                if not incidents:
                    print_error("No incidents available for testing!")
                    return
                    
            else:
                print_error("Failed to get incidents")
                return
        except Exception as e:
            print_error(f"Error getting incidents: {e}")
            return
        
        # Run tests for each attack scenario
        print_header("ATTACK SCENARIO TESTING")
        
        all_results = []
        
        for idx, (scenario_name, scenario) in enumerate(ATTACK_SCENARIOS.items()):
            # Use different incidents for variety
            incident_id = incidents[idx % len(incidents)]["id"]
            
            result = await test_attack_scenario(client, scenario_name, scenario, incident_id)
            all_results.append(result)
            
            # Small delay between scenarios
            await asyncio.sleep(0.5)
        
        # Summary
        print_header("TEST SUMMARY")
        
        total_scenarios = len(all_results)
        passed_scenarios = sum(1 for r in all_results if r["success"])
        total_workflows = sum(len(r["workflows_created"]) for r in all_results)
        total_investigations = sum(len(r["investigations_started"]) for r in all_results)
        total_commands = sum(len(r["commands_tested"]) for r in all_results)
        
        print_info(f"Scenarios Tested: {total_scenarios}")
        print_info(f"Scenarios Passed: {passed_scenarios}")
        print_info(f"Total Commands: {total_commands}")
        print_info(f"Workflows Created: {total_workflows}")
        print_info(f"Investigations Started: {total_investigations}")
        
        # Detailed results
        print("\n" + Color.BOLD + "Scenario Results:" + Color.END)
        for result in all_results:
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            workflows = len(result["workflows_created"])
            investigations = len(result["investigations_started"])
            print(f"  {status} - {result['attack_type']}: {workflows}W / {investigations}I")
        
        # Coverage metrics
        print("\n" + Color.BOLD + "Coverage Metrics:" + Color.END)
        coverage_pct = (passed_scenarios / total_scenarios * 100) if total_scenarios > 0 else 0
        print(f"  Attack Type Coverage: {coverage_pct:.1f}%")
        
        # Agent coverage
        agents_triggered = set()
        for result in all_results:
            agents_triggered.update(result.get("agents_triggered", []))
        
        expected_agents = {"containment", "forensics", "threat_hunting", "attribution", "deception"}
        agents_coverage = (len(agents_triggered) / len(expected_agents) * 100) if expected_agents else 0
        
        print(f"  Agent Coverage: {len(agents_triggered)}/{len(expected_agents)} agents")
        
        # Save results
        results_file = Path(__file__).parent / "comprehensive_coverage_results.json"
        with open(results_file, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_scenarios": total_scenarios,
                "passed_scenarios": passed_scenarios,
                "total_workflows": total_workflows,
                "total_investigations": total_investigations,
                "coverage_percentage": coverage_pct,
                "agent_coverage": agents_coverage,
                "results": all_results
            }, f, indent=2)
        
        print_info(f"\nDetailed results saved to: {results_file}")
        
        # Final status
        if coverage_pct >= 80:
            print_success(f"\n🎉 EXCELLENT COVERAGE: {coverage_pct:.1f}%")
        elif coverage_pct >= 60:
            print_info(f"\n⚠️  GOOD COVERAGE: {coverage_pct:.1f}%")
        else:
            print_error(f"\n❌ LOW COVERAGE: {coverage_pct:.1f}%")

if __name__ == "__main__":
    asyncio.run(run_comprehensive_tests())











