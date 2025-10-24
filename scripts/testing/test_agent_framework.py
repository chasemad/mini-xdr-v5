#!/usr/bin/env python3
"""
Comprehensive test suite for IAM, EDR, and DLP agents
Tests all agent actions, rollback functionality, and database logging
"""

import asyncio
import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from app.agents.iam_agent import iam_agent
from app.agents.edr_agent import edr_agent
from app.agents.dlp_agent import dlp_agent


class AgentTester:
    """Test harness for agent framework"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.results = []
    
    def test(self, name: str, condition: bool, details: str = ""):
        """Test a condition and track results"""
        if condition:
            self.passed += 1
            print(f"✅ PASS: {name}")
            if details:
                print(f"   {details}")
        else:
            self.failed += 1
            print(f"❌ FAIL: {name}")
            if details:
                print(f"   {details}")
        
        self.results.append({
            "name": name,
            "passed": condition,
            "details": details
        })
    
    def summary(self):
        """Print test summary"""
        total = self.passed + self.failed
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {total}")
        print(f"Passed: {self.passed} ({100*self.passed//total if total > 0 else 0}%)")
        print(f"Failed: {self.failed}")
        print("="*60)
        
        if self.failed == 0:
            print("🎉 All tests passed!")
            return True
        else:
            print(f"⚠️ {self.failed} test(s) failed")
            return False


async def test_iam_agent(tester: AgentTester):
    """Test IAM Agent"""
    print("\n" + "="*60)
    print("IAM AGENT TESTS")
    print("="*60 + "\n")
    
    # Test 1: Disable user
    print("[1/6] Testing disable_user_account...")
    result = await iam_agent.execute_action(
        "disable_user_account",
        {"username": "testuser@domain.local", "reason": "Test"}
    )
    tester.test(
        "IAM: Disable user",
        result.get("success") == True,
        f"Rollback ID: {result.get('rollback_id')}"
    )
    disable_rollback_id = result.get('rollback_id')
    
    # Test 2: Quarantine user
    print("\n[2/6] Testing quarantine_user...")
    result = await iam_agent.execute_action(
        "quarantine_user",
        {
            "username": "compromised@domain.local",
            "security_group": "CN=Quarantine,OU=Security,DC=domain,DC=local"
        }
    )
    tester.test(
        "IAM: Quarantine user",
        result.get("success") == True,
        f"Status: {result.get('result', {}).get('status')}"
    )
    
    # Test 3: Revoke Kerberos tickets
    print("\n[3/6] Testing revoke_kerberos_tickets...")
    result = await iam_agent.execute_action(
        "revoke_kerberos_tickets",
        {"username": "testuser@domain.local"}
    )
    tester.test(
        "IAM: Revoke Kerberos tickets",
        result.get("success") == True
    )
    
    # Test 4: Reset password
    print("\n[4/6] Testing reset_password...")
    result = await iam_agent.execute_action(
        "reset_password",
        {"username": "testuser@domain.local"}
    )
    res = result.get("result", {})
    password = res.get("new_password") or res.get("temporary_password", "")
    tester.test(
        "IAM: Reset password",
        result.get("success") == True and password,
        f"New password generated: {len(password)} chars"
    )
    
    # Test 5: Remove from privileged groups
    print("\n[5/6] Testing remove_from_group...")
    result = await iam_agent.execute_action(
        "remove_from_group",
        {"username": "testuser@domain.local", "group": "Domain Admins"}
    )
    tester.test(
        "IAM: Remove from privileged groups",
        result.get("success") == True
    )
    
    # Test 6: Rollback
    print("\n[6/6] Testing rollback...")
    if disable_rollback_id:
        result = await iam_agent.rollback_action(disable_rollback_id)
        tester.test(
            "IAM: Rollback disable_user_account",
            result.get("success") == True,
            f"Restored state: {result.get('restored_state', {}).get('action')}"
        )
    else:
        tester.test("IAM: Rollback disable_user_account", False, "No rollback ID")


async def test_edr_agent(tester: AgentTester):
    """Test EDR Agent"""
    print("\n" + "="*60)
    print("EDR AGENT TESTS")
    print("="*60 + "\n")
    
    # Test 1: Kill process
    print("[1/7] Testing kill_process...")
    result = await edr_agent.execute_action(
        "kill_process",
        {"hostname": "workstation01", "process_name": "malware.exe"}
    )
    tester.test(
        "EDR: Kill process",
        result.get("success") == True,
        f"Status: {result.get('result', {}).get('status')}"
    )
    
    # Test 2: Quarantine file
    print("\n[2/7] Testing quarantine_file...")
    result = await edr_agent.execute_action(
        "quarantine_file",
        {"hostname": "workstation01", "file_path": "C:\\malware.exe"}
    )
    tester.test(
        "EDR: Quarantine file",
        result.get("success") == True,
        f"Quarantine path: {result.get('result', {}).get('quarantine_path')}"
    )
    quarantine_rollback_id = result.get('rollback_id')
    
    # Test 3: Collect memory dump
    print("\n[3/7] Testing collect_memory_dump...")
    result = await edr_agent.execute_action(
        "collect_memory_dump",
        {"hostname": "workstation01"}
    )
    tester.test(
        "EDR: Collect memory dump",
        result.get("success") == True,
        f"Dump path: {result.get('result', {}).get('dump_path')}"
    )
    
    # Test 4: Isolate host
    print("\n[4/7] Testing isolate_host...")
    result = await edr_agent.execute_action(
        "isolate_host",
        {"hostname": "workstation01", "level": "strict"}
    )
    tester.test(
        "EDR: Isolate host",
        result.get("success") == True,
        f"Isolation level: {result.get('result', {}).get('isolation_level')}"
    )
    isolate_rollback_id = result.get('rollback_id')
    
    # Test 5: Delete registry key
    print("\n[5/7] Testing delete_registry_key...")
    result = await edr_agent.execute_action(
        "delete_registry_key",
        {
            "hostname": "workstation01",
            "key_path": "HKLM:\\Software\\Malware\\Persistence"
        }
    )
    tester.test(
        "EDR: Delete registry key",
        result.get("success") == True
    )
    
    # Test 6: Disable scheduled task
    print("\n[6/7] Testing disable_scheduled_task...")
    result = await edr_agent.execute_action(
        "disable_scheduled_task",
        {"hostname": "workstation01", "task_name": "MaliciousTask"}
    )
    tester.test(
        "EDR: Disable scheduled task",
        result.get("success") == True
    )
    
    # Test 7: Rollback isolation
    print("\n[7/7] Testing rollback...")
    if isolate_rollback_id:
        result = await edr_agent.rollback_action(isolate_rollback_id)
        tester.test(
            "EDR: Rollback isolate_host",
            result.get("success") == True,
            f"Un-isolated: {result.get('restored_state', {}).get('hostname')}"
        )
    else:
        tester.test("EDR: Rollback isolate_host", False, "No rollback ID")


async def test_dlp_agent(tester: AgentTester):
    """Test DLP Agent"""
    print("\n" + "="*60)
    print("DLP AGENT TESTS")
    print("="*60 + "\n")
    
    # Test 1: Scan file
    print("[1/3] Testing scan_file...")
    result = await dlp_agent.execute_action(
        "scan_file",
        {"file_path": "C:\\Users\\test\\documents\\data.csv"}
    )
    tester.test(
        "DLP: Scan file",
        result.get("success") == True,
        f"Findings: {len(result.get('result', {}).get('findings', []))}"
    )
    
    # Test 2: Block upload
    print("\n[2/3] Testing block_upload...")
    result = await dlp_agent.execute_action(
        "block_upload",
        {
            "hostname": "workstation01",
            "process_name": "chrome.exe",
            "destination": "http://malicious.com"
        }
    )
    tester.test(
        "DLP: Block upload",
        result.get("success") == True,
        f"Status: {result.get('result', {}).get('status')}"
    )
    block_rollback_id = result.get('rollback_id')
    
    # Test 3: Quarantine sensitive file
    print("\n[3/3] Testing quarantine_sensitive_file...")
    result = await dlp_agent.execute_action(
        "quarantine_sensitive_file",
        {"hostname": "workstation01", "file_path": "C:\\sensitive.xlsx"}
    )
    tester.test(
        "DLP: Quarantine sensitive file",
        result.get("success") == True
    )


async def test_detection_methods(tester: AgentTester):
    """Test agent detection capabilities"""
    print("\n" + "="*60)
    print("DETECTION TESTS")
    print("="*60 + "\n")
    
    # Mock event for testing
    class MockEvent:
        def __init__(self):
            self.raw = {}
    
    # Test EDR: Process injection detection
    print("[1/3] Testing EDR process injection detection...")
    event = MockEvent()
    event.raw = {
        "parent_process": "explorer.exe",
        "process_name": "powershell.exe",
        "api_calls": ["CreateRemoteThread", "WriteProcessMemory"]
    }
    
    result = await edr_agent.detect_process_injection(event)
    tester.test(
        "EDR: Process injection detection",
        result is not None and result.get("attack_type") == "process_injection",
        f"Confidence: {result.get('confidence') if result else 'N/A'}"
    )
    
    # Test EDR: PowerShell abuse detection
    print("\n[2/3] Testing EDR PowerShell abuse detection...")
    event = MockEvent()
    event.raw = {
        "command_line": "powershell -encodedcommand AAAABBBBCCCC downloadstring invoke-expression"
    }
    
    result = await edr_agent.detect_powershell_abuse(event)
    tester.test(
        "EDR: PowerShell abuse detection",
        result is not None and result.get("attack_type") == "powershell_abuse",
        f"Indicators: {len(result.get('indicators', [])) if result else 0}"
    )
    
    # Test DLP: Data exfiltration detection
    print("\n[3/3] Testing DLP data exfiltration detection...")
    event = MockEvent()
    event.raw = {
        "file_size": 15 * 1024 * 1024,  # 15MB
        "destination_ip": "8.8.8.8",
        "filename": "database_dump.zip"
    }
    
    result = await dlp_agent.detect_data_exfiltration(event)
    tester.test(
        "DLP: Data exfiltration detection",
        result is not None and result.get("attack_type") == "data_exfiltration",
        f"Confidence: {result.get('confidence') if result else 'N/A'}"
    )


async def main():
    """Run all tests"""
    print("\n" + "🧪 " + "="*58)
    print("MINI-XDR AGENT FRAMEWORK TEST SUITE")
    print("="*60 + "\n")
    
    tester = AgentTester()
    
    # Run all test suites
    await test_iam_agent(tester)
    await test_edr_agent(tester)
    await test_dlp_agent(tester)
    await test_detection_methods(tester)
    
    # Print summary
    success = tester.summary()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())

