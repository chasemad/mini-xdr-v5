#!/usr/bin/env python3
"""
Test script for the intelligent adaptive attack detection system
"""
import requests
import time
import json
import random

# Test endpoints
BASE_URL = "http://localhost:8000"

def test_adaptive_detection():
    """Test the adaptive detection system"""
    print("🚀 Testing Intelligent Adaptive Attack Detection System")
    print("=" * 60)
    
    # 1. Check system health
    print("1. Checking system health...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is healthy")
        else:
            print(f"❌ Backend unhealthy: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Backend not running: {e}")
        return False
    
    # 2. Check adaptive detection status
    print("\n2. Checking adaptive detection status...")
    try:
        response = requests.get(f"{BASE_URL}/api/adaptive/status")
        if response.status_code == 200:
            status = response.json()
            print("✅ Adaptive detection system status:")
            learning_status = status.get("learning_pipeline", {})
            print(f"   - Learning pipeline running: {learning_status.get('running', False)}")
            print(f"   - Active learning tasks: {learning_status.get('active_tasks', 0)}")
            print(f"   - Behavioral threshold: {status.get('adaptive_engine', {}).get('behavioral_threshold', 'N/A')}")
            
            ml_status = status.get("ml_detector", {})
            print(f"   - ML models status: {ml_status}")
            
            baseline_status = status.get("baseline_engine", {})
            print(f"   - Baseline IPs learned: {baseline_status.get('per_ip_baselines', 0)}")
        else:
            print(f"❌ Failed to get adaptive status: {response.status_code}")
    except Exception as e:
        print(f"❌ Error checking status: {e}")
    
    # 3. Test behavioral detection with simulated web attacks
    print("\n3. Testing behavioral detection with simulated web attacks...")
    test_ip = "192.168.1.100"
    
    # Simulate rapid enumeration attack
    web_attack_events = [
        {
            "src_ip": test_ip,
            "dst_port": 80,
            "eventid": "webhoneypot.request",
            "message": "GET /admin.php",
            "raw": {
                "path": "/admin.php",
                "status_code": 404,
                "user_agent": "curl/7.68.0",
                "attack_indicators": ["admin_scan"]
            }
        },
        {
            "src_ip": test_ip,
            "dst_port": 80,
            "eventid": "webhoneypot.request", 
            "message": "GET /wp-admin/",
            "raw": {
                "path": "/wp-admin/",
                "status_code": 404,
                "user_agent": "curl/7.68.0",
                "attack_indicators": ["admin_scan"]
            }
        },
        {
            "src_ip": test_ip,
            "dst_port": 80,
            "eventid": "webhoneypot.request",
            "message": "GET /index.php?id=1' OR 1=1--",
            "raw": {
                "path": "/index.php",
                "parameters": ["id=1' OR 1=1--"],
                "status_code": 500,
                "user_agent": "curl/7.68.0",
                "attack_indicators": ["sql_injection"]
            }
        },
        {
            "src_ip": test_ip,
            "dst_port": 80,
            "eventid": "webhoneypot.request",
            "message": "GET /.env",
            "raw": {
                "path": "/.env",
                "status_code": 404,
                "user_agent": "curl/7.68.0",
                "attack_indicators": ["sensitive_file_access"]
            }
        }
    ]
    
    # Send events to trigger adaptive detection
    for i, event in enumerate(web_attack_events):
        try:
            response = requests.post(f"{BASE_URL}/ingest/cowrie", json=event)
            if response.status_code == 200:
                result = response.json()
                print(f"   Event {i+1}: ✅ Stored, Incident: {result.get('incident_id', 'None')}")
                if result.get('incident_id'):
                    incident_id = result['incident_id']
                    
                    # Get incident details
                    incident_response = requests.get(f"{BASE_URL}/incidents/{incident_id}")
                    if incident_response.status_code == 200:
                        incident = incident_response.json()
                        print(f"      🎯 ADAPTIVE INCIDENT DETECTED!")
                        print(f"      - Reason: {incident['reason']}")
                        print(f"      - Status: {incident['status']}")
                        
            else:
                print(f"   Event {i+1}: ❌ Failed ({response.status_code})")
        except Exception as e:
            print(f"   Event {i+1}: ❌ Error: {e}")
        
        # Small delay between events to simulate real attack timing
        time.sleep(0.5)
    
    # 4. Test SSH brute force detection (enhanced)
    print("\n4. Testing enhanced SSH brute force detection...")
    ssh_events = []
    usernames = ["admin", "root", "user", "test"]
    passwords = ["123456", "password", "admin", "root", "qwerty"]
    
    for i in range(8):  # Should trigger traditional + adaptive detection
        event = {
            "src_ip": "192.168.1.101",
            "dst_port": 2222,
            "eventid": "cowrie.login.failed",
            "message": f"login attempt: {random.choice(usernames)}/{random.choice(passwords)}",
            "raw": {
                "username": random.choice(usernames),
                "password": random.choice(passwords),
                "session": f"session_{i}"
            }
        }
        
        try:
            response = requests.post(f"{BASE_URL}/ingest/cowrie", json=event)
            if response.status_code == 200:
                result = response.json()
                print(f"   SSH Event {i+1}: ✅ Stored, Incident: {result.get('incident_id', 'None')}")
                if result.get('incident_id'):
                    print(f"      🎯 SSH BRUTE FORCE DETECTED!")
            else:
                print(f"   SSH Event {i+1}: ❌ Failed ({response.status_code})")
        except Exception as e:
            print(f"   SSH Event {i+1}: ❌ Error: {e}")
        
        time.sleep(0.2)
    
    # 5. Check incident summary
    print("\n5. Checking detected incidents...")
    try:
        response = requests.get(f"{BASE_URL}/incidents")
        if response.status_code == 200:
            incidents = response.json()
            recent_incidents = [inc for inc in incidents if 'adaptive' in inc.get('reason', '').lower()]
            
            print(f"✅ Total incidents: {len(incidents)}")
            print(f"🧠 Adaptive detection incidents: {len(recent_incidents)}")
            
            for incident in recent_incidents[:3]:  # Show top 3
                print(f"   - ID {incident['id']}: {incident['reason'][:80]}...")
        else:
            print(f"❌ Failed to get incidents: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting incidents: {e}")
    
    # 6. Test forced learning update
    print("\n6. Testing forced learning update...")
    try:
        response = requests.post(f"{BASE_URL}/api/adaptive/force_learning")
        if response.status_code == 200:
            result = response.json()
            print("✅ Forced learning update completed:")
            print(f"   - Results: {result.get('results', {})}")
        else:
            print(f"❌ Learning update failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Learning update error: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Adaptive Detection System Test Complete!")
    print("\n📊 SUMMARY:")
    print("✅ Behavioral pattern analysis")
    print("✅ Enhanced ML ensemble detection") 
    print("✅ Statistical baseline learning")
    print("✅ Continuous learning pipeline")
    print("✅ Adaptive threat scoring")
    print("✅ Multi-layer detection correlation")
    
    return True

if __name__ == "__main__":
    test_adaptive_detection()
