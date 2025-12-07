#!/usr/bin/env python3
"""
Synthetic Training Data Generator for Adaptive Detection
Generates realistic baseline and attack data for ML model training
"""
import asyncio
import requests
import random
import time
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict
from pathlib import Path
import sys
import argparse

# Configuration
BASE_URL = os.getenv("MINIXDR_API_BASE_URL", "http://localhost:8000")

SCRIPT_DIR = Path(__file__).resolve().parent
AUTH_DIR = SCRIPT_DIR.parent / "auth"
if str(AUTH_DIR) not in sys.path:
    sys.path.insert(0, str(AUTH_DIR))

from agent_auth import load_agent_credentials, build_signed_headers

# Realistic data patterns
NORMAL_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "curl/7.68.0",
    "wget/1.20.3"
]

ATTACK_USER_AGENTS = [
    "sqlmap/1.4.7",
    "Nikto/2.1.6",
    "dirb/2.22",
    "gobuster/3.1.0",
    "BadBot/1.0",
    "AttackScript/2.0"
]

LEGITIMATE_PATHS = [
    "/", "/index.html", "/about.html", "/contact.html", "/products.html",
    "/login.html", "/register.html", "/favicon.ico", "/robots.txt",
    "/sitemap.xml", "/css/style.css", "/js/app.js", "/images/logo.png"
]

ATTACK_PATHS = [
    "/admin", "/admin.php", "/admin/", "/wp-admin", "/wp-admin/",
    "/phpmyadmin", "/phpmyadmin/", "/manager/html", "/console/",
    "/.env", "/.git/config", "/config.php", "/database.php",
    "/backup.sql", "/dump.sql", "/test.php", "/info.php"
]

SQL_INJECTION_PAYLOADS = [
    "1' OR '1'='1", "1 UNION SELECT NULL", "1' AND 1=1--",
    "'; DROP TABLE users--", "1' UNION SELECT 1,2,3--",
    "admin'--", "' OR 1=1#", "1'; WAITFOR DELAY '00:00:05'--"
]

LEGITIMATE_USERNAMES = ["admin", "user", "guest", "demo", "test"]
ATTACK_USERNAMES = [
    "administrator", "root", "sa", "oracle", "postgres", "mysql",
    "ftp", "mail", "www", "apache", "nginx", "tomcat", "jenkins"
]

LEGITIMATE_PASSWORDS = ["password123", "admin123", "welcome", "test123"]
ATTACK_PASSWORDS = [
    "123456", "password", "admin", "root", "qwerty", "letmein",
    "monkey", "dragon", "master", "shadow", "12345678", "football",
    "baseball", "welcome", "login", "passw0rd", "abc123"
]

class TrainingDataGenerator:
    def __init__(self, base_url: str = BASE_URL, agent_profile: str | None = None):
        self.base_url = base_url
        self.credentials = load_agent_credentials(agent_profile)
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mini-XDR-TrainingGenerator/1.0'})
    
    def generate_legitimate_web_traffic(self, ip_range: str = "10.1.1", count: int = 50) -> List[Dict]:
        """Generate normal web browsing patterns"""
        events = []
        
        for i in range(count):
            src_ip = f"{ip_range}.{random.randint(10, 250)}"
            
            # Normal browsing session
            session_length = random.randint(1, 8)
            base_time = datetime.now() - timedelta(hours=random.randint(1, 72))
            
            for j in range(session_length):
                event_time = base_time + timedelta(seconds=j * random.randint(2, 30))
                path = random.choice(LEGITIMATE_PATHS)
                
                event = {
                    "eventid": "webhoneypot.request",
                    "src_ip": src_ip,
                    "dst_port": 80,
                    "message": f"GET {path}",
                    "timestamp": event_time.isoformat(),
                    "raw": {
                        "method": "GET",
                        "path": path,
                        "status_code": random.choice([200, 200, 200, 404]),
                        "user_agent": random.choice(NORMAL_USER_AGENTS),
                        "response_size": random.randint(1024, 8192),
                        "response_time": random.uniform(0.1, 2.0)
                    }
                }
                events.append(event)
        
        return events
    
    def generate_ssh_legitimate_traffic(self, ip_range: str = "10.1.2", count: int = 30) -> List[Dict]:
        """Generate legitimate SSH activity"""
        events = []
        
        for i in range(count):
            src_ip = f"{ip_range}.{random.randint(10, 250)}"
            
            # Successful SSH sessions with normal commands
            base_time = datetime.now() - timedelta(hours=random.randint(1, 72))
            
            # Login success
            events.append({
                "eventid": "cowrie.login.success",
                "src_ip": src_ip,
                "dst_port": 2222,
                "message": "SSH login successful",
                "timestamp": base_time.isoformat(),
                "raw": {
                    "username": random.choice(["admin", "user"]),
                    "password": random.choice(["admin123", "user123"]),
                    "session": f"session_{i}"
                }
            })
            
            # Normal commands
            commands = ["ls", "pwd", "whoami", "ps aux", "df -h", "top", "exit"]
            for j, cmd in enumerate(random.sample(commands, random.randint(2, 5))):
                cmd_time = base_time + timedelta(seconds=j * random.randint(5, 30))
                events.append({
                    "eventid": "cowrie.command.input",
                    "src_ip": src_ip,
                    "dst_port": 2222,
                    "message": f"Command: {cmd}",
                    "timestamp": cmd_time.isoformat(),
                    "raw": {
                        "input": cmd,
                        "session": f"session_{i}"
                    }
                })
        
        return events
    
    def generate_web_attack_patterns(self, attack_type: str = "mixed", count: int = 20) -> List[Dict]:
        """Generate various web attack patterns"""
        events = []
        
        for i in range(count):
            src_ip = f"192.168.100.{random.randint(10, 250)}"
            base_time = datetime.now() - timedelta(minutes=random.randint(1, 60))
            
            if attack_type in ["admin_scan", "mixed"]:
                # Admin panel scanning
                for j, path in enumerate(random.sample(ATTACK_PATHS, random.randint(3, 8))):
                    event_time = base_time + timedelta(seconds=j * random.uniform(0.5, 3.0))
                    events.append({
                        "eventid": "webhoneypot.request",
                        "src_ip": src_ip,
                        "dst_port": 80,
                        "message": f"GET {path}",
                        "timestamp": event_time.isoformat(),
                        "raw": {
                            "method": "GET",
                            "path": path,
                            "status_code": 404,
                            "user_agent": random.choice(ATTACK_USER_AGENTS),
                            "attack_indicators": ["admin_scan", "directory_traversal"]
                        }
                    })
            
            if attack_type in ["sql_injection", "mixed"]:
                # SQL injection attempts
                for j in range(random.randint(2, 5)):
                    payload = random.choice(SQL_INJECTION_PAYLOADS)
                    event_time = base_time + timedelta(seconds=j * random.uniform(1.0, 5.0))
                    events.append({
                        "eventid": "webhoneypot.request",
                        "src_ip": src_ip,
                        "dst_port": 80,
                        "message": f"GET /index.php?id={payload}",
                        "timestamp": event_time.isoformat(),
                        "raw": {
                            "method": "GET",
                            "path": "/index.php",
                            "parameters": [f"id={payload}"],
                            "status_code": 500,
                            "user_agent": random.choice(ATTACK_USER_AGENTS),
                            "attack_indicators": ["sql_injection"]
                        }
                    })
        
        return events
    
    def generate_ssh_attack_patterns(self, attack_type: str = "brute_force", count: int = 15) -> List[Dict]:
        """Generate SSH attack patterns"""
        events = []
        
        for i in range(count):
            src_ip = f"192.168.200.{random.randint(10, 250)}"
            base_time = datetime.now() - timedelta(minutes=random.randint(1, 60))
            
            if attack_type in ["brute_force", "mixed"]:
                # Rapid brute force
                for j in range(random.randint(6, 12)):
                    username = random.choice(ATTACK_USERNAMES)
                    password = random.choice(ATTACK_PASSWORDS)
                    event_time = base_time + timedelta(seconds=j * random.uniform(0.5, 2.0))
                    
                    events.append({
                        "eventid": "cowrie.login.failed",
                        "src_ip": src_ip,
                        "dst_port": 2222,
                        "message": f"SSH login failed: {username}/{password}",
                        "timestamp": event_time.isoformat(),
                        "raw": {
                            "username": username,
                            "password": password,
                            "session": f"attack_session_{i}_{j}"
                        }
                    })
            
            elif attack_type == "password_spray":
                # Password spraying pattern
                passwords = random.sample(ATTACK_PASSWORDS, 5)
                for j, password in enumerate(passwords):
                    usernames = random.sample(ATTACK_USERNAMES, random.randint(2, 4))
                    for k, username in enumerate(usernames):
                        event_time = base_time + timedelta(seconds=(j*10) + k * random.uniform(1.0, 3.0))
                        
                        events.append({
                            "eventid": "cowrie.login.failed",
                            "src_ip": src_ip,
                            "dst_port": 2222,
                            "message": f"SSH login failed: {username}/{password}",
                            "timestamp": event_time.isoformat(),
                            "raw": {
                                "username": username,
                                "password": password,
                                "session": f"spray_session_{i}_{j}_{k}"
                            }
                        })
        
        return events
    
    async def send_events_batch(self, events: List[Dict], source_type: str = "training") -> Dict:
        """Send a batch of events to the API"""
        payload = {
            "source_type": source_type,
            "hostname": "training-generator",
            "events": events
        }
        headers, body_text = build_signed_headers(
            self.credentials,
            "POST",
            "/ingest/multi",
            payload,
        )

        try:
            response = self.session.post(
                f"{self.base_url}/ingest/multi",
                data=body_text,
                headers=headers,
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    async def generate_comprehensive_dataset(self, 
                                           legitimate_web: int = 100,
                                           legitimate_ssh: int = 50,
                                           web_attacks: int = 30,
                                           ssh_attacks: int = 25) -> Dict:
        """Generate a comprehensive training dataset"""
        
        print("🔄 Generating comprehensive training dataset...")
        results = {}
        
        # Generate legitimate traffic (for baseline learning)
        print(f"   📊 Generating {legitimate_web} legitimate web events...")
        web_events = self.generate_legitimate_web_traffic(count=legitimate_web)
        result = await self.send_events_batch(web_events, "webhoneypot")
        results['legitimate_web'] = result
        
        await asyncio.sleep(1)  # Rate limiting
        
        print(f"   🔐 Generating {legitimate_ssh} legitimate SSH events...")
        ssh_events = self.generate_ssh_legitimate_traffic(count=legitimate_ssh)
        result = await self.send_events_batch(ssh_events, "cowrie")
        results['legitimate_ssh'] = result
        
        await asyncio.sleep(1)
        
        # Generate attack patterns
        print(f"   ⚔️ Generating {web_attacks} web attack events...")
        web_attack_events = self.generate_web_attack_patterns(count=web_attacks)
        result = await self.send_events_batch(web_attack_events, "webhoneypot")
        results['web_attacks'] = result
        
        await asyncio.sleep(1)
        
        print(f"   🔓 Generating {ssh_attacks} SSH attack events...")
        ssh_attack_events = self.generate_ssh_attack_patterns(count=ssh_attacks)
        result = await self.send_events_batch(ssh_attack_events, "cowrie")
        results['ssh_attacks'] = result
        
        return results
    
    async def quick_training_boost(self) -> Dict:
        """Generate a quick set of diverse training data"""
        print("🚀 Quick Training Boost - Generating diverse dataset...")
        
        # Mix of everything
        events = []
        events.extend(self.generate_legitimate_web_traffic(count=50))
        events.extend(self.generate_ssh_legitimate_traffic(count=25))
        events.extend(self.generate_web_attack_patterns("mixed", count=20))
        events.extend(self.generate_ssh_attack_patterns("mixed", count=15))
        
        # Shuffle to simulate realistic timing
        random.shuffle(events)
        
        # Send in batches to avoid overwhelming the API
        batch_size = 25
        results = []
        
        for i in range(0, len(events), batch_size):
            batch = events[i:i + batch_size]
            print(f"   📤 Sending batch {i//batch_size + 1}/{(len(events) + batch_size - 1)//batch_size}...")
            result = await self.send_events_batch(batch, "training")
            results.append(result)
            await asyncio.sleep(0.5)  # Brief pause between batches
        
        return {"batches_sent": len(results), "results": results}

async def main():
    parser = argparse.ArgumentParser(description="Generate training data for adaptive detection")
    parser.add_argument("--mode", choices=["quick", "comprehensive", "custom"], 
                       default="quick", help="Training data generation mode")
    parser.add_argument("--web-count", type=int, default=100, 
                       help="Number of legitimate web events")
    parser.add_argument("--ssh-count", type=int, default=50,
                       help="Number of legitimate SSH events")
    parser.add_argument("--attack-web", type=int, default=30,
                       help="Number of web attack events")
    parser.add_argument("--attack-ssh", type=int, default=25,
                       help="Number of SSH attack events")
    parser.add_argument("--agent-profile", default=None,
                       help="Agent credential profile (default: MINIXDR_AGENT_PROFILE or HUNTER)")

    args = parser.parse_args()

    generator = TrainingDataGenerator(agent_profile=args.agent_profile)
    
    # Check if backend is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Backend not healthy. Please start Mini-XDR first.")
            return
    except Exception:
        print("❌ Cannot connect to backend. Please start Mini-XDR first:")
        print("   cd $(cd "$(dirname "$0")/../.." ${PROJECT_ROOT:-$(dirname $(dirname $(dirname $(realpath "$0"))))}${PROJECT_ROOT:-$(dirname $(dirname $(dirname $(realpath "$0"))))} pwd) && ./scripts/start-all.sh")
        return
    
    print("🧠 Training Data Generator for Adaptive Detection")
    print("=" * 50)
    
    if args.mode == "quick":
        results = await generator.quick_training_boost()
        print("\n✅ Quick training boost completed!")
        print(f"   📊 Batches sent: {results['batches_sent']}")
        
    elif args.mode == "comprehensive":
        results = await generator.generate_comprehensive_dataset(
            legitimate_web=args.web_count,
            legitimate_ssh=args.ssh_count,
            web_attacks=args.attack_web,
            ssh_attacks=args.attack_ssh
        )
        print("\n✅ Comprehensive dataset generated!")
        for key, result in results.items():
            processed = result.get('processed', 0)
            incidents = result.get('incidents_detected', 0)
            print(f"   {key}: {processed} events, {incidents} incidents")
    
    # Force learning update after data generation
    print("\n🔄 Triggering learning pipeline update...")
    try:
        headers, body_text = build_signed_headers(
            generator.credentials,
            "POST",
            "/api/adaptive/force_learning",
            "",
        )
        response = generator.session.post(
            f"{BASE_URL}/api/adaptive/force_learning",
            data=body_text,
            headers=headers,
        )
        if response.status_code == 200:
            learning_result = response.json()
            print("✅ Learning pipeline updated successfully!")
            print(f"   Results: {learning_result.get('results', {})}")
        else:
            print("⚠️ Learning update failed")
    except Exception as e:
        print(f"⚠️ Learning update error: {e}")
    
    print("\n🎯 Next Steps:")
    print("   • Check adaptive status: curl http://localhost:8000/api/adaptive/status")
    print("   • View incidents: curl http://localhost:8000/incidents")
    print("   • Test detection: ./scripts/test-adaptive-detection.sh")
    print("\n🧠 The models will learn from this data and improve detection accuracy!")

if __name__ == "__main__":
    asyncio.run(main())
