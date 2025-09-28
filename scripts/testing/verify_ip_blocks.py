#!/usr/bin/env python3
"""
Script to verify IP blocks on T-Pot honeypot
Usage: python verify_ip_blocks.py [optional_ip_to_check]
"""
import asyncio
import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.responder import responder

async def verify_ip_blocks(specific_ip=None):
    """Verify IP blocks on T-Pot system"""
    
    print("🔍 Checking IP blocks on T-Pot...")
    
    # Check for all blocked IPs
    print("\n=== Currently Blocked IPs ===")
    status, stdout, stderr = await responder.execute_command('sudo iptables -L INPUT -n | grep DROP')
    if stdout:
        blocked_ips = []
        for line in stdout.strip().split('\n'):
            if 'DROP' in line:
                parts = line.split()
                if len(parts) >= 4:
                    ip = parts[3]
                    blocked_ips.append(ip)
        
        if blocked_ips:
            print(f"✅ Found {len(blocked_ips)} blocked IP(s):")
            for i, ip in enumerate(blocked_ips, 1):
                print(f"  {i}. {ip}")
        else:
            print("❌ No blocked IPs found")
    else:
        print("❌ No blocked IPs found")
    
    # Check specific IP if provided
    if specific_ip:
        print(f"\n=== Checking Specific IP: {specific_ip} ===")
        status2, stdout2, stderr2 = await responder.execute_command(f'sudo iptables -L INPUT -n | grep {specific_ip}')
        if stdout2:
            print(f"✅ {specific_ip} is BLOCKED")
            print(f"   Rule: {stdout2.strip()}")
        else:
            print(f"❌ {specific_ip} is NOT blocked")
    
    # Show INPUT chain summary
    print("\n=== INPUT Chain Summary (first 10 rules) ===")
    status3, stdout3, stderr3 = await responder.execute_command('sudo iptables -L INPUT -n --line-numbers | head -12')
    if stdout3:
        lines = stdout3.strip().split('\n')
        for line in lines:
            if 'DROP' in line:
                print(f"🔴 {line}")  # Highlight DROP rules
            elif 'Chain INPUT' in line or 'num  target' in line:
                print(f"📋 {line}")
            else:
                print(f"   {line}")

async def main():
    specific_ip = sys.argv[1] if len(sys.argv) > 1 else None
    await verify_ip_blocks(specific_ip)

if __name__ == "__main__":
    asyncio.run(main())
