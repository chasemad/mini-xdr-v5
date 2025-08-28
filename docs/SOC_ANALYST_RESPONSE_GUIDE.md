# 🚨 SOC Analyst Response Guide: SQL Injection & Web Application Compromise

## 🎯 **Incident Assessment Framework**

When a SOC analyst sees a SQL injection incident, here's the **complete assessment and response workflow**:

### **Phase 1: Initial Triage (0-5 minutes)**

#### 🔍 **Compromise Indicators to Check:**

1. **Authentication Success Indicators:**
   - ✅ HTTP 200 responses to admin pages
   - ✅ Session tokens generated
   - ✅ "Welcome admin" or similar success messages
   - ✅ Extended session duration

2. **Database Access Confirmation:**
   - ✅ `SELECT * FROM users` in logs
   - ✅ `SHOW TABLES` commands
   - ✅ `information_schema` queries
   - ✅ Password hash retrieval patterns

3. **Data Exfiltration Signs:**
   - ✅ Large response sizes (>10KB)
   - ✅ `mysqldump` or `SELECT INTO OUTFILE`
   - ✅ Multiple rapid queries
   - ✅ Base64 encoded data transfers

### **Phase 2: Impact Assessment (5-15 minutes)**

#### 🚨 **Critical Questions:**
1. **Was the attack successful?**
   - Check for HTTP 200 responses vs 403/404
   - Look for data retrieval patterns
   - Monitor session establishment

2. **What data was accessed?**
   - User credentials table
   - Payment information
   - Personal identifiable information (PII)
   - Administrative accounts

3. **Did privilege escalation occur?**
   - `sudo`, `su root` commands
   - User creation (`CREATE USER`, `adduser`)
   - Permission modifications (`GRANT ALL`)

4. **Is persistence established?**
   - Backdoor files uploaded
   - Cron jobs created
   - New user accounts
   - Web shells installed

### **Phase 3: Immediate Response Actions**

#### 🔒 **Network Isolation:**
```bash
# Block source IP immediately
iptables -A INPUT -s ATTACKER_IP -j DROP
ufw deny from ATTACKER_IP

# Isolate compromised host
# Place in quarantine VLAN
# Block outbound connections
```

#### 🕵️ **Forensic Evidence Collection:**
```bash
# Capture network traffic
tcpdump -i eth0 -w incident_$(date +%Y%m%d_%H%M).pcap

# Preserve web logs
cp /var/log/apache2/access.log incident_access_$(date +%Y%m%d_%H%M).log
cp /var/log/mysql/mysql.log incident_db_$(date +%Y%m%d_%H%M).log

# Memory dump (if compromised host)
dd if=/dev/mem of=memory_dump_$(date +%Y%m%d_%H%M).raw
```

#### 🔍 **Database Integrity Check:**
```sql
-- Check for unauthorized changes
SELECT * FROM mysql.user WHERE user NOT IN ('expected_users');

-- Look for backdoor accounts
SELECT user, host, authentication_string FROM mysql.user 
WHERE Create_time > 'INCIDENT_START_TIME';

-- Check for data modifications
SELECT * FROM information_schema.tables 
WHERE table_schema = 'production' 
ORDER BY update_time DESC;
```

### **Phase 4: Damage Assessment**

#### 📊 **Data Compromise Evaluation:**
1. **Account Compromise:**
   - Reset all admin passwords
   - Revoke active sessions
   - Enable MFA if not present

2. **Data Exfiltration Assessment:**
   - Review outbound traffic logs
   - Check for large data transfers
   - Monitor for data appearing on dark web

3. **System Integrity:**
   - File integrity monitoring (FIM) alerts
   - Unauthorized file modifications
   - New backdoor files

### **Phase 5: Recovery Actions**

#### 🛠️ **System Remediation:**
```bash
# Remove backdoors
find /var/www -name "*.php" -newer /tmp/incident_start -exec rm {} \;

# Reset database permissions
DROP USER IF EXISTS 'suspicious_user'@'%';
FLUSH PRIVILEGES;

# Update application
# Patch SQL injection vulnerability
# Deploy WAF rules
```

#### 🔐 **Security Hardening:**
- Deploy Web Application Firewall (WAF)
- Implement input validation
- Enable SQL injection protection
- Regular security scanning

## 🎯 **Enhanced Honeypot Setup for Attack Chaining**

### **Current Limitations:**
Your honeypot currently only logs HTTP requests. To simulate realistic post-exploitation:

### **Recommended Enhancements:**

#### 1. **Interactive Web Shell Simulation:**
```php
// Add to your web honeypot: /admin/exec.php
<?php
if ($_POST['cmd']) {
    $cmd = $_POST['cmd'];
    // Log command to Mini-XDR
    $log_data = [
        'timestamp' => date('c'),
        'event_type' => 'command_execution',
        'src_ip' => $_SERVER['REMOTE_ADDR'],
        'command' => $cmd,
        'attack_indicators' => ['command_injection', 'post_exploitation']
    ];
    
    // Send to Mini-XDR
    $ch = curl_init('http://10.0.0.123:8000/ingest/multi');
    curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode([
        'source_type' => 'webhoneypot',
        'hostname' => 'honeypot-vm',
        'events' => [$log_data]
    ]));
    curl_setopt($ch, CURLOPT_HTTPHEADER, [
        'Content-Type: application/json',
        'Authorization: Bearer honeypot-agent-key-12345'
    ]);
    curl_exec($ch);
    
    // Simulate command responses
    echo "Command executed: $cmd\n";
    if (strpos($cmd, 'whoami') !== false) echo "root\n";
    if (strpos($cmd, 'id') !== false) echo "uid=0(root) gid=0(root) groups=0(root)\n";
}
?>
```

#### 2. **Database Access Simulation:**
```php
// Add to your web honeypot: /admin/users.php
<?php
if ($_GET['action'] == 'export') {
    // Log database export attempt
    $log_data = [
        'timestamp' => date('c'),
        'event_type' => 'data_exfiltration',
        'src_ip' => $_SERVER['REMOTE_ADDR'],
        'attack_indicators' => ['data_exfiltration', 'database_dump'],
        'success_indicators' => ['mysqldump executed', '10000 user records exported']
    ];
    
    // Send to Mini-XDR (same pattern as above)
    
    // Return fake user data
    echo "-- MySQL dump\n";
    echo "INSERT INTO users VALUES (1,'admin','$2y$10$hash...',1);\n";
    echo "INSERT INTO users VALUES (2,'john.doe','$2y$10$hash...',0);\n";
    // ... more fake data
}
?>
```

#### 3. **Persistence Mechanism Simulation:**
```php
// Add to your web honeypot: /admin/backdoor.php
<?php
if ($_POST['action'] == 'install_backdoor') {
    $log_data = [
        'timestamp' => date('c'),
        'event_type' => 'persistence',
        'src_ip' => $_SERVER['REMOTE_ADDR'],
        'attack_indicators' => ['persistence', 'backdoor_creation'],
        'success_indicators' => [
            'crontab entry created',
            'web shell uploaded to /var/www/html/system.php',
            'CREATE USER backdoor_user IDENTIFIED BY "password123"'
        ]
    ];
    
    // Log to Mini-XDR
    echo "Backdoor installed successfully\n";
    echo "Persistence mechanisms activated\n";
}
?>
```

## 🚀 **Testing the Complete Attack Chain**

### **Run the Advanced Simulation:**
```bash
# Execute the complete attack chain
./scripts/simulate-advanced-attack-chain.sh
```

### **Expected Mini-XDR Detection:**
After running the simulation, your Mini-XDR should detect:

✅ **Compromise Assessment:**
- 🚨 COMPROMISE CONFIRMED
- 🗄️ DATABASE ACCESS  
- ⬆️ PRIVILEGE ESCALATION

✅ **Post-Exploitation Activity:**
- 📤 DATA EXFILTRATION
- 🔒 PERSISTENCE
- ↔️ LATERAL MOVEMENT  
- 🔍 RECONNAISSANCE

✅ **SOC Response Triggers:**
- Immediate IP blocking
- Host isolation recommendations
- Incident escalation to Tier 2
- Forensic evidence collection
- Database integrity validation

## 🎯 **SOC Analyst Decision Tree**

```
SQL Injection Detected
├── HTTP 200 Response?
│   ├── YES → COMPROMISE CONFIRMED
│   │   ├── Check for data access patterns
│   │   ├── Look for privilege escalation
│   │   ├── Monitor for persistence mechanisms
│   │   └── IMMEDIATE ISOLATION + FORENSICS
│   └── NO → Attempted but failed
│       ├── Monitor for continued attempts
│       ├── Block source IP
│       └── Update WAF rules
├── Database queries in logs?
│   ├── SELECT statements → Data extraction likely
│   ├── INSERT/UPDATE → Data manipulation
│   └── CREATE USER → Account creation
└── Post-exploitation activity?
    ├── Command execution → Full compromise
    ├── File uploads → Persistence established
    └── Network scanning → Lateral movement prep
```

This gives SOC analysts the **complete picture** needed to assess compromise and respond appropriately! 🎯
