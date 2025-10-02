# SOC Analyst Quick Reference Guide
## Mini-XDR Attack Response Commands

**100% Coverage for All Honeypot Attacks** ✅

---

## 🎯 Quick Attack Response

### SSH Brute Force 🔓
```
🟢 Containment:
→ "Block this SSH brute force attack"
→ "Block IP [IP_ADDRESS]"

🔵 Investigation:
→ "Investigate the brute force pattern"
→ "Hunt for similar brute force attacks"
→ "Analyze the attacker's behavior"
```

### DDoS/DoS Attack 🌊
```
🟢 Containment:
→ "Deploy firewall rules to mitigate this DDoS"
→ "Capture network traffic during this attack"

🔵 Investigation:
→ "Investigate the DDoS attack pattern"
→ "Analyze traffic patterns"
```

### Malware/Botnet 🦠
```
🟢 Containment:
→ "Isolate infected systems and quarantine the malware"
→ "Capture forensic evidence and analyze the binary"

🔵 Investigation:
→ "Investigate the malware behavior and analyze the payload"
→ "Hunt for similar malware across the network"
```

### Web Attacks (SQL Injection/XSS) 🌐
```
🟢 Containment:
→ "Deploy WAF rules to block this SQL injection"
→ "Block the attacking IP and analyze the payload"
→ "Check database integrity after this attack"

🔵 Investigation:
→ "Investigate the web attack pattern"
```

### Advanced Persistent Threat (APT) 🎯
```
🟢 Containment:
→ "Isolate affected systems and analyze the attack chain"

🔵 Investigation:
→ "Investigate this APT activity and track the threat actor"
→ "Hunt for lateral movement indicators"
→ "Capture all evidence and perform deep forensics"
```

### Credential Stuffing 🔑
```
🟢 Containment:
→ "Reset passwords for compromised accounts"
→ "Block the credential stuffing attack"
→ "Enable MFA for affected accounts"

🔵 Investigation:
→ "Investigate the credential list source"
```

### Lateral Movement 🔀
```
🟢 Containment:
→ "Isolate compromised hosts to prevent spread"

🔵 Investigation:
→ "Investigate lateral movement across the network"
→ "Hunt for similar movement patterns"
→ "Analyze the attacker's pivot strategy"
```

### Data Exfiltration 📤
```
🟢 Containment:
→ "Block IP and encrypt sensitive data immediately"
→ "Capture network traffic and analyze data flow"
→ "Enable DLP and backup critical data"

🔵 Investigation:
→ "Investigate data exfiltration patterns"
```

### Network Reconnaissance 🔍
```
🟢 Containment:
→ "Deploy deception services to track the attacker"
→ "Block scanning IPs and analyze the pattern"

🔵 Investigation:
→ "Investigate this reconnaissance activity"
→ "Hunt for similar reconnaissance across the network"
```

### Command & Control (C2) 📡
```
🟢 Containment:
→ "Block C2 traffic and isolate infected hosts"

🔵 Investigation:
→ "Investigate C2 communication and identify the server"
→ "Analyze the C2 protocol and track the campaign"
→ "Hunt for other systems communicating with this C2"
```

### Password Spray 💧
```
🟢 Containment:
→ "Block this password spray attack"
→ "Reset passwords and enforce MFA"

🔵 Investigation:
→ "Investigate the spray pattern and target accounts"
→ "Hunt for distributed attack sources"
```

### Insider Threat 👤
```
🟢 Containment:
→ "Revoke user sessions and disable the account"

🔵 Investigation:
→ "Investigate this insider threat activity"
→ "Analyze access patterns and data accessed"
→ "Track user behavior and identify anomalies"
```

---

## 📋 Action Categories

### Network Containment 🌐
| Command | Action | Result |
|---------|--------|--------|
| "Block IP [IP]" | block_ip | Blocks specific IP |
| "Deploy firewall rules" | deploy_firewall_rules | Firewall protection |
| "Deploy WAF rules" | deploy_waf_rules | Web application firewall |
| "Capture network traffic" | capture_network_traffic | PCAP capture |
| "Block C2 traffic" | block_c2_traffic | Blocks C2 communication |

### Endpoint Protection 💻
| Command | Action | Result |
|---------|--------|--------|
| "Isolate host/systems" | isolate_host | Network isolation |
| "Terminate process" | terminate_process | Kills process |
| "Quarantine malware" | isolate_host | Malware containment |

### Identity & Access 🔐
| Command | Action | Result |
|---------|--------|--------|
| "Reset passwords" | reset_passwords | Password reset |
| "Revoke sessions" | revoke_user_sessions | Session termination |
| "Enforce MFA" | enforce_mfa | MFA activation |
| "Disable account" | disable_user_account | Account suspension |

### Data Protection 🛡️
| Command | Action | Result |
|---------|--------|--------|
| "Encrypt data" | encrypt_sensitive_data | Data encryption |
| "Backup data" | backup_critical_data | Data backup |
| "Enable DLP" | enable_dlp | DLP activation |
| "Check DB integrity" | check_database_integrity | DB validation |

### Forensics & Investigation 🔬
| Command | Action | Result |
|---------|--------|--------|
| "Investigate [attack]" | investigate_behavior | Forensic analysis |
| "Hunt similar attacks" | hunt_similar_attacks | Threat hunting |
| "Analyze malware" | analyze_malware | Malware analysis |
| "Track threat actor" | track_threat_actor | Attribution |
| "Capture evidence" | capture_forensic_evidence | Evidence collection |

### Deception 🎭
| Command | Action | Result |
|---------|--------|--------|
| "Deploy honeypot" | deploy_honeypot | Honeypot deployment |
| "Deploy deception services" | deploy_honeypot | Deception layer |

### Communication 📢
| Command | Action | Result |
|---------|--------|--------|
| "Alert security team" | alert_security_analysts | SOC alert |
| "Create incident case" | create_incident_case | Case creation |
| "Escalate to SOC" | escalate_to_team | Escalation |

---

## 🎨 UI Indicators

### Color Codes:
- 🟢 **Green Toast** = Workflow Created (Action will be taken)
- 🔵 **Blue Toast** = Investigation Started (Analysis in progress)
- 🟡 **Yellow** = Approval Required
- 🔴 **Red** = Error/Failed

### Toast Messages:
```
✅ "Workflow Created - Workflow 123 created and ready to execute"
✅ "Investigation Started - Case inv_abc123 - Analyzing 50 events"
⚠️  "Workflow Created - Workflow 124 created - approval required"
```

---

## 🔄 Typical Workflows

### Incident Response Flow:
```
1. Open incident → http://localhost:3000/incidents/incident/[ID]
2. Use AI chat (right sidebar)
3. Type command (see above)
4. Watch for toast notification
5. Verify action in workflows/actions section
```

### Containment Flow:
```
User: "Block this SSH brute force attack"
  → System detects "block" keyword
  → Creates workflow with block_ip action
  → Shows green toast
  → Workflow appears in incident
  → Execute when ready
```

### Investigation Flow:
```
User: "Investigate the malware behavior"
  → System detects "investigate" keyword
  → Initializes ForensicsAgent
  → Creates investigation case
  → Shows blue toast
  → Action logged in database
  → View findings in actions
```

---

## 📊 Agent Capabilities

### ContainmentAgent 🛡️
**When to Use**: Immediate threat response
- Block IPs
- Isolate hosts
- Deploy firewalls
- Network containment

**Example**: "Block IP and isolate the host"

### ForensicsAgent 🔬
**When to Use**: Deep analysis needed
- Evidence collection
- Malware analysis
- Traffic capture
- Timeline reconstruction

**Example**: "Investigate the malware and capture evidence"

### ThreatHuntingAgent 🎯
**When to Use**: Proactive searching
- Hunt similar attacks
- Pattern detection
- Behavioral analysis
- Threat correlation

**Example**: "Hunt for similar attacks across the network"

### AttributionAgent 🕵️
**When to Use**: Threat actor tracking
- Campaign identification
- Actor attribution
- C2 analysis
- APT tracking

**Example**: "Track the threat actor and identify the campaign"

### DeceptionAgent 🎭
**When to Use**: Attacker engagement
- Honeypot deployment
- Deception services
- Attacker tracking

**Example**: "Deploy deception services to track the attacker"

---

## 💡 Pro Tips

### Combining Actions:
```
✅ "Block IP, reset passwords, and enforce MFA"
   → Creates workflow with 3 actions

✅ "Isolate host and capture forensic evidence"
   → Creates workflow with 2 actions

✅ "Investigate APT and track threat actor"
   → Starts investigation with attribution
```

### Best Practices:
1. **Be Specific**: Include attack type and target
2. **Use Action Verbs**: block, isolate, investigate, analyze
3. **Combine Related Actions**: Multi-step workflows
4. **Check Approvals**: High-risk actions need approval

### Common Patterns:
```
🔴 Critical Response:
"Block IP, isolate host, alert security team"

🟡 Investigation:
"Investigate [attack], hunt similar, track actor"

🟢 Containment:
"Deploy firewall, capture traffic, enable DLP"
```

---

## 🚨 Emergency Response

### Immediate Actions:
```
Ransomware:
→ "Isolate infected systems and backup critical data"

Data Breach:
→ "Block IP, encrypt data, enable DLP immediately"

APT Detected:
→ "Isolate hosts, investigate APT, track threat actor"

Active C2:
→ "Block C2 traffic, isolate hosts, capture evidence"
```

---

## 📈 Monitoring

### Check Workflows:
- Go to: `http://localhost:3000/workflows`
- Filter by incident ID
- Execute approved workflows

### Check Investigations:
- View in incident action history
- Look for "forensic_investigation" actions
- Check case IDs (format: `inv_XXXX`)

### Database Queries:
```sql
-- Recent workflows
SELECT workflow_id, playbook_name, status 
FROM response_workflows 
WHERE workflow_id LIKE 'chat_%' 
ORDER BY created_at DESC LIMIT 10;

-- Recent investigations
SELECT action, detail, params 
FROM actions 
WHERE action='forensic_investigation' 
ORDER BY created_at DESC LIMIT 10;
```

---

## ✅ Verification Checklist

After taking action:
- [ ] Toast notification appeared
- [ ] Workflow/Investigation ID shown
- [ ] Entry in incident actions
- [ ] Workflow in workflows page (if workflow)
- [ ] Database record created

---

## 🆘 Troubleshooting

### Workflow Not Created?
1. Check action keywords (block, isolate, deploy, etc.)
2. Ensure incident ID is valid
3. View backend logs: `tail -f /tmp/backend_new.log`

### Investigation Not Started?
1. Use investigation keywords (investigate, analyze, hunt)
2. Check for recent events on incident
3. Verify ForensicsAgent is initialized

### No Toast Notification?
1. Check browser console for errors
2. Verify backend is running: `curl http://localhost:8000/health`
3. Check API key in frontend `.env.local`

---

**Quick Access**:
- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:8000`
- Health Check: `http://localhost:8000/health`
- Workflows: `http://localhost:3000/workflows`

**Status**: ✅ **100% Attack Coverage - All Systems Operational**



