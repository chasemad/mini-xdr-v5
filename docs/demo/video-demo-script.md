# Mini-XDR Platform Demo Script

> **Purpose**: This is a narration script for a video demo of the Mini-XDR platform. Read this aloud while navigating through the application.

---

## Opening (30 seconds)

**[Show login page or dashboard loading]**

> "Welcome to Mini-XDR — an AI-powered Extended Detection and Response platform that gives security teams superhuman capabilities to detect, investigate, and respond to threats in real-time.

> Let me walk you through how this platform can transform your security operations."

---

## Page 1: Main Dashboard (2-3 minutes)

**[Navigate to: http://localhost:3000]**

### The Command Center

> "This is your Security Operations Command Center — the nerve center of your entire security program. Let's break down what you're seeing."

**[Point to top metrics cards]**

> "At the top, you have four critical metrics that tell you the health of your security posture at a glance:

> - **Total Incidents** — The number of security events requiring attention
> - **High Priority** — Critical threats that need immediate action
> - **Contained** — Incidents we've successfully neutralized
> - **AI Detected** — Threats our machine learning models caught automatically

> Notice the percentages underneath? That's comparing today's numbers to yesterday, so you instantly know if things are getting better or worse."

**[Point to Recent Activity section]**

> "Below that is your **Recent Activity Feed**. This shows the latest incidents in real-time. Click any incident to dive deeper. Notice the colored dots — red for high severity, orange for medium, green for low."

**[Point to Live Event Stream and Attack Surface cards]**

> "On the bottom, you see **Live Event Stream** showing real-time network activity, and your **Attack Surface** — a snapshot of what we're protecting: network sensors, enrolled agents, and honeypots."

**[Point to Phase 2 Intelligence cards]**

> "These Phase 2 Intelligence cards show you how smart your system is:
> - **ML Accuracy** — How accurate our threat detection models are
> - **Feature Store** — The hit rate of our feature engineering cache
> - **Agent Hub** — How many AI agents are actively working
> - **Training Data** — Samples collected to continuously improve our models"

### Key Takeaway

> "The dashboard answers one question instantly: 'What do I need to worry about right now?'"

---

## Page 2: Incidents Page (2 minutes)

**[Click "Incidents" in the sidebar]**

### Your Incident Queue

> "This is your Incident Queue — every security event that needs investigation lives here."

**[Point to filters at top]**

> "Use these filters to slice and dice your view:
> - Search by IP address or description
> - Filter by severity — High, Medium, or Low
> - Filter by status — Open, Contained, or Dismissed

> If you have 50 incidents and need to focus on the critical ones, just filter by 'High' severity."

**[Click on an incident card]**

> "Each incident card gives you everything you need at a glance:
> - The **source IP** that triggered the alert
> - The **risk score** — a percentage showing how dangerous this threat is
> - The **confidence** level — how sure our AI is about this classification
> - A description of what happened

> You can hit **Quick View** for a fast summary, or **Full Analysis** to go deep."

### Key Takeaway

> "The Incidents page is your prioritized to-do list. Work from top to bottom, highest severity first."

---

## Page 3: Incident Details (3-4 minutes)

**[Click "Full Analysis" on any incident]**

### Deep Dive Investigation

> "Now we're looking at a single incident in full detail. This is where your investigation happens."

**[Point to header]**

> "The header shows you critical context:
> - Incident number and current status
> - The source IP being investigated
> - How long the incident has been active
> - Whether our AI Council has verified this as a real threat"

**[Click through tabs]**

### Overview Tab

> "The **Overview** tab gives you the big picture — the AI's analysis, threat classification, and what our machine learning models think about this incident."

### Council Analysis Tab

> "The **Council Analysis** shows you our multi-AI verification system. We don't rely on just one model — we use a council of AI judges including Gemini, Grok, and GPT to verify threats. This reduces false positives dramatically."

### AI Agents Tab

> "The **AI Agents** tab shows you which autonomous agents are working on this incident:
> - **Attribution Agent** — identifies who is attacking you
> - **Containment Agent** — takes defensive actions
> - **Forensics Agent** — collects evidence

> Each agent works independently but coordinates through our orchestrator."

### Timeline Tab

> "The **Timeline** shows every event in chronological order. You can see exactly what happened, when, and in what sequence. This is crucial for understanding the attack chain."

### Evidence Tab

> "The **Evidence** tab collects all the artifacts — IP addresses, domains, file hashes — anything that can be used for further investigation or shared with your threat intelligence team."

### Response Actions Tab

> "The **Response Actions** tab is where you take action. You can:
> - Block the attacker's IP
> - Isolate affected hosts
> - Run additional scans
> - Escalate to senior analysts

> Many of these can execute with one click."

### Key Takeaway

> "Incident details give you the full story and the power to act. Everything you need to investigate and respond is in one place."

---

## Page 4: AI Agents Command Center (2-3 minutes)

**[Click "AI Agents" in the sidebar]**

### Meet Your AI Security Team

> "This is the Agent Command Center — where you interact with your autonomous AI security agents."

**[Point to the four agent cards at top]**

> "You have four specialized agents:

> - **Containment Orchestrator** — Evaluates incidents and executes blocking actions. It can block IPs, isolate hosts, and enforce security policies automatically.

> - **Attribution Agent** — Figures out WHO is attacking you. It identifies threat actors, correlates campaigns, and analyzes tactics and techniques.

> - **Forensics Agent** — Collects and preserves evidence. It maintains chain of custody and creates forensic timelines.

> - **Deception Agent** — Manages honeypots and engages attackers. It profiles adversaries and gathers intelligence.

> Each shows its current status — online or offline — and when it was last active."

**[Point to the chat interface]**

> "This chat interface lets you communicate with agents directly. Select an agent from the dropdown, then type natural language commands like:
> - 'Evaluate incident 123'
> - 'Analyze IP 8.8.8.8'
> - 'Run a threat hunting scan'

> The agent responds with analysis, confidence scores, and any actions it took."

### Key Takeaway

> "Your AI agents work 24/7. They're like having a team of expert analysts that never sleep, never take breaks, and can work on hundreds of incidents simultaneously."

---

## Page 5: Threat Hunting (2 minutes)

**[Click "Threat Hunt" in the sidebar]**

### Proactive Defense

> "Threat Hunting is about being proactive — going out and FINDING threats before they find you."

**[Point to Hunt Query Builder]**

> "This query builder lets you search across all your security data. Type queries like:
> - `eventid:cowrie.login.failed` — Find failed SSH attempts
> - `src_ip:192.168.*` — Find traffic from internal addresses
> - Pattern matching with regex for sophisticated searches"

**[Point to Quick Hunt Templates]**

> "Don't want to write queries? Use these templates:
> - **SSH Brute Force** — Find password guessing attacks
> - **Lateral Movement** — Find attackers moving inside your network
> - **Suspicious Agents** — Find unusual user agents like curl or wget
> - **Off-Hours Activity** — Find activity when nobody should be working"

**[Point to tabs]**

> "The other tabs let you:
> - Save queries for reuse
> - Track Indicators of Compromise
> - View analytics on your hunting effectiveness"

### Key Takeaway

> "Threat hunting flips the script. Instead of waiting for alerts, you actively search for attackers hiding in your environment."

---

## Page 6: Threat Intelligence (2-3 minutes)

**[Click "Intelligence" in the sidebar]**

### Know Your Enemy

> "Threat Intelligence is your database of known threats — the bad IPs, malicious domains, file hashes, and the threat actors behind them."

**[Point to IOCs tab]**

> "The **IOCs tab** shows Indicators of Compromise:
> - IP addresses from known attackers
> - Malicious domain names
> - File hashes of malware
> - Each has a threat level, confidence score, and source"

**[Point to Add IOC panel]**

> "You can manually add IOCs your team discovers. Just enter the type, value, threat level, and any tags. This enriches your detection capability."

**[Click Threat Feeds tab]**

> "**Threat Feeds** are external sources of intelligence. We integrate with:
> - Commercial feeds
> - Open source feeds like AlienVault OTX
> - Your own internal feeds

> Each feed shows status, last update, and how many IOCs it contributes."

**[Click Threat Actors tab]**

> "The **Threat Actors** tab profiles who attacks you:
> - Their aliases and motivations
> - Geographic regions they operate from
> - Industries they target
> - Their tactics, techniques, and procedures (TTPs)"

**[Click Campaigns tab]**

> "**Campaigns** track organized attack efforts — like a phishing campaign or ransomware operation. You can see start dates, target sectors, and associated IOCs."

### Key Takeaway

> "Threat Intelligence answers: Who attacks us? What do they use? How do they operate? This knowledge makes your defenses smarter."

---

## Page 7: Workflows (2-3 minutes)

**[Click "Workflows" in the sidebar]**

### Automated Response

> "Workflows automate your incident response. Instead of manually clicking through steps, you define the response once and let the system execute it automatically."

**[Point to capability cards]**

> "You can build workflows four ways:

> - **Natural Language** — Just describe what you want in plain English: 'When we see a brute force attack, block the IP and notify the team'

> - **Visual Designer** — Drag and drop actions onto a canvas. Connect them to build a response flow.

> - **Playbook Templates** — Start with pre-built playbooks for common scenarios like ransomware, DDoS, or phishing.

> - **Execution Engine** — Monitor running workflows in real-time."

**[Click Natural Language tab and show interface]**

> "Watch this — I type: 'Block any IP that has more than 5 failed login attempts in 60 seconds'

> The AI parses this into an actual workflow with triggers, conditions, and actions. It's like having a conversation with your security automation."

**[Point to Auto Triggers tab]**

> "The **Auto Triggers** tab shows workflows that run automatically:
> - SSH Brute Force Response — triggers after 5 failed logins
> - Malware Upload Response — triggers when malware hits our honeypot
> - Each shows success rate, average response time, and toggle to enable/disable"

### Key Takeaway

> "Workflows are your force multiplier. A single analyst can respond to thousands of incidents because automation handles the repetitive work."

---

## Page 8: 3D Visualizations (2 minutes)

**[Click "Visualizations" in the sidebar]**

### See the Battle

> "This is our 3D Visualization dashboard — a real-time view of the global threat landscape."

**[Point to interactive globe]**

> "The globe shows threats geographically. Each point is an attack — the color indicates severity:
> - Red for critical
> - Orange for high
> - Yellow for medium
> - Green for low

> Watch the animated lines — those are active attack paths showing traffic flow from attacker to target."

**[Toggle to Timeline view]**

> "The 3D Timeline shows threats chronologically. You can see how attacks progress over time, identify patterns, and spot coordinated campaigns."

**[Point to controls panel]**

> "Use these controls to:
> - Adjust animation speed
> - Filter by severity or region
> - Toggle attack paths and labels
> - Pause for detailed inspection"

### Key Takeaway

> "Visualization turns abstract data into intuitive understanding. You can literally see where attacks originate and how they spread."

---

## Page 9: Honeypot Monitoring (2 minutes)

**[Click "Honeypot" in the sidebar]**

### Your Digital Tripwires

> "This page monitors our T-Pot honeypot infrastructure. Honeypots are decoys — fake systems that attract attackers and let us study their methods."

**[Point to status cards]**

> "At a glance:
> - **Connection Status** — Are we connected to the honeypot?
> - **Active Honeypots** — How many traps are running
> - **Blocked IPs** — Attackers we've already stopped
> - **Recent Attacks** — Activity in the last 5 minutes"

**[Point to container list]**

> "Each honeypot container mimics a different vulnerable service:
> - **Cowrie** — SSH/Telnet honeypot catches brute force attacks
> - **Dionaea** — Catches malware and exploits
> - **Suricata** — Network intrusion detection

> You can start, stop, or restart any container from here."

**[Point to Recent Attacks section]**

> "When an attacker hits our honeypot, it appears here instantly. You see:
> - Their IP address
> - What port they targeted
> - What attack signature was detected
> - A one-click button to block them

> This is real attacker behavior — gold for threat intelligence."

### Key Takeaway

> "Honeypots let you study attackers without risk. You learn their tools, techniques, and targets — then use that knowledge to protect real systems."

---

## Page 10: Analytics (1-2 minutes)

**[Click "Analytics" in the sidebar]**

### Under the Hood

> "The Analytics page shows how our AI and ML systems perform."

**[Click ML Monitoring tab]**

> "**ML Monitoring** tracks our machine learning models:
> - Model accuracy over time
> - Feature importance rankings
> - Prediction distributions
> - Training data quality

> If a model starts degrading, you'll see it here first."

**[Click Explainable AI tab]**

> "**Explainable AI** tells you WHY the AI made a decision. For any prediction, you can see:
> - Which features influenced the decision most
> - How confident the model is
> - What similar incidents looked like

> This transparency builds trust. You're not blindly following a black box."

### Key Takeaway

> "Analytics keeps your AI honest. You can always understand and verify what the machines are doing."

---

## Closing (30 seconds)

**[Return to Dashboard]**

> "That's Mini-XDR — a complete platform for modern security operations:

> - **Detect** threats with AI and ML-powered analysis
> - **Investigate** with rich context and autonomous agents
> - **Respond** with automated workflows that execute in seconds
> - **Learn** from every attack to get smarter over time

> This isn't the future of security operations — it's available today. Thank you for watching."

---

## Demo Tips

### Before Recording
- [ ] Start both backend and frontend servers
- [ ] Have at least 3-5 incidents in the system
- [ ] Test all page navigations work
- [ ] Clear browser cache for smooth performance

### Pacing
- Speak slowly and clearly
- Pause 2-3 seconds when changing pages
- Let animations complete before speaking
- Total runtime: 15-20 minutes

### If Something Goes Wrong
- "Let me refresh that..."
- "Sometimes data takes a moment to load..."
- Stay calm, it happens in live demos

### Engagement Tips
- Vary your tone — don't be monotone
- Use phrases like "Notice how..." and "Watch this..."
- Connect features to real-world benefits
- End each section with a clear takeaway

---

*Script Version: 1.0*
*Last Updated: November 2024*
