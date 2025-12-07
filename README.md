# Mini-XDR: Autonomous AI-Powered Detection & Response Platform

<div align="center">

**A next-generation Extended Detection and Response (XDR) platform featuring a swarm of 12 AI agents, multi-LLM reasoning, and real-time honeypot integration.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-15-black.svg)](https://nextjs.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Personal lab project for learning & portfolio. Not affiliated with any employer.*

</div>

---

## 🎯 Overview

Mini-XDR is an autonomous AI-powered threat detection and response system built in 2025. It implements a **two-layer intelligence architecture**:

- **Layer 1 (Fast ML)**: Traditional ML ensemble processes events in <50ms using 79-dimensional feature extraction
- **Layer 2 (Council of Models)**: Multi-LLM reasoning with Gemini, Grok, and OpenAI for deep analysis on uncertain predictions

The platform orchestrates **12 specialized AI agents** that collaborate on threat detection, containment, forensics, attribution, deception, and automated response—all coordinated through an advanced orchestration framework with conflict resolution.

---

## 🤖 AI Agent Swarm

Mini-XDR deploys **12 specialized AI agents** that work collaboratively:

| Agent | Capability | Key Features |
|-------|------------|--------------|
| **Containment Agent** | Autonomous threat response | LangChain orchestration, IP blocking, host isolation, honeypot-aware containment |
| **Forensics Agent** | Digital forensics & evidence | Evidence collection, chain of custody, timeline reconstruction, PCAP analysis |
| **Attribution Agent** | Threat actor identification | Campaign correlation, TTP analysis, infrastructure clustering, actor signatures |
| **Deception Agent** | Honeypot management | Dynamic honeypot deployment (Cowrie, Dionaea, Conpot), attacker profiling |
| **EDR Agent** | Endpoint detection & response | Process control, file quarantine, registry monitoring |
| **IAM Agent** | Identity & access management | Active Directory monitoring, credential protection, privilege analysis |
| **DLP Agent** | Data loss prevention | Sensitive data scanning, exfiltration blocking, upload monitoring |
| **Predictive Hunter** | Proactive threat hunting | Time-series forecasting, behavioral baselines, hypothesis generation |
| **NLP Analyzer** | Natural language interface | Semantic search, query parsing, analyst-friendly insights |
| **Ingestion Agent** | Multi-source ingestion | Syslog, CEF, JSON normalization, deduplication |
| **Coordination Hub** | Multi-agent orchestration | Conflict resolution, decision optimization, collaborative intelligence |
| **LangChain Orchestrator** | ReAct-style orchestration | GPT-4 powered tool selection, multi-step reasoning |

---

## 🧠 Council of Models

The Council provides **multi-LLM reasoning** for uncertain predictions (ML confidence between 0.5-0.9):

### Gemini Judge
Deep reasoning engine with 1M+ token context. Analyzes the 79-dimensional feature vector and event timeline to verify or override ML predictions. Provides explainable reasoning for SOC analysts.

### Grok Intel
Real-time threat intelligence from X (Twitter). Queries security researcher discussions about file hashes, domains, and IPs. Returns social proof of malicious activity.

### OpenAI Remediation
Generates precise remediation scripts:
- Firewall rules (Palo Alto, Cisco, iptables)
- PowerShell/Bash response scripts
- Network isolation commands
- Step-by-step action plans

---

## 🔬 Machine Learning Stack

### Feature Engineering
Extracts **79 features** from security events covering:
- Temporal patterns (event frequency, time windows)
- Network behavior (port diversity, connection patterns)
- Authentication metrics (failed logins, brute force indicators)
- Payload analysis (command patterns, malware signatures)
- Behavioral baselines (deviation scores)

### Detection Models

| Model | Type | Purpose |
|-------|------|---------|
| **XDRThreatDetector** | Deep Neural Network | Multi-class threat classification |
| **XDRAnomalyDetector** | Autoencoder | Unsupervised anomaly detection via reconstruction error |
| **LSTMAttentionDetector** | LSTM + Attention | Sequential attack pattern recognition |
| **IsolationForest** | Ensemble | Outlier-based anomaly detection |
| **XGBoost** | Gradient Boosting | High-precision threat scoring |

### Advanced Capabilities
- **Ensemble Optimizer**: Weighted voting with model agreement scoring
- **Federated Learning**: Distributed model training across deployments
- **Concept Drift Detection**: Automatic model retraining on distribution shift
- **Explainable AI**: Feature importance and decision reasoning

---

## 🍯 T-Pot Honeypot Integration

Real-time integration with [T-Pot](https://github.com/telekom-security/tpotce) honeypot infrastructure:

- **SSH Monitoring**: Secure tunnel to T-Pot for log streaming
- **Supported Honeypots**: Cowrie, Dionaea, Suricata, Conpot, ElasticHoney, and more
- **Automated Response**: UFW-based IP blocking, container management
- **Elasticsearch Queries**: Direct access to honeypot attack data

**Data Flow:**
1. Attackers hit T-Pot honeypots (SSH, HTTP, malware, etc.)
2. Honeypots log attacks to JSON files
3. `TPotConnector` streams events via SSH tunnel
4. Events processed through ML pipeline → AI agents
5. Defensive actions executed on T-Pot infrastructure

---

## 📋 Playbook Engine

SOAR-style workflow automation with built-in playbooks:

- **SSH Brute Force Response**: Rate limiting → IP blocking → forensic collection
- **Malware Detection**: File quarantine → hash analysis → attribution lookup
- **Lateral Movement**: Host isolation → credential reset → network sweep
- **Data Exfiltration**: Connection termination → data audit → compliance notification
- **Comprehensive Investigation**: Full forensic case with timeline reconstruction

Playbooks support:
- Conditional execution based on incident context
- Parallel step execution with dependency graphs
- Rollback actions for containment reversal
- AI-powered action suggestions

---

## 🖥️ SOC Dashboard

Modern Next.js 15 frontend with:

| Component | Description |
|-----------|-------------|
| **Tactical Decision Center** | Real-time incident queue with AI recommendations |
| **Enhanced AI Analysis** | LLM-generated threat insights and context |
| **Workflow Designer** | Visual drag-and-drop playbook builder |
| **Threat Status Bar** | Live attack metrics and trend indicators |
| **Action History Panel** | Full audit trail of response actions |
| **Honeypot Dashboard** | T-Pot attack monitoring and container control |
| **NLP Query Interface** | Natural language incident search |

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Git
- (Optional) Python 3.11+ for local development

### 1. Clone & Configure

```bash
git clone https://github.com/chasemad/mini-xdr-v5.git
cd mini-xdr-v5

# Configure API keys (optional - system works without them)
cp backend/env.example backend/.env
# Edit .env to add: OPENAI_API_KEY, XAI_API_KEY, ABUSEIPDB_API_KEY, VIRUSTOTAL_API_KEY
```

### 2. Start the Stack

```bash
docker-compose up -d
```

### 3. Verify Services

```bash
docker-compose ps
curl -s http://localhost:8000/health | jq
```

### 4. Access the Dashboard

Open http://localhost:3000 in your browser.

### 5. Generate Test Data (Optional)

```bash
./scripts/inject-fake-attack-auth.sh
```

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|--------------|
| **Backend** | FastAPI, Python 3.11+, SQLAlchemy (async), APScheduler |
| **Frontend** | Next.js 15, React, TypeScript, TailwindCSS |
| **Database** | PostgreSQL 15 (asyncpg) |
| **Cache** | Redis 7 |
| **ML/AI** | PyTorch, scikit-learn, XGBoost, LangChain |
| **LLMs** | OpenAI GPT-4, Google Gemini, xAI Grok |
| **Honeypot** | T-Pot, Cowrie, Dionaea, Suricata |
| **Infrastructure** | Docker, Kubernetes (optional) |

---

## 📁 Repository Structure

```
mini-xdr/
├── backend/                 # FastAPI services and AI agents
│   └── app/
│       ├── agents/         # 12 specialized AI agents
│       ├── council/        # Gemini, Grok, OpenAI nodes
│       ├── ai_models/      # ML model wrappers
│       ├── orchestrator/   # LangGraph state machine
│       └── ...
├── frontend/               # Next.js 15 SOC dashboard
│   ├── app/               # App Router pages
│   └── components/        # React components
├── docs/                   # Comprehensive documentation
├── models/                 # Trained ML model files
├── scripts/                # Utility and demo scripts
├── config/                 # Configuration templates
├── ops/                    # Infrastructure manifests
└── tests/                  # Test suite
```

---

## 🤝 Contributing

Contributions are welcome! This is an active project focused on advancing autonomous threat detection.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Ensure all tests pass (`./tests/run_all_tests.sh`)
5. Commit with a descriptive message
6. Push and open a Pull Request

### Areas for Contribution

- New AI agents or agent capabilities
- Additional ML models or detection techniques
- UI/UX improvements and visualizations
- Documentation and tutorials
- Threat intelligence source integrations
- Performance optimizations

### Development Setup

See [docs/getting-started/local-setup.md](docs/getting-started/local-setup.md) for detailed instructions.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ for the security community**

[Documentation](docs/README.md) · [Report Bug](../../issues) · [Request Feature](../../issues)

</div>
