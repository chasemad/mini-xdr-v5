# Mini-XDR: Autonomous AI-Powered Detection & Response Lab

**Disclaimer:** Personal lab project for learning & portfolio. Not affiliated with any employer. All credentials are dummy or managed via Secrets Manager.

Built in 2025 (78+ commits). 9 specialized autonomous AI agents handle threat hunting, triage, containment, forensics, reporting, and more. Real-time ingestion from Wazuh/Osquery/Suricata via Kafka + Redis. Multi-model LLM reasoning (Claude 3.5 + Grok-4 + local Llama-3.1-70B).

## Architecture

```mermaid
flowchart LR
    subgraph Agents
        A1[Ingestion]
        A2[Hunting]
        A3[Triager]
        A4[Containment]
        A5[Forensics]
        A6[Deception]
        A7[Attribution]
        A8[DLP]
        A9[Reporting]
    end

    subgraph Pipeline
        K[(Kafka)]
        R[(Redis Streams)]
        DB[(PostgreSQL)]
    end

    subgraph AI
        C1[Claude 3.5]
        G4[Grok-4]
        L7[Llama 3.1 70B (local)]
    end

    subgraph Frontend
        UI[Streamlit / Next.js Dashboard]
    end

    A1 --> K
    Agents --> R
    K --> R
    R --> DB
    DB --> UI
    R --> UI

    R --> C1
    R --> G4
    R --> L7
    C1 --> R
    G4 --> R
    L7 --> R

    UI -->|Feedback| Agents
```

_Source file: `docs/diagrams/mini-xdr-architecture.mmd`_

## Current Status

| Capability | Status | Notes |
| --- | --- | --- |
| 9 autonomous agents (ingestion, hunting, triage, containment, forensics, deception, attribution, DLP, reporting) | ✅ | LangChain + FastAPI orchestration with Redis/Kafka event bus |
| Real-time pipeline | ✅ | Kafka + Redis Streams + PostgreSQL; Wazuh/Osquery/Suricata inputs |
| Multi-LLM reasoning | ✅ | Claude 3.5 + Grok-4 + local Llama-3.1-70B with routing |
| Dashboard | ⚠️ | Basic Streamlit/Next.js flows; polishing UI/UX and charts |
| macOS/Linux EDR agents | ⚠️ | Enrollment + live response in progress |
| Production hardening | ⚠️ | K8s autoscaling + additional zero-trust policies underway |

## Highlights

- Autonomous pipeline: ingestion → enrichment → ML ensemble scoring → agent-driven containment and reporting
- AI everywhere: retrieval-augmented reasoning and cross-model debate (Claude/Grok/Llama) for difficult investigations
- Threat intel aware: AbuseIPDB/VirusTotal hooks, signature + behavior detection, and deception-driven IOC collection
- Observability: Prometheus metrics, health endpoints, and structured audit trails for every automated action
- Extensible: modular agents, pluggable playbooks, and message-driven integrations (Kafka, Redis, HTTP webhooks)

## How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/chasemad/mini-xdr-v5.git
   cd mini-xdr-v5
   ```
2. (Optional) Create a Python env and install deps:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Configure environment (dummy defaults provided):
   ```bash
   cp backend/env.example .env
   # Set API_KEY, OPENAI_API_KEY, XAI_API_KEY, etc. (demo-minixdr-api-key / demo-tpot-api-key are placeholders)
   ```
4. Bring up the stack:
   ```bash
   docker-compose up -d
   ```
5. Check services:
   ```bash
   docker-compose ps
   curl -s http://localhost:8000/health
   ```
6. Run a quick ingest/response smoke test (optional):
   ```bash
   ./tests/run_all_tests.sh  # or use tests/demo_chat_integration.sh for the UI flow
   ```

## Demo

- 5-minute overview (unlisted): [insert your YouTube link here]
- Agent cycle GIF:

  ![Agent cycle placeholder](docs/media/agent-cycle.gif)

  _(Replace `docs/media/agent-cycle.gif` with your own capture.)_

## Repo Layout

- `backend/` — FastAPI services, agents, ML pipelines
- `frontend/` — Next.js dashboard + Streamlit utilities
- `src/` — pointer to code locations for contributors
- `docs/` — runbooks, architecture notes, and Mermaid diagrams
- `examples/attack-logs/` — sanitized attack traces for sharing/demos

## Contributing

Pull requests welcome. This is a living project—improvements, docs, and integrations are appreciated.
