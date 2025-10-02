# NLP Workflow Parser Coverage & Gap Analysis

## Overview
The NLP workflow parser converts natural language requests into structured response workflows. High accuracy requires that the parser identify verbs, noun phrases, threat indicators, and context to map to the action registry (`backend/app/advanced_response_engine.py`).

This document enumerates:
- Core coverage by category.
- Known gaps & fallbacks.
- Testing strategy & datasets.
- Roadmap toward full chat coverage including agent capabilities.

## Action Categories & Supported Phrases

### Network Defense
- **block_ip / block_ip_advanced** – “block IP 10.0.0.5”, “ban the attacker”, “block attacking IP ranges”.
- **deploy_firewall_rules** – “deploy firewall rules”, “establish DDoS protection”, “implement network containment”.
- **api_rate_limiting** – “set up rate limiting”, “throttle API calls”.
- **capture_network_traffic** – “perform traffic analysis”, “collect netflow logs”.

**Gaps:**
- Need recognition of CDN/WAF specifics (e.g., “enable CloudFront shield”).
- No direct mapping for network segmentation actions beyond isolation.

### Endpoint Containment
- **isolate_host / isolate_host_advanced** – “isolate the host”, “host isolation”, “quarantine endpoint”.
- **memory_dump_collection** – “memory dump”, “capture forensic memory”.
- **terminate_process** – “kill malicious process”.

**Gaps:**
- Distinguish single host vs fleet isolation (should map to automation for multiple endpoints).
- Recognize synonyms like “lockdown workstation”.

### Identity & Access
- **reset_passwords** – “force password reset”, “rotate credentials”, “credential stuffing defense”.
- **enforce_mfa** – “enforce MFA”, “require multi-factor authentication”.
- **disable_user_account** – “disable compromised user account”, “suspend the user”.
- **revoke_user_sessions** – “terminate active sessions”, “logout all users”.

**Gaps:**
- Support for privileged access management actions (e.g., “rotate admin tokens”).

### Email Security
- **quarantine_email** – “quarantine emails”, “pull phishing message”.
- **block_sender** – “block sender domain”, “ban the sender”.

**Gaps:**
- Additional actions for campaign analysis, mailbox forensics.

### Data Protection
- **enable_dlp**, **encrypt_sensitive_data**, **backup_critical_data** – recognized via key phrases (“prevent data exfiltration”).

**Gaps:**
- Need specialized parsing for cloud data stores, SaaS connectors.

### Communication / Case Management
- **alert_security_analysts** – “notify the SOC”, “alert analysts”.
- **create_incident_case** – “open incident case”.

## Fallback Behavior
When no actions detected, `frontend/app/components/NaturalLanguageInput.tsx` surfaces curated templates and highlights parser feedback. Aim to minimize fallback reliance by expanding parser patterns and leveraging LLM-assisted extraction (via OpenAI) when key phrases missing.

## Testing Strategy
- Unit tests (`tests/test_nlp_parser.py`) cover malware, credential stuffing, DDoS, phishing, insider threat scenarios.
- Expand with dataset-driven tests (JSON or CSV) representing SOC analyst prompts.
- Consider fuzzing with synthetic variations (tense changes, typos, multi-step). Integrate with CI.

## Roadmap for Full Coverage
1. **Phrase Expansion & Synonyms** – maintain mapping table with synonyms, industry jargon, vendor-specific language.
2. **Intent Classification Layer** – categorize prompts (Response, Investigation, Automation, Reporting). Use heuristics or LLM.
3. **Dynamic Parameter Extraction** – parse IPs, domains, user IDs, hostnames using regex & spaCy.
4. **LLM Backoff** – when pattern-based parsing yields <2 actions or low confidence, call `parse_workflow_from_natural_language` with AI mode (requires OpenAI key). Return structured response after verifying action types exist in registry.
5. **Agent Mapping** – align actions with available agents (containment agent, forensics agent). If action lacks agent/tool support, warn user or propose alternative.
6. **Trigger Integration** – automatically propose trigger creation when NLP request fits repeatable pattern (e.g., “every time phishing occurs…”). Attach metadata to `WorkflowTrigger` model.
7. **Evaluation Metrics** – track parser accuracy by logging success/fallback, average actions detected, confidence score distribution.

## Distinguishing Request Types
Implement classifier that labels prompt as one of:
- `Incident Response` (actions & containment)
- `Investigation / Hunt`
- `Automation Configuration` (create triggers, schedules)
- `Reporting / Summary`
- `Informational Q&A`

Depending on label:
- Response/Automation → convert to workflow or trigger.
- Investigation → return search queries, hunts.
- Reporting → call analytics modules.
- Q&A → defer to knowledge base or LLM.

Each type should surface relevant agents (e.g., Response Engine vs Investigation Agent) and guardrails (approval requirements, simulation if required).

