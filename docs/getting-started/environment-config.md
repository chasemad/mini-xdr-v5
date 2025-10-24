# Environment Configuration

## Backend (`backend/.env`)

### Core Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `API_HOST` | `127.0.0.1` | Bind address for uvicorn. |
| `API_PORT` | `8000` | API port. Keep in sync with frontend `NEXT_PUBLIC_API_BASE`. |
| `UI_ORIGIN` | `http://localhost:3000` | Comma-separated origins allowed by CORS (`settings.cors_origins`). |
| `API_KEY` | _(none)_ | Required for API key-authenticated routes (`backend/app/security.py`). Generate 64+ char secrets per environment. |
| `JWT_SECRET_KEY` | _(none)_ | Secret key for JWT token signing (`backend/app/auth.py`). Required for user authentication. Generate a strong random secret. |
| `ENCRYPTION_KEY` | _(none)_ | Key for encrypting sensitive data. Generate a strong random secret. |

### Database Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `DATABASE_URL` | `sqlite+aiosqlite:///./xdr.db` | Async SQLAlchemy DSN. Replace with `postgresql+asyncpg://user:pass@host/db` for Postgres. |

### Detection & Response Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `FAIL_WINDOW_SECONDS` | `60` | Detection window (`settings.fail_window_seconds`). |
| `FAIL_THRESHOLD` | `6` | Failures before incident triggers. |
| `AUTO_CONTAIN` | `false` | Enables automatic responder actions. Mirrors `settings.auto_contain`. |
| `ALLOW_PRIVATE_IP_BLOCKING` | `true` | Enable blocking of private IP addresses (for testing). |

### Honeypot & Deception Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `HONEYPOT_HOST` | `34.193.101.171` | T-Pot honeypot host address. |
| `HONEYPOT_USER` | `admin` | SSH username for honeypot access. |
| `HONEYPOT_SSH_KEY` | `~/.ssh/mini-xdr-tpot-key.pem` | Path to SSH private key for honeypot access. |
| `HONEYPOT_SSH_PORT` | `64295` | SSH port for honeypot access. |
| `TPOT_API_KEY` | _(none)_ | API key for T-Pot honeypot management. |
| `TPOT_HOST` | _(none)_ | Alternative T-Pot host (overrides HONEYPOT_HOST). |
| `TPOT_SSH_PORT` | _(none)_ | Alternative T-Pot SSH port. |
| `TPOT_WEB_PORT` | _(none)_ | T-Pot web interface port. |

### AI & LLM Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `LLM_PROVIDER` | `openai` | LLM provider (openai, xai). |
| `OPENAI_API_KEY` | _(none)_ | OpenAI API key for AI features. |
| `OPENAI_MODEL` | `gpt-4` | OpenAI model to use. |
| `XAI_API_KEY` | _(none)_ | xAI API key for alternative LLM provider. |
| `XAI_MODEL` | `grok-beta` | xAI model to use. |

### Threat Intelligence Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `ABUSEIPDB_API_KEY` | _(none)_ | AbuseIPDB API key for threat intelligence. |
| `VIRUSTOTAL_API_KEY` | _(none)_ | VirusTotal API key for malware analysis. |

### ML & Model Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `ML_MODELS_PATH` | `./models` | Path to ML model files. |
| `AUTO_RETRAIN_ENABLED` | `true` | Enable automatic ML model retraining. |
| `AGENT_API_KEY` | _(none)_ | API key for agent operations. |

### Agent HMAC Credentials

Each agent requires four HMAC credential variables. Available agents: containment, attribution, forensics, deception, hunter (predictive), rollback.

| Variable Pattern | Description |
| --- | --- |
| `{AGENT_NAME}_AGENT_DEVICE_ID` | Unique device identifier for agent. |
| `{AGENT_NAME}_AGENT_PUBLIC_ID` | Public identifier for agent authentication. |
| `{AGENT_NAME}_AGENT_HMAC_KEY` | HMAC key for message signing. |
| `{AGENT_NAME}_AGENT_SECRET` | Shared secret for agent verification. |

Example for containment agent:
- `CONTAINMENT_AGENT_DEVICE_ID`
- `CONTAINMENT_AGENT_PUBLIC_ID`
- `CONTAINMENT_AGENT_HMAC_KEY`
- `CONTAINMENT_AGENT_SECRET`

### Secrets Management

| Variable | Default | Description |
| --- | --- | --- |
| `SECRETS_MANAGER_ENABLED` | `false` | When `true`, `backend/app/config.py` loads shared secrets from AWS Secrets Manager via `backend/app/secrets_manager.py`. |

### Policy Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `POLICIES_PATH` | `../policies` | Path to policy definition files. |

Store real values in a secrets manager for staging/production and export them before starting the backend.

## Frontend (`frontend/.env.local`)

| Variable | Default | Description |
| --- | --- | --- |
| `NEXT_PUBLIC_API_BASE` | `http://localhost:8000` | Base URL used by `frontend/app/lib/api.ts`. |
| `NEXT_PUBLIC_API_KEY` | _(none)_ | Propagated to `x-api-key` header for API requests. Required if backend enforces API keys. |

## Global Notes

- The backend automatically calls `_load_secrets_on_init` in `backend/app/config.py`; ensure IAM
  permissions are available when enabling AWS Secrets Manager.
- APScheduler runs background jobs (`process_scheduled_unblocks`, `background_retrain_ml_models`) once
  the FastAPI app starts. Tune intervals in `backend/app/main.py` if environment-specific behaviour is
  required.
- For container deployments, inject environment variables through task definitions or Kubernetes
  secrets; do **not** bake credentials into images.
