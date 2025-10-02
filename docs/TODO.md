# 🛡️ Mini‑XDR MASTER README — Cursor Build Script (Enterprise‑Grade, Full Update)

> **Goal**: Ship a production‑grade, secure, and elegant Mini‑XDR platform with a world‑class, no‑nonsense UI/UX. Build and validate each piece step‑by‑step in Cursor, with **no shortcuts** on security, reliability, or testing. This README is the **single source of truth** for local dev, CI, security hardening, and AWS deployment (staging → production), plus T‑Pot/Cowrie and realistic enterprise simulations.
>
> **Assumptions**
>
> * Existing stack: FastAPI backend (async SQLAlchemy/Alembic), Next.js/React (App Router) + Tailwind, Kafka + Redis for orchestration, SQLite→Postgres, Cowrie/T‑Pot telemetry, some LLM agents via LangChain/MCP, 3D/advanced visualizations.
> * **AWS Secrets Manager already provisioned**; we will verify integration and rotate secrets.
> * We standardize on **Docker for all runtime surfaces**, and **AWS ECS Fargate** for first production deployment (simpler ops than EKS; good blast‑radius controls). Kafka runs via **Amazon MSK** in prod; Kafka+Zookeeper via Docker in dev.
> * UI/UX is **clean, minimal, enterprise**. No clipart, no corny icons: we use typographic hierarchy, spacing, motion with intent, and **accessible** contrasts.

---

## 0) 🔑 High‑Level Architecture & Quality Gates

**Services**

* **backend/** FastAPI app: AuthN/AuthZ, ingest, workflows, response actions, AI advisors.
* **frontend/** Next.js app: SOC views, workflows (React Flow), analytics, reports.
* **orchestrator/** Worker(s) handling long‑running actions, agent calls, Kafka consumers, Redis queues.
* **infra/** IaC (Terraform) for AWS (ECR, ECS/Fargate, ALB+WAF, RDS Postgres, ElastiCache Redis, MSK Kafka, S3, CloudWatch, IAM, Secrets Manager, Parameter Store, VPC, subnets, SGs).
* **observability/** OTel Collector, Prometheus/Grafana dashboards, CloudWatch metrics/alerts; Sentry for FE/BE.
* **tests/** full pyramid: unit, contract, e2e, load (Locust), chaos; adversary simulation (CALDERA/Atomic‑Red‑Team) isolated.

**Quality Gates (must pass before promotion)**

1. ✅ All unit + contract tests green; mutation score ≥ 75% (mutmut/pytest‑mutation optional).
2. ✅ Static analysis: mypy (strict), ruff/flake8, bandit, Trivy (images) → none critical.
3. ✅ Secrets scanning (gitleaks) → zero leaks; supply‑chain attestations (SLSA provenance via GitHub OIDC → AWS).
4. ✅ SBOM (Syft) published; vulnerability scan (Grype) ≤ medium (document mitigations if exceptions).
5. ✅ Load: p95 < 300ms for read APIs (staging baseline), orchestrations complete within SLOs.
6. ✅ Playbooks validated in simulator + one “live” path against T‑Pot/Cowrie or safe AWS target hosts.

> **Doc‑backed tips**
>
> * FastAPI performance: prefer `uvicorn[standard]` workers behind an ALB; enable `httpx` connection pooling.
> * SQLAlchemy 2.0 async engine with `asyncpg`; set `pool_recycle` and `pool_pre_ping`.
> * Next.js App Router with `edge` runtime for non‑sensitive static/SSR where possible; use `node` runtime for authenticated dashboards.

---

## 1) 🧱 Repository Layout (authoritative)

```
.
├─ backend/
│  ├─ app/
│  │  ├─ api/                    # routers: /auth, /ingest, /incidents, /response, /workflows, /integrations
│  │  ├─ core/                   # config, logging, security, rate‑limit, middleware
│  │  ├─ models/                 # SQLAlchemy ORM
│  │  ├─ schemas/                # Pydantic models
│  │  ├─ services/               # domain logic (incidents, ioc, forensics, advisors)
│  │  ├─ integrations/           # siem/soar, cloud, email, idp, threat intel
│  │  ├─ workflows/              # engine, validator, templates, executor
│  │  ├─ zero_trust/             # identity verification, device attestation, behavior analytics
│  │  ├─ mcp/                    # agent command registry + allowlists
│  │  ├─ mq/                     # kafka producers/consumers; redis queues
│  │  ├─ telemetry/              # otel, metrics, audit logging
│  │  └─ main.py                 # FastAPI app factory
│  ├─ alembic/                   # db migrations
│  ├─ pyproject.toml
│  └─ Dockerfile
│
├─ frontend/
│  ├─ app/                       # Next.js App Router
│  │  ├─ (dashboard)/            # role‑aware layouts
│  │  ├─ workflows/              # React Flow designer + monitor
│  │  ├─ analytics/
│  │  ├─ incidents/
│  │  ├─ components/             # Cards, Tables, Modals, Forms, Charts
│  │  └─ lib/                    # api client, auth, swr hooks
│  ├─ tailwind.config.ts
│  └─ Dockerfile
│
├─ orchestrator/
│  ├─ worker/                    # background executors, long‑running tasks
│  ├─ pyproject.toml
│  └─ Dockerfile
│
├─ observability/
│  ├─ otel-collector.yaml
│  └─ grafana/ dashboards/
│
├─ infra/
│  ├─ terraform/                 # modules: vpc, ecr, ecs, rds, redis, msk, s3, alb, waf, iam, secrets
│  ├─ docker-compose.dev.yml
│  └─ makefile
│
├─ tests/
│  ├─ unit/
│  ├─ contract/
│  ├─ e2e/
│  ├─ load/
│  └─ adversary/
│
├─ .github/workflows/            # CI/CD pipelines
├─ .env.example
└─ README.md (this)
```

> **Naming conventions**
>
> * Python: Ruff default rules; enforce `snake_case` modules, `PascalCase` classes, `lower_snake` tables/columns.
> * TS/React: ESLint + TypeScript strict; components `PascalCase`, hooks `useCamelCase`.

---

## 2) 🔐 Secrets, Config, and Identity

**Golden rules**

* **Never** commit secrets. Use AWS Secrets Manager for prod/staging; `.env` only for local.
* Rotate keys; short TTL; scoping by least privilege.
* Every external action executed by agents goes through **allowlisted command maps** and **policy checks**.

**.env.example (minimal)**

```
# Common
APP_ENV=local
LOG_LEVEL=INFO
ALLOWED_ORIGINS=https://mini-xdr.localhost,https://staging.example.com

# DB (local)
DB_HOST=postgres
DB_PORT=5432
DB_USER=xdr
DB_PASSWORD=devpassword
DB_NAME=xdr

# Redis
REDIS_URL=redis://redis:6379/0

# Kafka (local)
KAFKA_BROKERS=kafka:9092
KAFKA_SECURITY_PROTOCOL=PLAINTEXT

# API
API_HMAC_SECRET=local-signing-secret
JWT_SIGNING_KEY=local-jwt-hs256
RATE_LIMIT_REQUESTS_PER_MIN=120

# AWS (staging/prod retrieved via IAM role/Secrets Manager at runtime)
AWS_REGION=us-east-1
SECRETS_PREFIX=/mini-xdr/
```

**Runtime secret resolution (backend/app/core/config.py)**

```python
from functools import lru_cache
from pydantic_settings import BaseSettings
import boto3, os, json

class Settings(BaseSettings):
    app_env: str = "local"
    allowed_origins: str = ""
    db_host: str = "localhost"
    db_port: int = 5432
    db_user: str = "xdr"
    db_password: str = ""
    db_name: str = "xdr"
    redis_url: str = "redis://localhost:6379/0"
    kafka_brokers: str = "localhost:9092"
    jwt_signing_key: str = ""
    api_hmac_secret: str = ""

    class Config:
        env_file = ".env.local", env_file_encoding = "utf-8"

@lru_cache
def get_settings() -> Settings:
    s = Settings()
    if os.getenv("AWS_REGION") and os.getenv("SECRETS_PREFIX"):
        sm = boto3.client("secretsmanager", region_name=os.environ["AWS_REGION"])
        # fail-closed: required secrets must exist in SM
        for key in ["DB_PASSWORD","JWT_SIGNING_KEY","API_HMAC_SECRET"]:
            name = f"{os.environ['SECRETS_PREFIX']}{key}"
            resp = sm.get_secret_value(SecretId=name)
            setattr(s, key.lower(), json.loads(resp.get("SecretString","{}")).get(key) or resp.get("SecretString"))
    return s
```

**AuthN/AuthZ**

* **Inbound**: user JWT (short‑lived, HS256/RS256) + session hardening (IP/device hints), service‑to‑service **mTLS** (prod), and **HMAC** signatures for ingest/agent webhooks (`X‑MiniXDR‑Sig` header with SHA‑256 over body + nonce + ts).
* **RBAC**: roles (viewer, analyst, responder, admin, tenant‑admin) with **scope‑based permissions**; enforce at route + service layers.
* **Rate limiting**: token bucket (per key, per IP) with Redis backend.
* **CORS**: explicit origin allowlist per environment; never `*` in staging/prod.

**JWT middleware (snippet)**

```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer
import jwt, time

security = HTTPBearer()

async def require_jwt(creds = Depends(security)):
    try:
        payload = jwt.decode(creds.credentials, get_settings().jwt_signing_key, algorithms=["HS256","RS256"], options={"require": ["exp","iat","sub"]})
        if payload.get("nbf") and payload["nbf"] > int(time.time()):
            raise ValueError("Token not yet valid")
        return payload
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")
```

---

## 3) 🧪 Local Dev Environment (Docker Compose)

**infra/docker-compose.dev.yml**

```yaml
version: '3.9'
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: xdr
      POSTGRES_PASSWORD: devpassword
      POSTGRES_DB: xdr
    ports: ["5432:5432"]
    healthcheck: { test: ["CMD-SHELL","pg_isready -U xdr"], interval: 5s, retries: 10 }

  redis:
    image: redis:7-alpine
    command: ["redis-server","--appendonly","yes"]
    ports: ["6379:6379"]

  zookeeper:
    image: bitnami/zookeeper:3.9
    environment: { ALLOW_ANONYMOUS_LOGIN: "yes" }

  kafka:
    image: bitnami/kafka:3.7
    depends_on: [zookeeper]
    environment:
      KAFKA_CFG_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_CFG_LISTENERS: PLAINTEXT://:9092
      KAFKA_CFG_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_CFG_AUTO_CREATE_TOPICS_ENABLE: "true"
    ports: ["9092:9092"]

  otel:
    image: otel/opentelemetry-collector:0.104.0
    volumes: ["../observability/otel-collector.yaml:/etc/otel/config.yaml"]
    command: ["--config=/etc/otel/config.yaml"]

  backend:
    build: ../backend
    env_file: ["../.env.local"]
    depends_on: [postgres, redis, kafka]
    ports: ["8000:8000"]

  orchestrator:
    build: ../orchestrator
    env_file: ["../.env.local"]
    depends_on: [backend, redis, kafka]

  frontend:
    build: ../frontend
    env_file: ["../.env.local"]
    depends_on: [backend]
    ports: ["3000:3000"]
```

**Makefile (infra/makefile)**

```
.PHONY: up down logs migrate seed fmt lint unit contract e2e load
up:    ; docker compose -f docker-compose.dev.yml up -d --build
logs:  ; docker compose -f docker-compose.dev.yml logs -f --tail=200
migrate: ; docker compose -f docker-compose.dev.yml exec backend alembic upgrade head
seed:  ; docker compose -f docker-compose.dev.yml exec backend python -m app.scripts.seed
fmt:   ; ruff --fix backend orchestrator && prettier -w frontend
lint:  ; ruff backend orchestrator && eslint frontend --max-warnings=0
unit:  ; pytest -q tests/unit
contract: ; pytest -q tests/contract
e2e:   ; playwright test -c tests/e2e
load:  ; locust -f tests/load/locustfile.py
```

**First‑run checklist**

1. `cp .env.example .env.local` → adjust as needed (dev only).
2. `cd infra && make up && sleep 10 && make migrate && make seed`
3. Open **[http://localhost:3000](http://localhost:3000)** (frontend), **[http://localhost:8000/docs](http://localhost:8000/docs)** (OpenAPI).
4. Verify topics auto‑create: `docker exec -it infra-kafka-1 kafka-topics.sh --list --bootstrap-server kafka:9092`.

**Official doc‑aligned notes**

* Postgres: use UTF‑8, `timezone=UTC`.
* Docker: prefer non‑root users in images; set `PYTHONDONTWRITEBYTECODE=1`, `PYTHONUNBUFFERED=1`.

---

## 4) 🧩 Database & Migrations

* Use **Alembic** with autogenerate for models, but review diffs **manually**.
* Naming convention: `snake_case`, tables prefixed by domain when helpful (`response_workflows`, `incident_events`).
* **Idempotent seeds** for: roles, sample users, sample playbooks, demo incidents/IOCs.

**Fix for missing table** `(sqlite3.OperationalError) no such table: response_workflows`

1. Ensure model exists in `backend/app/models/response_workflow.py` and imported in `app/models/__init__.py`.
2. Create and apply migration:

```bash
docker compose -f infra/docker-compose.dev.yml exec backend alembic revision -m "create response_workflows" --autogenerate
docker compose -f infra/docker-compose.dev.yml exec backend alembic upgrade head
```

3. Add integration test to assert table presence and CRUD basic ops.

**SQLAlchemy patterns**

```python
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

async def get_workflow(session: AsyncSession, wf_id: str):
    res = await session.execute(select(Workflow).where(Workflow.id == wf_id))
    return res.scalar_one_or_none()
```

---

## 5) 🔗 Kafka/Redis Orchestration & MCP Guardrails

**Kafka topics (dev)**

* `incidents.v1`, `actions.requests.v1`, `actions.results.v1`, `advisors.signals.v1`.

**Producer/consumer (Python, aiokafka)**

```python
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer

producer = AIOKafkaProducer(bootstrap_servers=get_settings().kafka_brokers)
await producer.start(); await producer.send_and_wait("incidents.v1", msg_bytes)

consumer = AIOKafkaConsumer("actions.requests.v1", bootstrap_servers=get_settings().kafka_brokers, group_id="orchestrator")
await consumer.start()
async for msg in consumer:
    handle_action_request(msg)
```

**Redis token bucket (Starlette middleware)**

```python
import time, aioredis

redis = await aioredis.from_url(get_settings().redis_url)

async def token_bucket(key: str, limit: int, period: int) -> int:
    now = int(time.time())
    bucket = f"tb:{key}:{now//period}"
    count = await redis.incr(bucket)
    if count == 1:
        await redis.expire(bucket, period)
    remaining = max(0, limit - count)
    if remaining <= 0:
        raise HTTPException(429, detail="Rate limit exceeded")
    return remaining
```

**MCP/Agents Guardrails**

* All agent capabilities are **named functions** with strict schemas.
* Shell commands only via **predefined adapters** mapping to safe operations; no arbitrary `exec`.
* Enforce **allowlists** + **dry‑run** + **blast radius** estimate.

```python
class Command(BaseModel):
    name: Literal['isolate_host','rotate_key','block_ip']
    params: dict
```

---

## 6) 🧠 Phase 2 — Visual Workflow System (React Flow)

**Install**

```
npm i reactflow @dnd-kit/core @dnd-kit/sortable @dnd-kit/utilities framer-motion zod
```

**Frontend structure**

```
/frontend/app/workflows/
  page.tsx
  designer/
    WorkflowCanvas.tsx
    NodeTypes.tsx
    EdgeTypes.tsx
    Sidebar.tsx
    Toolbar.tsx
  templates/
    PlaybookLibrary.tsx
    TemplateCard.tsx
    TemplateImporter.tsx
  execution/
    ExecutionMonitor.tsx
    ProgressVisualization.tsx
    ExecutionControls.tsx
```

**Workflow DTO (backend)**

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class Node(BaseModel):
    id: str
    type: Literal['action','condition','decision','parallel','wait','data']
    config: dict

class Edge(BaseModel):
    id: str
    source: str
    target: str
    condition: Optional[str] = None  # JMESPath or CEL expression

class Workflow(BaseModel):
    id: str
    name: str
    version: int
    nodes: List[Node]
    edges: List[Edge]
```

**Validator**

* Ensure DAG (no cycles unless explicitly supported).
* Validate required configs by node type.
* **Safety pre‑checks**: any `action` requires policy check (`can_execute(user, action, target)`), dry‑run support.

**Execution**

* Persist graph as versioned JSON; execution creates a **run record** with node status, timings, logs, and artifacts.
* Live progress via SSE/WebSocket; nodes emit telemetry → Kafka → UI subscribes.

---

## 7) 🧠 Phase 2 — AI Response Advisor

**Python deps**: `sentence-transformers`, `faiss-cpu`, `networkx`, `scikit-optimize`.

**Capabilities**

* Similar incident retrieval (vector index over incident summaries/IOCs/TTPs).
* Risk/benefit scoring per action (historical success, blast radius estimate, rollback cost).
* Confidence indicators surfaced in UI; require **human approval** above impact threshold unless policy says otherwise.

**Safety**

* Advisors **suggest**; execution path always returns through validator and policy engine.

---

## 8) ⚙️ Response Action Library (Phase 3 expansion)

**Pattern**

```python
class Action(BaseModel):
    name: str
    inputs: ActionInputs
    def validate(self): ...
    async def dry_run(self): ...
    async def execute(self): ...  # idempotent, audited
```

**Examples**

* Network: WAF rule add, SG restrict, BGP blackhole (via provider API).
* Endpoint: EDR deploy trigger, AV quick scan, session kill.
* Cloud: S3 block public ACL, IAM key rotate, Lambda disable.
* Identity: force MFA, OAuth revoke, session invalidate.

**Testing**: Each action has **simulator** and **prod** driver implementations. Simulator runs in CI; prod guarded by feature flags and safeties.

---

## 9) 📊 Analytics & Reporting (Phase 4)

* Real‑time dashboards sourcing from Prometheus/OTel and application DB.
* Executive/SOC/Compliance dashboards with consistent design system.
* Scheduled reports: server‑side generation (WeasyPrint/ReportLab) stored in S3 with signed URLs.

---

## 10) 🌐 AWS Deployment (Staging → Prod, ECS Fargate)

**Why ECS Fargate**: least operational overhead, strong IAM boundaries, network isolation, blue/green via CodeDeploy or feature flags behind ALB.

**Terraform modules (infra/terraform/)**

* `vpc/` (private subnets for tasks, NAT for egress; no IGW to DB/Redis/MSK)
* `ecr/` (repos: backend, frontend, orchestrator)
* `kms/` (CMKs for RDS, Secrets, S3)
* `rds/` (Postgres 16, Multi‑AZ; auth via IAM tokens or secret)
* `elasticache/` (Redis 7, transit encryption, AUTH token)
* `msk/` (MSK Serverless or provisioned; TLS only)
* `secrets/` (AWS Secrets Manager paths `/mini-xdr/*`)
* `ecs/` (Fargate services + task roles, execution roles; CloudWatch logs)
* `alb/` + `waf/` (ALB with WAF managed rules; TLS 1.2+, HSTS)
* `s3/` (artifacts, reports, evidence; block public access; VPC endpoints)

**Example: ECS Task Definition (Terraform snippet)**

```hcl
resource "aws_ecs_task_definition" "backend" {
  family                   = "mini-xdr-backend"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 512
  memory                   = 1024
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.backend_task.arn
  container_definitions    = jsonencode([
    {
      name      = "backend"
      image     = "${aws_ecr_repository.backend.repository_url}:${var.image_tag}"
      essential = true
      portMappings = [{ containerPort = 8000, protocol = "tcp" }]
      environment = [
        { name = "AWS_REGION", value = var.region },
        { name = "SECRETS_PREFIX", value = "/mini-xdr/" }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = "/ecs/mini-xdr/backend"
          awslogs-region        = var.region
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])
}
```

**ALB + WAF (high‑level)**

* ALB in public subnets terminates TLS; target groups for FE/BE; stickiness disabled.
* WAF with AWS Managed Core + Bot Control; custom rules for geo/IP allowlists as needed.
* HSTS via ALB header insert (or at app level, see Appendix A).

**GitHub OIDC → AWS**

* Configure federated identity provider + role with trust policy restricted to your repo and branch/tags.

---

## 11) 🔭 Observability & Audit

**OpenTelemetry**

* Instrument FastAPI with `opentelemetry-instrumentation-fastapi` and `opentelemetry-exporter-otlp` to Collector.
* Frontend: use `@opentelemetry/api` + web tracer; propagate `traceparent` to backend.

**Metrics**

* request latencies, queue depths, action success rate, rollback rate, advisor acceptance rate, DB pool stats.

**Audit**

* Immutable append‑only audit table + optional S3/Object Lock; checksum chain for tamper detection.

---

## 12) 🧪 Testing & Simulation Framework (Phase 11)

**Layers**

* *Unit*: fast, pure Python/TS.
* *Contract*: FE/BE contract tests using mocked API or Pact‑like setup.
* *E2E*: Playwright for UI flows; test containers for services.
* *Load*: Locust scenarios: burst ingest, workflow fan‑out, long‑running actions.
* *Chaos*: kill orchestrator container, network partitions on Kafka; verify graceful degradation.
* *Adversary Sim*: CALDERA/Atomic‑Red‑Team in isolated VPC or lab; **never** target prod assets.

**Pass/fail gates in CI**: any critical regression blocks deploy; artifacts (screenshots, traces, coverage) uploaded.

---

## 13) 🖼️ UI/UX Standards (no fluff, enterprise‑class)

**Principles**

* Clear information hierarchy; minimal color; purposeful motion; keyboard navigable; WCAG AA.
* Components: Cards, Tables (virtualized), Tabs, Breadcrumbs, Empty states, Non‑blocking toasts, Modal/Sheet.
* Charts: crisp axes, sensible defaults; enable raw data drill‑through.

**Design tokens (tailwind.config.ts)**

* Spacing scale aligned to 4px grid; radii `xl/2xl` sparingly; shadows subtle; transitions 150–250ms.
* Typography: professional sans‑serif (e.g., Inter); monospace for logs/traces; no decorative icons unless conveying state.

**React Flow Canvas**

* Node shapes are simple, labelled; colors convey status only. Zoom/pan smooth; toolbar with Save/Validate/Run.

---

## 14) 🧯 Bugfixes Called Out (Immediate)

### 14.1 Category filter 400 (`Invalid category: all`)

* **Backend**: Accept `all` or treat missing `category` as all. Validate against enum but allow `None`.

```python
# /response/actions endpoint
category = request.query_params.get('category')
if not category or category == 'all':
    actions = list_all_actions()
else:
    actions = list_actions_by_category(category)
```

* **Frontend**: Ensure dropdown uses `''` (empty) for All, not literal `all`, or align with backend.

### 14.2 Empty Incident Timeline & IOCs

* Confirm ORM relationships and eager loads; ensure ingest populates `attack_timeline` and `iocs`.
* Add backfill script to synthesize timeline from events when missing; add DB constraints to prevent null graphs.
* Verify serializer exposes fields; FE expects arrays with `{ts, type, summary, ref}`.

### 14.3 Workflow Creation DB Error

* Create migration for `response_workflows`; add index on `(name, version)` unique per tenant.
* Add health check in startup: verify critical tables exist, else log fatal with remediation hint.

---

## 15) 🔒 Security Hardening (Phase 7)

**HTTP Security Headers**

* HSTS, Frame‑Options: DENY, Referrer‑Policy: strict‑origin‑when‑cross‑origin, CSP (nonce‑based), X‑Content‑Type‑Options, TLS 1.2+.

**Input validation**

* pydantic strict mode; max body sizes; file type allowlists; reject `text/html` uploads.

**Rate limits**

* per route per principal; exponential backoff on auth failures; device fingerprint hints.

**Zero‑Trust posture**

* continuous re‑auth for sensitive actions; privilege time‑boxing; just‑in‑time elevation with approvals.

**Secrets**

* rotate quarterly or on incident; detect in commits with gitleaks pre‑commit.

**Supply chain**

* pin versions; verify signatures (cosign); attestation stored with artifacts.

**Data**

* encrypt at rest (KMS); in transit (TLS); row‑level security where multi‑tenant.

**FastAPI middleware (Appendix A shows code)**

---

## 16) 🧰 CI/CD (GitHub Actions)

**.github/workflows/ci.yml (sketch)**

```yaml
name: ci
on:
  push:
    branches: [ main ]
  pull_request:

permissions:
  id-token: write
  contents: read

jobs:
  lint-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.12' }
      - uses: actions/setup-node@v4
        with: { node-version: '20' }
      - run: pip install -r backend/requirements-dev.txt
      - run: npm ci --prefix frontend
      - run: ruff backend orchestrator && bandit -r backend -x tests
      - run: pytest -q
      - run: npm run -C frontend lint && npm run -C frontend test -- --ci

  build-scan-push:
    needs: lint-test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.DEPLOY_ROLE_ARN }}
          aws-region: us-east-1
      - uses: aws-actions/amazon-ecr-login@v2
      - name: Build images
        run: |
          docker build -t ${{ steps.login-ecr.outputs.registry }}/mini-xdr-backend:${{ github.sha }} backend
          docker build -t ${{ steps.login-ecr.outputs.registry }}/mini-xdr-frontend:${{ github.sha }} frontend
      - name: Scan images (Trivy)
        uses: aquasecurity/trivy-action@0.24.0
        with:
          image-ref: ${{ steps.login-ecr.outputs.registry }}/mini-xdr-backend:${{ github.sha }}
          format: 'table'
          exit-code: '1'
          vuln-type: 'os,library'
          ignore-unfixed: true
      - name: Push images
        run: |
          docker push ${{ steps.login-ecr.outputs.registry }}/mini-xdr-backend:${{ github.sha }}
          docker push ${{ steps.login-ecr.outputs.registry }}/mini-xdr-frontend:${{ github.sha }}

  deploy-staging:
    needs: build-scan-push
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.DEPLOY_ROLE_ARN }}
          aws-region: us-east-1
      - name: Update task defs & service
        run: |
          ./infra/scripts/update-ecs.sh staging ${{ github.sha }}
```

**Gates**

* Block if Trivy severe vulns; allow override via security council approval (two‑person rule) recorded in audit.
* SBOM via Syft attached to release; Grype scan ≤ medium.

---

## 17) 🛰️ T‑Pot/Cowrie & Enterprise Simulation

**Cowrie integration**

* Ingest JSON logs via secure S3 drop or HTTPS endpoint with HMAC auth.
* Map Cowrie events → internal schema: session start/stop, credential attempts, file downloads, commands.

**Replay harness**

```python
# transforms cowrie event json -> incidents.v1 Kafka message
```

**Simulated enterprise**

* Ephemeral ASG/EC2 with `Environment=Sandbox`, SG limiting access. All actions target sandbox only by tag.

---

## 18) 🔄 Promotion Flow

1. **Dev** (local Docker) → **Staging** (ECS Fargate, staged secrets, MSK‑Serverless, RDS‑single AZ) → **Prod** (Multi‑AZ, backups, WAF strict).
2. Each promotion requires: changelog, migration plan, rollback plan, risk assessment, security sign‑off.

---

## 19) ✅ Operational Runbooks

* **Incident ingest fails**: check Kafka topic lag, consumer group health, DLQ; inspect OTel traces; verify MSK auth.
* **Workflow stuck**: examine node state; retry policy; compensating actions; timeouts per node type.
* **Key rotation**: rotate Secrets Manager secrets, restart tasks via rolling deploy.
* **DB migration gone wrong**: `alembic downgrade -1`; restore from snapshot in staging before prod.

---

## 20) 📚 References & Exemplars (authoritative docs)

* FastAPI docs; SQLAlchemy/Alembic docs; Next.js docs; React Flow docs; OpenTelemetry docs; Kafka/MSK docs; Redis/ElastiCache docs; Terraform & AWS provider docs; ECS Fargate docs; ALB/WAF docs; AWS Secrets Manager docs; GitHub OIDC docs; SLSA; OWASP ASVS & Top 10; MITRE ATT&CK / CALDERA; Atomic Red Team; Syft/Grype; Trivy.

> Tip: When in doubt, prefer **official docs** over blogs. Keep provider versions pinned and updated intentionally.

---

## 21) 🧭 Cursor Workflow — How to Use This README

1. Create tasks from each section in order: **secrets → local compose → migrations → orchestration → workflows UI → advisors → tests → AWS**.
2. Copy snippets as boilerplate; refactor paths to match your repo.
3. After each step: run `make lint unit contract` locally; commit; open PR; CI must pass **all gates**.
4. Promote to staging only when simulation tests pass and **security sign‑off** is recorded.

---

## 22) 📌 Success Criteria Recap

* 100% local functionality (no console errors, healthy compose stack).
* Dummy honeypot incidents flow end‑to‑end; timeline + IOCs populated.
* All 16 baseline actions executable in simulator; prod drivers gated & tested on sandbox infra.
* React Flow designer ships with versioned playbooks, validator, and monitored execution.
* AWS staging live with ECS/RDS/Redis/MSK; secrets via Secrets Manager; WAF on ALB.
* CI/CD with SBOM, scans, signed images, OIDC‑based deploy; observability + immutable audit.

---

### Appendix A — Example FastAPI Hardenings

```python
app = FastAPI()

@app.middleware('http')
async def security_headers(request, call_next):
    resp = await call_next(request)
    resp.headers.update({
        'Strict-Transport-Security': 'max-age=63072000; includeSubDomains; preload',
        'X-Frame-Options': 'DENY',
        'Referrer-Policy': 'strict-origin-when-cross-origin',
        'X-Content-Type-Options': 'nosniff',
        'Content-Security-Policy': "default-src 'self'; img-src 'self' data:; script-src 'self' 'nonce-...'; style-src 'self' 'unsafe-inline'"
    })
    return resp
```

### Appendix B — Rate Limit Example (Starlette + Redis)

```python
# Pseudocode: token bucket per user/ip/api-key
async def rate_limit(key: str, limit: int, period: int):
    # use Redis INCR with EXPIRE; return remaining tokens or raise 429
    ...
```

### Appendix C — GitHub Actions OIDC → AWS (snippet)

```yaml
permissions:
  id-token: write
  contents: read

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/gh-oidc-deployer
          aws-region: us-east-1
      - name: Build & Push ECR
        run: |
          make build push
```

### Appendix D — Terraform Module Skeletons

```
infra/terraform/
  main.tf
  providers.tf
  variables.tf
  outputs.tf
  modules/
    vpc/
    ecr/
    ecs/
    rds/
    elasticache/
    msk/
    alb/
    waf/
    secrets/
```

### Appendix E — Next.js App Router Auth & Security

* Use `next-auth` (or custom) with **JWT** sessions (`session.strategy = 'jwt'`).
* Set `cookies: { sessionToken: { sameSite: 'lax', secure: true, httpOnly: true } }`.
* Add **Content Security Policy** via middleware; strip `X-Powered-By` header.
* Ensure API client sets `X-Request-ID` and forwards `traceparent`.

### Appendix F — MSK + Glue Schema Registry (optional)

* Register Avro/JSON schemas; enforce compatibility (BACKWARD).
* Producer/consumer validate against schema to prevent payload drift.

### Appendix G — Sentry Integration

* BE: `sentry-sdk[fastapi]` with `traces_sample_rate=0.1` in staging.
* FE: `@sentry/nextjs`; mask PII; sample rates low in prod.

---

**End of MASTER README** — Use this document as your Cursor playbook. Build deliberately, validate ruthlessly, and ship confidently.

---

## 23) 🎨 UI/UX MASTER SPEC — Enterprise‑Grade (Authoritative)

> **North Star**: Operators resolve incidents faster with higher confidence and fewer clicks. Design must be **calm, legible, and predictable** under pressure. Every interaction is accessible, keyboardable, and auditable.

### 23.1 Information Architecture (IA)

**Top‑level nav (left rail, collapsible)**

1. **Overview** (Exec & SOC snapshot)
2. **Incidents** (queue, triage, detail)
3. **Hunt** (queries, pivots, saved hunts)
4. **Workflows** (designer, runs, library)
5. **Analytics** (dashboards, reports)
6. **Assets** (hosts, identities, cloud)
7. **Advisors** (AI suggestions, rationale)
8. **Settings** (tenant, roles, integrations)
9. **Audit** (immutable event log)

**Global actions**: Search ⌘K, New Workflow, Export, Help. Global breadcrumbs reflect tenant→area→entity.

**RBAC‑aware visibility**: Hide routes and actions user cannot access; show why when blocked (policy tooltip + doc link).

### 23.2 Layout System & Grids

* **Breakpoints**: sm 640, md 768, lg 1024, xl 1280, 2xl 1536. Stick to a 12‑column grid ≥ lg, 8‑column on md.
* **Density**: Default compact for tables (SOC operators), comfortable for forms.
* **Panels**: Master‑detail split for lists → detail (35% / 65%). Side sheets (48rem max) for secondary tasks.

### 23.3 Design Tokens (Tailwind)

```ts
// tailwind.config.ts (extract)
export const theme = {
  extend: {
    fontFamily: { sans: ['Inter', 'system-ui'], mono: ['JetBrains Mono', 'ui-monospace'] },
    colors: {
      brand: { 50:'#eef2ff', 500:'#4f46e5', 600:'#4338ca', 700:'#3730a3' },
      ink: { 900:'#0b1020', 700:'#1c243a', 500:'#3b4965' },
      success: { 500:'#16a34a' }, warning: { 500:'#f59e0b' }, danger: { 500:'#ef4444' }
    },
    borderRadius: { xl: '0.75rem', '2xl': '1rem' },
    boxShadow: { card: '0 2px 10px rgba(0,0,0,0.06)' }
  }
}
```

**Motion**: 150–250ms, ease‑out for entrance, ease‑in for exit. Reduce motion if `prefers-reduced-motion`.

### 23.4 Core Screens (specs)

#### A) Overview

* **KPI strip**: MTTR, Open Incidents, P95 API latency, Advisor Acceptance %. Quick filters per tenant/time.
* **Incident pulse chart** (24h): area or bar with anomaly markers. Click to pivot to filtered Incidents.
* **Top tactics/techniques** (ATT&CK) with counts; clicking applies filters.
* **Live queue**: last 10 events with severity badges; accessible live region (polite).

#### B) Incidents (Queue)

* Toolbar: Severity, Status, Source, Time, Tenant chips; Saved views; Export.
* Table columns: ID, Severity, Status, Title, Entities (host/user), First Seen, Last Activity, Owner.
* Row affordances: selection for bulk actions, keyboard nav (↑ ↓, Enter to open detail).

#### C) Incident Detail

* Header: severity pill + title, owner avatar, status dropdown, SLA countdown.
* Tabs: **Timeline**, **IOCs**, **Entities**, **Evidence**, **Response**, **Audit**.
* **Timeline**: vertical, time‑grouped; each item shows source, summary, tags; expandable raw JSON.
* **Response panel**: recommended actions (Advisor) with rationale + blast radius; Require confirm; dry‑run toggle.
* **Artifacts**: code blocks (mono), download buttons with signed URLs.

#### D) Workflows (Designer)

* Canvas with snapping grid; node palette (Search, Condition, Action, Wait, Parallel, Human‑approve).
* Inspector panel (right): schema‑driven forms (zod) for node config; validation messages inline.
* Controls: Validate (lint errors appear as badges on nodes), Save version, Run (dry‑run default), Monitor.
* Execution Monitor: Gantt‑like per node with status (Queued/Running/Success/Failed/Skipped), logs drawer.

#### E) Analytics

* Filters on top, cards below; drill‑through links to Incidents/Hunt. Export as CSV/PDF.

#### F) Settings

* Integrations list (tiles) with connection status; edit opens side sheet form.
* Role editor: matrix (resource x action) with presets; changes require 2‑person approval (modal explanation).

### 23.5 Component Library (shadcn/ui baseline)

**Essentials**: Card, Button, Badge, Tabs, Drawer/Sheet, Dialog, Tooltip, Toast, DataTable (virtualized), Form, Breadcrumbs, EmptyState, CodeBlock, CopyToClipboard, DiffView, JSONViewer, Chart (Recharts).

**Data Table spec**

* Virtualization for >1k rows, sticky headers, column pinning, resize, saved views. Inline filters, server‑side sort.
* Row selection with bulk actions area that appears docked.

**Empty States**

* Illustration‑free (no clipart). Title, brief copy, primary action, and secondary link to docs.

**States**: skeleton loaders for cards/tables; optimistic UI for status changes; toasts for non‑blocking success/error.

### 23.6 Accessibility (WCAG 2.2 AA)

* Color contrast ≥ 4.5:1; keyboard nav across all interactives; focus visible; ARIA live regions for updates.
* Forms: labels tied to inputs; error text below inputs; descriptions via `aria‑describedby`.
* Tables: `<th scope="col">`, row headers where suitable; announce sort state.

### 23.7 Security UX

* Dangerous actions use **two‑step**: confirm dialog + require reason. Show audit preview pre‑submit.
* Permissions denials show **why** (policy rule) and **how** to request access.
* Secret values are write‑only: show last rotated date, not the value.

### 23.8 Performance UX

* Streaming responses for long tasks (SSE/WebSocket) with progressive logs.
* Cache lists with SWR; indicate freshness timestamp.
* Avoid blocking spinners > 700ms; replace with skeletons and contextual progress.

### 23.9 Copy & Semantics

* Tone: calm, precise, action‑oriented ("Quarantine host" not "Do quarantine").
* Avoid fear language; surface risk numerically ("Estimated impact: Low").
* Tooltips provide definitions; link to docs in footers.

### 23.10 Theming & Tenancy

* Light/dark by system; brand color override per tenant via CSS variables. Tenant switcher top‑left.
* Clear tenant scoping on all pages; badge with tenant name/color.

### 23.11 Keyboard Shortcuts (discoverable via ⌘/?)

* Global: ⌘K command palette, G then I (Incidents), G then W (Workflows).
* Incident detail: E (Edit title), A (Assign), S (Change status), R (Open Response panel), . (Add note).

### 23.12 Charts (Recharts)

* Consistent axes, grid lines subtle; legends clickable to isolate series.
* Tooltips show exact timestamp and value, plus link to pivot.

### 23.13 Forms & Validation

* Zod schemas mirrored backend; server errors mapped to fields.
* Destructive toggles require re‑type to confirm for high‑risk settings.

### 23.14 Observability in UI

* Top nav environment badge (Dev/Staging/Prod). Health menu shows BE/DB/MSK/Redis status.
* Per‑request trace link (if user has permission) opens viewer with traceId.

### 23.15 QA Checklists (per screen)

* **Incidents Queue**: 12+ filters combine without layout shift; 10k rows perf; a11y pass.
* **Incident Detail**: timeline virtualization; raw view toggle; copy buttons labeled.
* **Workflow Designer**: prevents cyclic graph unless explicitly allowed; auto‑save; undo/redo.

---

## 24) 🧩 Frontend Implementation Notes (Cursor‑ready)

### 24.1 Directory & Component Patterns

```
/frontend/app/(dashboard)/
  layout.tsx        // left rail + top bar
  page.tsx          // Overview
  incidents/
    page.tsx        // queue
    [id]/page.tsx   // detail
  workflows/
    page.tsx        // list
    designer/page.tsx
    runs/[runId]/page.tsx
  analytics/page.tsx
  settings/
    page.tsx
    roles/page.tsx
    integrations/page.tsx
  advisors/page.tsx

/frontend/app/components/
  data-table/
  forms/
  charts/
  layout/
  primitives/
```

**API client**: `/app/lib/api.ts` with fetch wrappers that add `X-Request-Id`, `traceparent`, and manage retries (idempotent GETs only).

**State**: SWR for server state, Zustand for local UI state (e.g., designer panels).

### 24.2 Example Components

**IncidentSeverityBadge.tsx**

```tsx
export function IncidentSeverityBadge({ level }: { level: 'low'|'medium'|'high'|'critical' }) {
  const map = { low: 'bg-emerald-100 text-emerald-700', medium: 'bg-amber-100 text-amber-700', high: 'bg-orange-100 text-orange-700', critical: 'bg-red-100 text-red-700' }
  return <span className={`inline-flex items-center rounded-md px-2 py-1 text-xs font-medium ${map[level]}`}>{level}</span>
}
```

**EmptyState.tsx**

```tsx
export function EmptyState({ title, action, description }: { title: string; description?: string; action?: React.ReactNode }) {
  return (
    <div className="text-center p-10 border border-dashed rounded-2xl">
      <h3 className="text-lg font-semibold">{title}</h3>
      {description && <p className="mt-2 text-sm text-ink-500">{description}</p>}
      {action && <div className="mt-6">{action}</div>}
    </div>
  )
}
```

**JSONViewer.tsx** (with collapse and copy)

### 24.3 Workflow Designer UX Rules

* Nodes snap to 8px grid, auto‑layout option for DAG.
* Validation badges (red/yellow) show count; clicking focuses first error.
* Running workflow locks structure, but allows viewing logs and cancel/retry per node.
* Template library with tags (Containment, Forensics, Identity, Cloud). Import dialog validates schema before add.

### 24.4 Incident Detail UX Rules

* Timeline groups by hour; Ctrl/Cmd‑F filters timeline locally.
* Evidence attachments preview (text, JSON, PCAP summary) inline; large files stream download.
* Response actions: show **required inputs** up front, advanced collapsed; dry‑run default, prod toggle with warning.

### 24.5 Command Palette

* Spotlight modal (⌘K) with providers: Navigate, Actions, Entities, Docs. Results grouped with shortcut hints.

---

## 25) 🔐 Trust, Safety & Compliance in UX

* Privacy hints: show data minimization notice on high‑sensitivity pages (PII).
* Provide export with redaction options (hash IPs, redact usernames) per policy.
* Every action form displays **what will be logged** and retention policy.

---

## 26) 🧪 UX Testing Playbook

* **Heuristics**: Nielsen 10; severity scoring for issues.
* **Scenarios**: (1) SSH brute‑force storm; (2) Credential leak; (3) Suspicious Lambda. Success = time‑to‑confirm ≤ target.
* **A/B** for queue density & column sets; measure click count and task completion.
* **Accessibility**: axe automated + manual keyboard walkthrough.

---

## 27) 🧠 Cursor Deep‑Think Prompts (paste into tasks)

**Global Audit Prompt**

> Think deeply about our current FE structure and align it to the IA in §23.1. Refactor pages/components into the specified directories, ensure all routes are RBAC‑aware, add keyboard shortcuts per §23.11, and replace any generic spinners with skeletons. Verify all tables meet the Data Table spec and that errors are mapped to fields with zod. Output a diff plan and apply changes step by step with tests.

**Incidents UX Prompt**

> Review the Incidents queue and detail pages. Ensure filters are server‑side, virtualize lists, add saved views, and implement the status change optimistic flow with toasts and rollback. On detail, implement the tab set and Timeline virtualization with raw JSON view. Confirm copy tone matches §23.9.

**Workflows Designer Prompt**

> Implement the designer per §23.4D and §24.3. Add node palette, inspector, validation badges, save/versioning, and execution monitor. Ensure dry‑run is default; prod run requires confirm + reason. Add undo/redo and auto‑layout. Create template library with tags.

**Analytics Prompt**

> Build KPI cards and pulse chart; wire drill‑through to Incidents/Hunt. Add CSV/PDF export and ensure charts meet §23.12.

**Settings & Roles Prompt**

> Implement the role matrix with presets and 2‑person approval. Integration tiles with status, side‑sheet edit forms, and write‑only secret fields showing last rotation.

**Observability Prompt**

> Add environment badge, health menu, and per‑request trace links. Ensure errors show request‑id and a "Report with context" button.

**Accessibility Prompt**

> Run an a11y pass: focus outlines, ARIA, label associations, table semantics, color contrast ≥ 4.5:1. Fix or document all issues.

**Performance Prompt**

> Add SWR caching with revalidation, debounce search, paginate long lists, and measure TTI/LCP. Replace any large bundle charts with dynamic imports.

---

## 28) ✅ UI/UX Exit Criteria

* Operators complete core tasks with ≤ 7 clicks from queue to mitigated response.
* No console errors; Lighthouse a11y ≥ 95, perf ≥ 85 on Incidents queue.
* Keyboard parity: end‑to‑end triage possible without mouse.
* Every destructive/prod action requires confirmation + reason; audit entry is created with user, tenant, traceId.

---
