# Documents 11-12: DevOps/CI/CD & Regulatory Standards

---

## 11: DevOps & CI/CD

**Current State:** Manual deployments, basic K8s configs  
**Target State:** Automated testing, zero-downtime deployments, GitOps  
**Priority:** P0 (Required for reliability)  
**Time:** 2-3 weeks

---

### Task 1: CI/CD Pipeline (GitHub Actions)

**File:** `.github/workflows/ci-cd.yml` (NEW)

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  AWS_REGION: us-east-1
  ECR_REPOSITORY: mini-xdr-backend
  EKS_CLUSTER: mini-xdr-prod

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          cd backend
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-asyncio
      
      - name: Run linting
        run: |
          pip install ruff
          ruff check backend/app
      
      - name: Run tests
        env:
          DATABASE_URL: postgresql+asyncpg://postgres:test@localhost:5432/test
          REDIS_URL: redis://localhost:6379
        run: |
          cd backend
          pytest --cov=app --cov-report=xml --cov-report=term
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./backend/coverage.xml

  build-backend:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}
      
      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v1
      
      - name: Build and push Docker image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG \
                       -t $ECR_REGISTRY/$ECR_REPOSITORY:latest \
                       -f ops/Dockerfile.backend .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest
  
  build-frontend:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'
          cache-dependency-path: frontend/package-lock.json
      
      - name: Install dependencies
        run: |
          cd frontend
          npm ci
      
      - name: Run tests
        run: |
          cd frontend
          npm test -- --passWithNoTests
      
      - name: Build
        run: |
          cd frontend
          npm run build
      
      - name: Build and push Docker image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        run: |
          docker build -t $ECR_REGISTRY/mini-xdr-frontend:${{ github.sha }} \
                       -f ops/Dockerfile.frontend .
          docker push $ECR_REGISTRY/mini-xdr-frontend:${{ github.sha }}
  
  deploy-staging:
    needs: [build-backend, build-frontend]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    environment: staging
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure kubectl
        uses: azure/k8s-set-context@v3
        with:
          method: kubeconfig
          kubeconfig: ${{ secrets.KUBE_CONFIG_STAGING }}
      
      - name: Deploy to staging
        run: |
          kubectl set image deployment/mini-xdr-backend \
            backend=${{ env.ECR_REGISTRY }}/${{ env.ECR_REPOSITORY }}:${{ github.sha }} \
            -n mini-xdr-staging
          
          kubectl rollout status deployment/mini-xdr-backend -n mini-xdr-staging
      
      - name: Run smoke tests
        run: |
          sleep 30
          curl -f https://staging.mini-xdr.com/health || exit 1
  
  deploy-production:
    needs: [build-backend, build-frontend]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure kubectl
        uses: azure/k8s-set-context@v3
        with:
          method: kubeconfig
          kubeconfig: ${{ secrets.KUBE_CONFIG_PROD }}
      
      - name: Deploy to production
        run: |
          kubectl set image deployment/mini-xdr-backend \
            backend=${{ env.ECR_REGISTRY }}/${{ env.ECR_REPOSITORY }}:${{ github.sha }} \
            -n mini-xdr
          
          kubectl rollout status deployment/mini-xdr-backend -n mini-xdr --timeout=5m
      
      - name: Verify deployment
        run: |
          sleep 30
          curl -f https://api.mini-xdr.com/health || exit 1
      
      - name: Notify deployment
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          text: 'Production deployment completed'
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

**Checklist:**
- [ ] Create GitHub Actions workflow
- [ ] Set up staging environment
- [ ] Configure secrets (AWS, kubectl)
- [ ] Test CI/CD pipeline
- [ ] Add deployment notifications

---

### Task 2: Infrastructure as Code

**File:** `/infrastructure/terraform/main.tf` (NEW)

```hcl
# Minimal Terraform for production infrastructure

terraform {
  required_version = ">= 1.0"
  
  backend "s3" {
    bucket = "mini-xdr-terraform-state"
    key    = "production/terraform.tfstate"
    region = "us-east-1"
  }
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# RDS PostgreSQL
resource "aws_db_instance" "postgres" {
  identifier     = "mini-xdr-db"
  engine         = "postgres"
  engine_version = "15"
  instance_class = "db.t3.medium"
  
  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_encrypted     = true
  
  multi_az               = true  # High availability
  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  username = "minixdr"
  password = var.db_password  # From secrets
  
  vpc_security_group_ids = [aws_security_group.db.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  skip_final_snapshot = false
  final_snapshot_identifier = "mini-xdr-final-snapshot"
  
  tags = {
    Environment = "production"
    Terraform   = "true"
  }
}

# ElastiCache Redis
resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "mini-xdr-cache"
  engine               = "redis"
  node_type            = "cache.t3.medium"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  engine_version       = "7.0"
  port                 = 6379
  
  subnet_group_name    = aws_elasticache_subnet_group.main.name
  security_group_ids   = [aws_security_group.redis.id]
  
  snapshot_retention_limit = 5
  snapshot_window         = "03:00-05:00"
}

# S3 bucket for backups
resource "aws_s3_bucket" "backups" {
  bucket = "mini-xdr-backups-${var.aws_account_id}"
  
  tags = {
    Environment = "production"
  }
}

resource "aws_s3_bucket_versioning" "backups" {
  bucket = aws_s3_bucket.backups.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "backups" {
  bucket = aws_s3_bucket.backups.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
      kms_master_key_id = aws_kms_key.backups.id
    }
  }
}

# KMS key for encryption
resource "aws_kms_key" "backups" {
  description             = "Mini-XDR backup encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = true
}
```

**Checklist:**
- [ ] Initialize Terraform
- [ ] Create production infrastructure
- [ ] Store Terraform state in S3
- [ ] Document infrastructure changes
- [ ] Add outputs for connection strings

---

### Task 3: Automated Testing

**File:** `/backend/tests/conftest.py` (Enhance existing)

```python
"""Pytest fixtures for testing"""
import pytest
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.pool import NullPool

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="function")
async def db_session():
    """Create test database session"""
    engine = create_async_engine(
        "postgresql+asyncpg://postgres:test@localhost:5432/test",
        poolclass=NullPool
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async with AsyncSession(engine) as session:
        yield session
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()

@pytest.fixture
def mock_llm():
    """Mock LLM for testing"""
    class MockLLM:
        def invoke(self, prompt):
            return "Mock LLM response"
    return MockLLM()
```

**File:** `/backend/tests/test_api.py` (NEW)

```python
"""API endpoint tests"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

@pytest.mark.asyncio
async def test_create_incident(db_session):
    response = client.post("/api/incidents", json={
        "src_ip": "192.168.1.100",
        "reason": "Brute force attack",
        "severity": "high"
    })
    assert response.status_code == 200
    assert "id" in response.json()
```

**Checklist:**
- [ ] Write tests for all endpoints
- [ ] Achieve 60%+ code coverage
- [ ] Add integration tests
- [ ] Add E2E tests
- [ ] Run tests in CI

---

### Solo Developer Quick Setup

**Week 1: CI/CD**
- [ ] Create GitHub Actions workflow
- [ ] Set up automated testing
- [ ] Configure Docker builds
- [ ] Test deployment to staging

**Week 2: IaC**
- [ ] Write Terraform configs
- [ ] Create staging infrastructure
- [ ] Create production infrastructure
- [ ] Document infrastructure

**Week 3: Testing**
- [ ] Write API tests
- [ ] Write ML tests
- [ ] Achieve 60% coverage
- [ ] Add pre-commit hooks

---

## 12: Regulatory & Industry Standards

**Priority:** P1 (Required for enterprise)  
**Time:** Ongoing (integrate into development process)

---

### NIST Cybersecurity Framework Mapping

**Map Mini-XDR features to NIST CSF:**

| NIST Function | Mini-XDR Feature | Implementation |
|---------------|------------------|----------------|
| **Identify** | Asset Discovery | `/backend/app/discovery/` |
| **Protect** | Access Control | RBAC (Doc 01) |
| **Detect** | ML Detection Engine | `/backend/app/ml_engine.py` |
| **Respond** | Incident Response | `/backend/app/responder.py` |
| **Recover** | Backup & DR | Automated backups (Doc 05) |

**Checklist:**
- [ ] Document NIST CSF alignment
- [ ] Create compliance matrix
- [ ] Include in security docs

---

### MITRE ATT&CK Integration

**File:** `/backend/app/mitre_attack.py` (NEW)

```python
"""MITRE ATT&CK technique mapping"""

ATTACK_TECHNIQUES = {
    "brute_force": {
        "technique_id": "T1110",
        "technique_name": "Brute Force",
        "tactic": "Credential Access"
    },
    "lateral_movement": {
        "technique_id": "T1021",
        "technique_name": "Remote Services",
        "tactic": "Lateral Movement"
    },
    "data_exfiltration": {
        "technique_id": "T1041",
        "technique_name": "Exfiltration Over C2 Channel",
        "tactic": "Exfiltration"
    }
}

def tag_incident_with_mitre(incident_type: str) -> dict:
    """Tag incident with MITRE ATT&CK technique"""
    return ATTACK_TECHNIQUES.get(incident_type, {
        "technique_id": "Unknown",
        "technique_name": "Unknown",
        "tactic": "Unknown"
    })
```

**Checklist:**
- [ ] Map detections to MITRE techniques
- [ ] Add technique tags to incidents
- [ ] Create ATT&CK coverage matrix
- [ ] Show coverage in dashboard

---

### PCI-DSS Compliance (If handling payment data)

**Requirements:**
1. ✅ Firewall protection (WAF - Doc 07)
2. ✅ Encrypted transmission (TLS 1.2+)
3. ✅ Encrypted storage (encryption at rest - Doc 02)
4. ✅ Access control (RBAC - Doc 01)
5. ✅ Monitoring & logging (Prometheus - Doc 05)
6. ✅ Security testing (Pen test - Doc 07)

**Checklist:**
- [ ] Complete PCI-DSS self-assessment
- [ ] Implement missing controls
- [ ] Annual PCI audit (if Level 1 merchant)

---

### FedRAMP (For US Government Customers)

**Reality Check:** FedRAMP takes 12-18 months and costs $250K-1M

**Only pursue if:**
- You have government customers lined up
- You have funding for compliance
- You have dedicated compliance team

**Alternative:** Partner with FedRAMP-authorized cloud (AWS GovCloud)

---

### ISO 27001 (Recommended for Enterprise)

**See Document 02, Task 3 for implementation**

Timeline: 9-12 months  
Cost: $30K-80K  
Value: Global trust, required by many enterprises

---

## Solo Developer Priority

**This Month:**
- [ ] Set up CI/CD pipeline (Week 1)
- [ ] Add automated tests (Week 2)
- [ ] Map to MITRE ATT&CK (Week 3)
- [ ] Create NIST CSF alignment doc (Week 4)

**This Quarter:**
- [ ] Achieve SOC 2 Type I (see Doc 02)
- [ ] Pass penetration test (see Doc 07)
- [ ] 60%+ test coverage
- [ ] Infrastructure as code for everything

**This Year:**
- [ ] SOC 2 Type II
- [ ] Consider ISO 27001
- [ ] PCI-DSS if needed
- [ ] FedRAMP if government customers

---

## Final Summary

You now have **13 complete production readiness documents** covering:

1. ✅ Master plan & roadmap
2. ✅ Authentication & authorization  
3. ✅ Compliance & privacy
4. ✅ Enterprise integrations
5. ✅ Scalability & performance
6. ✅ Reliability & HA
7. ✅ ML/AI hardening
8. ✅ Security hardening
9. ✅ Support & operations
10. ✅ Licensing & commercialization
11. ✅ UX & accessibility
12. ✅ DevOps & CI/CD
13. ✅ Regulatory standards

**Total documented tasks:** 400+  
**Total code examples:** 200+  
**Estimated solo effort:** 6-12 months  
**Estimated investment:** $20K-50K

**You can build this. Start with Document 01 (Auth) tomorrow.**

Good luck! 🚀


