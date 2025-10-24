# 05: Reliability & High Availability - Production Implementation

**Status:** Critical for SLA Commitments  
**Current State:** Single-instance, no failover, manual recovery  
**Target State:** 99.9% uptime, automated failover, disaster recovery tested  
**Priority:** P0 for paying customers

---

## Current State Analysis

### What EXISTS
- ✅ Basic health check endpoint (`/health`)
- ✅ Kubernetes deployment configs
- ⚠️ Single pod deployments (no redundancy)
- ⚠️ No automated backups
- ⚠️ No disaster recovery plan
- ⚠️ No uptime monitoring
- ⚠️ No incident response procedures

### SLA Requirements by Tier

| Tier | Uptime SLA | Max Downtime/Year | Max Downtime/Month |
|------|-----------|-------------------|---------------------|
| Free/Trial | Best effort | N/A | N/A |
| Starter | 99% | 3.65 days | 7.2 hours |
| Professional | 99.5% | 1.83 days | 3.6 hours |
| Enterprise | 99.9% | 8.76 hours | 43 minutes |

---

## Implementation Checklist

### Task 1: Multi-Instance Deployment

#### 1.1: Update Kubernetes Deployments for HA
**File:** `/ops/k8s/backend-deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mini-xdr-backend
  namespace: mini-xdr
spec:
  # High availability configuration
  replicas: 3  # CHANGED from 1
  
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1  # Add 1 new pod before removing old
      maxUnavailable: 0  # Zero downtime deployments
  
  # Pod anti-affinity - spread across nodes
  template:
    spec:
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - mini-xdr-backend
            topologyKey: kubernetes.io/hostname
      
      # Readiness and liveness probes
      containers:
      - name: backend
        image: mini-xdr-backend:latest
        
        # Readiness probe - when is pod ready for traffic?
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          successThreshold: 1
          failureThreshold: 3
        
        # Liveness probe - is pod still alive?
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          successThreshold: 1
          failureThreshold: 3
        
        # Startup probe - for slow-starting apps
        startupProbe:
          httpGet:
            path: /health/startup
            port: 8000
          initialDelaySeconds: 0
          periodSeconds: 5
          timeoutSeconds: 3
          successThreshold: 1
          failureThreshold: 30  # 150 seconds max startup time
        
        # Resource limits
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        
        # Graceful shutdown
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 15"]  # Wait for connections to drain
      
      # Graceful termination period
      terminationGracePeriodSeconds: 30
```

**Checklist:**
- [ ] Update backend deployment to 3 replicas
- [ ] Update frontend deployment to 3 replicas
- [ ] Add pod anti-affinity rules
- [ ] Configure rolling updates with zero downtime
- [ ] Test deployment with `kubectl rollout status`

#### 1.2: Implement Health Check Endpoints
**File:** `/backend/app/main.py`

```python
from sqlalchemy import text

@app.get("/health")
async def health_check():
    """Basic health check"""
    return {"status": "ok"}


@app.get("/health/ready")
async def readiness_check(db: AsyncSession = Depends(get_db)):
    """Readiness check - can we serve traffic?"""
    try:
        # Check database connection
        await db.execute(text("SELECT 1"))
        
        # Check Redis connection
        await redis_client.ping()
        
        # Check critical integrations
        # ... additional checks ...
        
        return {
            "status": "ready",
            "checks": {
                "database": "ok",
                "redis": "ok"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Not ready: {str(e)}")


@app.get("/health/live")
async def liveness_check():
    """Liveness check - is the app still running?"""
    # Simple check - if this endpoint responds, we're alive
    # Don't check external dependencies here
    return {"status": "alive"}


@app.get("/health/startup")
async def startup_check(db: AsyncSession = Depends(get_db)):
    """Startup check - has the app finished starting up?"""
    try:
        # Check if ML models are loaded
        if not ml_detector.models_loaded:
            raise HTTPException(503, "ML models still loading")
        
        # Check database is accessible
        await db.execute(text("SELECT 1"))
        
        return {"status": "started"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Still starting: {str(e)}")
```

**Checklist:**
- [ ] Implement /health/ready endpoint
- [ ] Implement /health/live endpoint
- [ ] Implement /health/startup endpoint
- [ ] Test health checks return correct status codes
- [ ] Verify Kubernetes uses health checks

---

### Task 2: Database High Availability

#### 2.1: PostgreSQL with Replication
**File:** `/ops/k8s/postgres-ha.yaml` (if self-hosted)

```yaml
# For managed PostgreSQL (recommended):
# - AWS RDS with Multi-AZ
# - Azure Database for PostgreSQL with HA
# - Google Cloud SQL with HA

# For self-hosted (using Patroni for HA):
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres-ha
  namespace: mini-xdr
spec:
  serviceName: postgres-ha
  replicas: 3
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        env:
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        - name: POSTGRES_REPLICATION_MODE
          value: "master"
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 100Gi
```

**Checklist:**
- [ ] Use managed PostgreSQL with Multi-AZ (AWS RDS, Azure, GCP)
- [ ] Configure automatic failover (< 30 second failover time)
- [ ] Set up read replicas for read scaling
- [ ] Test failover manually
- [ ] Monitor replication lag

#### 2.2: Connection Pooling with PgBouncer
**File:** `/ops/k8s/pgbouncer.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pgbouncer
  namespace: mini-xdr
spec:
  replicas: 2
  selector:
    matchLabels:
      app: pgbouncer
  template:
    metadata:
      labels:
        app: pgbouncer
    spec:
      containers:
      - name: pgbouncer
        image: pgbouncer/pgbouncer:latest
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: pgbouncer-config
          mountPath: /etc/pgbouncer
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "200m"
      volumes:
      - name: pgbouncer-config
        configMap:
          name: pgbouncer-config
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: pgbouncer-config
  namespace: mini-xdr
data:
  pgbouncer.ini: |
    [databases]
    minixdr = host=postgres-primary port=5432 dbname=minixdr
    
    [pgbouncer]
    listen_addr = 0.0.0.0
    listen_port = 5432
    auth_type = md5
    auth_file = /etc/pgbouncer/userlist.txt
    pool_mode = transaction
    max_client_conn = 1000
    default_pool_size = 25
    reserve_pool_size = 5
    reserve_pool_timeout = 3
    server_lifetime = 3600
    server_idle_timeout = 600
```

**Checklist:**
- [ ] Deploy PgBouncer for connection pooling
- [ ] Configure pool size based on database limits
- [ ] Update app to connect through PgBouncer
- [ ] Monitor connection pool usage
- [ ] Test with high connection load

---

### Task 3: Automated Backups & Disaster Recovery

#### 3.1: Automated Backup System
**New File:** `/scripts/backup/automated_backup.sh`

```bash
#!/bin/bash
# Automated backup system with retention and verification

set -euo pipefail

# Configuration
BACKUP_DIR="/var/backups/mini-xdr"
S3_BUCKET="s3://mini-xdr-backups-${AWS_ACCOUNT_ID}"
RETENTION_DAILY=7
RETENTION_WEEKLY=4
RETENTION_MONTHLY=6

# Database backup
backup_database() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_file="db_${timestamp}.sql.gz"
    
    echo "[$(date)] Starting database backup..."
    
    # Full database dump
    PGPASSWORD=$DB_PASSWORD pg_dump \
        -h $DB_HOST \
        -U $DB_USER \
        -d $DB_NAME \
        -F c \
        -Z 9 \
        -f "${BACKUP_DIR}/${backup_file}"
    
    # Calculate checksum
    sha256sum "${BACKUP_DIR}/${backup_file}" > "${BACKUP_DIR}/${backup_file}.sha256"
    
    # Upload to S3 with encryption
    aws s3 cp "${BACKUP_DIR}/${backup_file}" \
        "${S3_BUCKET}/daily/${backup_file}" \
        --sse aws:kms \
        --sse-kms-key-id "alias/mini-xdr-backup-key"
    
    aws s3 cp "${BACKUP_DIR}/${backup_file}.sha256" \
        "${S3_BUCKET}/daily/${backup_file}.sha256"
    
    echo "[$(date)] Database backup completed: ${backup_file}"
}

# Application state backup
backup_application_state() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local state_file="state_${timestamp}.tar.gz"
    
    echo "[$(date)] Starting application state backup..."
    
    # Backup ML models, configs, policies
    tar -czf "${BACKUP_DIR}/${state_file}" \
        /app/models \
        /app/policies \
        /app/config
    
    aws s3 cp "${BACKUP_DIR}/${state_file}" \
        "${S3_BUCKET}/state/${state_file}"
    
    echo "[$(date)] Application state backup completed"
}

# Verify backup integrity
verify_backup() {
    local backup_file=$1
    
    echo "[$(date)] Verifying backup: ${backup_file}"
    
    # Download from S3
    aws s3 cp "${S3_BUCKET}/daily/${backup_file}" /tmp/verify_backup.sql.gz
    aws s3 cp "${S3_BUCKET}/daily/${backup_file}.sha256" /tmp/verify_backup.sha256
    
    # Verify checksum
    cd /tmp
    sha256sum -c verify_backup.sha256
    
    # Test restore to temporary database
    pg_restore -d $TEST_DB -c -C /tmp/verify_backup.sql.gz
    
    echo "[$(date)] Backup verification successful"
    
    # Cleanup
    rm /tmp/verify_backup.*
}

# Retention policy
apply_retention() {
    echo "[$(date)] Applying retention policy..."
    
    # Delete daily backups older than 7 days
    find ${BACKUP_DIR} -name "db_*.sql.gz" -mtime +${RETENTION_DAILY} -delete
    
    # S3 lifecycle policy handles cloud retention
    echo "[$(date)] Retention policy applied"
}

# Main execution
main() {
    mkdir -p ${BACKUP_DIR}
    
    backup_database
    backup_application_state
    
    # Verify latest backup (weekly)
    if [ $(date +%u) -eq 7 ]; then
        latest_backup=$(ls -t ${BACKUP_DIR}/db_*.sql.gz | head -1)
        verify_backup $(basename $latest_backup)
    fi
    
    apply_retention
    
    echo "[$(date)] Backup process completed successfully"
}

main
```

**Checklist:**
- [ ] Create automated backup script
- [ ] Set up S3 bucket with versioning
- [ ] Configure S3 lifecycle policies (delete after 90 days)
- [ ] Set up cron job: `0 2 * * * /scripts/backup/automated_backup.sh`
- [ ] Test backup restoration monthly
- [ ] Document recovery procedures
- [ ] Set up backup monitoring/alerts

#### 3.2: Disaster Recovery Runbook
**New File:** `/docs/operations/DISASTER_RECOVERY.md`

```markdown
# Disaster Recovery Runbook

## Recovery Objectives
- RPO (Recovery Point Objective): < 1 hour
- RTO (Recovery Time Objective): < 4 hours

## Scenarios

### Scenario 1: Database Failure
**Detection:** Database health checks failing

**Recovery Steps:**
1. Verify database is truly down
2. Check RDS console for automatic failover status
3. If failover hasn't triggered:
   ```bash
   aws rds failover-db-cluster --db-cluster-identifier mini-xdr-prod
   ```
4. Verify application reconnects (check health endpoints)
5. Monitor for replication lag

**Expected Time:** 2-5 minutes

### Scenario 2: Complete Data Loss
**Detection:** Data corruption or accidental deletion

**Recovery Steps:**
1. Identify last known good backup
2. Create new database instance
3. Restore from backup:
   ```bash
   aws s3 cp s3://mini-xdr-backups/daily/db_YYYYMMDD.sql.gz /tmp/
   pg_restore -d mini_xdr_restored /tmp/db_YYYYMMDD.sql.gz
   ```
4. Verify data integrity
5. Update connection strings to new database
6. Rolling restart of application pods

**Expected Time:** 2-4 hours

### Scenario 3: Kubernetes Cluster Failure
**Recovery Steps:**
1. Deploy new cluster from IaC:
   ```bash
   cd infrastructure/
   terraform apply -target=module.eks_cluster
   ```
2. Restore application state:
   ```bash
   kubectl apply -f ops/k8s/
   ```
3. Verify all pods are running
4. Update DNS to point to new load balancer

**Expected Time:** 1-2 hours

## Testing Schedule
- Monthly: Test database restoration
- Quarterly: Full DR drill with simulated failure
- Annually: Complete region failover test
```

**Checklist:**
- [ ] Document all disaster recovery procedures
- [ ] Create runbooks for each scenario
- [ ] Test DR procedures quarterly
- [ ] Train team on DR procedures
- [ ] Maintain DR contact list

---

### Task 4: Monitoring & Alerting

#### 4.1: Prometheus Metrics
**New File:** `/backend/app/monitoring/metrics.py`

```python
"""Prometheus metrics for monitoring"""
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response
import time

# Request metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

# Application metrics
incidents_total = Counter(
    'incidents_total',
    'Total incidents created',
    ['severity', 'organization']
)

events_ingested_total = Counter(
    'events_ingested_total',
    'Total events ingested',
    ['source_type', 'organization']
)

active_incidents = Gauge(
    'active_incidents',
    'Number of active incidents',
    ['severity']
)

# Database metrics
db_connections_active = Gauge(
    'db_connections_active',
    'Active database connections'
)

db_query_duration_seconds = Histogram(
    'db_query_duration_seconds',
    'Database query duration',
    ['query_type']
)

# Cache metrics
cache_hits_total = Counter('cache_hits_total', 'Total cache hits')
cache_misses_total = Counter('cache_misses_total', 'Total cache misses')

# Integration metrics
integration_requests_total = Counter(
    'integration_requests_total',
    'Total integration API requests',
    ['integration', 'status']
)


# Middleware for automatic request tracking
async def track_request_metrics(request, call_next):
    """Track request metrics automatically"""
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    
    http_requests_total.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    http_request_duration_seconds.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    return response


# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")
```

**File:** `/backend/app/main.py` - Add middleware

```python
from .monitoring.metrics import track_request_metrics

app.middleware("http")(track_request_metrics)
```

**Checklist:**
- [ ] Add Prometheus metrics
- [ ] Expose /metrics endpoint
- [ ] Deploy Prometheus server
- [ ] Create Grafana dashboards
- [ ] Test metrics collection

#### 4.2: Alerting Rules
**New File:** `/ops/monitoring/alert-rules.yaml`

```yaml
groups:
- name: mini-xdr-alerts
  interval: 30s
  rules:
  
  # High error rate
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} errors/sec"
  
  # Slow response time
  - alert: SlowResponseTime
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1.0
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Slow API response time"
      description: "P95 latency is {{ $value }} seconds"
  
  # Database connection pool exhaustion
  - alert: DatabasePoolExhausted
    expr: db_connections_active > 18
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Database connection pool nearly exhausted"
      description: "{{ $value }} of 20 connections in use"
  
  # Pod crash loop
  - alert: PodCrashLooping
    expr: rate(kube_pod_container_status_restarts_total[15m]) > 0
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Pod is crash looping"
      description: "Pod {{ $labels.pod }} has restarted {{ $value }} times"
  
  # Low cache hit rate
  - alert: LowCacheHitRate
    expr: rate(cache_hits_total[5m]) / (rate(cache_hits_total[5m]) + rate(cache_misses_total[5m])) < 0.8
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Cache hit rate below 80%"
      description: "Cache hit rate is {{ $value | humanizePercentage }}"
```

**Checklist:**
- [ ] Create alerting rules
- [ ] Configure PagerDuty/OpsGenie integration
- [ ] Set up Slack alerts
- [ ] Test alert firing
- [ ] Create on-call rotation

---

### Task 5: Chaos Engineering

#### 5.1: Chaos Testing Plan
**New File:** `/tests/chaos/chaos_tests.md`

```markdown
# Chaos Engineering Tests

## Purpose
Verify system resilience by intentionally introducing failures

## Tests

### Test 1: Random Pod Deletion
**Hypothesis:** System should handle pod failures gracefully

**Procedure:**
1. Monitor dashboard and API
2. Delete random backend pod: `kubectl delete pod -n mini-xdr <pod-name>`
3. Observe:
   - Are requests still served?
   - How long until pod is replaced?
   - Any user-visible errors?

**Success Criteria:**
- No requests fail (handled by other pods)
- New pod starts within 30 seconds
- No data loss

### Test 2: Database Latency Injection
**Hypothesis:** System should degrade gracefully with slow database

**Procedure:**
1. Add 500ms delay to database queries
2. Monitor response times and error rates
3. Verify circuit breakers activate

**Success Criteria:**
- API returns 503 after timeout
- Cached responses still served
- System recovers when latency resolves

### Test 3: Network Partition
**Hypothesis:** System should handle network failures

**Procedure:**
1. Block traffic between backend and database
2. Monitor error rates and failover
3. Restore connectivity

**Success Criteria:**
- Read-only mode activated
- No data corruption
- Automatic recovery when network restored
```

**Checklist:**
- [ ] Document chaos tests
- [ ] Run monthly chaos tests
- [ ] Fix identified issues
- [ ] Update runbooks based on learnings

---

## Solo Developer Priority List

**Week 1: Basic HA**
- [ ] Increase replicas to 3
- [ ] Add health check endpoints
- [ ] Configure rolling updates

**Week 2: Backups**
- [ ] Set up automated backups
- [ ] Test restoration
- [ ] Document recovery procedures

**Week 3: Monitoring**
- [ ] Add Prometheus metrics
- [ ] Set up alerts
- [ ] Create dashboards

**Week 4: Testing**
- [ ] Perform DR drill
- [ ] Run chaos tests
- [ ] Fix identified issues

**Total Time:** 3-4 weeks

---

**Status:** Ready for implementation  
**Next Document:** `06_ML_AI_PRODUCTION_HARDENING.md`


