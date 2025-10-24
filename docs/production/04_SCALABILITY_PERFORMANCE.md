# 04: Scalability & Performance - Production Implementation

**Status:** Important - Determines customer size you can handle  
**Current State:** ~10K events/sec, single-instance database  
**Target State:** 100K-1M+ events/sec, horizontally scalable  
**Priority:** P0 for mid-market, P1 for small businesses

---

## Current State Analysis

### What EXISTS Now

**File:** `/backend/app/db.py`
```python
✅ AsyncIO SQLAlchemy (async ready)
✅ Connection pooling (basic)
⚠️ SQLite default (NOT scalable)
⚠️ No query optimization
⚠️ No caching layer
⚠️ No database sharding
```

**File:** `/backend/app/distributed/` (Kafka, Redis support)
```python
✅ Kafka integration prepared
✅ Redis integration prepared
⚠️ Not fully utilized in production
```

### Performance Bottlenecks

1. **Database**: SQLite is single-threaded, <1000 writes/sec
2. **No Caching**: Every query hits database
3. **No Indexing Strategy**: Missing compound indexes
4. **Synchronous Operations**: Some blocking calls in async code
5. **No Connection Pooling Limits**: Can exhaust database connections
6. **Large Table Scans**: Events table grows unbounded

---

## Implementation Checklist

### Task 1: Database Optimization

#### 1.1: PostgreSQL Production Configuration
**File:** `/backend/app/config.py`

```python
class Settings(BaseSettings):
    # ... existing fields ...
    
    # Production database settings
    database_url: str = "postgresql+asyncpg://user:pass@localhost:5432/minixdr"
    database_pool_size: int = 20  # Max concurrent connections
    database_max_overflow: int = 10  # Additional connections when pool exhausted
    database_pool_timeout: int = 30  # Seconds to wait for connection
    database_pool_recycle: int = 3600  # Recycle connections after 1 hour
    
    # Query optimization
    database_echo: bool = False  # Disable SQL logging in production
    database_statement_timeout: int = 30000  # 30 seconds max per query
```

**File:** `/backend/app/db.py`

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool, QueuePool
from .config import settings

# Production-grade async engine
engine = create_async_engine(
    settings.database_url,
    echo=settings.database_echo,
    future=True,
    pool_size=settings.database_pool_size,
    max_overflow=settings.database_max_overflow,
    pool_timeout=settings.database_pool_timeout,
    pool_recycle=settings.database_pool_recycle,
    pool_pre_ping=True,  # Verify connections before use
    poolclass=QueuePool,  # Use QueuePool for production
    connect_args={
        "statement_timeout": settings.database_statement_timeout,
        "command_timeout": settings.database_statement_timeout
    }
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,  # Manual flush control for better performance
)
```

**Checklist:**
- [ ] Update config with PostgreSQL settings
- [ ] Update db.py with production pooling
- [ ] Test connection pooling under load
- [ ] Monitor pool exhaustion
- [ ] Set statement timeout to prevent long-running queries

#### 1.2: Add Critical Indexes
**New File:** `/backend/migrations/versions/add_performance_indexes.py`

```python
"""Add performance indexes

Revision ID: perf_001
"""
from alembic import op


def upgrade():
    # Events table - most queried
    op.create_index(
        'ix_events_org_ts_severity',
        'events',
        ['organization_id', 'ts', 'severity'],
        postgresql_using='btree'
    )
    
    op.create_index(
        'ix_events_org_src_ip',
        'events',
        ['organization_id', 'src_ip'],
        postgresql_using='btree'
    )
    
    op.create_index(
        'ix_events_source_type_ts',
        'events',
        ['source_type', 'ts'],
        postgresql_where="deleted_at IS NULL",  # Partial index
        postgresql_using='btree'
    )
    
    # Incidents table
    op.create_index(
        'ix_incidents_org_status_severity',
        'incidents',
        ['organization_id', 'status', 'escalation_level'],
        postgresql_using='btree'
    )
    
    op.create_index(
        'ix_incidents_created_at_desc',
        'incidents',
        ['created_at'],
        postgresql_using='btree',
        postgresql_ops={'created_at': 'DESC'}  # Optimize for recent-first queries
    )
    
    # Audit logs - time-series optimized
    op.create_index(
        'ix_audit_logs_org_created',
        'audit_logs',
        ['organization_id', 'created_at'],
        postgresql_using='brin'  # BRIN index for time-series
    )


def downgrade():
    op.drop_index('ix_events_org_ts_severity')
    op.drop_index('ix_events_org_src_ip')
    op.drop_index('ix_events_source_type_ts')
    op.drop_index('ix_incidents_org_status_severity')
    op.drop_index('ix_incidents_created_at_desc')
    op.drop_index('ix_audit_logs_org_created')
```

**Checklist:**
- [ ] Create migration for performance indexes
- [ ] Run EXPLAIN ANALYZE on common queries
- [ ] Apply migration in staging first
- [ ] Verify query performance improvement (>50% faster)
- [ ] Monitor index usage with pg_stat_user_indexes

#### 1.3: Table Partitioning for Time-Series Data
**File:** `/backend/migrations/versions/partition_events_table.py`

```python
"""Partition events table by month for performance

Revision ID: part_001
"""
from alembic import op
from datetime import datetime, timedelta


def upgrade():
    # Convert events table to partitioned table
    op.execute("""
        -- Create new partitioned table
        CREATE TABLE events_partitioned (
            LIKE events INCLUDING ALL
        ) PARTITION BY RANGE (ts);
        
        -- Create partitions for last 6 months + next 3 months
        CREATE TABLE events_2025_01 PARTITION OF events_partitioned
            FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
        
        CREATE TABLE events_2025_02 PARTITION OF events_partitioned
            FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
        
        -- Add more partitions as needed...
        
        -- Copy data from old table
        INSERT INTO events_partitioned SELECT * FROM events;
        
        -- Swap tables
        ALTER TABLE events RENAME TO events_old;
        ALTER TABLE events_partitioned RENAME TO events;
        
        -- Drop old table (after backup!)
        -- DROP TABLE events_old;
    """)


def downgrade():
    # Reverse partitioning (complex - test thoroughly)
    pass
```

**Checklist:**
- [ ] Backup database before partitioning
- [ ] Create partition migration
- [ ] Test queries on partitioned table
- [ ] Set up monthly partition creation job
- [ ] Verify partition pruning with EXPLAIN

---

### Task 2: Caching Layer

#### 2.1: Redis Cache Implementation
**New File:** `/backend/app/cache.py`

```python
"""Redis caching layer for performance"""
import json
import hashlib
from typing import Any, Optional, Callable
from functools import wraps
from datetime import timedelta
import redis.asyncio as redis
from .config import settings

# Redis connection pool
redis_client = redis.from_url(
    settings.redis_url,
    encoding="utf-8",
    decode_responses=True,
    max_connections=50
)


async def get_cached(key: str) -> Optional[Any]:
    """Get value from cache"""
    try:
        value = await redis_client.get(key)
        return json.loads(value) if value else None
    except Exception as e:
        logger.error(f"Cache get error: {e}")
        return None


async def set_cached(
    key: str,
    value: Any,
    ttl: int = 300  # 5 minutes default
) -> bool:
    """Set value in cache with TTL"""
    try:
        await redis_client.setex(
            key,
            ttl,
            json.dumps(value, default=str)
        )
        return True
    except Exception as e:
        logger.error(f"Cache set error: {e}")
        return False


async def delete_cached(key: str) -> bool:
    """Delete from cache"""
    try:
        await redis_client.delete(key)
        return True
    except Exception as e:
        logger.error(f"Cache delete error: {e}")
        return False


async def invalidate_pattern(pattern: str):
    """Invalidate all keys matching pattern"""
    try:
        async for key in redis_client.scan_iter(match=pattern):
            await redis_client.delete(key)
    except Exception as e:
        logger.error(f"Cache invalidation error: {e}")


def cached(ttl: int = 300, key_prefix: str = ""):
    """Decorator for caching function results"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            key_parts = [key_prefix or func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
            
            cache_key = hashlib.md5(
                ":".join(key_parts).encode()
            ).hexdigest()
            
            # Try cache first
            cached_result = await get_cached(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Cache miss - call function
            result = await func(*args, **kwargs)
            
            # Store in cache
            await set_cached(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator


# Cache key patterns for invalidation
def get_incident_cache_key(incident_id: int) -> str:
    return f"incident:{incident_id}"


def get_user_incidents_cache_key(org_id: int, status: str = "all") -> str:
    return f"incidents:org:{org_id}:status:{status}"


async def invalidate_incident_cache(incident_id: int, org_id: int):
    """Invalidate all caches related to an incident"""
    await delete_cached(get_incident_cache_key(incident_id))
    await invalidate_pattern(f"incidents:org:{org_id}:*")
```

**Checklist:**
- [ ] Create cache.py with Redis client
- [ ] Add REDIS_URL to config.py and .env
- [ ] Install redis: `pip install redis[hiredis]`
- [ ] Test cache set/get
- [ ] Apply @cached decorator to expensive queries
- [ ] Implement cache invalidation on updates

#### 2.2: Apply Caching to API Endpoints
**File:** `/backend/app/main.py` - Update existing endpoints

```python
from .cache import cached, get_cached, set_cached, invalidate_incident_cache

# Cache incident list (expires in 2 minutes)
@app.get("/api/incidents")
@cached(ttl=120, key_prefix="incidents")
async def get_incidents(
    status: str = "all",
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    # Existing query logic...
    pass


# Cache individual incident (expires in 5 minutes)
@app.get("/api/incidents/{incident_id}")
@cached(ttl=300)
async def get_incident(
    incident_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    # Existing query logic...
    pass


# Invalidate cache on updates
@app.put("/api/incidents/{incident_id}")
async def update_incident(
    incident_id: int,
    update_data: dict,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    # Update logic...
    
    # Invalidate caches
    await invalidate_incident_cache(incident_id, current_user.organization_id)
    
    return {"updated": True}
```

**Checklist:**
- [ ] Add caching to /api/incidents endpoints
- [ ] Add caching to /api/events endpoints
- [ ] Add caching to /api/analytics endpoints
- [ ] Implement cache warming for common queries
- [ ] Monitor cache hit rate (aim for >80%)

---

### Task 3: Query Optimization

#### 3.1: Optimize Common Queries
**File:** `/backend/app/main.py` - Rewrite slow queries

```python
# BEFORE: N+1 query problem
@app.get("/api/incidents")
async def get_incidents(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Incident))
    incidents = result.scalars().all()
    
    # This triggers N additional queries!
    for incident in incidents:
        incident.user  # Lazy load user
        incident.action_logs  # Lazy load actions
    
    return incidents


# AFTER: Single query with joins
@app.get("/api/incidents")
async def get_incidents(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Incident)
        .options(
            selectinload(Incident.action_logs),  # Eager load
            selectinload(Incident.assigned_user)
        )
        .where(Incident.organization_id == org_id)
        .order_by(Incident.created_at.desc())
        .limit(100)  # Always limit results
    )
    incidents = result.scalars().all()
    return incidents


# Use pagination for large result sets
@app.get("/api/events")
async def get_events(
    page: int = 1,
    page_size: int = 100,
    db: AsyncSession = Depends(get_db)
):
    page_size = min(page_size, 1000)  # Cap at 1000
    offset = (page - 1) * page_size
    
    result = await db.execute(
        select(Event)
        .where(Event.organization_id == org_id)
        .order_by(Event.ts.desc())
        .limit(page_size)
        .offset(offset)
    )
    
    events = result.scalars().all()
    
    # Get total count (cached separately)
    total = await get_cached(f"events_count:{org_id}")
    if not total:
        count_result = await db.execute(
            select(func.count(Event.id))
            .where(Event.organization_id == org_id)
        )
        total = count_result.scalar()
        await set_cached(f"events_count:{org_id}", total, ttl=300)
    
    return {
        "events": events,
        "page": page,
        "page_size": page_size,
        "total": total,
        "pages": (total + page_size - 1) // page_size
    }
```

**Checklist:**
- [ ] Identify N+1 queries (use SQLAlchemy echo=True)
- [ ] Add selectinload() for relationships
- [ ] Add pagination to all list endpoints
- [ ] Limit max result size
- [ ] Cache count queries separately

#### 3.2: Database Query Monitoring
**New File:** `/backend/app/monitoring/query_monitor.py`

```python
"""Monitor slow queries and log for optimization"""
import time
import logging
from sqlalchemy import event
from sqlalchemy.engine import Engine

logger = logging.getLogger("query_monitor")

SLOW_QUERY_THRESHOLD = 100  # milliseconds


@event.listens_for(Engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    conn.info.setdefault('query_start_time', []).append(time.time())


@event.listens_for(Engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    total_time = time.time() - conn.info['query_start_time'].pop()
    total_time_ms = total_time * 1000
    
    if total_time_ms > SLOW_QUERY_THRESHOLD:
        logger.warning(
            f"Slow query ({total_time_ms:.2f}ms): {statement[:200]}"
        )
```

**Checklist:**
- [ ] Create query monitoring
- [ ] Log slow queries
- [ ] Set up alerting for queries >1 second
- [ ] Review slow query log weekly
- [ ] Add missing indexes for slow queries

---

### Task 4: Async Optimization

#### 4.1: Make ALL Operations Async
**File:** Scan all files for blocking calls

```python
# BLOCKING (BAD):
import requests
response = requests.get(url)  # Blocks event loop!

# ASYNC (GOOD):
import aiohttp
async with aiohttp.ClientSession() as session:
    async with session.get(url) as response:
        data = await response.json()


# BLOCKING (BAD):
import time
time.sleep(5)  # Blocks everything!

# ASYNC (GOOD):
import asyncio
await asyncio.sleep(5)  # Only blocks this coroutine


# BLOCKING (BAD):
with open("file.txt", "r") as f:
    data = f.read()  # Blocks I/O

# ASYNC (GOOD):
import aiofiles
async with aiofiles.open("file.txt", "r") as f:
    data = await f.read()
```

**Checklist:**
- [ ] Audit codebase for `requests` library usage → replace with `aiohttp`
- [ ] Audit for `time.sleep()` → replace with `asyncio.sleep()`
- [ ] Audit for `open()` → replace with `aiofiles`
- [ ] Run with `pytest-asyncio` to catch blocking calls
- [ ] Use `asyncio.gather()` for parallel operations

---

### Task 5: Horizontal Scaling

#### 5.1: Stateless Application Design
**File:** `/backend/app/main.py`

```python
# Store session state in Redis, not in-memory
from .cache import redis_client

# BEFORE (BAD): In-memory state
active_sessions = {}  # Lost on restart!

# AFTER (GOOD): Redis-backed state
async def get_session(session_id: str):
    session_data = await redis_client.get(f"session:{session_id}")
    return json.loads(session_data) if session_data else None

async def set_session(session_id: str, data: dict):
    await redis_client.setex(
        f"session:{session_id}",
        3600,  # 1 hour TTL
        json.dumps(data)
    )
```

**Checklist:**
- [ ] Move all state to Redis or database
- [ ] Remove in-memory caches
- [ ] Make app truly stateless
- [ ] Test with multiple instances
- [ ] Verify session sharing across instances

#### 5.2: Load Balancer Configuration
**File:** `/ops/k8s/backend-hpa.yaml` (Horizontal Pod Autoscaler)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mini-xdr-backend-hpa
  namespace: mini-xdr
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: mini-xdr-backend
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
```

**Checklist:**
- [ ] Create HPA configuration
- [ ] Test auto-scaling with load
- [ ] Set appropriate min/max replicas
- [ ] Monitor scaling events
- [ ] Tune CPU/memory thresholds

---

## Performance Targets

| Metric | Current | Target (Phase 1) | Target (Phase 3) |
|--------|---------|------------------|------------------|
| Events ingested/sec | ~1K | 10K | 100K+ |
| Query response time (p95) | ~500ms | <100ms | <50ms |
| Dashboard load time | ~2s | <500ms | <200ms |
| Incident creation time | ~200ms | <50ms | <20ms |
| Concurrent users | 10 | 100 | 1000+ |
| Database size | 1GB | 100GB | 1TB+ |

---

## Load Testing

### Test Script
**New File:** `/tests/load/locustfile.py`

```python
"""Load testing with Locust"""
from locust import HttpUser, task, between

class MiniXDRUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        # Login
        response = self.client.post("/api/auth/login", json={
            "email": "test@example.com",
            "password": "password123"
        })
        self.token = response.json()["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}
    
    @task(3)
    def view_incidents(self):
        self.client.get("/api/incidents", headers=self.headers)
    
    @task(2)
    def view_events(self):
        self.client.get("/api/events?page=1&page_size=50", headers=self.headers)
    
    @task(1)
    def create_incident(self):
        self.client.post("/api/incidents", json={
            "src_ip": "192.168.1.100",
            "reason": "Brute force detected",
            "severity": "high"
        }, headers=self.headers)
```

Run: `locust -f tests/load/locustfile.py --users 100 --spawn-rate 10`

**Checklist:**
- [ ] Install locust: `pip install locust`
- [ ] Create load test scenarios
- [ ] Run with 100 concurrent users
- [ ] Run with 1000 concurrent users
- [ ] Identify bottlenecks
- [ ] Optimize and re-test

---

## Solo Developer Quick Wins

**Week 1: Database**
- [ ] Migrate to PostgreSQL
- [ ] Add connection pooling
- [ ] Add critical indexes

**Week 2: Caching**
- [ ] Set up Redis
- [ ] Cache top 5 expensive queries
- [ ] Implement cache invalidation

**Week 3: Optimization**
- [ ] Fix N+1 queries
- [ ] Add pagination everywhere
- [ ] Make everything async

**Week 4: Testing**
- [ ] Load test with Locust
- [ ] Fix identified bottlenecks
- [ ] Document performance

**Total Time:** 3-4 weeks solo

---

**Status:** Ready for implementation  
**Next Document:** `05_RELIABILITY_HIGH_AVAILABILITY.md`


