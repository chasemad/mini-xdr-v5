# Data Management

This document covers data management operations for Mini-XDR, including cleanup procedures, data retention, and database maintenance.

## Database Cleanup

### Manual Database Cleanup

If you need to clear test/mock data or reset the system:

```bash
cd /Users/chasemad/Desktop/mini-xdr/backend

# Clear all incidents and events (requires API key)
curl -X DELETE http://localhost:8000/admin/clear-database \
  -H "x-api-key: your-api-key-here"

# Or manually via SQLite
sqlite3 xdr.db "BEGIN TRANSACTION; DELETE FROM incidents; DELETE FROM events; DELETE FROM actions; COMMIT;"

# Optimize database after cleanup
sqlite3 xdr.db "VACUUM; ANALYZE;"
```

### Clear Evidence Files

Evidence files are stored in `backend/evidence/` directory:

```bash
cd /Users/chasemad/Desktop/mini-xdr/backend
rm -rf evidence/*
```

### Clear Vector Storage

Qdrant vector embeddings are stored in `backend/qdrant_storage/`:

```bash
cd /Users/chasemad/Desktop/mini-xdr/backend
rm -rf qdrant_storage/*
```

The vector storage will be automatically recreated on next startup.

## Test Data Identification

Test/mock data typically uses reserved IP ranges:
- **203.0.113.0/24** (TEST-NET-3, RFC 5737)
- **192.0.2.0/24** (TEST-NET-1, RFC 5737)
- **198.51.100.0/24** (TEST-NET-2, RFC 5737)

To identify test data:

```bash
# Check for test IP events
sqlite3 xdr.db "SELECT COUNT(*) FROM events WHERE src_ip LIKE '203.0.113.%';"

# Check for test IP incidents
sqlite3 xdr.db "SELECT COUNT(*) FROM incidents WHERE src_ip LIKE '203.0.113.%';"
```

## Data Retention

### Production Data Retention Policy

Configure data retention policies based on your compliance requirements:

- **Events**: Recommend 90 days for active analysis, archive older data
- **Incidents**: Retain indefinitely or per regulatory requirements
- **Evidence**: Retain per incident retention policy
- **Logs**: Rotate daily, retain 30 days minimum
- **Training Samples**: Retain for model reproducibility

### Implementing Retention Policies

Create a scheduled job to archive/delete old data:

```python
# Example retention policy (add to scheduled jobs)
async def cleanup_old_events(db: AsyncSession, days: int = 90):
    """Remove events older than specified days"""
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    await db.execute(
        text("DELETE FROM events WHERE ts < :cutoff"),
        {"cutoff": cutoff_date}
    )
    await db.commit()
```

## Database Maintenance

### Regular Maintenance Tasks

Perform these tasks regularly for optimal performance:

```bash
# Optimize database (weekly)
sqlite3 xdr.db "VACUUM; ANALYZE;"

# Check database size
du -h xdr.db

# Check table sizes
sqlite3 xdr.db "SELECT name, SUM(pgsize) as size FROM dbstat GROUP BY name ORDER BY size DESC;"

# Backup database (daily recommended)
cp xdr.db xdr.db.backup.$(date +%Y%m%d)
```

### Database Backup

Regular backups are essential:

```bash
#!/bin/bash
# backup-database.sh
BACKUP_DIR="/path/to/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup SQLite database
cp /Users/chasemad/Desktop/mini-xdr/backend/xdr.db \
   $BACKUP_DIR/xdr.db.$DATE

# Backup evidence
tar -czf $BACKUP_DIR/evidence.$DATE.tar.gz \
   /Users/chasemad/Desktop/mini-xdr/backend/evidence/

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "xdr.db.*" -mtime +30 -delete
find $BACKUP_DIR -name "evidence.*.tar.gz" -mtime +30 -delete
```

## Monitoring Data Growth

Monitor data growth to plan for scaling:

```bash
# Check event count growth
sqlite3 xdr.db "SELECT DATE(ts) as date, COUNT(*) as events FROM events GROUP BY DATE(ts) ORDER BY date DESC LIMIT 7;"

# Check incident creation rate
sqlite3 xdr.db "SELECT DATE(created_at) as date, COUNT(*) as incidents FROM incidents GROUP BY DATE(created_at) ORDER BY date DESC LIMIT 7;"

# Check table sizes
sqlite3 xdr.db ".schema" | grep "CREATE TABLE" | wc -l
```

## Migration to Production Database

For production deployments, consider migrating from SQLite to PostgreSQL:

1. Export data from SQLite
2. Set up PostgreSQL database
3. Update connection strings in `backend/app/config.py`
4. Run Alembic migrations
5. Import data
6. Test thoroughly before cutover

See `docs/deployment/production-deployment.md` for detailed migration steps.

## Data Quality

### Verify Data Integrity

Regular data quality checks:

```bash
# Check for orphaned events (events without valid incident_id)
sqlite3 xdr.db "SELECT COUNT(*) FROM events WHERE incident_id IS NOT NULL AND incident_id NOT IN (SELECT id FROM incidents);"

# Check for incidents without events
sqlite3 xdr.db "SELECT COUNT(*) FROM incidents WHERE id NOT IN (SELECT DISTINCT incident_id FROM events WHERE incident_id IS NOT NULL);"

# Check for null/invalid data
sqlite3 xdr.db "SELECT COUNT(*) FROM incidents WHERE src_ip IS NULL OR src_ip = '';"
```

## Recent Cleanup (2025-11-21)

On November 21, 2025, the following cleanup was performed:
- Removed all test incidents from IP range 203.0.113.x (RFC 5737 TEST-NET-3)
- Cleared 52 test events from test IP ranges
- Removed test evidence files
- Cleared Qdrant vector storage
- Optimized database with VACUUM and ANALYZE
- System prepared for production T-Pot honeypot integration

**Result**: Database now contains only production data from real agents and models.

## Related Documentation

- [Monitoring & Alerts](./monitoring-and-alerts.md)
- [T-Pot Integration](../getting-started/tpot-integration.md)
- [Deployment Guide](../deployment/production-deployment.md)
