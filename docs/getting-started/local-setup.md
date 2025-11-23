# Mini-XDR Local Setup Instructions

## Congratulations! 🎉

Your Mini-XDR installation has been migrated to **100% local operation**. All AWS dependencies have been removed, and the system now runs entirely on your local infrastructure using Docker Compose.

## What Changed

### ✅ Removed
- ❌ AWS SageMaker ML endpoints
- ❌ AWS Secrets Manager integration
- ❌ AWS EKS/ECR deployments
- ❌ boto3 dependency
- ❌ All AWS-specific infrastructure code

### ✅ Added
- ✅ Docker Compose orchestration (postgres, redis, backend, frontend)
- ✅ Local PostgreSQL database (production-grade)
- ✅ Local Redis cache
- ✅ 100% local ML models (7 trained models, 2.1GB+)
- ✅ T-Pot honeypot integration (optional)
- ✅ Comprehensive documentation

## Quick Start

### 1. Create Environment File

```bash
cd /Users/chasemad/Desktop/mini-xdr

# Create .env from example (if doesn't exist)
if [ ! -f .env ]; then
    cat > .env << 'EOF'
# Database
DATABASE_URL=postgresql+asyncpg://xdr_user:local_dev_password@localhost:5432/mini_xdr

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
UI_ORIGIN=http://localhost:3000
API_KEY=demo-minixdr-api-key

# LLM Configuration
OPENAI_API_KEY=your_openai_key_here

# External Intelligence (optional)
ABUSEIPDB_API_KEY=your_abuseipdb_key_here
VIRUSTOTAL_API_KEY=your_virustotal_key_here

# Redis
REDIS_URL=redis://localhost:6379

# Honeypot (update when T-Pot is ready)
HONEYPOT_HOST=localhost
HONEYPOT_SSH_PORT=64295
EOF
    echo "✅ Created .env file - PLEASE UPDATE WITH YOUR API KEYS"
else
    echo "✅ .env file already exists"
fi
```

### 2. Start Mini-XDR

```bash
# Start all services
docker-compose up -d

# View logs (Ctrl+C to exit)
docker-compose logs -f
```

### 3. Verify Services

```bash
# Check all services are running
docker-compose ps

# Expected output:
# NAME                 STATUS              PORTS
# mini-xdr-backend     running             0.0.0.0:8000->8000/tcp
# mini-xdr-frontend    running             0.0.0.0:3000->3000/tcp
# mini-xdr-postgres    running (healthy)   0.0.0.0:5432->5432/tcp
# mini-xdr-redis       running (healthy)   0.0.0.0:6379->6379/tcp

# Test backend API
curl http://localhost:8000/health

# Expected: {"status":"ok"}
```

### 4. Access the Application

Open your browser to:
- **Dashboard**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Verify ML Models

Your locally trained models should automatically load:

```bash
# Check model status
curl http://localhost:8000/api/ml/status

# View model details
docker-compose logs backend | grep -i "model"

# Expected logs:
# ✅ Loaded model: general (97.98% accuracy)
# ✅ Loaded model: ddos_specialist
# ✅ Loaded model: brute_force_specialist
# ✅ Loaded model: web_attacks_specialist
```

## Database Initialization

Database is automatically initialized on first run. To manually reset:

```bash
# Stop services
docker-compose down

# Remove volumes (⚠️ this deletes all data)
docker-compose down -v

# Restart with fresh database
docker-compose up -d
```

## T-Pot Honeypot Setup (Optional)

When ready to set up your local T-Pot honeypot:

1. Follow the guide: `docs/getting-started/tpot-integration.md`
2. Update `.env` with T-Pot connection details
3. Uncomment T-Pot service in `docker-compose.yml`
4. Restart: `docker-compose up -d`

## Troubleshooting

### Port Already in Use

```bash
# Find what's using the port
lsof -i :8000
lsof -i :3000

# Kill the process or change ports in docker-compose.yml
```

### Backend Not Starting

```bash
# View backend logs
docker-compose logs backend

# Common issues:
# - ML models not found → check models/ directory exists
# - Database connection → ensure postgres is healthy
# - Missing API keys → check .env file
```

### Frontend Build Errors

```bash
# Rebuild frontend
docker-compose build frontend --no-cache
docker-compose up -d frontend
```

### ML Models Not Loading

```bash
# Check models directory is mounted
docker-compose exec backend ls -la /app/models

# Verify model files exist locally
ls -la /Users/chasemad/Desktop/mini-xdr/models/local_trained/

# If models are missing, you may need to retrain
# See: docs/ml/local-models.md
```

## Common Commands

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f backend

# Restart a service
docker-compose restart backend

# Rebuild a service
docker-compose build backend
docker-compose up -d backend

# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Check resource usage
docker stats
```

## Development Mode

For active development with hot-reloading:

```bash
# Backend auto-reloads on code changes (already configured)
# Frontend auto-reloads via Next.js

# Access backend shell
docker-compose exec backend /bin/bash

# Access PostgreSQL
docker-compose exec postgres psql -U xdr_user mini_xdr

# Run backend tests
docker-compose exec backend pytest

# Run frontend tests
docker-compose exec frontend npm test
```

## Performance Tips

### Increase Docker Resources

If you experience slow performance:

1. Open **Docker Desktop** → **Settings** → **Resources**
2. Increase Memory to **16GB** (32GB recommended)
3. Increase CPUs to **4+**
4. Apply & Restart

### Optimize PostgreSQL

For better database performance:

```yaml
# In docker-compose.yml, add to postgres service:
command:
  - postgres
  - -c
  - shared_buffers=2GB
  - -c
  - effective_cache_size=6GB
```

## Next Steps

1. **📚 Read Documentation**: Check `docs/getting-started/local-setup.md`
2. **🎯 Configure Policies**: Edit `policies/default_policies.yaml`
3. **🤖 Deploy Agents**: See `docs/agents/deployment.md`
4. **🔬 Test ML Models**: `docs/ml/local-models.md`
5. **🍯 Set up T-Pot**: `docs/getting-started/tpot-integration.md`

## Documentation

- **Local Setup Guide**: `docs/getting-started/local-setup.md`
- **T-Pot Integration**: `docs/getting-started/tpot-integration.md`
- **ML Models Architecture**: `docs/ml/local-models.md`
- **API Reference**: http://localhost:8000/docs

## Getting Help

- **Logs**: `docker-compose logs -f`
- **Backend API**: http://localhost:8000/docs
- **Health Status**: http://localhost:8000/health

## Success Criteria

✅ All services running: `docker-compose ps`
✅ Backend healthy: `curl http://localhost:8000/health`
✅ Frontend accessible: http://localhost:3000
✅ Database connected: Check backend logs
✅ ML models loaded: `curl http://localhost:8000/api/ml/status`
✅ Redis operational: `docker-compose logs redis`

---

**🎉 Congratulations! Your Mini-XDR is now running 100% locally with no AWS dependencies!**

For questions or issues, check the documentation in the `docs/` directory.
