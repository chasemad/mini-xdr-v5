# Mini-XDR Operations

This directory contains all deployment and operations scripts for the Mini-XDR system.

## 🚀 Quick Start - AWS Migration

For complete AWS migration, run:
```bash
./deploy-full-aws-migration.sh
```

## 📁 Scripts Overview

### AWS Migration Scripts
- **`deploy-full-aws-migration.sh`** - Master script for complete AWS migration
- **`deploy-mini-xdr-aws.sh`** - Deploy AWS infrastructure (EC2, RDS, S3)
- **`deploy-mini-xdr-code.sh`** - Upload and configure application code
- **`configure-tpot-aws-connection.sh`** - Setup TPOT → AWS data flow

### Container Deployment
- **`Dockerfile.backend`** - Backend container image
- **`Dockerfile.frontend`** - Frontend container image  
- **`Dockerfile.ingestion-agent`** - Data ingestion agent

### CloudFormation Templates
- **`aws-cloudformation.yaml`** - Complete infrastructure template
- **`k8s/`** - Kubernetes deployment manifests

### Honeypot Management
- **`aws-tpot-honeypot-setup.sh`** - Deploy TPOT honeypot
- **`honeypot-*.sh`** - Various honeypot management scripts
- **`fluent-bit.conf`** - Log forwarding configuration

## 🎯 Migration Workflow

1. **Infrastructure**: Creates AWS resources (EC2, RDS, S3, Security Groups)
2. **Application**: Uploads code, installs dependencies, configures services  
3. **Integration**: Connects TPOT honeypot to AWS backend
4. **Validation**: Tests complete data flow and API endpoints

## 📊 Monitoring

Created during deployment:
- `~/monitor-tpot-connection.sh` - Monitor TPOT → AWS data flow
- CloudWatch logging for all services
- Health check endpoints

## 🔧 Management

Common operations:
```bash
# Check deployment status
aws cloudformation describe-stacks --stack-name mini-xdr-backend

# SSH to backend
ssh -i ~/.ssh/mini-xdr-tpot-key.pem ubuntu@BACKEND_IP

# View service logs
sudo journalctl -u mini-xdr -f

# Restart services
sudo systemctl restart mini-xdr
```

## 📚 Documentation

See `../docs/AWS_MIGRATION_GUIDE.md` for complete migration guide.
