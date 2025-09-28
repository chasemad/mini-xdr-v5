# Mini-XDR AWS Migration Guide

This guide provides complete instructions for migrating your Mini-XDR cybersecurity system to AWS for full cloud-based operation.

## 🎯 Overview

The AWS migration moves all Mini-XDR components to the cloud, solving connectivity issues and creating a scalable, production-ready cybersecurity platform.

### Migration Components

- **✅ Mini-XDR Backend**: EC2 instance with full ML/AI capabilities
- **✅ PostgreSQL Database**: RDS for production scalability  
- **✅ ML Models Storage**: S3 bucket for model artifacts
- **✅ Direct TPOT Connection**: Cloud-to-cloud data flow
- **✅ Security Configuration**: Proper IAM roles and security groups
- **✅ Monitoring**: Health checks and data flow monitoring

## 🚀 Quick Start

### One-Command Migration

Run the master deployment script to migrate everything:

```bash
cd /Users/chasemad/Desktop/mini-xdr/ops
./deploy-full-aws-migration.sh
```

This script will:
1. Deploy AWS infrastructure (15-20 minutes)
2. Upload and configure application code (10-15 minutes)
3. Establish direct TPOT → AWS connection
4. Validate the complete deployment

## 📋 Prerequisites

### Required Tools
- AWS CLI installed and configured
- SSH access to existing TPOT instance
- Key pair: `mini-xdr-tpot-key` in AWS
- Tools: `jq`, `curl`, `ssh`, `scp`

### AWS Configuration
```bash
# Configure AWS CLI
aws configure

# Verify access
aws sts get-caller-identity

# Check key pair exists
aws ec2 describe-key-pairs --key-names mini-xdr-tpot-key
```

### Current Infrastructure
- **TPOT Instance**: 34.193.101.171 (will remain unchanged)
- **Your IP**: 24.11.0.176 (for security group access)
- **SSH Key**: ~/.ssh/mini-xdr-tpot-key.pem

## 🏗️ Architecture

### AWS Resources Created

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   TPOT Honeypot │───▶│  Mini-XDR EC2   │───▶│   RDS Database  │
│  34.193.101.171 │    │   (Backend)     │    │  (PostgreSQL)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   S3 Bucket     │
                       │  (ML Models)    │
                       └─────────────────┘
```

### Data Flow
1. **TPOT** captures real attacks and generates logs
2. **Fluent Bit** forwards logs to AWS Mini-XDR
3. **Mini-XDR Backend** processes logs with ML models and AI agents
4. **Database** stores security events and analysis results
5. **Frontend** (local or AWS) displays real-time attack data

## 🔧 Manual Step-by-Step Deployment

If you prefer manual control or need to troubleshoot:

### Step 1: Deploy Infrastructure
```bash
cd /Users/chasemad/Desktop/mini-xdr/ops
./deploy-mini-xdr-aws.sh
```

This creates:
- VPC with public/private subnets
- EC2 instance (t3.medium) for Mini-XDR backend
- RDS PostgreSQL database
- S3 bucket for ML models
- Security groups and IAM roles
- Elastic IP for consistent addressing

### Step 2: Deploy Application
```bash
./deploy-mini-xdr-code.sh
```

This handles:
- Code upload to EC2 instance
- Python dependencies installation
- Database schema initialization
- ML models upload to S3
- Service configuration and startup

### Step 3: Configure TPOT Connection
```bash
./configure-tpot-aws-connection.sh
```

This establishes:
- Direct TPOT → AWS Mini-XDR data flow
- Updated Fluent Bit configuration
- Security group rules for communication
- Data flow validation

## ⚙️ Configuration

### Environment Variables

The deployment creates a comprehensive `.env` file on the EC2 instance. Key settings:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
UI_ORIGIN=http://localhost:3000,http://YOUR_IP:3000

# Database (automatically configured)
DATABASE_URL=postgresql://postgres:PASSWORD@RDS_ENDPOINT:5432/postgres

# TPOT Integration
HONEYPOT_HOST=34.193.101.171
HONEYPOT_SSH_PORT=64295

# AWS Resources
MODELS_BUCKET=mini-xdr-models-ACCOUNT-REGION
AWS_REGION=us-east-1
```

### Frontend Configuration

To connect your local frontend to AWS backend:

```bash
# Copy AWS configuration
cp frontend/env.local.aws frontend/.env.local

# Start frontend
cd frontend
npm run dev
```

## 🔐 Security

### Access Control
- **Management**: Restricted to your IP (24.11.0.176/32)
- **API Access**: Controlled by security groups
- **Database**: Private subnet, backend access only
- **SSH Keys**: Required for all management access

### Security Groups

#### Backend Security Group
- Port 22: SSH from your IP only
- Port 8000: API access from your IP and TPOT
- Internal VPC communication allowed

#### Database Security Group  
- Port 5432: PostgreSQL from backend only
- No public access

### IAM Roles
- EC2 instance has minimal permissions for CloudWatch and S3
- No unnecessary privileges granted

## 📊 Monitoring

### Health Checks
```bash
# Check API health
curl http://BACKEND_IP:8000/health

# Check events endpoint
curl http://BACKEND_IP:8000/events

# Monitor data flow
~/monitor-tpot-connection.sh BACKEND_IP
```

### Service Management
```bash
# SSH to backend
ssh -i ~/.ssh/mini-xdr-tpot-key.pem ubuntu@BACKEND_IP

# Check service status
sudo systemctl status mini-xdr

# View logs
sudo journalctl -u mini-xdr -f

# Restart service
sudo systemctl restart mini-xdr
```

### Log Locations
- **Application**: `journalctl -u mini-xdr`
- **Setup**: `/var/log/mini-xdr-setup.log`
- **CloudInit**: `/var/log/cloud-init-output.log`

## 🧪 Testing

### Validate Real Attack Data Flow

1. **Generate Test Attack**:
   ```bash
   # SSH to TPOT will generate logs
   ssh admin@34.193.101.171
   ```

2. **Check Data Reception**:
   ```bash
   curl http://BACKEND_IP:8000/events | jq .
   ```

3. **Monitor Globe Visualization**:
   - Update frontend to use AWS backend
   - View real-time attack data on globe

### Performance Testing
```bash
# Load test API
curl -w "@curl-format.txt" -s http://BACKEND_IP:8000/events

# Database connectivity
ssh ubuntu@BACKEND_IP
cd /opt/mini-xdr/backend
source ../venv/bin/activate
python -c "from app.models import database; print('DB connected!')"
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Health Check Fails
```bash
# Check service status
ssh ubuntu@BACKEND_IP
sudo systemctl status mini-xdr
sudo journalctl -u mini-xdr -n 50
```

#### 2. Database Connection Issues
```bash
# Test database connectivity
python -c "
import asyncio
import asyncpg
async def test():
    conn = await asyncpg.connect('postgresql://postgres:PASSWORD@ENDPOINT:5432/postgres')
    print('Connected!')
    await conn.close()
asyncio.run(test())
"
```

#### 3. TPOT Data Not Flowing
```bash
# Check TPOT Fluent Bit
ssh -i ~/.ssh/mini-xdr-tpot-key.pem -p 64295 admin@34.193.101.171
sudo docker logs $(sudo docker ps -q --filter "name=fluent")

# Test connectivity from TPOT
curl -X POST http://BACKEND_IP:8000/ingest/multi \
  -H "Authorization: Bearer tpot-honeypot-key" \
  -d '{"test": "connectivity"}'
```

#### 4. Frontend Connection Issues
```bash
# Check CORS settings
curl -H "Origin: http://localhost:3000" \
     -H "Access-Control-Request-Method: GET" \
     -H "Access-Control-Request-Headers: X-Requested-With" \
     -X OPTIONS http://BACKEND_IP:8000/events
```

### Recovery Procedures

#### Restart All Services
```bash
ssh ubuntu@BACKEND_IP
sudo systemctl restart mini-xdr
sudo systemctl restart nginx
```

#### Redeploy Application
```bash
cd /Users/chasemad/Desktop/mini-xdr/ops
./deploy-mini-xdr-code.sh
```

#### Full Infrastructure Recovery
```bash
# Delete existing stack
aws cloudformation delete-stack --stack-name mini-xdr-backend

# Wait for deletion
aws cloudformation wait stack-delete-complete --stack-name mini-xdr-backend

# Redeploy everything
./deploy-full-aws-migration.sh
```

## 💰 Cost Optimization

### Monthly Cost Estimate
- **EC2 t3.medium**: ~$30/month
- **RDS db.t3.micro**: ~$15/month  
- **EIP**: ~$4/month
- **S3 storage**: ~$1/month
- **Data transfer**: ~$5/month
- **Total**: ~$55/month

### Cost Reduction Options
1. **Use t3.small**: Save ~$15/month
2. **Stop instances when not needed**: Save 70%+
3. **Reserved instances**: Save 40% with 1-year commitment

## 🔄 Maintenance

### Regular Tasks
- **Weekly**: Check system logs and performance
- **Monthly**: Update dependencies and security patches
- **Quarterly**: Review costs and optimize resources

### Update Procedures
```bash
# Update application code
cd /Users/chasemad/Desktop/mini-xdr/ops
./deploy-mini-xdr-code.sh

# Update TPOT configuration
./configure-tpot-aws-connection.sh
```

## 📈 Scaling

### Horizontal Scaling
- Deploy additional Mini-XDR instances in other regions
- Use Application Load Balancer for distribution
- Implement Redis for shared state

### Vertical Scaling
- Upgrade to t3.large or t3.xlarge for more ML processing
- Increase RDS instance size for larger datasets
- Add read replicas for database scaling

## 🎯 Success Criteria

After successful migration, you should see:

✅ **Real Attack Data**: TPOT sending live attack data to AWS  
✅ **ML Processing**: AI agents analyzing real threats  
✅ **Globe Visualization**: Real-time global attack display  
✅ **Scalable Architecture**: Production-ready cloud deployment  
✅ **Security**: Proper access controls and encryption  

## 📞 Support

### Getting Help
- Check logs first: `sudo journalctl -u mini-xdr -f`
- Review troubleshooting section above
- Use monitoring script for data flow issues
- AWS CloudFormation console for infrastructure issues

### Useful Commands
```bash
# Quick status check
curl -s http://BACKEND_IP:8000/health | jq .

# Complete system status
ssh ubuntu@BACKEND_IP 'sudo systemctl status mini-xdr nginx'

# Data flow monitoring
~/monitor-tpot-connection.sh BACKEND_IP
```

---

## 🎉 Deployment Complete!

Your Mini-XDR system is now fully deployed on AWS with:
- ✅ Real attack data collection from TPOT honeypot
- ✅ ML-powered threat analysis in the cloud  
- ✅ Scalable, production-ready architecture
- ✅ Direct cloud-to-cloud data flow
- ✅ Enhanced security and monitoring

The system is ready for real-world cybersecurity threat detection and response!
