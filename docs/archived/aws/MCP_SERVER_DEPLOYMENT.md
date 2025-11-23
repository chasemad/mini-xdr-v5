# Mini-XDR MCP Server Deployment Guide

**Model Context Protocol Server for AI Assistant Integration**

This guide covers deploying the Mini-XDR MCP server to AWS EKS, enabling AI assistants like Claude Code to access your XDR platform's 37 tools for incident management, threat hunting, and orchestration.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Deployment](#quick-deployment)
4. [Detailed Steps](#detailed-steps)
5. [Connecting AI Assistants](#connecting-ai-assistants)
6. [Available Tools](#available-tools)
7. [Troubleshooting](#troubleshooting)
8. [Security Considerations](#security-considerations)

---

## Overview

The MCP server exposes **37 specialized tools** to AI assistants, enabling them to:

- 📋 Query and analyze security incidents
- 🔍 Perform threat hunting and forensic investigations
- 🤖 Orchestrate multi-agent responses
- 🕵️ Lookup threat intelligence
- 🎯 Execute containment actions
- 📊 Analyze patterns and correlations

### Architecture

```
┌─────────────────────┐
│  AI Assistant       │
│  (Claude Code)      │
└──────────┬──────────┘
           │ HTTP/SSE
           │
┌──────────▼──────────┐
│  AWS ALB            │
│  /mcp route         │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  MCP Server         │
│  (2 pods, port 3001)│
└──────────┬──────────┘
           │ HTTP API
┌──────────▼──────────┐
│  Backend Service    │
│  (port 8000)        │
└─────────────────────┘
```

---

## Prerequisites

### Required Tools
- ✅ Docker (for building images)
- ✅ AWS CLI configured with credentials
- ✅ kubectl configured for your EKS cluster
- ✅ Git (for version tracking)

### AWS Resources
- ✅ EKS cluster running (e.g., `mini-xdr-cluster`)
- ✅ ECR repository access
- ✅ ALB Ingress Controller installed
- ✅ Mini-XDR backend deployed and running

### Verify Prerequisites

```bash
# Check AWS credentials
aws sts get-caller-identity

# Check kubectl context
kubectl config current-context

# Verify backend is running
kubectl get pods -n mini-xdr -l app=mini-xdr-backend

# Check ALB ingress controller
kubectl get pods -n kube-system | grep aws-load-balancer-controller
```

---

## Quick Deployment

### One-Command Deployment

```bash
cd /path/to/mini-xdr
./ops/deploy-mcp-server.sh --all
```

This will:
1. ✅ Build the Docker image
2. ✅ Create ECR repository (if needed)
3. ✅ Push image to ECR
4. ✅ Deploy to Kubernetes
5. ✅ Wait for rollout completion
6. ✅ Verify health checks

### Expected Output

```
[12:00:00] ✅ ECR repository exists
[12:00:05] 🐳 Building Docker image...
[12:02:30] ✅ Image built: 123456.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-mcp-server:latest
[12:03:00] ✅ Image pushed to ECR
[12:03:30] 📦 Deploying to Kubernetes...
[12:04:00] ✅ Deployment ready!
[12:04:05] ✅ MCP server is healthy!
```

---

## Detailed Steps

### Step 1: Build Docker Image

```bash
cd /path/to/mini-xdr

# Build with custom tag
IMAGE_TAG=v1.0.0 ./ops/deploy-mcp-server.sh --build

# Or manually
docker build -f ops/Dockerfile.mcp-server -t mini-xdr-mcp-server:latest .
```

**Image Size:** ~200MB (Node.js 20 slim + dependencies)

### Step 2: Push to ECR

```bash
# Using deployment script
./ops/deploy-mcp-server.sh --push

# Or manually
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# Authenticate
aws ecr get-login-password --region $AWS_REGION | \
    docker login --username AWS --password-stdin $ECR_REGISTRY

# Tag and push
docker tag mini-xdr-mcp-server:latest ${ECR_REGISTRY}/mini-xdr-mcp-server:latest
docker push ${ECR_REGISTRY}/mini-xdr-mcp-server:latest
```

### Step 3: Deploy to Kubernetes

```bash
# Using deployment script
./ops/deploy-mcp-server.sh --deploy

# Or manually
kubectl apply -f ops/k8s/mcp-server-deployment.yaml
kubectl apply -f ops/k8s/mcp-server-service.yaml
```

### Step 4: Update ALB Ingress

```bash
# Apply updated ingress configuration
kubectl apply -f ops/k8s/ingress-with-mcp.yaml

# Wait for ALB to update (2-3 minutes)
kubectl get ingress -n mini-xdr -w
```

### Step 5: Verify Deployment

```bash
# Check pods
kubectl get pods -n mini-xdr -l app=mini-xdr-mcp-server

# Check service
kubectl get svc -n mini-xdr mcp-server-service

# Check health
MCP_POD=$(kubectl get pods -n mini-xdr -l app=mini-xdr-mcp-server -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n mini-xdr $MCP_POD -- curl http://localhost:3001/health

# Expected output:
# {"status":"healthy","service":"mini-xdr-mcp-server","transport":"http",...}
```

---

## Connecting AI Assistants

### Option A: Claude Code (via HTTP)

#### 1. Get your ALB URL

```bash
kubectl get ingress -n mini-xdr mini-xdr-ingress -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
```

Example: `k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com`

#### 2. Test MCP endpoint

```bash
ALB_URL="your-alb-url-here"
curl http://${ALB_URL}/mcp/health
```

#### 3. Add to Claude Code

```bash
# Add MCP server to Claude Code
claude mcp add --transport http mini-xdr \
  http://${ALB_URL}/mcp
```

#### 4. Verify connection

```bash
# List configured MCP servers
claude mcp list

# Test a tool
# In Claude Code conversation:
# "Use mini-xdr to show me recent incidents"
```

### Option B: Port Forward (Local Testing)

```bash
# Forward MCP server port
kubectl port-forward -n mini-xdr svc/mcp-server-service 3001:3001

# In another terminal, add to Claude Code
claude mcp add --transport http mini-xdr-local \
  http://localhost:3001/mcp

# Test
curl http://localhost:3001/health
```

### Option C: Direct MCP Server Access (Advanced)

For production deployments with custom domain:

```bash
# Apply separate ingress with SSL
kubectl apply -f ops/k8s/ingress-mcp-separate.yaml

# Point mcp.yourdomain.com DNS to ALB
# Add to Claude Code with HTTPS
claude mcp add --transport http mini-xdr \
  https://mcp.yourdomain.com/
```

---

## Available Tools

The MCP server provides **37 tools** across 8 categories:

### 1. Incident Management (4 tools)
- `get_incidents` - List security incidents with filtering
- `get_incident` - Get detailed incident information
- `analyze_incident_deep` - Deep AI-powered analysis
- `contain_incident` - Execute containment actions

### 2. Natural Language Processing (3 tools)
- `natural_language_query` - Ask questions in plain English
- `nlp_threat_analysis` - AI-powered threat analysis
- `semantic_incident_search` - Search using natural language

### 3. Threat Hunting (4 tools)
- `threat_hunt` - Proactive threat discovery
- `forensic_investigation` - Evidence collection and analysis
- `query_threat_patterns` - Pattern-based searching
- `correlation_analysis` - Find related incidents

### 4. AI Orchestration (6 tools)
- `orchestrate_response` - Multi-agent coordination
- `get_orchestrator_status` - Check orchestration status
- `get_workflow_status` - Track workflow execution
- `get_workflow_execution_status` - Detailed execution info
- `get_agent_actions` - View agent action history
- `rollback_agent_action` - Undo agent actions

### 5. Threat Intelligence (2 tools)
- `threat_intel_lookup` - Query external threat feeds
- `attribution_analysis` - Threat actor profiling

### 6. Response Workflows (5 tools)
- `create_visual_workflow` - Design custom workflows
- `get_available_response_actions` - List action types
- `execute_response_workflow` - Run automated responses
- `execute_enterprise_action` - EDR/IAM/DLP/TPot actions
- `get_response_impact_metrics` - Measure effectiveness

### 7. Real-time Monitoring (4 tools)
- `start_incident_stream` - Stream live incidents
- `stop_incident_stream` - Stop streaming
- `get_system_health` - System status check
- `get_auto_contain_setting` - Check auto-containment

### 8. Advanced Operations (9 tools)
- `unblock_incident` - Remove containment
- `schedule_unblock` - Scheduled unblocking
- `set_auto_contain_setting` - Configure auto-response
- `test_tpot_integration` - Test honeypot connection
- `execute_tpot_command` - Control honeypot
- `execute_iam_action` - IAM operations
- `execute_edr_action` - EDR commands
- `execute_dlp_action` - DLP controls
- `get_response_impact_metrics` - Analytics

### Example Tool Usage (in Claude Code)

```
You: "Show me all critical incidents from the last 24 hours"
Claude: [Uses get_incidents tool with status filter]

You: "Analyze incident #42"
Claude: [Uses analyze_incident_deep with AI orchestration]

You: "Hunt for signs of lateral movement"
Claude: [Uses threat_hunt with MITRE ATT&CK techniques]
```

---

## Troubleshooting

### Issue: Pods Not Starting

```bash
# Check pod status
kubectl describe pod -n mini-xdr -l app=mini-xdr-mcp-server

# Check logs
kubectl logs -n mini-xdr -l app=mini-xdr-mcp-server

# Common fixes:
# 1. Image pull errors - verify ECR permissions
# 2. Backend connection - check API_BASE env var
# 3. Resource limits - check node capacity
```

### Issue: Health Check Failing

```bash
# Port forward and test directly
kubectl port-forward -n mini-xdr svc/mcp-server-service 3001:3001
curl http://localhost:3001/health

# Check backend connectivity
MCP_POD=$(kubectl get pods -n mini-xdr -l app=mini-xdr-mcp-server -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n mini-xdr $MCP_POD -- curl http://backend-service:8000/health
```

### Issue: ALB Not Routing to MCP

```bash
# Check ingress rules
kubectl get ingress -n mini-xdr mini-xdr-ingress -o yaml

# Verify target groups in AWS console
aws elbv2 describe-target-groups --region us-east-1

# Check target health
aws elbv2 describe-target-health --target-group-arn <arn>
```

### Issue: Claude Code Can't Connect

```bash
# Test MCP endpoint externally
ALB_URL=$(kubectl get ingress -n mini-xdr mini-xdr-ingress -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
curl -v http://${ALB_URL}/mcp/health

# Check security groups allow inbound HTTP/HTTPS

# Verify MCP is responding
kubectl logs -n mini-xdr -l app=mini-xdr-mcp-server --tail=50
```

### Issue: Tools Not Working

```bash
# Test backend API directly
kubectl exec -n mini-xdr $MCP_POD -- curl http://backend-service:8000/api/incidents

# Check API key if required
kubectl get secret -n mini-xdr mini-xdr-secrets -o yaml

# View MCP server logs for errors
kubectl logs -n mini-xdr -l app=mini-xdr-mcp-server -f
```

---

## Security Considerations

### 1. Network Security

**✅ Implemented:**
- Network policies restrict pod-to-pod communication
- ClusterIP service keeps MCP internal by default
- Security groups control ALB ingress

**⚠️ Recommendations:**
```bash
# Restrict ALB source IPs (if possible)
kubectl annotate ingress mini-xdr-ingress -n mini-xdr \
  service.beta.kubernetes.io/load-balancer-source-ranges="YOUR_IP/32"

# Enable SSL/TLS
# Update ingress-with-mcp.yaml with ACM certificate ARN
```

### 2. Authentication

**⚠️ Current State:**
- Backend API key optional
- No MCP-level authentication (relies on network security)

**🔐 Recommendations:**
```bash
# Set API key for backend
kubectl create secret generic mini-xdr-secrets -n mini-xdr \
  --from-literal=api-key="your-secure-api-key" \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart MCP server to pick up secret
kubectl rollout restart deployment/mini-xdr-mcp-server -n mini-xdr
```

### 3. SSL/TLS

**Strongly Recommended for Production:**

1. Request ACM certificate:
```bash
aws acm request-certificate \
  --domain-name mcp.yourdomain.com \
  --validation-method DNS
```

2. Update ingress:
```yaml
annotations:
  alb.ingress.kubernetes.io/listen-ports: '[{"HTTPS":443}]'
  alb.ingress.kubernetes.io/certificate-arn: arn:aws:acm:...
  alb.ingress.kubernetes.io/ssl-redirect: '443'
```

### 4. Audit Logging

```bash
# Enable pod logs persistence
kubectl logs -n mini-xdr -l app=mini-xdr-mcp-server --tail=-1 > mcp-audit.log

# Send to CloudWatch
# Install Fluent Bit DaemonSet for log forwarding
```

---

## Maintenance

### Updating MCP Server

```bash
# Update code and redeploy
cd /path/to/mini-xdr
git pull
./ops/deploy-mcp-server.sh --all

# Rolling update happens automatically
```

### Scaling

```bash
# Manual scaling
kubectl scale deployment mini-xdr-mcp-server -n mini-xdr --replicas=4

# HPA automatically scales based on CPU/memory (configured in deployment)
```

### Monitoring

```bash
# Watch pods
kubectl get pods -n mini-xdr -l app=mini-xdr-mcp-server -w

# View metrics
kubectl top pods -n mini-xdr -l app=mini-xdr-mcp-server

# Check HPA status
kubectl get hpa -n mini-xdr mini-xdr-mcp-server-hpa
```

---

## Next Steps

1. ✅ Deploy MCP server: `./ops/deploy-mcp-server.sh --all`
2. ✅ Update ingress: `kubectl apply -f ops/k8s/ingress-with-mcp.yaml`
3. ✅ Connect Claude Code: `claude mcp add --transport http mini-xdr http://ALB_URL/mcp`
4. ✅ Test tools: "Show me recent incidents"
5. 🔐 Enable SSL/TLS for production
6. 📊 Set up monitoring and alerting

---

## Support

- **Logs:** `kubectl logs -n mini-xdr -l app=mini-xdr-mcp-server -f`
- **Status:** `kubectl get all -n mini-xdr -l app=mini-xdr-mcp-server`
- **Describe:** `kubectl describe deployment mini-xdr-mcp-server -n mini-xdr`

For issues, check:
1. Pod status and logs
2. Backend API connectivity
3. ALB target group health
4. Network policies and security groups
