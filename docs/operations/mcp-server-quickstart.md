# Mini-XDR MCP Server - Quick Start Guide

**Deploy AI Assistant Integration to AWS in 5 Minutes**

---

## What You're Deploying

A **Model Context Protocol (MCP) server** with **37 tools** that lets AI assistants like Claude Code interact with your Mini-XDR platform:

- 📋 Query incidents and events
- 🔍 Hunt threats and run forensics
- 🤖 Orchestrate multi-agent responses
- 🕵️ Lookup threat intelligence
- 🎯 Execute containment actions

---

## Prerequisites Checklist

```bash
# 1. AWS CLI configured
aws sts get-caller-identity

# 2. kubectl configured for EKS
kubectl get nodes

# 3. Backend deployed and healthy
kubectl get pods -n mini-xdr | grep backend
```

✅ All three working? Let's deploy!

---

## Deployment (One Command)

```bash
cd /Users/chasemad/Desktop/mini-xdr
./ops/deploy-mcp-server.sh --all
```

**Expected time:** 3-5 minutes

**What it does:**
1. Builds Docker image (~200MB)
2. Creates ECR repository
3. Pushes to ECR
4. Deploys to Kubernetes (2 pods)
5. Waits for health checks

---

## Update ALB Ingress

```bash
# Add /mcp route to your ALB
kubectl apply -f ops/k8s/ingress-with-mcp.yaml

# Wait 2-3 minutes for ALB to update
kubectl get ingress -n mini-xdr -w
```

---

## Connect Claude Code

### 1. Get your ALB URL

```bash
ALB_URL=$(kubectl get ingress -n mini-xdr mini-xdr-ingress -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
echo "Your MCP URL: http://${ALB_URL}/mcp"
```

### 2. Test it works

```bash
curl http://${ALB_URL}/mcp/health
# Should return: {"status":"healthy",...}
```

### 3. Add to Claude Code

```bash
claude mcp add --transport http mini-xdr http://${ALB_URL}/mcp
```

---

## Test It Out

In Claude Code, try these commands:

```
"Show me all incidents from the last 24 hours"
"Analyze incident #1"
"Hunt for SSH brute force attacks"
"What's the system health?"
```

Claude will automatically use the MCP tools to query your live XDR data!

---

## Verify Deployment

```bash
# Check pods are running
kubectl get pods -n mini-xdr -l app=mini-xdr-mcp-server

# Check logs
kubectl logs -n mini-xdr -l app=mini-xdr-mcp-server --tail=50

# Test health endpoint
MCP_POD=$(kubectl get pods -n mini-xdr -l app=mini-xdr-mcp-server -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n mini-xdr $MCP_POD -- curl http://localhost:3001/health
```

---

## Troubleshooting

### Pods won't start?
```bash
kubectl describe pod -n mini-xdr -l app=mini-xdr-mcp-server
kubectl logs -n mini-xdr -l app=mini-xdr-mcp-server
```

### Health check failing?
```bash
# Test backend connection from MCP pod
kubectl exec -n mini-xdr $MCP_POD -- curl http://backend-service:8000/health
```

### Claude Code can't connect?
```bash
# Port forward and test locally
kubectl port-forward -n mini-xdr svc/mcp-server-service 3001:3001
curl http://localhost:3001/health

# Add local version
claude mcp add --transport http mini-xdr-local http://localhost:3001
```

---

## What's Next?

1. **Read full docs:** `docs/deployment/MCP_SERVER_DEPLOYMENT.md`
2. **Enable SSL/TLS** for production (recommended)
3. **Set up monitoring** with CloudWatch
4. **Explore all 37 tools** - full list in docs

---

## Architecture Diagram

```
┌─────────────────────────────┐
│   Claude Code (Your Mac)    │
└─────────────┬───────────────┘
              │ HTTP
┌─────────────▼───────────────┐
│   AWS ALB                   │
│   /mcp → mcp-server:3001    │
└─────────────┬───────────────┘
              │
┌─────────────▼───────────────┐
│   MCP Server Pods (2x)      │
│   - 37 tools                │
│   - Node.js 20              │
└─────────────┬───────────────┘
              │ API calls
┌─────────────▼───────────────┐
│   Backend Service           │
│   - FastAPI                 │
│   - ML Models               │
│   - AI Agents               │
└─────────────────────────────┘
```

---

## Production Checklist

Before going live:

- [ ] Enable SSL/TLS with ACM certificate
- [ ] Set API key for backend authentication
- [ ] Configure CloudWatch log forwarding
- [ ] Restrict ALB source IPs (if possible)
- [ ] Set up monitoring and alerts
- [ ] Document MCP URL for your team

---

## Need Help?

**View logs:**
```bash
kubectl logs -n mini-xdr -l app=mini-xdr-mcp-server -f
```

**Check status:**
```bash
kubectl get all -n mini-xdr -l app=mini-xdr-mcp-server
```

**Full documentation:**
```bash
cat docs/deployment/MCP_SERVER_DEPLOYMENT.md
```

---

**Deployed successfully?** 🎉

Try asking Claude Code: *"Show me all high-severity incidents and help me investigate the most recent one"*

Claude will use your MCP tools to query live data and orchestrate responses!
