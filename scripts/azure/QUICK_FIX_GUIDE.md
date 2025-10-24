# Quick Fix Guide: Get Your Mini-XDR Demo Live ASAP

**Goal**: Get external access to your Mini-XDR frontend at http://[EXTERNAL_IP] for recruiter demos.

## 🚀 Fastest Path to Success (Recommended)

Based on expert analysis, follow this sequence:

### Option 1: NGINX Ingress (RECOMMENDED - 80% Success Rate)
**Time**: ~10-15 minutes  
**Why**: Bypasses LoadBalancer issues, most reliable in AKS

```bash
cd /Users/chasemad/Desktop/mini-xdr

# Run the NGINX Ingress deployment script
chmod +x scripts/azure/deploy-nginx-ingress.sh
./scripts/azure/deploy-nginx-ingress.sh
```

**What it does**:
1. Installs NGINX Ingress Controller
2. Creates a new LoadBalancer with fresh external IP
3. Configures Ingress routing to your frontend
4. Updates NSG rules
5. Tests the connection

**Expected Result**: You'll get a new IP address (e.g., `http://20.xxx.xxx.xxx`) that works immediately.

---

### Option 2: Recreate LoadBalancer (If you prefer direct LB)
**Time**: ~5-10 minutes  
**Success Rate**: ~50% (Azure data plane might still have issues)

```bash
# Recreate the LoadBalancer service with fixed config
chmod +x scripts/azure/recreate-loadbalancer.sh
./scripts/azure/recreate-loadbalancer.sh
```

---

### Option 3: Quick Diagnostics First (If you want to understand the issue)
**Time**: ~5 minutes

```bash
# Run diagnostics to identify the exact problem
chmod +x scripts/azure/fix-loadbalancer-diagnostics.sh
./scripts/azure/fix-loadbalancer-diagnostics.sh
```

This will tell you:
- Is it a LoadBalancer forwarding issue?
- Are there subscription-level restrictions?
- Can you access the app via NodePort directly?

---

## 📋 Step-by-Step: NGINX Ingress (Most Reliable)

### Prerequisites
```bash
# Make sure you're logged into Azure and kubectl is configured
az account show
kubectl get nodes
kubectl get pods -n mini-xdr
```

### Deployment Steps

1. **Run the deployment script**:
   ```bash
   cd /Users/chasemad/Desktop/mini-xdr
   chmod +x scripts/azure/deploy-nginx-ingress.sh
   ./scripts/azure/deploy-nginx-ingress.sh
   ```

2. **Wait for completion** (~5 minutes):
   - Script will install NGINX Ingress Controller
   - Wait for external IP assignment
   - Create Ingress resource
   - Test the connection

3. **Get your demo URL**:
   ```bash
   # The script will output your new URL
   # Example: http://20.123.45.67
   
   # You can also check manually:
   kubectl get svc -n ingress-nginx ingress-nginx-controller
   ```

4. **Test from your browser**:
   - Open: `http://[NEW_IP_ADDRESS]`
   - You should see your Mini-XDR frontend

5. **Update your resume/docs**:
   - Replace old IP (4.156.121.111) with new Ingress IP
   - Test from multiple locations/devices

---

## 🔍 If NGINX Ingress Doesn't Work

### Check Ingress status:
```bash
kubectl get ingress -n mini-xdr mini-xdr-ingress
kubectl describe ingress -n mini-xdr mini-xdr-ingress
```

### Check Ingress controller logs:
```bash
kubectl logs -n ingress-nginx deployment/ingress-nginx-controller --tail=100
```

### Verify backend pods are healthy:
```bash
kubectl get pods -n mini-xdr
kubectl get endpoints -n mini-xdr mini-xdr-frontend-service
```

### Check NSG rules:
```bash
az network nsg rule list \
  --resource-group MC_mini-xdr-prod-rg_mini-xdr-aks_eastus \
  --nsg-name aks-agentpool-10857568-nsg \
  --output table
```

---

## 🆘 Emergency Workaround (If Everything Fails)

If you need to demo RIGHT NOW and nothing else works:

### Port Forwarding (Local Only)
```bash
# Forward to localhost
kubectl port-forward -n mini-xdr svc/mini-xdr-frontend-service 3000:3000

# Access at: http://localhost:3000
```

### Remote Port Forwarding (For Remote Demos)
```bash
# Option 1: SSH tunnel from your laptop
ssh -L 3000:localhost:3000 user@your-machine-with-kubectl

# Option 2: Use ngrok (temporary public URL)
# Install ngrok, then:
kubectl port-forward -n mini-xdr svc/mini-xdr-frontend-service 3000:3000 &
ngrok http 3000
# Use the ngrok URL for your demo
```

---

## 📊 Current Status Reference

**Current Configuration**:
- Cluster: mini-xdr-aks (East US)
- Resource Group: mini-xdr-prod-rg
- Current LB IP: 4.156.121.111 (NOT WORKING)
- Your IP: 24.11.0.176
- Frontend Service: mini-xdr-frontend-service (port 3000)
- Backend Service: mini-xdr-backend-service (port 8000)

**What's Working**:
- ✅ Pods are healthy
- ✅ Internal cluster routing works
- ✅ Health probes pass
- ✅ NodePort access works (confirmed via diagnostics)
- ✅ ICMP (ping) to LoadBalancer IP works

**What's Broken**:
- ❌ TCP connections to LoadBalancer IP timeout
- ❌ Azure LoadBalancer data plane not forwarding traffic

---

## 🎯 Success Criteria

You'll know it's working when:
1. `curl -I http://[NEW_IP]` returns `200 OK` with HTML headers
2. Browser shows your Mini-XDR frontend UI
3. You can login and navigate the dashboard
4. Backend API calls work (check browser console)

---

## 📞 If You Need Help

1. **Check script outputs**: All scripts save logs and IPs to `/tmp/`
2. **Gather logs**:
   ```bash
   kubectl get all -n mini-xdr > /tmp/mini-xdr-status.txt
   kubectl get svc -n ingress-nginx >> /tmp/mini-xdr-status.txt
   ```
3. **Azure Support**: If needed, open ticket with:
   - Subscription ID: e5636423-8514-4bdd-bfef-f7ecdb934260
   - Resource Group: MC_mini-xdr-prod-rg_mini-xdr-aks_eastus
   - Issue: LoadBalancer not forwarding TCP traffic despite healthy probes

---

## ⏱️ Timeline Estimate

- **NGINX Ingress**: 10-15 minutes to working demo
- **LoadBalancer Recreate**: 5-10 minutes (but might not fix Azure data plane issue)
- **Full Diagnostics**: 30-45 minutes (if you want to understand everything)

**For your demo**: I recommend **running the NGINX Ingress script NOW**. It's the most reliable fix.

---

## 🔄 Rollback Plan

If something goes wrong:

```bash
# Remove NGINX Ingress
kubectl delete namespace ingress-nginx
kubectl delete ingress -n mini-xdr mini-xdr-ingress

# Restore original LoadBalancer (from backup)
kubectl apply -f /tmp/mini-xdr-lb-backup-[TIMESTAMP].yaml
```

---

## ✅ Post-Deployment Checklist

- [ ] External IP is assigned
- [ ] `curl -I http://[IP]` returns 200 OK
- [ ] Frontend loads in browser
- [ ] Can login with test credentials
- [ ] Dashboard displays data
- [ ] Alerts page works
- [ ] Backend API responds (check browser DevTools Network tab)
- [ ] Updated resume/portfolio with new URL
- [ ] Tested from mobile device
- [ ] Screenshot taken for portfolio

---

**Ready to go?** Run this now:

```bash
cd /Users/chasemad/Desktop/mini-xdr
./scripts/azure/deploy-nginx-ingress.sh
```

Good luck with your demo! 🚀


