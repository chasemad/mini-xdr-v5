# Azure AKS LoadBalancer Debugging - Handoff Prompt

## COPY THIS ENTIRE PROMPT TO START A NEW DEBUGGING SESSION

---

I need you to act as a senior software engineer debugging an Azure AKS LoadBalancer connectivity issue. Here's the complete context:

## Current Deployment State

**Project:** Mini-XDR (Security Operations Center platform)  
**Cloud:** Azure AKS (East US region)  
**Cluster:** mini-xdr-aks  
**Resource Group:** mini-xdr-prod-rg  
**Managed RG:** MC_mini-xdr-prod-rg_mini-xdr-aks_eastus  
**External IP:** 4.156.121.111  
**My IP (to whitelist):** 24.11.0.176  

### Kubernetes Resources Deployed

**Namespace:** mini-xdr

**Pods (all healthy):**
```
mini-xdr-backend-55544d45b9-7625t    1/1 Running
mini-xdr-frontend-6787c58cff-snbgb   1/1 Running  
mini-xdr-frontend-6787c58cff-wf5nz   1/1 Running
```

**Services:**
```
mini-xdr-backend-service    ClusterIP      10.1.8.89      8000/TCP,9090/TCP
mini-xdr-frontend-service   ClusterIP      10.1.75.224    3000/TCP
mini-xdr-loadbalancer       LoadBalancer   10.1.174.232   4.156.121.111   80:32662/TCP,443:30699/TCP
```

**Critical Service Config:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: mini-xdr-loadbalancer
  namespace: mini-xdr
  annotations:
    service.beta.kubernetes.io/azure-load-balancer-health-probe-request-path: /health
spec:
  type: LoadBalancer
  externalTrafficPolicy: Local
  selector:
    app: mini-xdr-frontend
  ports:
    - name: http
      port: 80
      targetPort: 3000
      nodePort: 32662
    - name: https
      port: 443
      targetPort: 3000
      nodePort: 30699
  sessionAffinity: ClientIP
  healthCheckNodePort: 32600
```

**Endpoints (healthy):**
- 10.0.4.36:3000 (frontend pod 1)
- 10.0.4.100:3000 (frontend pod 2)

---

## ✅ What's Working

1. **All Kubernetes components are healthy:**
   - Pods: 3/3 Running
   - Services: All configured correctly
   - Endpoints: 2 frontend pods registered
   - Internal communication: Perfect

2. **Internal connectivity verified:**
   ```bash
   # From within cluster, NodePort 32662 works perfectly:
   kubectl exec -n mini-xdr deployment/mini-xdr-backend -- curl -s http://10.0.4.4:32662
   # Returns: Full HTML of Mini-XDR frontend ✅
   
   # Health check endpoint works:
   kubectl exec -n mini-xdr deployment/mini-xdr-backend -- curl -s http://10.0.4.4:32600/healthz
   # Returns: {"serviceProxyHealthy": true, "localEndpoints": 2} ✅
   ```

3. **Azure LoadBalancer properly configured:**
   - Public IP: 4.156.121.111 (allocated, static)
   - Frontend IP: ac790b1d443c64d62929b1bca031f087
   - Backend Pool: kubernetes (1 node registered: aks-system-17665817-vmss000000)
   - **Backend Ports MANUALLY FIXED:**
     - HTTP: frontend 80 → backend 32662 ✅
     - HTTPS: frontend 443 → backend 30699 ✅
   - Health Probe: Port 32600, path /healthz, protocol HTTP, status: Passing ✅

4. **Network connectivity:**
   - ICMP works: `ping 4.156.121.111` returns 70ms ✅
   - Public IP is reachable from the internet ✅

5. **NSG rules created for my IP (24.11.0.176/32):**
   - AllowHTTPFromMyIP (priority 100): port 80
   - AllowHTTPSFromMyIP (priority 110): port 443
   - AllowNodePortHTTP (priority 120): port 32662
   - AllowNodePortHTTPS (priority 130): port 30699
   - AllowHealthProbe (priority 140): port 32600 from AzureLoadBalancer

6. **No blocking infrastructure:**
   - No Azure Firewall configured ✅
   - No Application Gateway configured ✅
   - No route tables blocking traffic ✅

---

## ❌ What's NOT Working

**CRITICAL ISSUE:** External HTTP/HTTPS access to LoadBalancer completely fails.

```bash
# All of these timeout after 10 seconds:
curl -I http://4.156.121.111         # Connection timeout ❌
curl -I https://4.156.121.111        # Connection timeout ❌
telnet 4.156.121.111 80              # Connection timeout ❌
```

**Symptom:** TCP SYN packets appear to reach the IP (no immediate rejection), but no SYN-ACK is returned. Connection hangs until timeout.

---

## 🔍 Extensive Debugging Already Completed

### Azure Cloud Controller Manager Bug
**Known Issue:** Azure CCM incorrectly sets LoadBalancer backend ports to service port (80) instead of NodePort (32662).

**Fix Applied:**
```bash
az network lb rule update \
  --resource-group MC_mini-xdr-prod-rg_mini-xdr-aks_eastus \
  --lb-name kubernetes \
  --name ac790b1d443c64d62929b1bca031f087-TCP-80 \
  --backend-port 32662

az network lb rule update \
  --resource-group MC_mini-xdr-prod-rg_mini-xdr-aks_eastus \
  --lb-name kubernetes \
  --name ac790b1d443c64d62929b1bca031f087-TCP-443 \
  --backend-port 30699
```

Verification shows backend ports are now correct, but connectivity still fails.

### NSG Testing
**Test 1:** Created specific allow rules for my IP (24.11.0.176/32) on all required ports.
- Result: Still timeout ❌

**Test 2:** Created wide-open rule (priority 101, allow all sources, all ports, all protocols).
- Result: Still timeout ❌

**Conclusion:** NSG is NOT the blocker. Even with completely open NSG, traffic doesn't reach pods.

### Service Configuration Testing
**Test 1:** Removed `loadBalancerSourceRanges` restriction.
```bash
kubectl patch svc mini-xdr-loadbalancer -n mini-xdr --type='json' \
  -p='[{"op": "remove", "path": "/spec/loadBalancerSourceRanges"}]'
```
- Result: Still timeout ❌

**Test 2:** Verified `externalTrafficPolicy: Local` is set (for source IP preservation).
- Result: Configured correctly ✅, but still timeout ❌

### iptables and kube-proxy Verification
```bash
# Verified iptables rules exist for the LoadBalancer:
kubectl exec -n kube-system daemonset/kube-proxy -- iptables -t nat -L KUBE-SERVICES | grep mini-xdr
# Shows: KUBE-EXT-2H57UZIDCDIQRGXY rules for both HTTP and HTTPS ✅
```

---

## 🚨 Root Cause Assessment

After exhaustive testing, **this appears to be an Azure platform-level networking issue**, NOT a configuration problem.

### Evidence:
1. ✅ All Kubernetes configs correct (verified multiple times)
2. ✅ LoadBalancer backend ports manually fixed
3. ✅ NSG rules created (and tested wide-open)
4. ✅ Health probes passing
5. ✅ Internal connectivity perfect
6. ✅ No Azure Firewall/App Gateway/route tables
7. ❌ **Even with NSG fully open, traffic doesn't reach the pods**
8. ✅ ICMP (ping) works, but TCP doesn't

### Likely Azure Infrastructure Issues:
1. **Azure subscription network policy** blocking inbound LoadBalancer traffic
2. **Azure East US region networking issue** (regional problem)
3. **Public IP DDoS protection** being overly aggressive (inherited from VNet)
4. **Azure Cloud Controller Manager data plane bug** (control plane shows correct config, but data plane isn't forwarding)
5. **Standard SKU LoadBalancer restriction** we're not aware of

---

## 🎯 What I Need You To Do

**Debug this like a senior software engineer.** Specifically:

### 1. Deep Dive Network Path Analysis
```bash
# Check if there's any Azure resource we missed:
az network watcher list
az network ddos-protection-plan list
az network service-endpoint-policy list

# Check effective NSG rules on the actual node NIC:
az network nic list --resource-group MC_mini-xdr-prod-rg_mini-xdr-aks_eastus
az network nic show-effective-nsg --resource-group [RG] --name [NIC_NAME]

# Check public IP restrictions:
az network public-ip show \
  --resource-group MC_mini-xdr-prod-rg_mini-xdr-aks_eastus \
  --name kubernetes-ac790b1d443c64d62929b1bca031f087 \
  --query "{ip:ipAddress, sku:sku, ddos:ddosSettings, zones:zones}"
```

### 2. Verify LoadBalancer Data Plane
```bash
# Check if backend health is actually reporting healthy:
az network lb show \
  --resource-group MC_mini-xdr-prod-rg_mini-xdr-aks_eastus \
  --name kubernetes \
  --query "probes[*].{name:name, port:port, protocol:protocol, state:provisioningState}"

# Check if there are any LoadBalancer inbound NAT rules interfering:
az network lb inbound-nat-rule list \
  --resource-group MC_mini-xdr-prod-rg_mini-xdr-aks_eastus \
  --lb-name kubernetes
```

### 3. Test Alternative Approaches
**Option A: Try accessing NodePort directly from external IP**
If Azure allows it, try hitting the NodePort directly to see if LoadBalancer is the problem:
```bash
curl -I http://[NODE_EXTERNAL_IP]:32662
```

**Option B: Deploy NGINX Ingress Controller**
Sometimes NGINX Ingress works when plain LoadBalancer doesn't:
```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.11.3/deploy/static/provider/cloud/deploy.yaml

# Wait for external IP:
kubectl get svc -n ingress-nginx ingress-nginx-controller

# Create Ingress resource for Mini-XDR
```

**Option C: Recreate LoadBalancer service from scratch**
Current service may be in a corrupted state:
```bash
kubectl delete svc mini-xdr-loadbalancer -n mini-xdr
# Then recreate with proper annotations
```

### 4. Check Azure Subscription Policies
```bash
# Check for any Azure policies blocking LoadBalancer:
az policy assignment list --query "[?contains(displayName,'network') || contains(displayName,'security')]"

# Check subscription features:
az feature list --namespace Microsoft.Network --output table | grep LoadBalancer
```

### 5. Enable Azure Network Watcher
```bash
# Capture packets to see where they're being dropped:
# (This requires Network Watcher to be enabled in the region)
az network watcher packet-capture create \
  --resource-group MC_mini-xdr-prod-rg_mini-xdr-aks_eastus \
  --vm [VM_NAME] \
  --name mini-xdr-packet-capture \
  --filters "[{\"protocol\":\"TCP\",\"localPort\":\"32662\"}]"
```

---

## 📁 Relevant Files

- `/Users/chasemad/Desktop/mini-xdr/AZURE_LOADBALANCER_STATUS.md` - Current status doc (updated with findings)
- `/Users/chasemad/Desktop/mini-xdr/LOADBALANCER_DEBUG_SUMMARY.md` - Technical debugging summary

---

## 🎯 Success Criteria

**Goal:** Make http://4.156.121.111 accessible from my IP (24.11.0.176).

**Test:**
```bash
curl -I http://4.156.121.111
# Should return: HTTP/1.1 200 OK or similar response ✅
```

**Current Result:**
```bash
curl -I http://4.156.121.111
# Returns: Connection timeout after 10 seconds ❌
```

---

## 💡 Recommended Debugging Order

1. **First:** Check for any Azure subscription-level network policies or restrictions
2. **Second:** Deploy NGINX Ingress Controller as alternative (this often works when LoadBalancer doesn't)
3. **Third:** Use Network Watcher to packet capture and see exactly where traffic is dropped
4. **Fourth:** If all else fails, contact Azure support with all the evidence we've gathered

---

## ⚙️ Quick Access Commands

**View LoadBalancer:**
```bash
kubectl get svc mini-xdr-loadbalancer -n mini-xdr
```

**View NSG Rules:**
```bash
az network nsg rule list \
  --resource-group MC_mini-xdr-prod-rg_mini-xdr-aks_eastus \
  --nsg-name aks-agentpool-10857568-nsg \
  --output table
```

**View LoadBalancer Rules:**
```bash
az network lb rule list \
  --resource-group MC_mini-xdr-prod-rg_mini-xdr-aks_eastus \
  --lb-name kubernetes \
  --output table
```

**Test Internal Connectivity:**
```bash
kubectl exec -n mini-xdr deployment/mini-xdr-backend -- \
  curl -s http://10.0.4.4:32662 | head -20
```

**Port Forward for Immediate Access:**
```bash
kubectl port-forward -n mini-xdr svc/mini-xdr-frontend-service 3000:3000
# Then access http://localhost:3000
```

---

## 🔧 Azure CLI Context

**Logged in:** Yes  
**Subscription:** e5636423-8514-4bdd-bfef-f7ecdb934260  
**kubectl context:** Connected to mini-xdr-aks  

---

## 📝 Additional Context

- This is a production demo system for job interviews/LinkedIn
- Application is a Security Operations Center (SOC) platform with ML-based threat detection
- Frontend: React/Next.js on port 3000
- Backend: FastAPI on port 8000
- System has been running for ~6 hours
- All components working perfectly except external LoadBalancer access

---

**START YOUR ANALYSIS HERE. Think systematically, check everything, and find the root cause.**

