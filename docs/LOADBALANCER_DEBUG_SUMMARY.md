# LoadBalancer Debugging Summary

## Date: October 9, 2025

## Issue
External LoadBalancer (4.156.121.111) not accessible despite all internal components being healthy.

## Root Cause
After exhaustive debugging, the issue is at the Azure platform networking layer, NOT with:
- Kubernetes configuration ✅
- Pod health ✅  
- Service configuration ✅
- NSG rules ✅ (tested wide-open)
- LoadBalancer backend ports ✅ (manually fixed from 80→32662, 443→30699)
- Health probes ✅ (passing)
- Backend pool ✅ (node registered)

## What We Tested
1. ✅ Created NSG rules for ports 80, 443, 32662, 30699, 32600
2. ✅ Tested with NSG completely open (priority 101, allow all)
3. ✅ Removed loadBalancerSourceRanges restrictions
4. ✅ Fixed LoadBalancer backend ports (was 80, now 32662)
5. ✅ Verified externalTrafficPolicy: Local
6. ✅ Confirmed internal connectivity (pods accessible on NodePorts)
7. ✅ Checked for Azure Firewall (none found)
8. ✅ Checked for Application Gateway (none found)
9. ✅ Verified Public IP is allocated and pingable
10. ✅ Confirmed no route tables blocking traffic

## Symptoms
- PING works (ICMP passes through)
- HTTP/HTTPS connections timeout after 10 seconds
- Telnet to port 80 times out
- curl times out
- Connection attempt reaches the IP but gets no response

## Likely Platform Issues
1. Azure subscription network policy blocking inbound LoadBalancer traffic
2. Azure East US region networking problem
3. Public IP DDoS protection overly aggressive
4. Azure Cloud Controller Manager data plane configuration bug

## Recommended Fix
Deploy NGINX Ingress Controller instead of using LoadBalancer service type directly.
Ingress Controllers are more mature and reliable in AKS.

## Configuration State
### NSG Rules Created
- AllowHTTPFromMyIP (priority 100): 80 from 24.11.0.176/32
- TempAllowAll (priority 101): * from * (for testing)
- AllowHTTPSFromMyIP (priority 110): 443 from 24.11.0.176/32
- AllowNodePortHTTP (priority 120): 32662 from 24.11.0.176/32
- AllowNodePortHTTPS (priority 130): 30699 from 24.11.0.176/32
- AllowHealthProbe (priority 140): 32600 from AzureLoadBalancer

### LoadBalancer Rules
- Frontend: 4.156.121.111:80 → Backend: 32662 ✅
- Frontend: 4.156.121.111:443 → Backend: 30699 ✅
- Health Probe: 32600/healthz (passing) ✅

### Kubernetes Service
```yaml
apiVersion: v1
kind: Service
metadata:
  name: mini-xdr-loadbalancer
  namespace: mini-xdr
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
```

## Next Steps
1. Deploy NGINX Ingress Controller
2. Create Ingress resource for Mini-XDR
3. Update DNS/documentation with new external IP
4. Remove temporary NSG rules
5. Clean up LoadBalancer service if Ingress works

