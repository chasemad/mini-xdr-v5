# 07: Security Hardening

**Current State:** Basic HMAC auth, TLS, some input validation  
**Target State:** Penetration tested, vulnerability managed, security monitoring  
**Priority:** P0 (Required for any customer)  
**Solo Effort:** 4-5 weeks + external pen test

---

## Critical Security Gaps

- ❌ No penetration testing done
- ❌ No vulnerability scanning in CI/CD
- ❌ No secrets rotation policy
- ❌ No WAF (Web Application Firewall)
- ❌ No DDoS protection
- ❌ No security incident response plan
- ❌ No bug bounty program

---

## Implementation Checklist

### Task 1: Input Validation & Sanitization

**File:** `/backend/app/validation.py` (NEW)

```python
"""Input validation and sanitization"""
import re
from typing import Any
import bleach
from pydantic import validator

# IP address validation
def is_valid_ip(ip: str) -> bool:
    pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    if not re.match(pattern, ip):
        return False
    octets = [int(x) for x in ip.split('.')]
    return all(0 <= octet <= 255 for octet in octets)

# SQL injection prevention
def sanitize_sql_input(value: str) -> str:
    """Remove SQL injection attempts"""
    dangerous_patterns = [
        r"(\s*(union|select|insert|update|delete|drop|create|alter)\s+)",
        r"(--|\#|\/\*|\*\/)",
        r"(\bor\b.*=.*|and.*=.*)",
    ]
    for pattern in dangerous_patterns:
        if re.search(pattern, value, re.IGNORECASE):
            raise ValueError("Potential SQL injection detected")
    return value

# XSS prevention
def sanitize_html(value: str) -> str:
    """Remove XSS attempts"""
    allowed_tags = ['b', 'i', 'u', 'p', 'br']
    return bleach.clean(value, tags=allowed_tags, strip=True)

# Path traversal prevention
def is_safe_path(path: str) -> bool:
    """Prevent directory traversal"""
    if '..' in path or path.startswith('/'):
        return False
    return True
```

**Checklist:**
- [ ] Add input validation to all endpoints
- [ ] Sanitize user inputs before storage
- [ ] Validate file uploads
- [ ] Test with OWASP ZAP

### Task 2: Secrets Management

**File:** `/backend/app/secrets_manager.py` - Enhance existing

```python
"""Secrets management with rotation"""

class SecretRotationPolicy:
    """Define secret rotation policies"""
    
    def __init__(self):
        self.policies = {
            "api_keys": 90,  # days
            "database_passwords": 60,
            "jwt_secrets": 180,
            "encryption_keys": 365
        }
    
    async def check_rotation_needed(self, secret_name: str, last_rotated: datetime) -> bool:
        """Check if secret needs rotation"""
        secret_type = self.get_secret_type(secret_name)
        max_age_days = self.policies.get(secret_type, 90)
        
        age_days = (datetime.now() - last_rotated).days
        return age_days >= max_age_days
    
    async def rotate_secret(self, secret_name: str):
        """Rotate a secret"""
        # 1. Generate new secret
        new_secret = self.generate_secure_secret()
        
        # 2. Store new secret in AWS Secrets Manager
        await self.store_secret(secret_name, new_secret)
        
        # 3. Update application config (zero-downtime)
        await self.update_config(secret_name, new_secret)
        
        # 4. Wait for propagation
        await asyncio.sleep(60)
        
        # 5. Delete old secret
        await self.delete_old_secret(secret_name)
        
        logger.info(f"Rotated secret: {secret_name}")
```

**Checklist:**
- [ ] Move all secrets to AWS Secrets Manager
- [ ] Implement rotation policies
- [ ] Set up rotation alerts
- [ ] Test rotation without downtime
- [ ] Document emergency rotation procedure

### Task 3: Vulnerability Scanning

**File:** `.github/workflows/security-scan.yml` (NEW)

```yaml
name: Security Scanning

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  dependency-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Run Snyk dependency scan
        uses: snyk/actions/python@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          args: --severity-threshold=high
      
      - name: Run Safety check
        run: |
          pip install safety
          safety check --json
  
  sast-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Run Bandit SAST
        run: |
          pip install bandit
          bandit -r backend/app -f json -o bandit-report.json
      
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: bandit-report
          path: bandit-report.json
  
  container-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build image
        run: docker build -t mini-xdr-backend:test -f ops/Dockerfile.backend .
      
      - name: Run Trivy scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'mini-xdr-backend:test'
          format: 'sarif'
          output: 'trivy-results.sarif'
      
      - name: Upload to GitHub Security
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'
```

**Checklist:**
- [ ] Set up Snyk for dependency scanning
- [ ] Add Bandit for Python SAST
- [ ] Add Trivy for container scanning
- [ ] Configure GitHub Security Alerts
- [ ] Fix all HIGH and CRITICAL findings
- [ ] Set up weekly security review

### Task 4: Web Application Firewall

**File:** `/ops/k8s/waf-ingress.yaml` (NEW)

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mini-xdr-waf
  namespace: mini-xdr
  annotations:
    # AWS WAF integration
    alb.ingress.kubernetes.io/wafv2-acl-arn: "arn:aws:wafv2:us-east-1:ACCOUNT:regional/webacl/mini-xdr-waf/ID"
    
    # Rate limiting
    nginx.ingress.kubernetes.io/limit-rps: "100"
    nginx.ingress.kubernetes.io/limit-connections: "10"
    
    # Security headers
    nginx.ingress.kubernetes.io/configuration-snippet: |
      more_set_headers "X-Frame-Options: DENY";
      more_set_headers "X-Content-Type-Options: nosniff";
      more_set_headers "X-XSS-Protection: 1; mode=block";
      more_set_headers "Strict-Transport-Security: max-age=31536000; includeSubDomains";
spec:
  ingressClassName: nginx
  rules:
  - host: api.mini-xdr.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: mini-xdr-backend-service
            port:
              number: 8000
```

**Checklist:**
- [ ] Deploy WAF (AWS WAF or ModSecurity)
- [ ] Configure rate limiting
- [ ] Block common attack patterns
- [ ] Add geo-blocking if needed
- [ ] Monitor WAF logs

### Task 5: Penetration Testing

**Process:**
1. **Internal Testing (Week 1)**
   - Run OWASP ZAP automated scan
   - Run Burp Suite manual testing
   - Test authentication bypass
   - Test authorization flaws
   - Test injection attacks

2. **External Pen Test (Week 2-3)**
   - Hire pen testing firm ($10K-$15K)
   - Provide scope document
   - Run test in staging environment
   - Receive findings report

3. **Remediation (Week 4)**
   - Fix all CRITICAL findings
   - Fix all HIGH findings
   - Document MEDIUM/LOW for future
   - Retest fixes

4. **Annual Schedule:**
   - Full pen test: Annually
   - Automated scans: Weekly
   - Dependency scans: Daily

**Checklist:**
- [ ] Run OWASP ZAP scan
- [ ] Fix automated findings
- [ ] Hire external pen tester
- [ ] Remediate all findings
- [ ] Get pen test report for compliance
- [ ] Schedule annual pen test

---

## Quick Security Wins (This Week)

**Day 1:**
- [ ] Enable GitHub security alerts
- [ ] Run `pip-audit` on requirements.txt
- [ ] Update all dependencies with known CVEs

**Day 2:**
- [ ] Add rate limiting to all endpoints
- [ ] Implement CORS properly
- [ ] Add security headers

**Day 3:**
- [ ] Set up Snyk scanning
- [ ] Fix HIGH severity issues
- [ ] Document security findings

**Day 4:**
- [ ] Add input validation
- [ ] Test with OWASP ZAP
- [ ] Create security incident response plan

**Day 5:**
- [ ] Move secrets to Secrets Manager
- [ ] Enable 2FA for all admin accounts
- [ ] Review and update access controls

---

**Next:** `08_SUPPORT_OPERATIONS.md`


