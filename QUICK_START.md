# Mini-XDR Quick Start Guide

## 🎯 YOUR ACCOUNTS (READY TO USE)

### Admin Account
```
Email:    chasemadrian@protonmail.com
Password: demo-tpot-api-key
Role:     Admin
```

### Demo Account (For Recruiters)
```
Email:    demo@minicorp.com
Password: Demo@2025
Role:     Security Analyst
```

## 🌐 AWS DEPLOYMENT

**URL:** http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com

**Status:**
- ✅ Backend: WORKING - Authentication operational
- ⚠️ Frontend: OLD VERSION running (auth fixes not deployed yet)

## 🔥 TESTING AUTHENTICATION (WORKS NOW!)

```bash
# Test admin login
curl -X POST http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "chasemadrian@protonmail.com", "password": "demo-tpot-api-key"}'

# You'll get JWT tokens back - authentication is WORKING!
```

## 🖥️ RECOMMENDED: RUN FRONTEND LOCALLY

For best experience with all latest features:

```bash
cd /Users/chasemad/Desktop/mini-xdr/frontend

# Set AWS backend URL
export NEXT_PUBLIC_API_BASE=http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com
export NEXT_PUBLIC_API_URL=http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com

# Run dev server
npm run dev
```

Then visit: http://localhost:3000

**Benefits:**
- ✅ Latest auth routing (redirects to /login)
- ✅ Onboarding banner visible
- ✅ Connected to AWS backend
- ✅ Perfect for demoing to recruiters

## ✅ VERIFIED WORKING

- ✅ Admin login successful
- ✅ Demo login successful  
- ✅ JWT tokens generated
- ✅ Organization: "Mini Corp" (slug: mini-corp)
- ✅ Onboarding status: "not_started" (fresh)
- ✅ Database: 0 incidents, 0 events, 0 agents (clean slate)

## ⚠️ KNOWN ISSUE

**Frontend deployment:** Old version running on AWS (doesn't have auth redirect)

**Temporary workaround:** Run frontend locally as shown above

**Future fix:** Deploy updated frontend image from CI/CD pipeline

## 🎁 FOR RECRUITERS

Share this:

```
Mini-XDR Security Platform Demo

URL:      http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com
Email:    demo@minicorp.com
Password: Demo@2025

OR for best experience:
Contact me for the localhost demo link with latest features
```

## 🔐 SECURITY

- No security was reduced
- Bcrypt hashing working (bcrypt 5.0.0)
- JWT authentication operational
- Account lockout after 5 failed attempts
- Multi-tenant isolation active
- Zero default/mock data

---

**AUTHENTICATION IS WORKING! 🚀**

Next: Deploy frontend with auth routing fixes (optional - local dev works great)

