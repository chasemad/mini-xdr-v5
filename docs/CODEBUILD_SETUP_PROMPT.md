# AWS CodeBuild Setup for Mini-XDR - Complete Context

## 🎯 OBJECTIVE
Set up AWS CodeBuild to build and push a 5.25GB Docker image to Amazon ECR from source code in my local repository. The local push approach has failed repeatedly due to network/size constraints.

---

## 📦 PROJECT DETAILS

**Project:** Mini-XDR Security Platform  
**Repository:** `/Users/chasemad/Desktop/mini-xdr/backend/`  
**Docker Image Size:** 5.25 GB (contains TensorFlow, PyTorch, CUDA libraries)  
**Target:** AWS ECR in us-east-1  
**ECR Repository:** `116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend`  
**Desired Tag:** `onboarding-final` or `latest`  
**Platform:** linux/amd64 (for AWS EKS)

---

## 🔴 PROBLEMS WE'VE ENCOUNTERED

### Attempt #1: Direct `docker push`
```bash
docker push 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:onboarding-v1.0
```
**Problem:** 
- Push kept getting stuck after showing "Layer already exists" for multiple layers
- Created wrong multi-architecture manifest (arm64 instead of amd64)
- EKS couldn't pull: `no match for platform in manifest: not found`
- Push ran for 20+ minutes with no completion

### Attempt #2: Docker `buildx` with `--push`
```bash
docker buildx build --platform linux/amd64 \
  --tag 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:onboarding-final \
  --push .
```
**Problem:**
- Build completed successfully (all layers cached)
- Stuck on "exporting to image" / "pushing layers" for 10+ minutes
- No progress updates, no errors, just hangs
- Process still running but not completing

### Attempt #3: EC2 Instance for Building
**Approach:** Launch EC2, transfer code, build there (closer to ECR)
```bash
# Launched t3.large with Amazon Linux 2
# Installed Docker
# Tried to transfer code via tar over SSH
```
**Problem:**
- EC2 instance came with only 8GB root disk
- Disk filled to 100% during code transfer
- `No space left on device` errors
- Terminated instance

### Attempt #4: S3 Transfer then EC2 Build
**Approach:** Save Docker image to tar, upload to S3, download on EC2
```bash
docker save ... | gzip > /tmp/mini-xdr-backend-onboarding.tar.gz  # 4.9GB
aws s3 cp /tmp/mini-xdr-backend-onboarding.tar.gz s3://bucket/
```
**Problem:**
- S3 upload started fine (reached 160MB at 6-16 MiB/s)
- Upload timed out: `Read timeout on endpoint URL`
- Network too slow/unstable for 4.9GB upload

### Attempt #5: Multiple Docker Push Retries
**Problem:**
- Multiple push processes got stuck simultaneously
- Had to kill stuck processes repeatedly
- Seems to be network/bandwidth issue from local machine to ECR

---

## 💾 LOCAL ENVIRONMENT CONSTRAINTS

**Docker Status:**
```
TYPE            TOTAL     ACTIVE    SIZE      RECLAIMABLE
Images          47        13        76.25GB   66GB (86%)
Containers      15        1         421.9kB   380.9kB (90%)
Local Volumes   33        10        53.63GB   1.652GB (3%)
Build Cache     221       0         16.25GB   16.25GB
```

**Docker VM:** 155GB total, heavily utilized  
**Mac Disk Space:** 365GB free (plenty of space)  
**Network:** Unreliable for large uploads (S3 and ECR timeouts)

---

## 📂 WHAT NEEDS TO BE BUILT

### Dockerfile Location
`/Users/chasemad/Desktop/mini-xdr/backend/Dockerfile`

### Key Files
```
backend/
├── Dockerfile (multi-stage build)
├── requirements.txt (150+ Python packages)
├── app/
│   ├── main.py
│   ├── onboarding_routes.py (NEW - needs to be deployed)
│   ├── discovery_service.py (NEW)
│   ├── agent_enrollment_service.py (NEW)
│   └── models.py (updated)
├── migrations/
│   └── versions/5093d5f3c7d4_add_onboarding_tables.py (NEW)
└── alembic.ini
```

### Build Requirements
- **Platform:** linux/amd64
- **Python:** 3.12-slim base image
- **Large Packages:** TensorFlow (~620MB), PyTorch (~887MB), CUDA libraries (2+ GB)
- **Multi-stage build:** Yes (builder stage + final stage)
- **Build time:** ~2.5 hours locally (due to ML packages)

---

## 🏗️ EXISTING AWS INFRASTRUCTURE

**AWS Account ID:** 116912495274  
**Region:** us-east-1

### EKS Cluster
- **Name:** mini-xdr-cluster
- **Namespace:** mini-xdr
- **Current backend deployment:** Running old version without onboarding features
- **Pods:** 2 backend, 3 frontend (all healthy)

### ECR Repositories
```bash
# Backend repo exists
116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend

# Current images in ECR:
- amd64 (old working version)
- onboarding-v1.0 (BROKEN - wrong manifest)
- latest (old version)
```

### RDS Database
- **Endpoint:** mini-xdr-postgres.ccnkck4wij3r.us-east-1.rds.amazonaws.com
- **Migration ready:** Yes, needs `alembic upgrade head` after deployment

### Other Resources
- VPC, subnets, security groups all configured
- ALB with health checks
- Redis cluster

---

## 🎯 WHAT I WANT TO ACHIEVE WITH AWS CODEBUILD

### Primary Goal
**Build the Docker image IN AWS and push to ECR** so I don't have to transfer 5GB over my local network.

### Specific Requirements

1. **Source Code Upload**
   - Upload just the `backend/` source code (~500MB without venv)
   - Either via S3, CodeCommit, or GitHub (I can choose)
   - Avoid transferring the full 5.25GB built image

2. **Build Process**
   - Build the Docker image using the Dockerfile in AWS
   - Platform: linux/amd64
   - Tag: `116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:onboarding-final`
   - Push to ECR with CORRECT manifest

3. **CodeBuild Configuration**
   - Compute type: Need enough resources for 5.25GB image build
   - Build environment: Docker/linux
   - Build time estimate: 2.5 hours (acceptable)
   - Cost: ~$15 one-time (acceptable)

4. **After Build Completes**
   - Verify image in ECR
   - Deploy to EKS: `kubectl set image deployment/mini-xdr-backend backend=...`
   - Apply database migration
   - Test onboarding endpoints

---

## 📋 WHAT I NEED HELP WITH

### Step-by-Step Guide For:

1. **Uploading Source Code**
   - What's the best method to get my local `backend/` folder into AWS?
   - S3? GitHub? CodeCommit? Direct upload?
   - How to exclude venv, node_modules, __pycache__, etc.

2. **Creating buildspec.yml**
   - Complete buildspec.yml for multi-stage Docker build
   - Proper ECR login steps
   - Correct docker buildx commands with --platform linux/amd64
   - Environment variables needed

3. **Setting Up CodeBuild Project**
   - Exact AWS Console steps or CLI commands
   - Compute type selection (need recommendation)
   - IAM role configuration (ECR push permissions)
   - Timeout settings for 2.5 hour build

4. **Monitoring and Troubleshooting**
   - How to watch build progress
   - Where to find build logs
   - Common errors and fixes

5. **Deployment After Build**
   - Commands to deploy new image to EKS
   - How to verify correct platform/manifest
   - Rollback strategy if it fails

---

## 🔑 IAM RESOURCES I HAVE

- AWS CLI configured with admin credentials
- kubectl configured for EKS cluster
- Existing IAM roles:
  - EKS node role
  - EKS service account roles
  - (May need CodeBuild service role)

---

## ✅ SUCCESS CRITERIA

1. ✅ Source code uploaded to AWS
2. ✅ CodeBuild project created and configured
3. ✅ Build completes successfully (~2.5 hours)
4. ✅ Image pushed to ECR with correct amd64 manifest
5. ✅ EKS can pull and run the image
6. ✅ Onboarding endpoints accessible: `http://ALB_URL/api/onboarding/status`

---

## 📝 ADDITIONAL CONTEXT

### Why This is Urgent
- Onboarding system is complete and tested locally
- Customer demo scheduled soon
- Local push has failed 5+ times over 3+ hours
- Need reliable build/deploy pipeline

### Budget
- Willing to spend $15-20 for one-time CodeBuild
- Want reusable pipeline for future updates

### Technical Skills
- Comfortable with AWS CLI, kubectl, Docker
- Can follow step-by-step instructions
- Prefer CLI commands over Console UI (but can do either)

---

## 🚀 REQUESTED OUTPUT

Please provide:

1. **Complete buildspec.yml** file ready to use
2. **Step-by-step CLI commands** to:
   - Prepare and upload source code
   - Create S3 bucket (if needed)
   - Create IAM role for CodeBuild
   - Create CodeBuild project
   - Start the build
   - Monitor progress
3. **Deployment commands** for after build completes
4. **Troubleshooting guide** for common issues
5. **Cost estimate** for this specific build

---

## 📞 IMMEDIATE QUESTIONS

1. What's the fastest way to get my 500MB backend folder into AWS?
2. What CodeBuild compute type do I need for a 5.25GB Docker image?
3. Can I use Docker layer caching in CodeBuild to speed up future builds?
4. How do I ensure the manifest is correct (linux/amd64 only)?
5. What's the typical build time for ML-heavy images in CodeBuild?

---

**Thank you! This is blocking our production deployment and I've spent 3+ hours troubleshooting local push issues. Looking forward to getting this working!**

