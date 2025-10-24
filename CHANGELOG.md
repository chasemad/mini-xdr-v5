# Changelog

All notable changes to Mini-XDR will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2025-10-24

### Fixed
- **Backend security middleware now allows JWT authentication for `/api/onboarding` endpoints**
  - Updated SIMPLE_AUTH_PREFIXES to include /api/onboarding
  - Onboarding wizard endpoints now work with Bearer tokens
  - Fixes 401 errors on onboarding status/start/profile endpoints
- Frontend build issues resolved via GitHub Actions AMD64 builds
- Proper environment variable injection during Docker builds

### Added
- **Complete onboarding wizard (4 steps: Profile, Network Scan, Agents, Validation)**
- Network discovery service for asset scanning
- Agent enrollment service with token generation
- Enterprise-grade CI/CD pipeline with GitHub Actions
  - PR validation with linting and security scans
  - Staging deployment automation
  - Production blue/green deployment with auto-rollback
  - Weekly security audits
- Multi-environment infrastructure (staging + production)
- Kubernetes Kustomize-based deployments
- AWS Secrets Manager integration (code ready, deployment pending)
- Automated security scanning in CI/CD (Trivy, TruffleHog, CodeQL)
- Operational runbooks for deployment, rollback, and incident response
- **SECURITY.md** - Comprehensive security policy and vulnerability reporting
- **SECURITY_AUDIT_REPORT.md** - Complete security assessment documentation
- **VERIFICATION_COMPLETE.md** - Production readiness verification
- OIDC federation documentation for GitHub Actions

### Changed
- Migrated from ConfigMap workaround to immutable Docker images
- Reorganized Kubernetes manifests into base and overlays structure
- Updated Dockerfiles with multi-stage builds for smaller images
- Improved health checks and resource limits (90s startup for ML models)
- Enhanced documentation structure with operational guides

### Security
- **CRITICAL FIX:** Removed hardcoded fallback secret in auth.py
- Enforced JWT_SECRET_KEY requirement at application startup
- Completed comprehensive security audit (Score: 75/100)
- Verified authentication working in production
- AWS Secrets Manager infrastructure configured
- Container images built with security scanning (Trivy)
- Added Trivy container scanning in CI/CD
- Enabled pod security standards
- Regular automated security audits scheduled
- Added security policy documentation (SECURITY.md)
- Added vulnerability reporting process

### Infrastructure
- Deployed to AWS EKS cluster (mini-xdr-cluster, us-east-1)
- RDS PostgreSQL with all migrations applied (5093d5f3c7d4)
- Redis cluster for session management
- Application Load Balancer with health checks and IP whitelisting
- ECR repositories with multi-tag strategy (version, minor, major, latest)

## [1.0.0-auth-fix] - 2025-10-24

### Fixed
- **CRITICAL**: Resolved bcrypt/passlib compatibility issue causing authentication failures
  - Replaced passlib wrapper with direct bcrypt implementation
  - Fixed "password cannot be longer than 72 bytes" error
  - Compatible with bcrypt 5.0.0 while maintaining bcrypt 4.1.2 pin
  
### Added
- Direct bcrypt password hashing and verification (auth.py)
- Explicit bcrypt==4.1.2 dependency pin in requirements.txt
- Frontend authentication redirect to login page
- Onboarding status banner in dashboard
- ConfigMap-based hot-fix deployment (temporary solution)
- Enhanced frontend API URL configuration via build args

### Changed
- Updated frontend/app/page.tsx with improved auth flow
  - Added auth loading state handling
  - Non-blocking onboarding banner (allows dashboard access)
  - Better redirect logic for unauthenticated users
- Updated frontend/Dockerfile with API URL build arguments
- Created backend-deployment-patched.yaml for ConfigMap mounting

### Database
- Reset admin password: chasemadrian@protonmail.com
- Created demo account: demo@minicorp.com
- Reset organization onboarding status to 'not_started'
- Cleared all mock/test data for clean production state

### Infrastructure
- Deployed auth.py fix via Kubernetes ConfigMap (workaround)
- Updated backend deployment with ConfigMap volume mount
- Backend running on EKS with fixed authentication
- Frontend code ready but awaiting proper Docker deployment

### Security
- Maintained bcrypt 12 rounds (no security reduction)
- JWT tokens working (8-hour access, 30-day refresh)
- Account lockout: 5 failed attempts → 15 min lock
- Multi-tenant isolation preserved

### Documentation
- Created COMPLETE_SUMMARY.md - comprehensive accounting of all changes
- Created AUTHENTICATION_SUCCESS.md - technical success report
- Created QUICK_START.md - quick reference for accounts/access
- Created TEST_AND_DEPLOY_GUIDE.md - deployment procedures
- Updated AWS_DEPLOYMENT_GUIDE.md

## [1.0.0] - 2025-10-20

### Added
- **Complete Enterprise Onboarding System**
  - 4-step wizard: Profile → Network Scan → Agents → Validation
  - Network discovery service with NMAP integration
  - Agent enrollment with token-based authentication
  - Real-time asset tracking and agent heartbeat monitoring
  - 10 new API endpoints for onboarding workflow

- **Multi-Tenant Architecture**
  - Organization-based tenant isolation
  - Tenant middleware for all API requests
  - Tenant-aware service layer
  - Database migration for multi-tenancy

- **New AI Agents**
  - DLP Agent: Data loss prevention with pattern detection
  - EDR Agent: Endpoint detection and response
  - IAM Agent: Identity and access management

- **Frontend Components**
  - DashboardLayout with role-based navigation
  - SeverityBadge, StatusChip, ActionButton UI components
  - Agent enrollment tracking interface
  - Network discovery visualization

### Infrastructure
- AWS EKS cluster deployment in us-east-1
- Multi-AZ RDS PostgreSQL (mini-xdr-postgres)
- Redis cluster for session management
- Application Load Balancer with health checks
- ECR repositories for backend/frontend images

### Database
- Migration 8976084bce10: Multi-tenant support
- Migration 5093d5f3c7d4: Onboarding state and assets
- Migration 04c95f3f8bee: Action log table
- Models: DiscoveredAsset, AgentEnrollment, ActionLog

## [0.9.0] - 2025-10-15

### Added
- AI-powered multi-agent orchestration system
- 6 specialized security agents (Triage, Containment, Attribution, Threat Intel, Deception, MCP)
- ML ensemble detection with 83+ features
- Real-time incident response automation
- 3D threat globe visualization
- Advanced threat intelligence integration
- Policy-driven response engine
- Natural language agent interface
- Explainable AI with SHAP/LIME

### Infrastructure
- FastAPI backend with async support
- Next.js 14 frontend with React 19
- PostgreSQL with async SQLAlchemy
- Redis for caching and sessions
- Kubernetes manifests for deployment

### ML/AI
- PyTorch deep learning models
- TensorFlow federated learning
- XGBoost ensemble models
- Feature engineering pipeline
- Model serving infrastructure

## [0.1.0] - 2025-09-01

### Added
- Initial project structure
- Basic threat detection
- Simple web dashboard
- SQLite database
- Local development setup

---

## Version Numbering Guide

- **MAJOR** (X.0.0): Breaking changes, major architecture updates
- **MINOR** (0.X.0): New features, backwards-compatible
- **PATCH** (0.0.X): Bug fixes, security patches

## Links

- [GitHub Repository](https://github.com/chasemad/mini-xdr)
- [Documentation](https://github.com/chasemad/mini-xdr/tree/main/docs)
- [Issue Tracker](https://github.com/chasemad/mini-xdr/issues)

