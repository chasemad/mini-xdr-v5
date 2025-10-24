# Authentication & Access Control

## Authentication Paths

1. **JWT (Users/Admins)**
   - `/api/auth/login`, `/api/auth/register`, `/api/auth/me`, `/api/auth/invite`, `/api/auth/logout`
     implemented in `backend/app/auth.py`.
   - Tokens signed with `JWT_SECRET_KEY`; access tokens expire after 8 hours, refresh tokens after 30 days.
   - Password policy enforced via `validate_password_strength` (>=12 chars, complexity requirements).
   - Multi-tenant support with organization-based user isolation.

2. **API Key Authentication**
   - Routes listed in `SIMPLE_AUTH_PREFIXES` (`backend/app/security.py`) accept the `x-api-key` header.
   - Used for UI-to-API communication, automation endpoints, and service-to-service authentication.
   - Configurable per environment with rotation capabilities.

3. **HMAC Agent Credentials**
   - `AuthMiddleware` enforces signed requests for ingest and high-trust routes via `backend/app/security.py`.
   - 12 specialized agents each with unique HMAC credential sets (device_id, public_id, hmac_key, secret).
   - Agent enrollment and verification managed through `backend/app/agent_enrollment_service.py` and
     `backend/app/agent_verification_service.py`.
   - Credentials stored securely in `agent_credentials` table with tenant isolation.

4. **Onboarding Authentication**
   - Multi-step onboarding process with temporary credentials and progressive access granting.
   - Agent token generation and validation via `/api/onboarding/generate-agent-token` and
     `/api/onboarding/verify-agent-access`.

## Authorization

- **Multi-tenant Isolation**: Organization scoping enforced through `get_current_user` and tenant middleware
  (`backend/app/tenant_middleware.py`). All database queries include `organization_id` filters to preserve
  tenant isolation and prevent data leakage.
- **Policy Engine**: Comprehensive policy engine (`backend/app/policy_engine.py`) evaluates all response
  actions, workflow executions, and agent operations against organizational policies.
- **Workflow Approvals**: Response workflows support approval routing with role-based access control
  (`/api/response/workflows/{id}/approve`, `/api/response/workflows/{id}/reject`).
- **Agent Permissions**: Each agent has defined permission scopes and operation limits enforced through
  the orchestration layer.
- **Audit Logging**: All authorization decisions and access attempts are logged for compliance and
  security monitoring.

## Session Management

- Access tokens expire after 8 hours (configurable via `ACCESS_TOKEN_EXPIRE_MINUTES` in
  `backend/app/auth.py`).
- Refresh tokens expire after 30 days (configurable via `REFRESH_TOKEN_EXPIRE_DAYS`).
- Refresh tokens can be exchanged for new access tokens via `/api/auth/refresh`.
- Frontend stores tokens in `localStorage`; clearing storage signs the user out.
- Logout endpoint (`/api/auth/logout`) invalidates refresh tokens server-side (see `logout` handler
  in `backend/app/auth.py`).

## Best Practices

- Keep `JWT_SECRET_KEY` random and rotate on a regular cadence; invalidate sessions when rotating.
- Enforce HTTPS when transmitting tokens. When deploying behind reverse proxies, set
  `USE_FORWARDED_HEADERS` if required.
- For automation clients, prefer HMAC credentials over API keys and scope actions through policies.
