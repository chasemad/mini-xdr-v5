# 01: Authentication & Authorization - Production Implementation

**Status:** Critical Gap - Must Complete for Phase 1  
**Current State:** Basic HMAC auth, simple API keys  
**Target State:** Enterprise SSO, multi-tenancy, granular RBAC  
**Priority:** P0 (Blocker for any customer deployment)  

---

## Current State Analysis

### What EXISTS in `/backend/app/security.py`
```python
✅ HMAC Authentication (Lines 1-237)
   - Device ID + HMAC signature validation
   - Nonce-based replay protection
   - Rate limiting (burst + sustained windows)
   - Timestamp validation (±5 min clock skew)
   
✅ Simple API Key Auth
   - Basic X-API-Key header validation
   - Used for frontend-to-backend communication
```

### What EXISTS in `/backend/app/models.py`
```python
✅ AgentCredential (Lines 213-224)
   - device_id, public_id, secret_hash
   - Expiration and revocation support
   - Created/expires timestamps
   
✅ RequestNonce (Lines 226-237)
   - Replay attack prevention
   - Device ID + nonce tracking
```

### What's MISSING
- ❌ SSO/SAML integration (Okta, Azure AD, Google Workspace)
- ❌ OAuth 2.0 / OpenID Connect
- ❌ Multi-tenant data isolation
- ❌ Role-Based Access Control (RBAC)
- ❌ Session management
- ❌ Multi-Factor Authentication (MFA)
- ❌ User management UI
- ❌ Password policies
- ❌ Audit logging for auth events
- ❌ API key rotation policies

---

## Implementation Checklist

### Task 1: Multi-Tenancy Foundation

#### 1.1: Add Organization/Tenant Model
**File:** `/backend/app/models.py`  
**Location:** After `RequestNonce` class (around line 240)

```python
class Organization(Base):
    """Multi-tenant organization model"""
    __tablename__ = "organizations"
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Organization identity
    uuid = Column(String(36), unique=True, index=True, nullable=False)
    name = Column(String(128), nullable=False)
    display_name = Column(String(256))
    domain = Column(String(128), unique=True, index=True)  # company.com
    
    # Subscription & limits
    plan_tier = Column(String(32), default="trial")  # trial|starter|professional|enterprise
    max_users = Column(Integer, default=5)
    max_agents = Column(Integer, default=100)
    max_events_per_day = Column(Integer, default=10000)
    
    # Status
    status = Column(String(32), default="active")  # active|suspended|trial|cancelled
    trial_ends_at = Column(DateTime(timezone=True), nullable=True)
    
    # Configuration
    sso_enabled = Column(Boolean, default=False)
    sso_provider = Column(String(32), nullable=True)  # okta|azure_ad|google
    sso_config = Column(JSON, nullable=True)
    
    # Data residency
    data_region = Column(String(32), default="us-east-1")
    data_classification = Column(String(32), default="standard")  # standard|pii|phi|pci
    
    # Relationships (defined later)
    users = relationship("User", back_populates="organization")
    incidents = relationship("Incident", back_populates="organization")
```

**Checklist:**
- [ ] Add Organization class to models.py
- [ ] Create database migration: `alembic revision --autogenerate -m "add_organizations"`
- [ ] Run migration: `alembic upgrade head`
- [ ] Add organization_id foreign key to ALL existing tables (Event, Incident, Action, etc.)
- [ ] Test migration with sample data
- [ ] Update all queries to filter by organization_id

#### 1.2: Add User Model
**File:** `/backend/app/models.py`  
**Location:** After Organization class

```python
class User(Base):
    """User account with RBAC"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    
    # Organization association
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False, index=True)
    
    # Identity
    uuid = Column(String(36), unique=True, index=True, nullable=False)
    email = Column(String(256), unique=True, index=True, nullable=False)
    email_verified = Column(Boolean, default=False)
    username = Column(String(64), index=True)
    
    # Authentication
    password_hash = Column(String(256), nullable=True)  # Null if SSO-only
    mfa_enabled = Column(Boolean, default=False)
    mfa_secret = Column(String(256), nullable=True)
    
    # SSO
    sso_provider = Column(String(32), nullable=True)
    sso_user_id = Column(String(256), nullable=True, index=True)
    
    # Profile
    first_name = Column(String(128))
    last_name = Column(String(128))
    avatar_url = Column(String(512), nullable=True)
    timezone = Column(String(64), default="UTC")
    
    # Status
    status = Column(String(32), default="active")  # active|suspended|invited|deactivated
    invited_by_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    invitation_token = Column(String(256), nullable=True)
    invitation_expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Security
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime(timezone=True), nullable=True)
    password_changed_at = Column(DateTime(timezone=True), nullable=True)
    must_change_password = Column(Boolean, default=False)
    
    # Relationships
    organization = relationship("Organization", back_populates="users")
    roles = relationship("UserRole", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user")
```

**Checklist:**
- [ ] Add User class to models.py
- [ ] Install password hashing library: `pip install passlib[bcrypt]`
- [ ] Create migration: `alembic revision --autogenerate -m "add_users"`
- [ ] Create initial admin user seed script
- [ ] Add password validation (min 12 chars, complexity requirements)
- [ ] Implement password reset workflow

#### 1.3: Add RBAC Models
**File:** `/backend/app/models.py`

```python
class Role(Base):
    """Predefined system roles"""
    __tablename__ = "roles"
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Role definition
    name = Column(String(64), unique=True, index=True, nullable=False)
    display_name = Column(String(128), nullable=False)
    description = Column(Text, nullable=True)
    
    # Permissions (JSON array of permission strings)
    permissions = Column(JSON, nullable=False)
    
    # Scope
    is_system_role = Column(Boolean, default=True)  # System vs. custom
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=True)
    
    # Relationships
    user_roles = relationship("UserRole", back_populates="role")


class UserRole(Base):
    """Many-to-many user-role assignment"""
    __tablename__ = "user_roles"
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    role_id = Column(Integer, ForeignKey("roles.id"), nullable=False, index=True)
    
    # Assignment metadata
    assigned_by_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="roles", foreign_keys=[user_id])
    role = relationship("Role", back_populates="user_roles")
    assigned_by = relationship("User", foreign_keys=[assigned_by_id])
    
    __table_args__ = (
        Index("ix_user_roles_user_role", "user_id", "role_id", unique=True),
    )
```

**Checklist:**
- [ ] Add Role and UserRole classes
- [ ] Create migration: `alembic revision --autogenerate -m "add_rbac"`
- [ ] Define 5 standard roles (see Task 1.4)
- [ ] Create permission constants file
- [ ] Seed roles in migration or startup script

#### 1.4: Define Standard Permissions
**New File:** `/backend/app/permissions.py`

```python
"""
RBAC Permission Definitions
Format: resource:action
"""

# Incident permissions
INCIDENTS_VIEW = "incidents:view"
INCIDENTS_CREATE = "incidents:create"
INCIDENTS_UPDATE = "incidents:update"
INCIDENTS_DELETE = "incidents:delete"
INCIDENTS_ASSIGN = "incidents:assign"
INCIDENTS_CLOSE = "incidents:close"

# Response permissions
RESPONSE_VIEW = "response:view"
RESPONSE_EXECUTE = "response:execute"
RESPONSE_APPROVE = "response:approve"
RESPONSE_ROLLBACK = "response:rollback"
RESPONSE_CREATE_PLAYBOOK = "response:create_playbook"

# Agent permissions
AGENTS_VIEW = "agents:view"
AGENTS_INTERACT = "agents:interact"
AGENTS_CONFIGURE = "agents:configure"
AGENTS_OVERRIDE = "agents:override"

# Analytics permissions
ANALYTICS_VIEW = "analytics:view"
ANALYTICS_EXPORT = "analytics:export"

# ML permissions
ML_VIEW_MODELS = "ml:view_models"
ML_TRAIN_MODELS = "ml:train_models"
ML_DEPLOY_MODELS = "ml:deploy_models"
ML_DELETE_MODELS = "ml:delete_models"

# Integration permissions
INTEGRATIONS_VIEW = "integrations:view"
INTEGRATIONS_CONFIGURE = "integrations:configure"
INTEGRATIONS_DELETE = "integrations:delete"

# Settings permissions
SETTINGS_VIEW = "settings:view"
SETTINGS_UPDATE = "settings:update"
SETTINGS_USERS = "settings:users"
SETTINGS_BILLING = "settings:billing"

# Audit permissions
AUDIT_VIEW = "audit:view"
AUDIT_EXPORT = "audit:export"

# System admin permissions
SYSTEM_ADMIN = "system:admin"


# Standard role definitions
STANDARD_ROLES = {
    "super_admin": {
        "display_name": "Super Administrator",
        "description": "Full system access including user management",
        "permissions": [SYSTEM_ADMIN]  # Wildcard grants all permissions
    },
    "security_admin": {
        "display_name": "Security Administrator",
        "description": "Manage security policies, integrations, ML models",
        "permissions": [
            INCIDENTS_VIEW, INCIDENTS_CREATE, INCIDENTS_UPDATE, INCIDENTS_ASSIGN, INCIDENTS_CLOSE,
            RESPONSE_VIEW, RESPONSE_EXECUTE, RESPONSE_APPROVE, RESPONSE_CREATE_PLAYBOOK,
            AGENTS_VIEW, AGENTS_INTERACT, AGENTS_CONFIGURE,
            ANALYTICS_VIEW, ANALYTICS_EXPORT,
            ML_VIEW_MODELS, ML_TRAIN_MODELS, ML_DEPLOY_MODELS,
            INTEGRATIONS_VIEW, INTEGRATIONS_CONFIGURE,
            SETTINGS_VIEW, SETTINGS_UPDATE,
            AUDIT_VIEW, AUDIT_EXPORT
        ]
    },
    "soc_analyst": {
        "display_name": "SOC Analyst",
        "description": "Investigate incidents, execute approved responses",
        "permissions": [
            INCIDENTS_VIEW, INCIDENTS_UPDATE, INCIDENTS_ASSIGN,
            RESPONSE_VIEW, RESPONSE_EXECUTE,
            AGENTS_VIEW, AGENTS_INTERACT,
            ANALYTICS_VIEW,
            ML_VIEW_MODELS,
            INTEGRATIONS_VIEW,
            AUDIT_VIEW
        ]
    },
    "read_only_analyst": {
        "display_name": "Read-Only Analyst",
        "description": "View-only access for auditing and reporting",
        "permissions": [
            INCIDENTS_VIEW,
            RESPONSE_VIEW,
            AGENTS_VIEW,
            ANALYTICS_VIEW,
            ML_VIEW_MODELS,
            INTEGRATIONS_VIEW,
            AUDIT_VIEW
        ]
    },
    "incident_responder": {
        "display_name": "Incident Responder",
        "description": "Execute responses and manage active incidents",
        "permissions": [
            INCIDENTS_VIEW, INCIDENTS_UPDATE, INCIDENTS_CLOSE,
            RESPONSE_VIEW, RESPONSE_EXECUTE, RESPONSE_ROLLBACK,
            AGENTS_VIEW, AGENTS_INTERACT,
            ANALYTICS_VIEW,
            AUDIT_VIEW
        ]
    }
}
```

**Checklist:**
- [ ] Create permissions.py file
- [ ] Define all permission constants
- [ ] Create STANDARD_ROLES dictionary
- [ ] Add permission check decorator
- [ ] Seed roles in database

#### 1.5: Implement Permission Checking
**New File:** `/backend/app/rbac.py`

```python
"""RBAC enforcement utilities"""
from functools import wraps
from fastapi import HTTPException, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List

from .db import get_db
from .models import User, UserRole, Role
from .permissions import SYSTEM_ADMIN


async def get_current_user(request: Request, db: AsyncSession = Depends(get_db)) -> User:
    """Extract and validate current user from session/token"""
    # TODO: Extract from JWT token or session cookie
    user_id = request.state.user_id  # Set by auth middleware
    
    result = await db.execute(
        select(User)
        .where(User.id == user_id)
        .where(User.status == "active")
    )
    user = result.scalars().first()
    
    if not user:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    
    return user


async def get_user_permissions(user: User, db: AsyncSession) -> List[str]:
    """Get all permissions for a user"""
    result = await db.execute(
        select(Role)
        .join(UserRole, UserRole.role_id == Role.id)
        .where(UserRole.user_id == user.id)
    )
    roles = result.scalars().all()
    
    permissions = set()
    for role in roles:
        if SYSTEM_ADMIN in role.permissions:
            # System admin has all permissions
            return [SYSTEM_ADMIN]
        permissions.update(role.permissions)
    
    return list(permissions)


def require_permission(*required_permissions: str):
    """Decorator to require specific permissions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user from kwargs (injected by Depends)
            current_user = kwargs.get('current_user')
            db = kwargs.get('db')
            
            if not current_user or not db:
                raise HTTPException(status_code=500, detail="RBAC dependency injection failed")
            
            user_perms = await get_user_permissions(current_user, db)
            
            # Check if user has system admin (grants all)
            if SYSTEM_ADMIN in user_perms:
                return await func(*args, **kwargs)
            
            # Check if user has any of the required permissions
            if not any(perm in user_perms for perm in required_permissions):
                raise HTTPException(
                    status_code=403,
                    detail=f"Missing required permission: {required_permissions}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator
```

**Checklist:**
- [ ] Create rbac.py file
- [ ] Implement get_current_user function
- [ ] Implement get_user_permissions function
- [ ] Implement require_permission decorator
- [ ] Add permission checks to all sensitive endpoints in main.py
- [ ] Test permission denial returns 403

---

### Task 2: Session Management & JWT Tokens

#### 2.1: Add Session Model
**File:** `/backend/app/models.py`

```python
class UserSession(Base):
    """User session tracking"""
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    last_activity_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Session identification
    session_id = Column(String(64), unique=True, index=True, nullable=False)
    access_token_jti = Column(String(64), unique=True, index=True)  # JWT ID
    refresh_token_jti = Column(String(64), unique=True, index=True, nullable=True)
    
    # Session metadata
    ip_address = Column(String(64))
    user_agent = Column(String(512))
    device_fingerprint = Column(String(128), nullable=True)
    
    # Expiration
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    refresh_expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True, index=True)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    revoke_reason = Column(String(256), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="sessions")
```

**Checklist:**
- [ ] Add UserSession model
- [ ] Install PyJWT: `pip install PyJWT[crypto]`
- [ ] Create migration
- [ ] Implement session cleanup job (delete expired sessions)

#### 2.2: Implement JWT Authentication
**New File:** `/backend/app/auth.py`

```python
"""JWT authentication and session management"""
import jwt
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from fastapi import HTTPException, Cookie, Header
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from .config import settings
from .models import User, UserSession
from passlib.context import CryptContext

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
JWT_SECRET = settings.jwt_secret_key  # Add to config
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # 1 hour
REFRESH_TOKEN_EXPIRE_DAYS = 30  # 30 days


def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(
    user_id: int,
    organization_id: int,
    permissions: list[str],
    session_id: str
) -> tuple[str, str]:
    """Create JWT access token"""
    jti = secrets.token_urlsafe(32)
    exp = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    payload = {
        "sub": str(user_id),
        "org_id": organization_id,
        "permissions": permissions,
        "session_id": session_id,
        "jti": jti,
        "exp": exp,
        "iat": datetime.now(timezone.utc),
        "type": "access"
    }
    
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token, jti


def create_refresh_token(user_id: int, session_id: str) -> tuple[str, str]:
    """Create JWT refresh token"""
    jti = secrets.token_urlsafe(32)
    exp = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    payload = {
        "sub": str(user_id),
        "session_id": session_id,
        "jti": jti,
        "exp": exp,
        "iat": datetime.now(timezone.utc),
        "type": "refresh"
    }
    
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token, jti


def decode_token(token: str) -> Dict[str, Any]:
    """Decode and validate JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def validate_session(
    session_id: str,
    jti: str,
    db: AsyncSession
) -> UserSession:
    """Validate session is still active"""
    result = await db.execute(
        select(UserSession)
        .where(UserSession.session_id == session_id)
        .where(UserSession.access_token_jti == jti)
        .where(UserSession.is_active == True)
        .where(UserSession.expires_at > datetime.now(timezone.utc))
    )
    session = result.scalars().first()
    
    if not session:
        raise HTTPException(status_code=401, detail="Session invalid or expired")
    
    # Update last activity
    session.last_activity_at = datetime.now(timezone.utc)
    await db.commit()
    
    return session
```

**Checklist:**
- [ ] Create auth.py file
- [ ] Add JWT_SECRET_KEY to config.py and .env
- [ ] Implement password hashing functions
- [ ] Implement JWT token creation/validation
- [ ] Add session validation middleware
- [ ] Test token expiration and refresh

#### 2.3: Create Login Endpoints
**File:** `/backend/app/main.py`  
**Location:** Add new endpoints around line 300

```python
from pydantic import BaseModel, EmailStr
from .auth import (
    hash_password, verify_password,
    create_access_token, create_refresh_token,
    decode_token, validate_session
)
from .rbac import get_user_permissions


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict


@app.post("/api/auth/login", response_model=LoginResponse)
async def login(
    credentials: LoginRequest,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """Authenticate user and create session"""
    # Find user
    result = await db.execute(
        select(User).where(User.email == credentials.email)
    )
    user = result.scalars().first()
    
    if not user or not user.password_hash:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Verify password
    if not verify_password(credentials.password, user.password_hash):
        # Increment failed attempts
        user.failed_login_attempts += 1
        if user.failed_login_attempts >= 5:
            user.locked_until = datetime.now(timezone.utc) + timedelta(minutes=30)
        await db.commit()
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Check if account is locked
    if user.locked_until and user.locked_until > datetime.now(timezone.utc):
        raise HTTPException(status_code=423, detail="Account temporarily locked")
    
    # Check organization status
    org_result = await db.execute(
        select(Organization).where(Organization.id == user.organization_id)
    )
    org = org_result.scalars().first()
    
    if org.status != "active":
        raise HTTPException(status_code=403, detail="Organization suspended")
    
    # Get user permissions
    permissions = await get_user_permissions(user, db)
    
    # Create session
    session_id = secrets.token_urlsafe(32)
    access_token, access_jti = create_access_token(
        user.id, user.organization_id, permissions, session_id
    )
    refresh_token, refresh_jti = create_refresh_token(user.id, session_id)
    
    # Store session
    session = UserSession(
        user_id=user.id,
        session_id=session_id,
        access_token_jti=access_jti,
        refresh_token_jti=refresh_jti,
        ip_address=request.client.host,
        user_agent=request.headers.get("user-agent", ""),
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        refresh_expires_at=datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    )
    db.add(session)
    
    # Reset failed attempts and update last login
    user.failed_login_attempts = 0
    user.locked_until = None
    user.last_login_at = datetime.now(timezone.utc)
    
    await db.commit()
    
    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user={
            "id": user.id,
            "email": user.email,
            "name": f"{user.first_name} {user.last_name}",
            "organization": org.name,
            "permissions": permissions
        }
    )
```

**Checklist:**
- [ ] Add LoginRequest/LoginResponse models
- [ ] Implement /api/auth/login endpoint
- [ ] Implement /api/auth/logout endpoint
- [ ] Implement /api/auth/refresh endpoint
- [ ] Implement /api/auth/me endpoint (get current user)
- [ ] Add rate limiting to login endpoint (max 5 attempts/min)
- [ ] Test all auth flows

---

### Task 3: SSO/SAML Integration

#### 3.1: Install SSO Libraries
**File:** `/backend/requirements.txt`

```
# Add these lines
python-saml==1.15.0       # SAML 2.0 support
authlib==1.3.0            # OAuth 2.0 / OpenID Connect
PyJWT[crypto]==2.8.0      # JWT with crypto support
```

**Checklist:**
- [ ] Add SSO libraries to requirements.txt
- [ ] Install: `pip install -r requirements.txt`
- [ ] Verify installations

#### 3.2: Add SSO Configuration Model
**File:** `/backend/app/models.py`

```python
class SSOConfiguration(Base):
    """SSO provider configuration per organization"""
    __tablename__ = "sso_configurations"
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False, index=True)
    
    # Provider details
    provider = Column(String(32), nullable=False)  # okta|azure_ad|google|onelogin
    provider_name = Column(String(128))  # Display name
    
    # SAML configuration
    saml_entity_id = Column(String(512), nullable=True)
    saml_sso_url = Column(String(512), nullable=True)
    saml_x509_cert = Column(Text, nullable=True)
    saml_name_id_format = Column(String(256), nullable=True)
    
    # OAuth/OIDC configuration
    oauth_client_id = Column(String(256), nullable=True)
    oauth_client_secret = Column(String(512), nullable=True)
    oauth_authorization_endpoint = Column(String(512), nullable=True)
    oauth_token_endpoint = Column(String(512), nullable=True)
    oauth_userinfo_endpoint = Column(String(512), nullable=True)
    oauth_scopes = Column(JSON, nullable=True)  # ["openid", "email", "profile"]
    
    # Attribute mapping
    attribute_mapping = Column(JSON, nullable=True)  # {"email": "emailAddress", "first_name": "givenName"}
    
    # Configuration
    is_active = Column(Boolean, default=True)
    enforce_sso = Column(Boolean, default=False)  # Disable password login
    auto_provision_users = Column(Boolean, default=True)
    default_role_id = Column(Integer, ForeignKey("roles.id"), nullable=True)
    
    # Relationships
    organization = relationship("Organization", backref="sso_configurations")
```

**Checklist:**
- [ ] Add SSOConfiguration model
- [ ] Create migration
- [ ] Add SSO config UI endpoints

#### 3.3: Implement SAML Authentication
**New File:** `/backend/app/sso/saml.py`

```python
"""SAML 2.0 authentication provider"""
from onelogin.saml2.auth import OneLogin_Saml2_Auth
from onelogin.saml2.settings import OneLogin_Saml2_Settings
from fastapi import HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

# Implementation placeholder - requires detailed SAML setup
# See: https://github.com/onelogin/python3-saml
```

**Checklist:**
- [ ] Create sso/ directory
- [ ] Implement SAML authentication flow
- [ ] Create /api/auth/saml/login endpoint
- [ ] Create /api/auth/saml/acs (Assertion Consumer Service) endpoint
- [ ] Create /api/auth/saml/metadata endpoint
- [ ] Test with Okta sandbox
- [ ] Test with Azure AD
- [ ] Document SSO setup for customers

#### 3.4: Implement OAuth/OIDC Authentication
**New File:** `/backend/app/sso/oauth.py`

```python
"""OAuth 2.0 / OpenID Connect provider"""
from authlib.integrations.starlette_client import OAuth
from starlette.config import Config

# Implementation placeholder
# See: https://docs.authlib.org/en/latest/client/starlette.html
```

**Checklist:**
- [ ] Implement OAuth flow
- [ ] Create /api/auth/oauth/login endpoint
- [ ] Create /api/auth/oauth/callback endpoint
- [ ] Test with Google Workspace
- [ ] Test with Azure AD (OAuth flow)
- [ ] Test with Okta (OAuth flow)

---

### Task 4: Audit Logging

#### 4.1: Add Audit Log Model
**File:** `/backend/app/models.py`

```python
class AuditLog(Base):
    """Immutable audit log for compliance"""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Actor
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False, index=True)
    actor_type = Column(String(32), nullable=False)  # user|agent|system
    actor_identifier = Column(String(256), nullable=False)  # email or agent ID
    
    # Action
    action_type = Column(String(64), nullable=False, index=True)  # login|incident.create|setting.update
    resource_type = Column(String(64), nullable=True, index=True)  # incident|user|integration
    resource_id = Column(String(128), nullable=True, index=True)
    
    # Context
    ip_address = Column(String(64))
    user_agent = Column(String(512))
    session_id = Column(String(64), nullable=True)
    
    # Details
    action_details = Column(JSON, nullable=True)  # What changed
    old_values = Column(JSON, nullable=True)  # Before state
    new_values = Column(JSON, nullable=True)  # After state
    
    # Result
    status = Column(String(32), nullable=False)  # success|failed|denied
    error_message = Column(Text, nullable=True)
    
    # Integrity
    checksum = Column(String(64), nullable=False)  # SHA256 of record for tamper detection
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")
    organization = relationship("Organization")
```

**Checklist:**
- [ ] Add AuditLog model
- [ ] Create migration
- [ ] Implement audit_log() helper function
- [ ] Add audit logging to all sensitive operations
- [ ] Implement checksum verification
- [ ] Create audit log export API
- [ ] Test audit log query performance with indexes

#### 4.2: Implement Audit Logging Middleware
**New File:** `/backend/app/audit.py`

```python
"""Audit logging utilities"""
import hashlib
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession
from .models import AuditLog, User


async def create_audit_log(
    db: AsyncSession,
    user: Optional[User],
    organization_id: int,
    action_type: str,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    action_details: Optional[Dict[str, Any]] = None,
    old_values: Optional[Dict[str, Any]] = None,
    new_values: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    status: str = "success",
    error_message: Optional[str] = None
) -> AuditLog:
    """Create immutable audit log entry"""
    
    actor_type = "user" if user else "system"
    actor_identifier = user.email if user else "system"
    
    log = AuditLog(
        user_id=user.id if user else None,
        organization_id=organization_id,
        actor_type=actor_type,
        actor_identifier=actor_identifier,
        action_type=action_type,
        resource_type=resource_type,
        resource_id=resource_id,
        action_details=action_details,
        old_values=old_values,
        new_values=new_values,
        ip_address=ip_address,
        user_agent=user_agent,
        status=status,
        error_message=error_message
    )
    
    # Calculate checksum for integrity
    checksum_data = {
        "organization_id": organization_id,
        "actor": actor_identifier,
        "action": action_type,
        "resource": f"{resource_type}:{resource_id}" if resource_type else None,
        "timestamp": log.created_at.isoformat() if log.created_at else datetime.now(timezone.utc).isoformat()
    }
    checksum = hashlib.sha256(
        json.dumps(checksum_data, sort_keys=True).encode()
    ).hexdigest()
    log.checksum = checksum
    
    db.add(log)
    await db.commit()
    
    return log
```

**Checklist:**
- [ ] Create audit.py file
- [ ] Implement create_audit_log function
- [ ] Add audit logging to: login, logout, incident creation, response execution
- [ ] Add audit logging to: user management, settings changes, integration changes
- [ ] Create audit log viewer UI
- [ ] Implement audit log export (CSV, JSON)

---

### Task 5: Frontend Integration

#### 5.1: Update Frontend Authentication
**File:** `/frontend/app/login/page.tsx` (create new)

```typescript
'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'

export default function LoginPage() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const router = useRouter()

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')

    try {
      const res = await fetch('http://localhost:8000/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      })

      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || 'Login failed')
      }

      const data = await res.json()
      
      // Store tokens
      localStorage.setItem('access_token', data.access_token)
      localStorage.setItem('refresh_token', data.refresh_token)
      localStorage.setItem('user', JSON.stringify(data.user))

      router.push('/')
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100">
      <div className="max-w-md w-full bg-white p-8 rounded-lg shadow">
        <h1 className="text-2xl font-bold mb-6">Mini-XDR Login</h1>
        
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded mb-4">
            {error}
          </div>
        )}

        <form onSubmit={handleLogin}>
          <div className="mb-4">
            <label className="block text-gray-700 mb-2">Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full px-4 py-2 border rounded"
              required
            />
          </div>

          <div className="mb-6">
            <label className="block text-gray-700 mb-2">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full px-4 py-2 border rounded"
              required
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700 disabled:bg-gray-400"
          >
            {loading ? 'Logging in...' : 'Login'}
          </button>
        </form>
      </div>
    </div>
  )
}
```

**Checklist:**
- [ ] Create login page component
- [ ] Implement token storage (localStorage or httpOnly cookies)
- [ ] Create authentication context/provider
- [ ] Add auth check to protected routes
- [ ] Implement auto token refresh
- [ ] Add logout functionality
- [ ] Create user profile dropdown
- [ ] Show current user's permissions

---

## Testing Checklist

### Unit Tests
- [ ] Test password hashing and verification
- [ ] Test JWT token creation and validation
- [ ] Test permission checking logic
- [ ] Test session validation
- [ ] Test audit log checksum calculation

### Integration Tests
- [ ] Test complete login flow
- [ ] Test login with invalid credentials (account lockout)
- [ ] Test token refresh flow
- [ ] Test session expiration
- [ ] Test logout
- [ ] Test permission denial (403)
- [ ] Test SSO flow (if implemented)
- [ ] Test multi-tenant data isolation

### Security Tests
- [ ] Test for SQL injection in login
- [ ] Test for timing attacks on password verification
- [ ] Test session hijacking prevention
- [ ] Test CSRF protection
- [ ] Test XSS in user profile fields
- [ ] Verify audit logs are immutable
- [ ] Test rate limiting on login endpoint

---

## Production Deployment Checklist

### Before Go-Live
- [ ] All database migrations tested in staging
- [ ] JWT secret key is cryptographically random (64+ bytes)
- [ ] Password policy enforced (min 12 chars, complexity)
- [ ] Session timeout configured (default 1 hour)
- [ ] Rate limiting enabled on auth endpoints
- [ ] Audit logging enabled for all sensitive operations
- [ ] HTTPS enforced (no HTTP)
- [ ] httpOnly, Secure, SameSite cookies for sessions
- [ ] Password reset flow implemented and tested
- [ ] Email verification implemented
- [ ] Account lockout policy configured (5 failed attempts)
- [ ] MFA available (even if optional)

### Documentation
- [ ] User management guide for admins
- [ ] SSO setup guide for customers
- [ ] RBAC permission reference
- [ ] API authentication guide for integrations
- [ ] Troubleshooting guide for auth issues

---

## Estimated Effort

**Total Effort:** 6-8 weeks, 1-2 engineers

| Task | Effort | Priority |
|------|--------|----------|
| Multi-tenancy foundation | 2 weeks | P0 |
| Session management & JWT | 1 week | P0 |
| RBAC implementation | 1.5 weeks | P0 |
| SSO/SAML integration | 2 weeks | P1 |
| Audit logging | 1 week | P0 |
| Frontend integration | 1 week | P0 |
| Testing & documentation | 1 week | P0 |

---

**Status:** Ready for implementation  
**Next Document:** `02_DATA_COMPLIANCE_PRIVACY.md`


