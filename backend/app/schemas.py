"""
Pydantic schemas for request/response validation
"""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ==================== AUTHENTICATION SCHEMAS ====================

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    email: Optional[str] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RegisterOrganizationRequest(BaseModel):
    organization_name: str = Field(..., min_length=2, max_length=255)
    admin_email: EmailStr
    admin_password: str = Field(..., min_length=12)
    admin_name: str = Field(..., min_length=2, max_length=255)


class UserResponse(BaseModel):
    id: int
    organization_id: int
    email: str
    full_name: Optional[str]
    role: str
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]
    
    class Config:
        from_attributes = True


class OrganizationResponse(BaseModel):
    id: int
    name: str
    slug: str
    status: str
    created_at: datetime
    max_users: int
    max_log_sources: int
    onboarding_status: Optional[str]
    onboarding_step: Optional[str]
    onboarding_completed_at: Optional[datetime]

    # Seamless onboarding (v2) fields
    onboarding_flow_version: Optional[str] = "seamless"
    auto_discovery_enabled: Optional[bool] = True
    integration_settings: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class MeResponse(BaseModel):
    user: UserResponse
    organization: OrganizationResponse


class InviteUserRequest(BaseModel):
    email: EmailStr
    full_name: str
    role: str = Field(..., pattern="^(viewer|analyst|soc_lead|admin)$")


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=12)


class UpdateProfileRequest(BaseModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None


# ==================== ONBOARDING SCHEMAS ====================

class OnboardingProfileRequest(BaseModel):
    """Step 1: Organization profile"""
    organization_name: Optional[str] = None
    region: Optional[str] = None
    industry: Optional[str] = None
    company_size: Optional[str] = None  # small|medium|large|enterprise


class NetworkScanRequest(BaseModel):
    """Step 2: Network scan configuration"""
    network_ranges: List[str] = Field(..., min_items=1, description="CIDR ranges like 10.0.0.0/24")
    scan_type: str = Field("quick", pattern="^(quick|full)$")
    port_ranges: Optional[List[int]] = None
    

class NetworkScanResponse(BaseModel):
    """Network scan result"""
    scan_id: str
    status: str  # pending|running|completed|failed
    assets_discovered: int
    network_ranges: List[str]
    scan_type: str
    completed_at: Optional[str] = None
    error: Optional[str] = None


class DiscoveredAssetResponse(BaseModel):
    """Discovered asset details"""
    id: int
    ip: str
    hostname: Optional[str]
    os_type: Optional[str]
    os_role: Optional[str]
    classification: Optional[str]
    classification_confidence: float
    open_ports: List[int]
    services: Dict[str, Any]
    deployment_profile: Dict[str, Any]
    deployment_priority: str
    agent_compatible: bool
    discovered_at: Optional[str]
    last_seen: Optional[str]


class GenerateAgentTokenRequest(BaseModel):
    """Request to generate agent enrollment token"""
    platform: str = Field(..., pattern="^(windows|linux|macos|docker)$")
    hostname: Optional[str] = None
    discovered_asset_id: Optional[int] = None


class AgentTokenResponse(BaseModel):
    """Agent enrollment token and install scripts"""
    enrollment_id: int
    agent_token: str
    platform: str
    hostname: Optional[str]
    status: str
    install_scripts: Dict[str, str]
    created_at: str


class AgentEnrollmentResponse(BaseModel):
    """Enrolled agent details"""
    enrollment_id: int
    agent_id: Optional[str]
    agent_token: str
    hostname: Optional[str]
    platform: Optional[str]
    ip_address: Optional[str]
    status: str
    first_checkin: Optional[str]
    last_heartbeat: Optional[str]
    agent_metadata: Optional[Dict[str, Any]]
    created_at: str


class ValidationCheckResponse(BaseModel):
    """Validation check result"""
    check_name: str
    status: str  # pass|fail|pending
    message: str
    details: Optional[Dict[str, Any]] = None


class OnboardingStatusResponse(BaseModel):
    """Current onboarding status"""
    onboarding_status: str  # not_started|in_progress|completed
    onboarding_step: Optional[str]
    onboarding_data: Optional[Dict[str, Any]]
    onboarding_completed_at: Optional[str]
    completion_percentage: int
