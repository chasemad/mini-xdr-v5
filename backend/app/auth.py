"""
Authentication and Authorization Module for Mini-XDR
Provides JWT-based authentication with organization-based multi-tenancy
"""
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Optional
import bcrypt  # Direct bcrypt usage for compatibility
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import User, Organization
from .db import get_db
from .config import settings

# JWT configuration
SECRET_KEY = settings.JWT_SECRET_KEY or os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    raise ValueError(
        "JWT_SECRET_KEY is required but not set. "
        "Please set JWT_SECRET_KEY environment variable or configure it in settings."
    )
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 8
REFRESH_TOKEN_EXPIRE_DAYS = 30

# HTTP Bearer token scheme
security = HTTPBearer()

# Password validation regex
PASSWORD_REGEX = re.compile(
    r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&#^()_\-+=\[\]{}|\\:;\"\'<>,.?/~`])[A-Za-z\d@$!%*?&#^()_\-+=\[\]{}|\\:;\"\'<>,.?/~`]{12,}$'
)


def hash_password(password: str) -> str:
    """Hash a password using bcrypt - direct implementation for bcrypt 5.x compatibility"""
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash - direct bcrypt for compatibility"""
    try:
        password_bytes = plain_password.encode('utf-8')
        hash_bytes = hashed_password.encode('utf-8')
        return bcrypt.checkpw(password_bytes, hash_bytes)
    except Exception as e:
        print(f"Password verification error: {e}")
        return False


def validate_password_strength(password: str) -> tuple[bool, str]:
    """
    Validate password meets security requirements
    Returns: (is_valid, error_message)
    """
    if len(password) < 12:
        return False, "Password must be at least 12 characters long"
    
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    
    if not re.search(r'\d', password):
        return False, "Password must contain at least one number"
    
    if not re.search(r'[@$!%*?&#^()_\-+=\[\]{}|\\:;\"\'<>,.?/~`]', password):
        return False, "Password must contain at least one special character"
    
    return True, ""


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "access"
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """Create a JWT refresh token"""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "refresh"
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> dict:
    """Decode and verify a JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Dependency to get the current authenticated user
    Validates JWT token and returns User object
    """
    token = credentials.credentials
    
    try:
        payload = decode_token(token)
        user_id: int = payload.get("user_id")
        org_id: int = payload.get("organization_id")
        
        if user_id is None or org_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Fetch user from database
    stmt = select(User).where(User.id == user_id, User.is_active == True)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if account is locked
    if user.locked_until and user.locked_until > datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Account locked until {user.locked_until.isoformat()}",
        )
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Dependency to ensure user is active"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


async def get_current_user_optional(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Optional[User]:
    """
    Optional dependency to get the current authenticated user
    Returns User object if authenticated, None if not authenticated
    """
    try:
        payload = decode_token(credentials.credentials)
        user_id: int = payload.get("user_id")

        if user_id is None:
            return None

        # Get database session
        from .db import AsyncSessionLocal
        async with AsyncSessionLocal() as db:
            stmt = select(User).where(User.id == user_id, User.is_active == True)
            result = await db.execute(stmt)
            user = result.scalar_one_or_none()

            if user is None:
                return None

            # Check if account is locked
            if user.locked_until and user.locked_until > datetime.now(timezone.utc):
                return None

            return user

    except (JWTError, Exception):
        return None


def require_role(required_role: str):
    """Dependency factory to require specific role"""
    async def role_checker(current_user: User = Depends(get_current_user)) -> User:
        role_hierarchy = {
            "viewer": 1,
            "analyst": 2,
            "soc_lead": 3,
            "admin": 4
        }
        
        user_role_level = role_hierarchy.get(current_user.role, 0)
        required_role_level = role_hierarchy.get(required_role, 99)
        
        if user_role_level < required_role_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role}"
            )
        
        return current_user
    
    return role_checker


async def authenticate_user(db: AsyncSession, email: str, password: str) -> Optional[User]:
    """
    Authenticate user with email and password
    Returns User object if successful, None otherwise
    Implements account lockout after failed attempts
    """
    stmt = select(User).where(User.email == email)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    
    if not user:
        return None
    
    # Check if account is locked
    if user.locked_until and user.locked_until > datetime.now(timezone.utc):
        return None
    
    # Verify password
    if not verify_password(password, user.hashed_password):
        # Increment failed login attempts
        user.failed_login_attempts += 1
        
        # Lock account after 5 failed attempts for 15 minutes
        if user.failed_login_attempts >= 5:
            user.locked_until = datetime.now(timezone.utc) + timedelta(minutes=15)
        
        await db.commit()
        return None
    
    # Reset failed login attempts on successful login
    user.failed_login_attempts = 0
    user.locked_until = None
    user.last_login = datetime.now(timezone.utc)
    await db.commit()
    
    return user


async def create_organization(
    db: AsyncSession,
    name: str,
    slug: str,
    admin_email: str,
    admin_password: str,
    admin_name: str
) -> tuple[Organization, User]:
    """
    Create a new organization with an admin user
    Returns: (Organization, User)
    """
    # Validate password
    is_valid, error_msg = validate_password_strength(admin_password)
    if not is_valid:
        raise ValueError(error_msg)
    
    # Check if org slug or email already exists
    stmt = select(Organization).where(Organization.slug == slug)
    result = await db.execute(stmt)
    if result.scalar_one_or_none():
        raise ValueError("Organization slug already exists")
    
    stmt = select(User).where(User.email == admin_email)
    result = await db.execute(stmt)
    if result.scalar_one_or_none():
        raise ValueError("Email already registered")
    
    # Create organization
    org = Organization(
        name=name,
        slug=slug,
        status="active",
        settings={}
    )
    db.add(org)
    await db.flush()  # Get org.id without committing
    
    # Create admin user
    admin_user = User(
        organization_id=org.id,
        email=admin_email,
        hashed_password=hash_password(admin_password),
        full_name=admin_name,
        role="admin",
        is_active=True
    )
    db.add(admin_user)
    await db.commit()
    await db.refresh(org)
    await db.refresh(admin_user)
    
    return org, admin_user


def generate_slug(name: str) -> str:
    """Generate URL-safe slug from organization name"""
    slug = name.lower()
    slug = re.sub(r'[^a-z0-9\s-]', '', slug)
    slug = re.sub(r'[\s-]+', '-', slug)
    slug = slug.strip('-')
    return slug[:100]  # Limit to 100 chars

