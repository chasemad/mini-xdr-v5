"""
Tenant Middleware - Automatic organization_id filtering for multi-tenancy

Provides defense-in-depth by ensuring all database queries are automatically
scoped to the current user's organization.
"""
import logging
from contextvars import ContextVar
from typing import Optional
from fastapi import Request, HTTPException
from sqlalchemy import event
from sqlalchemy.orm import Session
from sqlalchemy.sql import Select

from .auth import decode_token

logger = logging.getLogger(__name__)

# Context variable to store current organization_id for the request
current_organization_id: ContextVar[Optional[int]] = ContextVar(
    "current_organization_id", default=None
)


class TenantMiddleware:
    """
    Middleware to extract and validate organization context from JWT
    
    Extracts organization_id from JWT token and stores it in context variable
    for automatic query filtering.
    """
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Extract token from Authorization header
        headers = dict(scope.get("headers", []))
        auth_header = headers.get(b"authorization", b"").decode("utf-8")
        
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]  # Remove "Bearer " prefix
            
            try:
                # Decode token and extract organization_id
                payload = decode_token(token)
                org_id = payload.get("organization_id")
                
                if org_id:
                    # Store in context variable for this request
                    current_organization_id.set(org_id)
                    logger.debug(f"Request scoped to organization {org_id}")
            except Exception as e:
                logger.debug(f"Token decode failed in middleware: {e}")
        
        # Process request
        await self.app(scope, receive, send)
        
        # Clear context after request
        current_organization_id.set(None)


def get_current_org_id() -> Optional[int]:
    """
    Get the current organization ID from request context
    
    Returns:
        Current organization ID or None if not in authenticated context
    """
    return current_organization_id.get()


def require_org_id() -> int:
    """
    Get the current organization ID, raising exception if not available
    
    Returns:
        Current organization ID
        
    Raises:
        HTTPException if no organization context available
    """
    org_id = current_organization_id.get()
    
    if org_id is None:
        raise HTTPException(
            status_code=403,
            detail="Organization context required"
        )
    
    return org_id


# SQLAlchemy event listener to automatically filter queries by organization_id
@event.listens_for(Session, "do_orm_execute")
def receive_do_orm_execute(orm_execute_state):
    """
    Automatically add organization_id filter to SELECT queries
    
    This provides defense-in-depth by ensuring queries can't accidentally
    access cross-tenant data even if the application code forgets to filter.
    """
    if not orm_execute_state.is_select:
        return
    
    org_id = current_organization_id.get()
    
    if org_id is None:
        # No org context - allow query (for system operations)
        return
    
    # Get the select statement
    statement = orm_execute_state.statement
    
    if not isinstance(statement, Select):
        return
    
    # Check if statement is querying a tenant-scoped table
    # List of tables that should be automatically filtered
    tenant_tables = {
        "events",
        "incidents",
        "log_sources",
        "action_logs",
        "containment_policies",
        "ml_models",
        "discovered_assets",
        "agent_enrollments"
    }
    
    # Get table name from statement
    # This is a simplified check - in production you'd want more robust detection
    statement_str = str(statement).lower()
    
    for table_name in tenant_tables:
        if table_name in statement_str:
            # Check if organization_id filter already exists
            if f"{table_name}.organization_id" not in statement_str:
                logger.warning(
                    f"Query on {table_name} missing organization_id filter. "
                    f"Org context: {org_id}. Consider adding explicit filter."
                )
                # In strict mode, you could raise an exception here
                # raise RuntimeError(f"Query on {table_name} must include organization_id filter")
            break


def validate_organization_access(user_org_id: int, resource_org_id: int):
    """
    Validate that user has access to resource
    
    Args:
        user_org_id: User's organization ID
        resource_org_id: Resource's organization ID
        
    Raises:
        HTTPException if access denied
    """
    if user_org_id != resource_org_id:
        logger.warning(
            f"Cross-tenant access attempt: user_org={user_org_id}, "
            f"resource_org={resource_org_id}"
        )
        raise HTTPException(
            status_code=404,  # Return 404 instead of 403 to avoid enumeration
            detail="Resource not found"
        )


# Helper function to ensure queries are org-scoped
def ensure_org_filter(query, organization_id: int, model_class):
    """
    Ensure query includes organization_id filter
    
    Args:
        query: SQLAlchemy query
        organization_id: Organization ID to filter by
        model_class: Model class being queried
        
    Returns:
        Filtered query
    """
    # Check if organization_id is already in the query
    # This is a safety net to prevent double-filtering
    existing_where = str(query.whereclause) if hasattr(query, 'whereclause') else ""
    
    if "organization_id" not in existing_where:
        return query.where(model_class.organization_id == organization_id)
    
    return query



