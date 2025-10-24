"""
Agent Enrollment Service - Tenant-aware agent registration

Handles agent token generation, enrollment tracking, and heartbeat monitoring.
"""
import logging
import secrets
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from .models import AgentEnrollment, Organization, DiscoveredAsset

logger = logging.getLogger(__name__)


class AgentEnrollmentService:
    """
    Tenant-scoped agent enrollment service
    
    Manages agent tokens, registration, and lifecycle tracking per organization.
    """
    
    def __init__(self, organization_id: int, db: AsyncSession):
        self.organization_id = organization_id
        self.db = db
    
    async def generate_agent_token(
        self,
        platform: str = "linux",
        discovered_asset_id: Optional[int] = None,
        hostname: Optional[str] = None
    ) -> Dict:
        """
        Generate a new agent enrollment token
        
        Args:
            platform: Target platform (windows|linux|macos|docker)
            discovered_asset_id: Optional link to discovered asset
            hostname: Optional target hostname
            
        Returns:
            Token details including token, install scripts, and enrollment info
        """
        # Generate cryptographically secure token
        token = f"xdr-{self.organization_id}-{secrets.token_urlsafe(32)}"
        
        # Create enrollment record
        enrollment = AgentEnrollment(
            organization_id=self.organization_id,
            agent_token=token,
            platform=platform,
            hostname=hostname,
            status="pending",
            enrollment_source="onboarding_wizard",
            discovered_asset_id=discovered_asset_id
        )
        
        self.db.add(enrollment)
        await self.db.commit()
        await self.db.refresh(enrollment)
        
        logger.info(
            f"Generated agent token for org {self.organization_id}, "
            f"platform {platform}, enrollment_id {enrollment.id}"
        )
        
        # Generate install scripts
        install_scripts = self._generate_install_scripts(token, platform)
        
        return {
            "enrollment_id": enrollment.id,
            "agent_token": token,
            "platform": platform,
            "hostname": hostname,
            "status": "pending",
            "install_scripts": install_scripts,
            "created_at": enrollment.created_at.isoformat()
        }
    
    def _generate_install_scripts(self, token: str, platform: str) -> Dict[str, str]:
        """
        Generate platform-specific install scripts
        
        Args:
            token: Agent enrollment token
            platform: Target platform
            
        Returns:
            Dictionary of script types to script content
        """
        # Backend API endpoint (should be configurable in production)
        backend_url = "http://backend-service:8000"  # K8s internal service
        
        scripts = {}
        
        if platform == "linux":
            scripts["bash"] = f"""#!/bin/bash
# Mini-XDR Agent Installation Script - Linux
set -e

echo "Installing Mini-XDR Agent..."

# Download agent
curl -fsSL {backend_url}/api/agents/download/linux -o /tmp/minixdr-agent
chmod +x /tmp/minixdr-agent

# Install as systemd service
sudo mv /tmp/minixdr-agent /usr/local/bin/minixdr-agent

# Create config
sudo mkdir -p /etc/minixdr
cat <<EOF | sudo tee /etc/minixdr/config.json
{{
  "agent_token": "{token}",
  "backend_url": "{backend_url}",
  "log_level": "info"
}}
EOF

# Create systemd service
cat <<EOF | sudo tee /etc/systemd/system/minixdr-agent.service
[Unit]
Description=Mini-XDR Security Agent
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/minixdr-agent --config /etc/minixdr/config.json
Restart=always
User=root

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable minixdr-agent
sudo systemctl start minixdr-agent

echo "✅ Mini-XDR Agent installed and started"
echo "Token: {token[:20]}..."
"""
            
        elif platform == "windows":
            scripts["powershell"] = f"""# Mini-XDR Agent Installation Script - Windows
# Run as Administrator

Write-Host "Installing Mini-XDR Agent..." -ForegroundColor Green

# Download agent
$agentUrl = "{backend_url}/api/agents/download/windows"
$agentPath = "$env:ProgramFiles\\MiniXDR\\agent.exe"

New-Item -Path "$env:ProgramFiles\\MiniXDR" -ItemType Directory -Force | Out-Null
Invoke-WebRequest -Uri $agentUrl -OutFile $agentPath

# Create config
$configPath = "$env:ProgramFiles\\MiniXDR\\config.json"
$config = @{{
    agent_token = "{token}"
    backend_url = "{backend_url}"
    log_level = "info"
}} | ConvertTo-Json

$config | Out-File -FilePath $configPath -Encoding UTF8

# Install as Windows Service
New-Service -Name "MiniXDRAgent" `
    -BinaryPathName "$agentPath --config $configPath" `
    -DisplayName "Mini-XDR Security Agent" `
    -StartupType Automatic `
    -Description "Mini-XDR endpoint security agent"

Start-Service -Name "MiniXDRAgent"

Write-Host "✅ Mini-XDR Agent installed and started" -ForegroundColor Green
Write-Host "Token: {token[:20]}..." -ForegroundColor Yellow
"""
            
        elif platform == "macos":
            scripts["bash"] = f"""#!/bin/bash
# Mini-XDR Agent Installation Script - macOS
set -e

echo "Installing Mini-XDR Agent..."

# Download agent
curl -fsSL {backend_url}/api/agents/download/macos -o /tmp/minixdr-agent
chmod +x /tmp/minixdr-agent

# Install to /usr/local/bin
sudo mv /tmp/minixdr-agent /usr/local/bin/minixdr-agent

# Create config directory
sudo mkdir -p /etc/minixdr
cat <<EOF | sudo tee /etc/minixdr/config.json
{{
  "agent_token": "{token}",
  "backend_url": "{backend_url}",
  "log_level": "info"
}}
EOF

# Create LaunchDaemon
cat <<EOF | sudo tee /Library/LaunchDaemons/com.minixdr.agent.plist
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.minixdr.agent</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/minixdr-agent</string>
        <string>--config</string>
        <string>/etc/minixdr/config.json</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
EOF

# Load and start service
sudo launchctl load /Library/LaunchDaemons/com.minixdr.agent.plist

echo "✅ Mini-XDR Agent installed and started"
echo "Token: {token[:20]}..."
"""
        
        scripts["docker"] = f"""# Mini-XDR Agent - Docker Compose
version: '3.8'

services:
  minixdr-agent:
    image: minixdr/agent:latest
    container_name: minixdr-agent
    restart: unless-stopped
    environment:
      - AGENT_TOKEN={token}
      - BACKEND_URL={backend_url}
      - LOG_LEVEL=info
    volumes:
      - /var/log:/host/logs:ro
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
    network_mode: host
    privileged: true
"""
        
        return scripts
    
    async def register_agent(
        self,
        agent_token: str,
        agent_id: str,
        hostname: str,
        platform: str,
        ip_address: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Register an agent after first check-in
        
        Args:
            agent_token: Enrollment token
            agent_id: Agent's self-reported ID
            hostname: Agent hostname
            platform: Agent platform
            ip_address: Agent IP address
            metadata: Additional metadata (OS version, agent version, etc.)
            
        Returns:
            Registration confirmation
        """
        # Find enrollment by token
        stmt = select(AgentEnrollment).where(
            AgentEnrollment.agent_token == agent_token,
            AgentEnrollment.organization_id == self.organization_id
        )
        result = await self.db.execute(stmt)
        enrollment = result.scalar_one_or_none()
        
        if not enrollment:
            raise ValueError(f"Invalid enrollment token for org {self.organization_id}")
        
        if enrollment.status == "revoked":
            raise ValueError("Enrollment token has been revoked")
        
        # Update enrollment with agent details
        enrollment.agent_id = agent_id
        enrollment.hostname = hostname
        enrollment.platform = platform
        enrollment.ip_address = ip_address
        enrollment.agent_metadata = metadata or {}
        enrollment.status = "active"
        enrollment.first_checkin = datetime.now(timezone.utc)
        enrollment.last_heartbeat = datetime.now(timezone.utc)
        
        await self.db.commit()
        await self.db.refresh(enrollment)
        
        logger.info(
            f"Agent registered: {agent_id} ({hostname}) for org {self.organization_id}"
        )
        
        return {
            "enrollment_id": enrollment.id,
            "agent_id": agent_id,
            "hostname": hostname,
            "platform": platform,
            "status": "active",
            "registered_at": enrollment.first_checkin.isoformat()
        }
    
    async def update_heartbeat(
        self,
        agent_token: str,
        agent_id: str
    ) -> bool:
        """
        Update agent heartbeat timestamp
        
        Args:
            agent_token: Enrollment token
            agent_id: Agent ID
            
        Returns:
            True if successful
        """
        stmt = select(AgentEnrollment).where(
            and_(
                AgentEnrollment.agent_token == agent_token,
                AgentEnrollment.agent_id == agent_id,
                AgentEnrollment.organization_id == self.organization_id
            )
        )
        result = await self.db.execute(stmt)
        enrollment = result.scalar_one_or_none()
        
        if not enrollment:
            return False
        
        enrollment.last_heartbeat = datetime.now(timezone.utc)
        
        # Update status if currently inactive
        if enrollment.status == "inactive":
            enrollment.status = "active"
            logger.info(f"Agent {agent_id} reactivated for org {self.organization_id}")
        
        await self.db.commit()
        return True
    
    async def get_enrolled_agents(
        self,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get list of enrolled agents for this organization
        
        Args:
            status: Optional status filter (pending|active|inactive|revoked)
            limit: Maximum number of results
            
        Returns:
            List of agent enrollment dictionaries
        """
        query = select(AgentEnrollment).where(
            AgentEnrollment.organization_id == self.organization_id
        )
        
        if status:
            query = query.where(AgentEnrollment.status == status)
        
        query = query.order_by(AgentEnrollment.created_at.desc()).limit(limit)
        
        result = await self.db.execute(query)
        enrollments = result.scalars().all()
        
        # Mark agents as inactive if no heartbeat in last 5 minutes
        current_time = datetime.now(timezone.utc)
        inactive_threshold = current_time - timedelta(minutes=5)
        
        return [
            {
                "enrollment_id": e.id,
                "agent_id": e.agent_id,
                "agent_token": e.agent_token[:20] + "..." if e.agent_token else None,  # Truncate for security
                "hostname": e.hostname,
                "platform": e.platform,
                "ip_address": e.ip_address,
                "status": "inactive" if (e.last_heartbeat and e.last_heartbeat < inactive_threshold and e.status == "active") else e.status,
                "first_checkin": e.first_checkin.isoformat() if e.first_checkin else None,
                "last_heartbeat": e.last_heartbeat.isoformat() if e.last_heartbeat else None,
                "agent_metadata": e.agent_metadata,
                "created_at": e.created_at.isoformat()
            }
            for e in enrollments
        ]
    
    async def revoke_agent(self, enrollment_id: int, reason: str) -> bool:
        """
        Revoke an agent enrollment
        
        Args:
            enrollment_id: Enrollment ID to revoke
            reason: Revocation reason
            
        Returns:
            True if successful
        """
        stmt = select(AgentEnrollment).where(
            and_(
                AgentEnrollment.id == enrollment_id,
                AgentEnrollment.organization_id == self.organization_id
            )
        )
        result = await self.db.execute(stmt)
        enrollment = result.scalar_one_or_none()
        
        if not enrollment:
            return False
        
        enrollment.status = "revoked"
        enrollment.revoked_at = datetime.now(timezone.utc)
        enrollment.revoked_reason = reason
        
        await self.db.commit()
        
        logger.info(
            f"Agent enrollment {enrollment_id} revoked for org {self.organization_id}: {reason}"
        )
        
        return True



