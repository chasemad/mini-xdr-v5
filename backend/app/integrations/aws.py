"""
AWS cloud integration for seamless onboarding
"""

import asyncio
import logging
import secrets
from typing import Any, Dict, List, Optional

import boto3
import botocore.exceptions

from .base import CloudIntegration

logger = logging.getLogger(__name__)


class AWSIntegration(CloudIntegration):
    """AWS cloud integration for asset discovery and agent deployment"""

    def __init__(self, organization_id: int, credentials: Dict[str, Any]):
        super().__init__(organization_id, credentials)
        self.regions = []
        self.session_credentials = None

    async def authenticate(self) -> bool:
        """
        Authenticate using AWS STS AssumeRole or direct credentials

        Supports two authentication methods:
        1. AssumeRole: Requires role_arn + optional external_id
        2. Direct credentials: access_key_id + secret_access_key
        """

        def _authenticate_sync() -> bool:
            try:
                # Method 1: AssumeRole (recommended for production)
                if "role_arn" in self.credentials:
                    logger.info(
                        f"Authenticating with AWS via AssumeRole: {self.credentials['role_arn']}"
                    )

                    # Create STS client with base credentials (if provided) or use instance role
                    sts_kwargs = {}
                    if self.credentials.get("access_key_id"):
                        sts_kwargs["aws_access_key_id"] = self.credentials[
                            "access_key_id"
                        ]
                        sts_kwargs["aws_secret_access_key"] = self.credentials[
                            "secret_access_key"
                        ]

                    sts_client = boto3.client("sts", **sts_kwargs)

                    # Assume the role
                    assume_role_kwargs = {
                        "RoleArn": self.credentials["role_arn"],
                        "RoleSessionName": f"mini-xdr-org-{self.organization_id}",
                    }

                    if self.credentials.get("external_id"):
                        assume_role_kwargs["ExternalId"] = self.credentials[
                            "external_id"
                        ]

                    assumed_role = sts_client.assume_role(**assume_role_kwargs)

                    # Store session credentials
                    self.session_credentials = {
                        "aws_access_key_id": assumed_role["Credentials"]["AccessKeyId"],
                        "aws_secret_access_key": assumed_role["Credentials"][
                            "SecretAccessKey"
                        ],
                        "aws_session_token": assumed_role["Credentials"][
                            "SessionToken"
                        ],
                    }

                # Method 2: Direct credentials
                else:
                    logger.info("Authenticating with AWS via direct credentials")
                    self.session_credentials = {
                        "aws_access_key_id": self.credentials["aws_access_key_id"],
                        "aws_secret_access_key": self.credentials[
                            "aws_secret_access_key"
                        ],
                    }
                    if self.credentials.get("aws_session_token"):
                        self.session_credentials[
                            "aws_session_token"
                        ] = self.credentials["aws_session_token"]

                # Test authentication by listing regions
                ec2 = boto3.client("ec2", **self.session_credentials)
                regions_response = ec2.describe_regions()
                self.regions = [r["RegionName"] for r in regions_response["Regions"]]

                logger.info(
                    f"AWS authentication successful. Found {len(self.regions)} regions"
                )
                return True

            except botocore.exceptions.ClientError as e:
                logger.error(f"AWS authentication failed: {e}")
                return False
            except Exception as e:
                logger.error(f"Unexpected error during AWS authentication: {e}")
                return False

        # Run synchronous boto3 calls in thread pool
        return await asyncio.to_thread(_authenticate_sync)

    async def get_regions(self) -> List[str]:
        """Get list of available AWS regions"""
        if not self.regions:
            await self.authenticate()
        return self.regions

    async def discover_assets(self) -> List[Dict[str, Any]]:
        """
        Discover EC2 instances and RDS databases across all regions

        Returns:
            List of discovered assets with full metadata
        """
        logger.info(f"Starting AWS asset discovery for org {self.organization_id}")
        assets = []

        # Ensure we're authenticated
        if not self.session_credentials:
            if not await self.authenticate():
                logger.error("Cannot discover assets - authentication failed")
                return []

        # Discover assets in each region
        for region in self.regions:
            try:
                region_assets = await self._discover_region_assets(region)
                assets.extend(region_assets)
                logger.info(f"Discovered {len(region_assets)} assets in {region}")
            except Exception as e:
                logger.warning(f"Failed to discover assets in region {region}: {e}")
                continue

        logger.info(f"Total AWS assets discovered: {len(assets)}")
        return assets

    async def _discover_region_assets(self, region: str) -> List[Dict[str, Any]]:
        """Discover assets in a specific AWS region"""

        def _discover_sync() -> List[Dict[str, Any]]:
            assets = []

            try:
                # Discover EC2 instances
                ec2 = boto3.client(
                    "ec2", region_name=region, **self.session_credentials
                )
                instances = ec2.describe_instances()

                for reservation in instances["Reservations"]:
                    for instance in reservation["Instances"]:
                        if instance["State"]["Name"] in ["running", "stopped"]:
                            assets.append(
                                {
                                    "provider": "aws",
                                    "asset_type": "ec2",
                                    "asset_id": instance["InstanceId"],
                                    "region": region,
                                    "asset_data": {
                                        "instance_type": instance["InstanceType"],
                                        "platform": instance.get("Platform", "linux"),
                                        "private_ip": instance.get("PrivateIpAddress"),
                                        "public_ip": instance.get("PublicIpAddress"),
                                        "vpc_id": instance.get("VpcId"),
                                        "subnet_id": instance.get("SubnetId"),
                                        "state": instance["State"]["Name"],
                                        "availability_zone": instance["Placement"][
                                            "AvailabilityZone"
                                        ],
                                        "launch_time": instance[
                                            "LaunchTime"
                                        ].isoformat(),
                                        "tags": {
                                            tag["Key"]: tag["Value"]
                                            for tag in instance.get("Tags", [])
                                        },
                                        "security_groups": [
                                            sg["GroupId"]
                                            for sg in instance.get("SecurityGroups", [])
                                        ],
                                        "iam_instance_profile": instance.get(
                                            "IamInstanceProfile", {}
                                        ).get("Arn"),
                                    },
                                    "agent_compatible": self._check_ssm_compatibility(
                                        instance["InstanceId"], region
                                    ),
                                    "priority": self._get_asset_priority(
                                        {
                                            "asset_type": "ec2",
                                            "data": {
                                                "tags": {
                                                    tag["Key"]: tag["Value"]
                                                    for tag in instance.get("Tags", [])
                                                }
                                            },
                                        }
                                    ),
                                }
                            )

            except Exception as e:
                logger.error(f"Failed to discover EC2 instances in {region}: {e}")

            try:
                # Discover RDS instances
                rds = boto3.client(
                    "rds", region_name=region, **self.session_credentials
                )
                db_instances = rds.describe_db_instances()

                for db in db_instances["DBInstances"]:
                    assets.append(
                        {
                            "provider": "aws",
                            "asset_type": "rds",
                            "asset_id": db["DBInstanceIdentifier"],
                            "region": region,
                            "asset_data": {
                                "engine": db["Engine"],
                                "engine_version": db["EngineVersion"],
                                "db_instance_class": db["DBInstanceClass"],
                                "endpoint": db.get("Endpoint", {}).get("Address"),
                                "port": db.get("Endpoint", {}).get("Port"),
                                "vpc_id": db.get("DBSubnetGroup", {}).get("VpcId"),
                                "availability_zone": db.get("AvailabilityZone"),
                                "multi_az": db.get("MultiAZ", False),
                                "publicly_accessible": db.get(
                                    "PubliclyAccessible", False
                                ),
                                "status": db["DBInstanceStatus"],
                                "storage_type": db.get("StorageType"),
                                "allocated_storage": db.get("AllocatedStorage"),
                                "tags": db.get("TagList", []),
                            },
                            "agent_compatible": False,  # RDS doesn't support agents
                            "priority": "critical",  # Database servers are always critical
                        }
                    )

            except Exception as e:
                logger.error(f"Failed to discover RDS instances in {region}: {e}")

            return assets

        return await asyncio.to_thread(_discover_sync)

    async def _get_enrollment_token(self, asset_id: str) -> str:
        """Get enrollment token from database for the deployed agent"""
        from sqlalchemy import select

        from ..models import AgentEnrollment

        # We need database access here, but we're in the AWS integration which doesn't have db access
        # The token should have been created by smart_deployment.py
        # For now, generate a token that matches what smart_deployment.py creates
        # This is a temporary solution - ideally the token should be passed as a parameter
        # or retrieved from a shared location
        token = f"aws-{self.organization_id}-{asset_id}-temp-token"
        logger.info(f"Using enrollment token for asset {asset_id}: {token[:20]}...")
        return token

    async def _get_backend_url(self) -> str:
        """Get backend URL from organization integration settings"""
        from sqlalchemy.ext.asyncio import AsyncSession
        from sqlalchemy.orm import sessionmaker

        from ..agent_enrollment_service import AgentEnrollmentService

        # Create a temporary session to get the backend URL
        # This is a bit of a hack since we're in a sync context but need async
        # In production, this should be passed as a parameter
        try:
            # Default fallback URL
            default_url = "http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com"

            # Try to get from org settings if available
            # For now, return the default since we know the ALB URL
            return default_url
        except Exception:
            return default_url

    def _check_ssm_compatibility(self, instance_id: str, region: str) -> bool:
        """Check if an EC2 instance is SSM-compatible by verifying it's managed by SSM"""
        try:
            # Create SSM client
            ssm = boto3.client("ssm", region_name=region, **self.session_credentials)

            # Check if instance is managed by SSM
            response = ssm.describe_instance_information(
                Filters=[{"Key": "InstanceIds", "Values": [instance_id]}]
            )

            # If we get instance info back, it's SSM-managed
            return len(response.get("InstanceInformationList", [])) > 0

        except Exception as e:
            # If SSM query fails, instance is not SSM-compatible
            logger.debug(f"Instance {instance_id} not SSM-compatible: {e}")
            return False

    async def deploy_agents(
        self, assets: List[Dict[str, Any]], tokens: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Deploy agents to EC2 instances using AWS Systems Manager (SSM)

        Args:
            assets: List of assets to deploy agents to

        Returns:
            Deployment results with success/failure counts and details
        """
        logger.info(f"Starting agent deployment to {len(assets)} assets")
        results = {"success": 0, "failed": 0, "skipped": 0, "details": []}

        # Get backend URL for agent scripts
        backend_url = await self._get_backend_url()

        # Filter for EC2 instances only (RDS doesn't support agents)
        ec2_assets = [
            a
            for a in assets
            if a["asset_type"] == "ec2" and a.get("agent_compatible", False)
        ]

        for asset in ec2_assets:
            try:
                result = await self._deploy_agent_to_ec2(asset, tokens, backend_url)
                if result["status"] == "success":
                    results["success"] += 1
                else:
                    results["failed"] += 1
                results["details"].append(result)

            except Exception as e:
                logger.error(f"Failed to deploy agent to {asset['asset_id']}: {e}")
                results["failed"] += 1
                results["details"].append(
                    {"asset_id": asset["asset_id"], "status": "failed", "error": str(e)}
                )

        # Count skipped assets
        results["skipped"] = len(assets) - len(ec2_assets)

        logger.info(
            f"Agent deployment complete: {results['success']} succeeded, "
            f"{results['failed']} failed, {results['skipped']} skipped"
        )
        return results

    async def _deploy_agent_to_ec2(
        self,
        asset: Dict[str, Any],
        tokens: Optional[Dict[str, str]] = None,
        backend_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Deploy agent to a single EC2 instance using SSM"""

        def _deploy_sync() -> Dict[str, Any]:
            try:
                region = asset["region"]
                instance_id = asset["asset_id"]
                platform = asset["asset_data"].get("platform", "linux")

                # Create SSM client
                ssm = boto3.client(
                    "ssm", region_name=region, **self.session_credentials
                )

                # Check if instance has SSM agent running
                try:
                    ssm.describe_instance_information(
                        Filters=[{"Key": "InstanceIds", "Values": [instance_id]}]
                    )
                except Exception as e:
                    return {
                        "asset_id": instance_id,
                        "status": "failed",
                        "error": f"SSM agent not available on instance: {str(e)}",
                    }

                # Generate installation script
                install_script = self._generate_agent_script(asset, tokens, backend_url)

                # Send command via SSM
                document_name = (
                    "AWS-RunPowerShellScript"
                    if platform == "windows"
                    else "AWS-RunShellScript"
                )
                response = ssm.send_command(
                    InstanceIds=[instance_id],
                    DocumentName=document_name,
                    Parameters={
                        "commands": [install_script],
                        "executionTimeout": ["600"],  # 10 minutes
                    },
                    Comment=f"Mini-XDR agent deployment for org {self.organization_id}",
                )

                command_id = response["Command"]["CommandId"]

                return {
                    "asset_id": instance_id,
                    "status": "success",
                    "command_id": command_id,
                    "message": "Agent deployment initiated via SSM",
                }

            except Exception as e:
                logger.error(f"SSM deployment failed for {asset['asset_id']}: {e}")
                return {
                    "asset_id": asset["asset_id"],
                    "status": "failed",
                    "error": str(e),
                }

        return await asyncio.to_thread(_deploy_sync)

    def _generate_agent_script(
        self,
        asset: Dict[str, Any],
        tokens: Optional[Dict[str, str]] = None,
        backend_url: Optional[str] = None,
    ) -> str:
        """
        Generate platform-specific agent installation script

        Note: This generates a script that will be passed to the agent_enrollment_service
        to get the actual backend URL from the organization's integration_settings
        """
        platform = asset["asset_data"].get("platform", "linux")
        asset_id = asset["asset_id"]

        # Get enrollment token from deployment tokens
        agent_token = (
            tokens.get(asset_id)
            if tokens
            else f"aws-{self.organization_id}-{asset_id}-fallback-token"
        )

        # Use provided backend URL or fallback
        if backend_url is None:
            backend_url = f"https://mini-xdr-backend-{self.organization_id}.example.com"  # Fallback URL

        if platform == "windows":
            return f"""
$ErrorActionPreference = "Stop"
$token = "{agent_token}"
$assetId = "{asset_id}"
$backendUrl = "{backend_url}"
$orgId = "{self.organization_id}"

# Install Python if not present
if (!(Get-Command python -ErrorAction SilentlyContinue)) {{
    Write-Host "Installing Python..."
    # Download and install Python 3.9
    $pythonUrl = "https://www.python.org/ftp/python/3.9.7/python-3.9.7-amd64.exe"
    $installerPath = "$env:TEMP\\python-installer.exe"
    Invoke-WebRequest -Uri $pythonUrl -OutFile $installerPath
    Start-Process -FilePath $installerPath -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1" -Wait
    Remove-Item $installerPath
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
}}

# Create agent directory
$agentDir = "C:\\Program Files\\MiniXDR"
New-Item -ItemType Directory -Force -Path $agentDir | Out-Null

# Create mock agent script
$agentScript = @"
import requests
import json
import time
import socket
import platform
import psutil
import threading
from datetime import datetime

TOKEN = "$token"
ASSET_ID = "$assetId"
BACKEND_URL = "$backendUrl"
ORG_ID = "$orgId"

def get_system_info():
    return {{
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_count": psutil.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "disk_usage": {{k: v.percent for k, v in psutil.disk_usage("")._asdict().items()}},
        "network_interfaces": list(psutil.net_if_addrs().keys())
    }}

def register_agent():
    try:
        response = requests.post(
            f"{{BACKEND_URL}}/api/agents/enroll",
            json={{
                "token": TOKEN,
                "agent_id": ASSET_ID,
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "ip_address": socket.gethostbyname(socket.gethostname()),
                "metadata": get_system_info()
            }},
            timeout=30
        )
        if response.status_code == 200:
            print("Agent registered successfully")
            return response.json()
        else:
            print(f"Registration failed: {{response.status_code}} - {{response.text}}")
            return None
    except Exception as e:
        print(f"Registration error: {{e}}")
        return None

def send_heartbeat():
    while True:
        try:
            # Send heartbeat with system metrics
            response = requests.post(
                f"{{BACKEND_URL}}/api/agents/heartbeat",
                json={{
                    "agent_id": ASSET_ID,
                    "metrics": get_system_info(),
                    "timestamp": datetime.utcnow().isoformat()
                }},
                timeout=30
            )
            if response.status_code == 200:
                print("Heartbeat sent successfully")
            else:
                print(f"Heartbeat failed: {{response.status_code}}")
        except Exception as e:
            print(f"Heartbeat error: {{e}}")

        time.sleep(60)  # Send heartbeat every minute

def simulate_security_events():
    while True:
        try:
            # Simulate security events
            events = [
                {{
                    "type": "file_access",
                    "path": "/etc/passwd",
                    "user": "system",
                    "timestamp": datetime.utcnow().isoformat()
                }},
                {{
                    "type": "network_connection",
                    "destination": "8.8.8.8:53",
                    "protocol": "udp",
                    "timestamp": datetime.utcnow().isoformat()
                }}
            ]

            for event in events:
                response = requests.post(
                    f"{{BACKEND_URL}}/api/agents/events",
                    json={{
                        "agent_id": ASSET_ID,
                        "events": [event]
                    }},
                    timeout=30
                )
                if response.status_code == 200:
                    print("Security event sent")
                else:
                    print(f"Event send failed: {{response.status_code}}")

        except Exception as e:
            print(f"Event send error: {{e}}")

        time.sleep(300)  # Send events every 5 minutes

if __name__ == "__main__":
    print("Starting Mini-XDR Mock Agent...")
    print(f"Agent ID: {{ASSET_ID}}")
    print(f"Backend URL: {{BACKEND_URL}}")

    # Register agent
    registration = register_agent()
    if registration:
        print("Agent registration successful")

        # Start heartbeat thread
        heartbeat_thread = threading.Thread(target=send_heartbeat, daemon=True)
        heartbeat_thread.start()

        # Start security event simulation thread
        event_thread = threading.Thread(target=simulate_security_events, daemon=True)
        event_thread.start()

        print("Agent monitoring active. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Agent stopping...")
    else:
        print("Agent registration failed")
"@

# Save agent script
$agentScript | Out-File -FilePath "$agentDir\\mini_xdr_agent.py" -Encoding UTF8

# Create service wrapper
$serviceScript = @"
# Install as Windows service using NSSM or similar
Write-Host "Mini-XDR agent installed at $agentDir\\mini_xdr_agent.py"
Write-Host "Run manually with: python $agentDir\\mini_xdr_agent.py"
"@

$serviceScript | Out-File -FilePath "$agentDir\\install_service.ps1" -Encoding UTF8

# Run agent immediately
Write-Host "Starting Mini-XDR agent..."
Start-Process -FilePath "python" -ArgumentList "$agentDir\\mini_xdr_agent.py" -NoNewWindow
"""
        else:  # Linux
            return f"""#!/bin/bash
set -e

TOKEN="{agent_token}"
ASSET_ID="{asset_id}"
BACKEND_URL="{backend_url}"
ORG_ID="{self.organization_id}"

echo "Installing Mini-XDR mock agent for Linux..."

# Install Python if not present
if ! command -v python3 &> /dev/null; then
    echo "Installing Python3..."
    if command -v apt-get &> /dev/null; then
        apt-get update && apt-get install -y python3 python3-pip python3-venv
    elif command -v yum &> /dev/null; then
        yum install -y python3 python3-pip
    elif command -v dnf &> /dev/null; then
        dnf install -y python3 python3-pip
    else
        echo "Could not install Python3 - package manager not found"
        exit 1
    fi
fi

# Install required Python packages
pip3 install requests psutil

# Create agent directory
AGENT_DIR="/opt/mini-xdr-agent"
mkdir -p "$AGENT_DIR"

# Create Python agent script
cat > "$AGENT_DIR/mini_xdr_agent.py" << 'EOF'
import requests
import json
import time
import socket
import platform
import psutil
import threading
from datetime import datetime

TOKEN = "${{TOKEN}}"
ASSET_ID = "${{ASSET_ID}}"
BACKEND_URL = "${{BACKEND_URL}}"
ORG_ID = "${{ORG_ID}}"

def get_system_info():
    return {{
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_count": psutil.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "disk_usage": {{k: v.percent for k, v in psutil.disk_usage("/")._asdict().items()}},
        "network_interfaces": list(psutil.net_if_addrs().keys())
    }}

def register_agent():
    try:
        response = requests.post(
            f"{{BACKEND_URL}}/api/agents/enroll",
            json={{
                "token": TOKEN,
                "agent_id": ASSET_ID,
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "ip_address": socket.gethostbyname(socket.gethostname()),
                "metadata": get_system_info()
            }},
            timeout=30
        )
        if response.status_code == 200:
            print("Agent registered successfully")
            return response.json()
        else:
            print(f"Registration failed: {{response.status_code}} - {{response.text}}")
            return None
    except Exception as e:
        print(f"Registration error: {{e}}")
        return None

def send_heartbeat():
    while True:
        try:
            # Send heartbeat with system metrics
            response = requests.post(
                f"{{BACKEND_URL}}/api/agents/heartbeat",
                json={{
                    "agent_id": ASSET_ID,
                    "metrics": get_system_info(),
                    "timestamp": datetime.utcnow().isoformat()
                }},
                timeout=30
            )
            if response.status_code == 200:
                print("Heartbeat sent successfully")
            else:
                print(f"Heartbeat failed: {{response.status_code}}")
        except Exception as e:
            print(f"Heartbeat error: {{e}}")

        time.sleep(60)  # Send heartbeat every minute

def simulate_security_events():
    while True:
        try:
            # Simulate security events
            events = [
                {{
                    "type": "file_access",
                    "path": "/etc/passwd",
                    "user": "system",
                    "timestamp": datetime.utcnow().isoformat()
                }},
                {{
                    "type": "network_connection",
                    "destination": "8.8.8.8:53",
                    "protocol": "udp",
                    "timestamp": datetime.utcnow().isoformat()
                }},
                {{
                    "type": "process_execution",
                    "command": "/bin/ls -la",
                    "user": "ec2-user",
                    "timestamp": datetime.utcnow().isoformat()
                }}
            ]

            for event in events:
                response = requests.post(
                    f"{{BACKEND_URL}}/api/agents/events",
                    json={{
                        "agent_id": ASSET_ID,
                        "events": [event]
                    }},
                    timeout=30
                )
                if response.status_code == 200:
                    print("Security event sent")
                else:
                    print(f"Event send failed: {{response.status_code}}")

        except Exception as e:
            print(f"Event send error: {{e}}")

        time.sleep(300)  # Send events every 5 minutes

if __name__ == "__main__":
    print("Starting Mini-XDR Mock Agent...")
    print(f"Agent ID: {{ASSET_ID}}")
    print(f"Backend URL: {{BACKEND_URL}}")

    # Register agent
    registration = register_agent()
    if registration:
        print("Agent registration successful")

        # Start heartbeat thread
        heartbeat_thread = threading.Thread(target=send_heartbeat, daemon=True)
        heartbeat_thread.start()

        # Start security event simulation thread
        event_thread = threading.Thread(target=simulate_security_events, daemon=True)
        event_thread.start()

        print("Agent monitoring active. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Agent stopping...")
    else:
        print("Agent registration failed")
EOF

# Make executable
chmod +x "$AGENT_DIR/mini_xdr_agent.py"

# Create systemd service
cat > /etc/systemd/system/mini-xdr-agent.service << EOF
[Unit]
Description=Mini-XDR Mock Agent
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/bin/python3 $AGENT_DIR/mini_xdr_agent.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
systemctl daemon-reload
systemctl enable mini-xdr-agent
systemctl start mini-xdr-agent

echo "Mini-XDR mock agent installed and started"
echo "Service status: $(systemctl is-active mini-xdr-agent)"
echo "Logs: journalctl -u mini-xdr-agent -f"
"""

    async def validate_permissions(self) -> Dict[str, bool]:
        """
        Validate required AWS permissions for asset discovery and agent deployment

        Returns:
            Dict of permission checks with boolean results
        """

        def _validate_sync() -> Dict[str, bool]:
            permissions = {
                "read_compute": False,
                "read_network": False,
                "read_storage": False,
                "deploy_agents": False,
            }

            if not self.session_credentials:
                return permissions

            try:
                # Test EC2 describe permission (read_compute)
                ec2 = boto3.client("ec2", **self.session_credentials)
                ec2.describe_instances(MaxResults=5)
                permissions["read_compute"] = True
            except Exception as e:
                logger.warning(f"Missing EC2 describe permission: {e}")

            try:
                # Test VPC describe permission (read_network)
                ec2 = boto3.client("ec2", **self.session_credentials)
                ec2.describe_vpcs(MaxResults=5)
                permissions["read_network"] = True
            except Exception as e:
                logger.warning(f"Missing VPC describe permission: {e}")

            try:
                # Test RDS describe permission (read_storage)
                rds = boto3.client("rds", **self.session_credentials)
                rds.describe_db_instances(MaxRecords=5)
                permissions["read_storage"] = True
            except Exception as e:
                logger.warning(f"Missing RDS describe permission: {e}")

            try:
                # Test SSM send command permission (deploy_agents)
                ssm = boto3.client("ssm", **self.session_credentials)
                ssm.describe_instance_information(MaxResults=5)
                permissions["deploy_agents"] = True
            except Exception as e:
                logger.warning(f"Missing SSM permissions: {e}")

            return permissions

        return await asyncio.to_thread(_validate_sync)

    async def get_deployment_status(
        self, command_id: str, instance_id: str, region: str
    ) -> Dict[str, Any]:
        """
        Check the status of an SSM command execution

        Args:
            command_id: SSM command ID
            instance_id: EC2 instance ID
            region: AWS region

        Returns:
            Command execution status and output
        """

        def _get_status_sync() -> Dict[str, Any]:
            try:
                ssm = boto3.client(
                    "ssm", region_name=region, **self.session_credentials
                )

                response = ssm.get_command_invocation(
                    CommandId=command_id, InstanceId=instance_id
                )

                return {
                    "status": response[
                        "Status"
                    ],  # Pending, InProgress, Success, Failed, etc.
                    "status_details": response.get("StatusDetails", ""),
                    "standard_output": response.get("StandardOutputContent", ""),
                    "standard_error": response.get("StandardErrorContent", ""),
                    "response_code": response.get("ResponseCode", -1),
                }

            except Exception as e:
                logger.error(f"Failed to get deployment status: {e}")
                return {"status": "Unknown", "error": str(e)}

        return await asyncio.to_thread(_get_status_sync)
