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
                                    "agent_compatible": instance["State"]["Name"]
                                    == "running",
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
                            "agent_compatible": False,  # RDS instances don't support agent deployment
                            "priority": "critical",  # Database servers are always critical
                        }
                    )

            except Exception as e:
                logger.error(f"Failed to discover RDS instances in {region}: {e}")

            return assets

        return await asyncio.to_thread(_discover_sync)

    async def deploy_agents(self, assets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Deploy agents to EC2 instances using AWS Systems Manager (SSM)

        Args:
            assets: List of assets to deploy agents to

        Returns:
            Deployment results with success/failure counts and details
        """
        logger.info(f"Starting agent deployment to {len(assets)} assets")
        results = {"success": 0, "failed": 0, "skipped": 0, "details": []}

        # Filter for EC2 instances only (RDS doesn't support agents)
        ec2_assets = [
            a
            for a in assets
            if a["asset_type"] == "ec2" and a.get("agent_compatible", False)
        ]

        for asset in ec2_assets:
            try:
                result = await self._deploy_agent_to_ec2(asset)
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
            f"Agent deployment complete: {results['success']} succeeded, {results['failed']} failed, {results['skipped']} skipped"
        )
        return results

    async def _deploy_agent_to_ec2(self, asset: Dict[str, Any]) -> Dict[str, Any]:
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
                install_script = self._generate_agent_script(asset)

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

    def _generate_agent_script(self, asset: Dict[str, Any]) -> str:
        """
        Generate platform-specific agent installation script

        Note: This generates a script that will be passed to the agent_enrollment_service
        to get the actual backend URL from the organization's integration_settings
        """
        platform = asset["asset_data"].get("platform", "linux")
        asset_id = asset["asset_id"]

        # Generate a unique agent token
        agent_token = (
            f"aws-{self.organization_id}-{asset_id}-{secrets.token_urlsafe(16)}"
        )

        if platform == "windows":
            return f"""
$ErrorActionPreference = "Stop"
$token = "{agent_token}"
$assetId = "{asset_id}"

# Download and install Mini-XDR agent
# Note: The backend URL will be dynamically resolved from org settings
Write-Host "Downloading Mini-XDR agent for Windows..."
# TODO: Implement Windows agent download and installation
Write-Host "Agent token: $token"
Write-Host "Asset ID: $assetId"
"""
        else:  # Linux
            return f"""#!/bin/bash
set -e

TOKEN="{agent_token}"
ASSET_ID="{asset_id}"
ORG_ID="{self.organization_id}"

echo "Installing Mini-XDR agent for Linux..."
# TODO: Implement Linux agent download and installation
# The agent will use the token to enroll and get the backend URL from the enrollment service
echo "Agent token: $TOKEN"
echo "Asset ID: $ASSET_ID"
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
