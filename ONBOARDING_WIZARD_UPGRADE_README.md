# Mini-XDR Seamless Onboarding Upgrade

## Executive Summary

**Replace Mini-XDR's legacy 4-step onboarding wizard with a seamless 5-minute, one-click experience** that reduces time-to-value from hours to minutes while maintaining enterprise security and multi-tenancy.

### Current State → Future State

| Aspect | Legacy (4-step wizard) | New (Seamless Onboarding) |
|--------|-------------------------|---------------------------|
| **Time to First Alert** | 2-4 hours | <5 minutes |
| **User Steps** | 10+ manual steps | 1-2 clicks |
| **Technical Knowledge** | High (network ranges, agent deployment) | None required |
| **Completion Rate** | ~70% | Target >95% |
| **Support Burden** | High | Minimal |
| **Mini Corp Status** | `not_started` | Ready for deployment |

---

## 🎯 **Replacement Strategy**

**This upgrade COMPLETELY REPLACES the legacy 4-step onboarding wizard** with the new seamless experience. There is no gradual rollout or A/B testing - the new system becomes the default and only onboarding method.

### **Why Complete Replacement?**
- **Legacy system is fundamentally flawed**: Complex, error-prone, high support burden
- **New system is superior**: Cloud-native, automated, user-friendly
- **Mini Corp as pilot**: Test with Mini Corp organization first, then roll to all users
- **No migration needed**: New organizations start with new system, legacy data is archived

### **What Gets Replaced**
- ❌ Legacy 4-step wizard (`/onboarding` routes)
- ❌ Manual network scanning and agent deployment
- ❌ Complex credential management
- ❌ Multi-step validation process

### **What's New**
- ✅ One-click cloud integration
- ✅ Automatic asset discovery and classification
- ✅ Smart agent deployment and configuration
- ✅ Real-time progress monitoring
- ✅ Instant security monitoring

---

## 🏗️ **Implementation Architecture**

### Core Components

#### **New Database Models**

```sql
-- New integration credentials table
CREATE TABLE integration_credentials (
    id SERIAL PRIMARY KEY,
    organization_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    provider VARCHAR(50) NOT NULL, -- aws, azure, gcp, crowdstrike, etc.
    credential_type VARCHAR(50) NOT NULL, -- oauth, api_key, service_account
    credential_data JSONB NOT NULL, -- Encrypted credential data
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    last_used_at TIMESTAMPTZ,
    UNIQUE(organization_id, provider)
);

-- Enhanced onboarding tracking
ALTER TABLE organizations ADD COLUMN IF NOT EXISTS
    onboarding_flow_version VARCHAR(20) DEFAULT 'legacy';

ALTER TABLE organizations ADD COLUMN IF NOT EXISTS
    auto_discovery_enabled BOOLEAN DEFAULT true;

ALTER TABLE organizations ADD COLUMN IF NOT EXISTS
    integration_settings JSONB DEFAULT '{}';

-- New cloud asset discovery table
CREATE TABLE cloud_assets (
    id SERIAL PRIMARY KEY,
    organization_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    provider VARCHAR(50) NOT NULL,
    asset_type VARCHAR(100) NOT NULL, -- ec2, vm, storage, etc.
    asset_id VARCHAR(255) NOT NULL,
    asset_data JSONB NOT NULL,
    region VARCHAR(50),
    discovered_at TIMESTAMPTZ DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ DEFAULT NOW(),
    agent_deployed BOOLEAN DEFAULT false,
    agent_status VARCHAR(20) DEFAULT 'pending',
    tags JSONB DEFAULT '{}',
    UNIQUE(organization_id, provider, asset_id)
);
```

#### **New Backend Services**

```
backend/app/
├── integrations/                    # NEW: Integration management
│   ├── __init__.py
│   ├── base.py                     # Base integration class
│   ├── aws.py                      # AWS integration
│   ├── azure.py                    # Azure integration (placeholder)
│   ├── gcp.py                      # GCP integration (placeholder)
│   ├── crowdstrike.py              # Crowdstrike integration (placeholder)
│   └── manager.py                  # Integration manager
├── onboarding_v2/                  # NEW: Seamless onboarding
│   ├── __init__.py
│   ├── auto_discovery.py           # Auto-discovery engine
│   ├── smart_deployment.py         # Smart agent deployment
│   ├── validation.py               # Onboarding validation
│   └── routes.py                   # New onboarding API
├── cloud_providers/               # NEW: Cloud provider clients
│   ├── aws_client.py               # AWS API client
│   ├── azure_client.py             # Azure API client (placeholder)
│   └── gcp_client.py               # GCP API client (placeholder)
└── models.py                       # Updated with new models
```

---

## 🔧 **Backend Implementation**

### **1. New Integration Base Classes**

```python
# backend/app/integrations/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

class CloudIntegration(ABC):
    """Base class for cloud provider integrations"""

    def __init__(self, organization_id: int, credentials: Dict[str, Any]):
        self.organization_id = organization_id
        self.credentials = credentials
        self.client = None

    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate with the cloud provider"""
        pass

    @abstractmethod
    async def discover_assets(self) -> List[Dict[str, Any]]:
        """Discover all assets in the cloud environment"""
        pass

    @abstractmethod
    async def deploy_agents(self, assets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Deploy agents to discovered assets"""
        pass

    @abstractmethod
    async def get_regions(self) -> List[str]:
        """Get available regions"""
        pass

    async def validate_permissions(self) -> Dict[str, bool]:
        """Validate required permissions"""
        return {
            "read_compute": False,
            "read_network": False,
            "read_storage": False,
            "deploy_agents": False
        }
```

### **2. AWS Integration Implementation**

```python
# backend/app/integrations/aws.py
import boto3
import botocore.exceptions
from typing import Dict, List, Optional, Any
from .base import CloudIntegration

class AWSIntegration(CloudIntegration):
    """AWS cloud integration for seamless onboarding"""

    def __init__(self, organization_id: int, credentials: Dict[str, Any]):
        super().__init__(organization_id, credentials)
        self.regions = []

    async def authenticate(self) -> bool:
        """Authenticate using AWS STS AssumeRole or direct credentials"""
        try:
            # Try STS AssumeRole first (recommended)
            if 'role_arn' in self.credentials:
                sts_client = boto3.client(
                    'sts',
                    aws_access_key_id=self.credentials.get('access_key_id'),
                    aws_secret_access_key=self.credentials.get('secret_access_key')
                )

                assumed_role = sts_client.assume_role(
                    RoleArn=self.credentials['role_arn'],
                    RoleSessionName=f'mini-xdr-onboarding-{self.organization_id}'
                )

                self.credentials = {
                    'aws_access_key_id': assumed_role['Credentials']['AccessKeyId'],
                    'aws_secret_access_key': assumed_role['Credentials']['SecretAccessKey'],
                    'aws_session_token': assumed_role['Credentials']['SessionToken']
                }

            # Test authentication
            ec2 = boto3.client(
                'ec2',
                aws_access_key_id=self.credentials['aws_access_key_id'],
                aws_secret_access_key=self.credentials['aws_secret_access_key'],
                aws_session_token=self.credentials.get('aws_session_token')
            )

            # Get available regions
            regions_response = ec2.describe_regions()
            self.regions = [r['RegionName'] for r in regions_response['Regions']]

            return True

        except Exception as e:
            logger.error(f"AWS authentication failed: {e}")
            return False

    async def discover_assets(self) -> List[Dict[str, Any]]:
        """Discover EC2 instances, RDS databases, Lambda functions, etc."""
        assets = []

        for region in self.regions:
            try:
                # Discover EC2 instances
                ec2 = boto3.client(
                    'ec2',
                    region_name=region,
                    aws_access_key_id=self.credentials['aws_access_key_id'],
                    aws_secret_access_key=self.credentials['aws_secret_access_key'],
                    aws_session_token=self.credentials.get('aws_session_token')
                )

                instances = ec2.describe_instances()
                for reservation in instances['Reservations']:
                    for instance in reservation['Instances']:
                        if instance['State']['Name'] == 'running':
                            assets.append({
                                'provider': 'aws',
                                'asset_type': 'ec2',
                                'asset_id': instance['InstanceId'],
                                'region': region,
                                'data': {
                                    'instance_type': instance['InstanceType'],
                                    'platform': instance.get('Platform', 'linux'),
                                    'private_ip': instance.get('PrivateIpAddress'),
                                    'public_ip': instance.get('PublicIpAddress'),
                                    'tags': {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])},
                                    'security_groups': [sg['GroupId'] for sg in instance.get('SecurityGroups', [])]
                                },
                                'agent_compatible': True,
                                'priority': 'high'
                            })

                # Discover RDS instances
                rds = boto3.client(
                    'rds',
                    region_name=region,
                    aws_access_key_id=self.credentials['aws_access_key_id'],
                    aws_secret_access_key=self.credentials['aws_secret_access_key'],
                    aws_session_token=self.credentials.get('aws_session_token')
                )

                db_instances = rds.describe_db_instances()
                for db in db_instances['DBInstances']:
                    assets.append({
                        'provider': 'aws',
                        'asset_type': 'rds',
                        'asset_id': db['DBInstanceIdentifier'],
                        'region': region,
                        'data': {
                            'engine': db['Engine'],
                            'engine_version': db['EngineVersion'],
                            'db_instance_class': db['DBInstanceClass'],
                            'endpoint': db['Endpoint']['Address']
                        },
                        'agent_compatible': True,
                        'priority': 'critical'
                    })

            except Exception as e:
                logger.warning(f"Failed to discover assets in region {region}: {e}")
                continue

        return assets

    async def deploy_agents(self, assets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Deploy agents using AWS Systems Manager"""
        results = {'success': 0, 'failed': 0, 'details': []}

        for asset in assets:
            try:
                if asset['asset_type'] == 'ec2':
                    # Use SSM to deploy agent
                    ssm = boto3.client(
                        'ssm',
                        region_name=asset['region'],
                        aws_access_key_id=self.credentials['aws_access_key_id'],
                        aws_secret_access_key=self.credentials['aws_secret_access_key'],
                        aws_session_token=self.credentials.get('aws_session_token')
                    )

                    # Generate agent install script
                    install_script = self._generate_agent_script(asset)

                    # Send command via SSM
                    response = ssm.send_command(
                        InstanceIds=[asset['asset_id']],
                        DocumentName='AWS-RunShellScript',
                        Parameters={
                            'commands': [install_script],
                            'executionTimeout': ['300']  # 5 minutes
                        }
                    )

                    results['success'] += 1
                    results['details'].append({
                        'asset_id': asset['asset_id'],
                        'status': 'deployed',
                        'command_id': response['Command']['CommandId']
                    })

            except Exception as e:
                results['failed'] += 1
                results['details'].append({
                    'asset_id': asset['asset_id'],
                    'status': 'failed',
                    'error': str(e)
                })

        return results

    def _generate_agent_script(self, asset: Dict[str, Any]) -> str:
        """Generate platform-specific agent installation script"""
        platform = asset['data'].get('platform', 'linux')
        backend_url = "http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com"

        if platform == 'windows':
            return f'''
            powershell -Command "& {{
                $token = '{self._generate_agent_token(asset)}'
                Invoke-WebRequest -Uri '{backend_url}/api/agents/download/windows' -OutFile 'C:\\Temp\\minixdr-agent.msi'
                Start-Process 'msiexec.exe' -ArgumentList '/i C:\\Temp\\minixdr-agent.msi TOKEN=$token /quiet' -Wait
            }}"
            '''
        else:  # Linux
            return f'''
            #!/bin/bash
            TOKEN="{self._generate_agent_token(asset)}"
            curl -fsSL {backend_url}/api/agents/download/linux -o /tmp/minixdr-agent
            chmod +x /tmp/minixdr-agent
            sudo /tmp/minixdr-agent --token $TOKEN --backend {backend_url}
            '''

    def _generate_agent_token(self, asset: Dict[str, Any]) -> str:
        """Generate secure agent token for this asset"""
        # Implementation would integrate with AgentEnrollmentService
        return f"aws-{self.organization_id}-{asset['asset_id'][:8]}-{secrets.token_urlsafe(16)}"
```

### **3. New Onboarding API Routes**

```python
# backend/app/onboarding_v2/routes.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, List, Optional, Any
import logging

from ..db import get_db
from ..auth import get_current_user
from ..models import User, Organization
from .auto_discovery import AutoDiscoveryEngine
from .smart_deployment import SmartDeploymentEngine
from .validation import OnboardingValidator
from ..integrations.manager import IntegrationManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/onboarding/v2", tags=["Onboarding V2"])

@router.post("/quick-start")
async def quick_start_onboarding(
    provider: str,  # aws, azure, gcp
    credentials: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """One-click onboarding with cloud integration"""

    # Get organization
    org = await get_organization(current_user, db)

    # Initialize integration manager
    integration_mgr = IntegrationManager(org.id, db)

    # Validate and store credentials
    success = await integration_mgr.setup_integration(provider, credentials)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to authenticate with cloud provider")

    # Start auto-discovery in background
    background_tasks.add_task(
        auto_discover_and_deploy,
        org.id,
        provider,
        db
    )

    return {
        "status": "initiated",
        "message": f"Auto-discovery started for {provider}. This may take a few minutes.",
        "estimated_completion": "5-10 minutes"
    }

@router.get("/progress")
async def get_onboarding_progress(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get real-time onboarding progress"""

    org = await get_organization(current_user, db)

    # Get discovery progress
    discovery_engine = AutoDiscoveryEngine(org.id, db)
    discovery_status = await discovery_engine.get_status()

    # Get deployment progress
    deployment_engine = SmartDeploymentEngine(org.id, db)
    deployment_status = await deployment_engine.get_status()

    # Get validation status
    validator = OnboardingValidator(org.id, db)
    validation_status = await validator.get_status()

    return {
        "discovery": discovery_status,
        "deployment": deployment_status,
        "validation": validation_status,
        "overall_progress": calculate_overall_progress([
            discovery_status, deployment_status, validation_status
        ])
    }

async def auto_discover_and_deploy(org_id: int, provider: str, db: AsyncSession):
    """Background task for auto-discovery and deployment"""

    try:
        # Step 1: Auto-discovery
        discovery_engine = AutoDiscoveryEngine(org_id, db)
        assets = await discovery_engine.discover_cloud_assets(provider)

        # Step 2: Smart deployment
        deployment_engine = SmartDeploymentEngine(org_id, db)
        await deployment_engine.deploy_to_assets(assets, provider)

        # Step 3: Validation
        validator = OnboardingValidator(org_id, db)
        await validator.validate_deployment()

        # Update organization status
        await update_org_onboarding_status(org_id, "completed", db)

        logger.info(f"Auto-onboarding completed for org {org_id}")

    except Exception as e:
        logger.error(f"Auto-onboarding failed for org {org_id}: {e}")
        await update_org_onboarding_status(org_id, "failed", db)
```

### **4. Integration Manager**

```python
# backend/app/integrations/manager.py
from typing import Dict, List, Optional, Any
import logging
from sqlalchemy.ext.asyncio import AsyncSession

from .aws import AWSIntegration
from .azure import AzureIntegration  # Placeholder
from .gcp import GCPIntegration      # Placeholder
from ..models import IntegrationCredentials

logger = logging.getLogger(__name__)

class IntegrationManager:
    """Manages cloud provider integrations"""

    def __init__(self, organization_id: int, db: AsyncSession):
        self.organization_id = organization_id
        self.db = db
        self.integrations = {
            'aws': AWSIntegration,
            'azure': AzureIntegration,
            'gcp': GCPIntegration
        }

    async def setup_integration(self, provider: str, credentials: Dict[str, Any]) -> bool:
        """Setup and validate integration credentials"""

        if provider not in self.integrations:
            return False

        # Create integration instance
        integration_class = self.integrations[provider]
        integration = integration_class(self.organization_id, credentials)

        # Test authentication
        if not await integration.authenticate():
            return False

        # Validate permissions
        permissions = await integration.validate_permissions()
        if not all(permissions.values()):
            logger.warning(f"Insufficient permissions for {provider}: {permissions}")
            return False

        # Store encrypted credentials
        await self._store_credentials(provider, credentials)

        return True

    async def get_integration(self, provider: str):
        """Get configured integration instance"""

        credentials = await self._get_credentials(provider)
        if not credentials:
            return None

        integration_class = self.integrations[provider]
        return integration_class(self.organization_id, credentials)

    async def list_integrations(self) -> List[Dict[str, Any]]:
        """List all configured integrations"""

        integrations = []
        for provider in self.integrations.keys():
            credentials = await self._get_credentials(provider)
            if credentials:
                status = await self._test_integration(provider)
                integrations.append({
                    'provider': provider,
                    'status': status,
                    'configured_at': credentials.created_at.isoformat()
                })

        return integrations

    async def _store_credentials(self, provider: str, credentials: Dict[str, Any]):
        """Store encrypted credentials in database"""

        # Encrypt sensitive data
        encrypted_data = await self._encrypt_credentials(credentials)

        # Store in database
        cred_record = IntegrationCredentials(
            organization_id=self.organization_id,
            provider=provider,
            credential_type='oauth',  # or api_key, service_account
            credential_data=encrypted_data,
            status='active'
        )

        self.db.add(cred_record)
        await self.db.commit()

    async def _get_credentials(self, provider: str) -> Optional[Dict[str, Any]]:
        """Retrieve and decrypt credentials"""

        # Query database
        # Return decrypted credentials
        pass

    async def _encrypt_credentials(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive credential data"""
        # Use organization's encryption key
        pass

    async def _test_integration(self, provider: str) -> str:
        """Test integration connectivity"""
        try:
            integration = await self.get_integration(provider)
            if await integration.authenticate():
                return 'connected'
            else:
                return 'error'
        except Exception:
            return 'error'
```

---

## 🎨 **Frontend Implementation**

### **1. New Onboarding Components**

```tsx
// frontend/app/components/onboarding/QuickStartOnboarding.tsx
"use client";

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { CheckCircle, Loader2, Cloud, Shield } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';

interface OnboardingProgress {
  discovery: { status: string; progress: number; assets_found: number };
  deployment: { status: string; progress: number; agents_deployed: number };
  validation: { status: string; progress: number; checks_passed: number };
  overall_progress: number;
}

export function QuickStartOnboarding() {
  const [selectedProvider, setSelectedProvider] = useState<string | null>(null);
  const [isConnecting, setIsConnecting] = useState(false);
  const [progress, setProgress] = useState<OnboardingProgress | null>(null);
  const { user } = useAuth();

  const handleCloudConnect = async (provider: string) => {
    setIsConnecting(true);
    setSelectedProvider(provider);

    try {
      // OAuth flow for cloud provider
      const authResult = await initiateCloudAuth(provider);

      // Start onboarding
      const response = await fetch('/api/onboarding/v2/quick-start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider,
          credentials: authResult
        })
      });

      if (response.ok) {
        // Start polling for progress
        startProgressPolling();
      }
    } catch (error) {
      console.error('Onboarding failed:', error);
    } finally {
      setIsConnecting(false);
    }
  };

  const startProgressPolling = () => {
    const poll = async () => {
      try {
        const response = await fetch('/api/onboarding/v2/progress');
        const progressData = await response.json();
        setProgress(progressData);

        if (progressData.overall_progress < 100) {
          setTimeout(poll, 2000); // Poll every 2 seconds
        }
      } catch (error) {
        console.error('Progress polling failed:', error);
      }
    };

    poll();
  };

  if (progress) {
    return <OnboardingProgress progress={progress} />;
  }

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">
          Welcome to Mini-XDR
        </h1>
        <p className="text-lg text-gray-600 mb-6">
          Get started in minutes with seamless cloud integration
        </p>
      </div>

      <div className="grid md:grid-cols-3 gap-6">
        {/* AWS Integration */}
        <Card className="cursor-pointer hover:shadow-lg transition-shadow">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Cloud className="h-5 w-5 text-orange-500" />
              Amazon Web Services
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-gray-600 mb-4">
              Automatically discover EC2 instances, RDS databases, and more across all regions
            </p>
            <Button
              onClick={() => handleCloudConnect('aws')}
              disabled={isConnecting}
              className="w-full"
            >
              {isConnecting && selectedProvider === 'aws' ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Connecting...
                </>
              ) : (
                'Connect AWS Account'
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Future Integrations - Coming Soon */}
        <div className="mt-8 text-center col-span-2">
          <p className="text-sm text-gray-500">
            🔄 Azure and GCP integrations coming soon
          </p>
        </div>
      </div>

      {/* No legacy fallback - this is the only onboarding method */}
    </div>
  );
}
```

### **2. Progress Monitoring Component**

```tsx
// frontend/app/components/onboarding/OnboardingProgress.tsx
"use client";

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { CheckCircle, Loader2, Server, Shield, Check } from 'lucide-react';

interface OnboardingProgressProps {
  progress: {
    discovery: { status: string; progress: number; assets_found: number };
    deployment: { status: string; progress: number; agents_deployed: number };
    validation: { status: string; progress: number; checks_passed: number };
    overall_progress: number;
  };
}

export function OnboardingProgress({ progress }: OnboardingProgressProps) {
  const steps = [
    {
      id: 'discovery',
      title: 'Discovering Assets',
      icon: Server,
      data: progress.discovery,
      description: `${progress.discovery.assets_found} assets found`
    },
    {
      id: 'deployment',
      title: 'Deploying Agents',
      icon: Shield,
      data: progress.deployment,
      description: `${progress.deployment.agents_deployed} agents deployed`
    },
    {
      id: 'validation',
      title: 'Validating Setup',
      icon: Check,
      data: progress.validation,
      description: `${progress.validation.checks_passed} checks passed`
    }
  ];

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">
          Setting up Mini-XDR
        </h1>
        <div className="w-full bg-gray-200 rounded-full h-2 mb-4">
          <div
            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
            style={{ width: `${progress.overall_progress}%` }}
          />
        </div>
        <p className="text-lg text-gray-600">
          {progress.overall_progress}% Complete
        </p>
      </div>

      <div className="grid gap-6">
        {steps.map((step) => {
          const Icon = step.icon;
          const isCompleted = step.data.status === 'completed';
          const isInProgress = step.data.status === 'in_progress';

          return (
            <Card key={step.id}>
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  {isCompleted ? (
                    <CheckCircle className="h-6 w-6 text-green-500" />
                  ) : isInProgress ? (
                    <Loader2 className="h-6 w-6 text-blue-500 animate-spin" />
                  ) : (
                    <Icon className="h-6 w-6 text-gray-400" />
                  )}
                  {step.title}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-600">{step.description}</span>
                  <span className="text-sm font-medium">{step.data.progress}%</span>
                </div>
                <Progress value={step.data.progress} className="w-full" />
              </CardContent>
            </Card>
          );
        })}
      </div>

      {progress.overall_progress === 100 && (
        <div className="mt-8 text-center">
          <div className="bg-green-50 border border-green-200 rounded-lg p-6">
            <CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-4" />
            <h2 className="text-xl font-semibold text-green-900 mb-2">
              Setup Complete!
            </h2>
            <p className="text-green-700 mb-4">
              Mini-XDR is now monitoring your environment and will alert you to security threats.
            </p>
            <Button onClick={() => window.location.href = '/incidents'}>
              View Dashboard
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
```

### **3. Admin Integration Management**

**Note**: This replaces all legacy integration management. Admin users can access `/settings` → "Cloud Integrations" to manage connected cloud providers, reconfigure credentials, and monitor integration health.

```tsx
// frontend/app/settings/integrations/page.tsx
"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Cloud, Plus, Settings, Trash2 } from 'lucide-react';

interface Integration {
  provider: string;
  status: 'connected' | 'disconnected' | 'error';
  configured_at: string;
  regions?: string[];
  asset_count?: number;
}

export default function IntegrationsSettings() {
  const [integrations, setIntegrations] = useState<Integration[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadIntegrations();
  }, []);

  const loadIntegrations = async () => {
    try {
      const response = await fetch('/api/integrations');
      const data = await response.json();
      setIntegrations(data);
    } catch (error) {
      console.error('Failed to load integrations:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleConnect = async (provider: string) => {
    try {
      const authResult = await initiateCloudAuth(provider);
      await fetch('/api/integrations/setup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider, credentials: authResult })
      });
      await loadIntegrations();
    } catch (error) {
      console.error('Failed to connect integration:', error);
    }
  };

  const handleDisconnect = async (provider: string) => {
    if (confirm(`Disconnect ${provider} integration?`)) {
      try {
        await fetch(`/api/integrations/${provider}`, { method: 'DELETE' });
        await loadIntegrations();
      } catch (error) {
        console.error('Failed to disconnect integration:', error);
      }
    }
  };

  const getStatusBadge = (status: string) => {
    const variants = {
      connected: 'bg-green-100 text-green-800',
      disconnected: 'bg-gray-100 text-gray-800',
      error: 'bg-red-100 text-red-800'
    };
    return <Badge className={variants[status as keyof typeof variants] || variants.disconnected}>{status}</Badge>;
  };

  const getProviderIcon = (provider: string) => {
    return <Cloud className="h-5 w-5" />;
  };

  const availableProviders = [
    { id: 'aws', name: 'Amazon Web Services', description: 'EC2, RDS, Lambda, and more' },
    { id: 'azure', name: 'Microsoft Azure', description: 'VMs, AKS, Azure resources', disabled: true },
    { id: 'gcp', name: 'Google Cloud Platform', description: 'GCE, GKE, Cloud assets', disabled: true }
  ];

  return (
    <div className="max-w-6xl mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900 mb-2">Cloud Integrations</h1>
        <p className="text-gray-600">
          Connect cloud providers for automatic asset discovery and agent deployment
        </p>
      </div>

      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
        {availableProviders.map((provider) => {
          const existing = integrations.find(i => i.provider === provider.id);

          return (
            <Card key={provider.id}>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  {getProviderIcon(provider.id)}
                  {provider.name}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-gray-600 mb-4">{provider.description}</p>

                {existing ? (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Status:</span>
                      {getStatusBadge(existing.status)}
                    </div>
                    <div className="text-xs text-gray-500">
                      Configured: {new Date(existing.configured_at).toLocaleDateString()}
                    </div>
                    {existing.asset_count && (
                      <div className="text-xs text-gray-500">
                        Assets: {existing.asset_count}
                      </div>
                    )}
                    <div className="flex gap-2">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handleConnect(provider.id)}
                      >
                        <Settings className="h-4 w-4 mr-1" />
                        Reconfigure
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handleDisconnect(provider.id)}
                      >
                        <Trash2 className="h-4 w-4 mr-1" />
                        Disconnect
                      </Button>
                    </div>
                  </div>
                ) : (
                  <Button
                    onClick={() => handleConnect(provider.id)}
                    disabled={provider.disabled}
                    className="w-full"
                  >
                    <Plus className="h-4 w-4 mr-1" />
                    {provider.disabled ? 'Coming Soon' : 'Connect'}
                  </Button>
                )}
              </CardContent>
            </Card>
          );
        })}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Integration Settings</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <label className="flex items-center gap-2">
                <input type="checkbox" defaultChecked />
                <span className="text-sm">Enable automatic agent deployment</span>
              </label>
              <p className="text-xs text-gray-500 ml-5">
                Automatically deploy agents to newly discovered cloud assets
              </p>
            </div>

            <div>
              <label className="flex items-center gap-2">
                <input type="checkbox" defaultChecked />
                <span className="text-sm">Enable auto-discovery for new regions</span>
              </label>
              <p className="text-xs text-gray-500 ml-5">
                Automatically scan new cloud regions as they become available
              </p>
            </div>

            <div>
              <label className="flex items-center gap-2">
                <input type="checkbox" />
                <span className="text-sm">Require approval for agent deployment</span>
              </label>
              <p className="text-xs text-gray-500 ml-5">
                Require admin approval before deploying agents to discovered assets
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
```

---

## 🔄 **Migration Strategy**

### **Phase 1: Database Migration**

```sql
-- Migration script to add new tables and columns
BEGIN;

-- Add new columns to organizations table
ALTER TABLE organizations ADD COLUMN IF NOT EXISTS onboarding_flow_version VARCHAR(20) DEFAULT 'legacy';
ALTER TABLE organizations ADD COLUMN IF NOT EXISTS auto_discovery_enabled BOOLEAN DEFAULT true;
ALTER TABLE organizations ADD COLUMN IF NOT EXISTS integration_settings JSONB DEFAULT '{}';

-- Create integration_credentials table
CREATE TABLE IF NOT EXISTS integration_credentials (
    id SERIAL PRIMARY KEY,
    organization_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    provider VARCHAR(50) NOT NULL,
    credential_type VARCHAR(50) NOT NULL,
    credential_data JSONB NOT NULL,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    last_used_at TIMESTAMPTZ,
    UNIQUE(organization_id, provider)
);

-- Create cloud_assets table
CREATE TABLE IF NOT EXISTS cloud_assets (
    id SERIAL PRIMARY KEY,
    organization_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    provider VARCHAR(50) NOT NULL,
    asset_type VARCHAR(100) NOT NULL,
    asset_id VARCHAR(255) NOT NULL,
    asset_data JSONB NOT NULL,
    region VARCHAR(50),
    discovered_at TIMESTAMPTZ DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ DEFAULT NOW(),
    agent_deployed BOOLEAN DEFAULT false,
    agent_status VARCHAR(20) DEFAULT 'pending',
    tags JSONB DEFAULT '{}',
    UNIQUE(organization_id, provider, asset_id)
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_integration_credentials_org_provider ON integration_credentials(organization_id, provider);
CREATE INDEX IF NOT EXISTS idx_cloud_assets_org_provider ON cloud_assets(organization_id, provider);
CREATE INDEX IF NOT EXISTS idx_cloud_assets_type ON cloud_assets(asset_type);

COMMIT;
```

### **Phase 2: Backend Deployment**

1. **Deploy new services alongside existing ones**
2. **Update requirements.txt** with new dependencies:
   ```
   boto3>=1.28.0          # AWS SDK
   azure-identity>=1.12.0 # Azure SDK (for future)
   google-cloud-compute>=1.12.0  # GCP SDK (for future)
   ```

3. **Update main.py** to include new routers:
   ```python
   # In backend/app/main.py
   from .onboarding_v2.routes import router as onboarding_v2_router
   from .integrations.manager import IntegrationManager

   app.include_router(onboarding_v2_router)

   # Initialize integration manager on startup
   @app.on_event("startup")
   async def startup_event():
       # Existing startup code...
       # Add integration cleanup/health checks
       pass
   ```

### **Phase 3: Frontend Deployment**

1. **Create new components** in `frontend/app/components/onboarding/`
2. **Replace legacy onboarding completely**:
   ```typescript
   // In frontend/app/onboarding/page.tsx - COMPLETELY REPLACE LEGACY
   import { QuickStartOnboarding } from '@/components/onboarding/QuickStartOnboarding';
   import { OnboardingProgress } from '@/components/onboarding/OnboardingProgress';

   export default function OnboardingPage() {
     // NEW: Always use seamless onboarding - no legacy fallback
     return <QuickStartOnboarding />;
   }
   ```

3. **Update navigation to remove legacy links**:
   ```typescript
   // Remove any links to legacy onboarding
   // Update any settings that reference old onboarding flow
   ```

### **Phase 4: Mini Corp First Implementation**

1. **Immediate Activation**: Enable new onboarding for Mini Corp organization (ID: 2)
2. **Test with Mini Corp**: Deploy mini-corp network and test complete flow
3. **Validate Integration**: Ensure AWS discovery, agent deployment, and monitoring work
4. **Gather Feedback**: Use Mini Corp as pilot for refinements

### **Phase 5: Complete Replacement**

1. **Full Migration**: After Mini Corp success, enable for all organizations
2. **Legacy Cleanup**: Remove old onboarding routes and components
3. **Database Cleanup**: Archive old onboarding data structures
4. **Documentation Update**: Update all docs to reference new onboarding only

---

## 🧪 **Testing Strategy**

### **Unit Tests**

```python
# backend/tests/test_aws_integration.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from app.integrations.aws import AWSIntegration

@pytest.mark.asyncio
async def test_aws_authentication():
    """Test AWS authentication flow"""
    integration = AWSIntegration(
        organization_id=1,
        credentials={
            'aws_access_key_id': 'test_key',
            'aws_secret_access_key': 'test_secret'
        }
    )

    # Mock boto3 client
    with patch('boto3.client') as mock_client:
        mock_ec2 = MagicMock()
        mock_client.return_value = mock_ec2
        mock_ec2.describe_regions.return_value = {'Regions': [{'RegionName': 'us-east-1'}]}

        result = await integration.authenticate()
        assert result is True

@pytest.mark.asyncio
async def test_asset_discovery():
    """Test EC2 instance discovery"""
    integration = AWSIntegration(1, {})
    integration.regions = ['us-east-1']

    with patch('boto3.client') as mock_client:
        mock_ec2 = MagicMock()
        mock_client.return_value = mock_ec2
        mock_ec2.describe_instances.return_value = {
            'Reservations': [{
                'Instances': [{
                    'InstanceId': 'i-1234567890abcdef0',
                    'State': {'Name': 'running'},
                    'InstanceType': 't3.medium',
                    'PrivateIpAddress': '10.0.1.10'
                }]
            }]
        }

        assets = await integration.discover_assets()
        assert len(assets) == 1
        assert assets[0]['asset_type'] == 'ec2'
        assert assets[0]['asset_id'] == 'i-1234567890abcdef0'
```

### **Integration Tests**

```python
# backend/tests/test_onboarding_flow.py
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_quick_start_flow():
    """Test complete quick start onboarding flow"""
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        # Mock AWS credentials
        credentials = {
            'aws_access_key_id': 'test_key',
            'aws_secret_access_key': 'test_secret'
        }

        # Start onboarding
        response = await client.post(
            "/api/onboarding/v2/quick-start",
            json={"provider": "aws", "credentials": credentials}
        )

        assert response.status_code == 200
        data = response.json()
        assert "initiated" in data["status"]

        # Check progress
        progress_response = await client.get("/api/onboarding/v2/progress")
        assert progress_response.status_code == 200
```

### **E2E Tests**

```typescript
// frontend/cypress/integration/onboarding.spec.ts
describe('Onboarding Flow', () => {
  it('completes AWS onboarding successfully', () => {
    cy.visit('/onboarding');

    // Click AWS connect
    cy.contains('Connect AWS Account').click();

    // Mock OAuth flow
    cy.window().then((win) => {
      win.postMessage({ type: 'OAUTH_SUCCESS', credentials: mockCredentials }, '*');
    });

    // Verify progress monitoring
    cy.contains('Setting up Mini-XDR').should('be.visible');
    cy.contains('100% Complete').should('be.visible');

    // Verify redirect to dashboard
    cy.url().should('include', '/incidents');
  });
});
```

---

## 📊 **Success Metrics & Monitoring**

### **Key Performance Indicators**

| Metric | Target | Current (Legacy) | Expected (New) |
|--------|--------|------------------|-----------------|
| Time to First Alert | <5 minutes | 2-4 hours | <5 minutes |
| Onboarding Completion | >95% | ~70% | >95% |
| User Satisfaction | >4.5/5 | N/A | >4.5/5 |
| Support Tickets | <20% of current | 100% | <20% |

### **Monitoring Dashboards**

```typescript
// Admin dashboard for onboarding metrics
export function OnboardingAnalytics() {
  const metrics = useOnboardingMetrics();

  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
      <MetricCard
        title="Completion Rate"
        value={`${metrics.completionRate}%`}
        trend={metrics.completionTrend}
        icon={CheckCircle}
      />
      <MetricCard
        title="Average Time"
        value={metrics.avgTime}
        trend={metrics.timeTrend}
        icon={Clock}
      />
      <MetricCard
        title="AWS Connections"
        value={metrics.awsConnections}
        trend={metrics.awsTrend}
        icon={Cloud}
      />
      <MetricCard
        title="Support Tickets"
        value={metrics.supportTickets}
        trend={metrics.supportTrend}
        icon={HelpCircle}
      />
    </div>
  );
}
```

---

## 🚀 **Mini Corp Go-Live Checklist**

### **Pre-Launch (Complete Replacement)**
- [ ] AWS integration fully tested and validated
- [ ] Database migrations applied (new tables + Mini Corp org)
- [ ] Mini Corp network deployed (VPC, instances, security)
- [ ] Frontend components deployed and tested
- [ ] API endpoints documented and secured
- [ ] Legacy onboarding routes disabled for Mini Corp

### **Mini Corp Launch**
- [ ] Login as Mini Corp user (organization_id: 2)
- [ ] Navigate to `/onboarding` - should show new seamless flow
- [ ] Click "Connect AWS Account" - test OAuth flow
- [ ] Verify auto-discovery finds mini-corp assets
- [ ] Confirm agent deployment to discovered systems
- [ ] Validate monitoring dashboard shows incidents

### **Post-Launch Validation**
- [ ] Check Mini Corp onboarding status changes to "completed"
- [ ] Verify agents are enrolled and reporting
- [ ] Test incident detection and alerting
- [ ] Confirm cross-VPC connectivity works
- [ ] Gather Mini Corp user feedback for improvements

---

## 📚 **Additional Resources**

### **API Documentation**
- `/api/onboarding/v2/quick-start` - Initiate seamless onboarding
- `/api/onboarding/v2/progress` - Real-time progress monitoring
- `/api/integrations` - Manage cloud integrations
- `/api/integrations/setup` - Configure new integrations

### **Configuration Files**
- `backend/app/integrations/aws.py` - AWS integration implementation
- `frontend/app/components/onboarding/QuickStartOnboarding.tsx` - Main onboarding UI
- `infrastructure/aws/mini-corp-vpc.yaml` - Test environment VPC

### **Security Considerations**
- All credentials encrypted at rest using organization-specific keys
- OAuth flows use short-lived tokens with minimal permissions
- Integration access logged and auditable
- Credential rotation automated where possible

---

## 🎯 **Mini Corp Implementation Plan**

### **Immediate Next Steps**

1. **Deploy Mini Corp Network** (follow `MINI_CORP_AWS_NETWORK_README.md`)
2. **Implement New Onboarding** (replace legacy system completely)
3. **Test with Mini Corp Org** (organization_id: 2, slug: 'mini-corp')
4. **Validate End-to-End** (AWS discovery → agent deployment → monitoring)

### **What Mini Corp Users Will Experience**

When a Mini Corp user logs in and goes to `/onboarding`:

1. **Welcome Screen**: "Welcome to Mini-XDR - Get started in minutes with seamless cloud integration"
2. **AWS Connect Button**: One prominent "Connect AWS Account" button
3. **OAuth Flow**: Secure AWS authentication (no manual credentials)
4. **Auto-Discovery**: Automatically finds DC-01, FS-01, WEB-01, DB-01, WK-01, WK-02, HP-01
5. **Smart Deployment**: Automatically chooses and deploys appropriate agents
6. **Live Progress**: Real-time progress updates every 2 seconds
7. **Completion**: "Setup Complete! Mini-XDR is now monitoring your environment"

### **Expected Timeline**
- **Setup**: 5 minutes (vs 2-4 hours with legacy)
- **First Alert**: <5 minutes after completion
- **Full Monitoring**: Immediate after agent deployment

---

## 📋 **Mini Corp Action Items**

### **Immediate (This Week)**
1. ✅ **Deploy Mini Corp Network** using `MINI_CORP_AWS_NETWORK_README.md`
2. ✅ **Implement New Onboarding Code** (backend + frontend components above)
3. ✅ **Test Mini Corp Onboarding** (org ID: 2)
4. ✅ **Validate AWS Integration** works end-to-end

### **Mini Corp User Flow**
```
Login as Mini Corp user → /onboarding → Connect AWS → Auto-Discovery → Smart Deploy → Monitor
```

### **Success Criteria**
- [ ] Mini Corp onboarding completes in <5 minutes
- [ ] All 7 systems (DC-01 through HP-01) discovered automatically
- [ ] Appropriate agents deployed to each system type
- [ ] Incidents start appearing in dashboard within minutes
- [ ] No manual configuration required

### **If Issues Arise**
- Check VPC peering between Mini-XDR VPC and Mini-Corp VPC
- Verify AWS credentials have proper permissions
- Confirm agent download URLs resolve correctly
- Review backend logs for integration errors

---

**This upgrade completely replaces Mini-XDR's legacy onboarding with a modern, cloud-native experience that transforms the user journey from complex setup to instant value.**
