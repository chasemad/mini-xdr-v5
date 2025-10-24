#!/usr/bin/env python3
"""
Set up Azure ML Workspace for Fast Mini-XDR Training
Configures GPU compute clusters for 4M+ event training
"""

import os
import sys
import json
import logging
from pathlib import Path
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    AmlCompute, 
    Environment,
    ComputeInstance,
    Data
)
from azure.identity import DefaultAzureCredential, AzureCliCredential
from azure.core.exceptions import ResourceExistsError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AzureMLWorkspaceSetup:
    """Set up Azure ML workspace for fast training"""
    
    def __init__(self, 
                 subscription_id=None,
                 resource_group="mini-xdr-ml-rg",
                 workspace_name="mini-xdr-ml-workspace",
                 location="eastus"):
        
        self.subscription_id = subscription_id or os.getenv('AZURE_SUBSCRIPTION_ID')
        self.resource_group = resource_group
        self.workspace_name = workspace_name
        self.location = location
        
        if not self.subscription_id:
            logger.error("❌ Azure subscription ID not set!")
            logger.info("   Set via: export AZURE_SUBSCRIPTION_ID=<your-sub-id>")
            logger.info("   Or pass as argument: --subscription-id <your-sub-id>")
            sys.exit(1)
        
        # Try Azure CLI credentials first, then default
        try:
            self.credential = AzureCliCredential()
            logger.info("✅ Using Azure CLI credentials")
        except Exception:
            logger.info("🔄 Falling back to DefaultAzureCredential")
            self.credential = DefaultAzureCredential()
        
        self.ml_client = None
    
    def create_workspace(self):
        """Create or get Azure ML workspace"""
        logger.info(f"🏢 Setting up Azure ML Workspace: {self.workspace_name}")
        logger.info(f"   Resource Group: {self.resource_group}")
        logger.info(f"   Location: {self.location}")
        
        try:
            # Create ML client
            self.ml_client = MLClient(
                credential=self.credential,
                subscription_id=self.subscription_id,
                resource_group_name=self.resource_group,
                workspace_name=self.workspace_name
            )
            
            # Try to get existing workspace
            workspace = self.ml_client.workspaces.get(self.workspace_name)
            logger.info(f"✅ Using existing workspace: {workspace.name}")
            
        except Exception as e:
            logger.info(f"📝 Creating new workspace (not found or error: {e})")
            
            # Create via Azure CLI (simpler for initial setup)
            import subprocess
            
            # Create resource group
            logger.info(f"   Creating resource group: {self.resource_group}")
            subprocess.run([
                'az', 'group', 'create',
                '--name', self.resource_group,
                '--location', self.location
            ], check=False)
            
            # Create workspace
            logger.info(f"   Creating ML workspace: {self.workspace_name}")
            result = subprocess.run([
                'az', 'ml', 'workspace', 'create',
                '--name', self.workspace_name,
                '--resource-group', self.resource_group,
                '--location', self.location
            ], check=True, capture_output=True, text=True)
            
            logger.info("✅ Workspace created successfully")
            
            # Now create ML client
            self.ml_client = MLClient(
                credential=self.credential,
                subscription_id=self.subscription_id,
                resource_group_name=self.resource_group,
                workspace_name=self.workspace_name
            )
        
        return self.ml_client
    
    def create_gpu_compute(self):
        """Create GPU compute cluster for fast training"""
        logger.info("\n🖥️  Setting up GPU Compute Cluster")
        
        compute_configs = [
            {
                'name': 'gpu-cluster-v100',
                'size': 'Standard_NC6s_v3',  # 1x V100 GPU, 6 cores, 112GB RAM
                'min_nodes': 0,
                'max_nodes': 4,
                'tier': 'dedicated',
                'description': 'V100 GPU cluster for deep learning training'
            },
            {
                'name': 'gpu-cluster-t4',
                'size': 'Standard_NC4as_T4_v3',  # 1x T4 GPU, 4 cores, 28GB RAM  
                'min_nodes': 0,
                'max_nodes': 4,
                'tier': 'low_priority',  # Cheaper
                'description': 'T4 GPU cluster for cost-effective training'
            }
        ]
        
        created_clusters = []
        
        for config in compute_configs:
            try:
                logger.info(f"\n   Creating cluster: {config['name']}")
                logger.info(f"      Size: {config['size']}")
                logger.info(f"      Nodes: {config['min_nodes']}-{config['max_nodes']}")
                logger.info(f"      Tier: {config['tier']}")
                
                cluster = AmlCompute(
                    name=config['name'],
                    type="amlcompute",
                    size=config['size'],
                    min_instances=config['min_nodes'],
                    max_instances=config['max_nodes'],
                    tier=config['tier'],
                    idle_time_before_scale_down=300  # 5 minutes
                )
                
                compute = self.ml_client.compute.begin_create_or_update(cluster).result()
                logger.info(f"   ✅ Created: {compute.name}")
                created_clusters.append(compute.name)
                
            except ResourceExistsError:
                logger.info(f"   ✅ Cluster already exists: {config['name']}")
                created_clusters.append(config['name'])
            except Exception as e:
                logger.warning(f"   ⚠️  Failed to create {config['name']}: {e}")
                logger.info(f"      Trying alternative configuration...")
        
        if not created_clusters:
            logger.warning("⚠️  No GPU clusters created - will use CPU")
            self.create_cpu_compute()
        
        return created_clusters
    
    def create_cpu_compute(self):
        """Create CPU compute cluster as fallback"""
        logger.info("\n🖥️  Setting up CPU Compute Cluster (Fallback)")
        
        try:
            cluster = AmlCompute(
                name='cpu-cluster',
                type="amlcompute",
                size='Standard_D4s_v3',  # 4 cores, 16GB RAM
                min_instances=0,
                max_instances=4,
                tier='dedicated'
            )
            
            compute = self.ml_client.compute.begin_create_or_update(cluster).result()
            logger.info(f"   ✅ Created CPU cluster: {compute.name}")
            return compute.name
            
        except ResourceExistsError:
            logger.info("   ✅ CPU cluster already exists")
            return 'cpu-cluster'
    
    def create_training_environment(self):
        """Create custom environment for Mini-XDR training"""
        logger.info("\n📦 Creating Training Environment")
        
        # Create conda environment file
        conda_yaml = """
name: mini-xdr-training
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - pytorch>=2.0.0
  - torchvision
  - cudatoolkit=11.8
  - numpy
  - pandas
  - scikit-learn
  - joblib
  - tqdm
  - pip:
    - azure-ai-ml
    - azureml-core
    - boto3
"""
        
        env_dir = Path("/Users/chasemad/Desktop/mini-xdr/scripts/azure-ml/environment")
        env_dir.mkdir(parents=True, exist_ok=True)
        
        conda_file = env_dir / "conda.yaml"
        with open(conda_file, 'w') as f:
            f.write(conda_yaml)
        
        logger.info(f"   📝 Created conda.yaml at {conda_file}")
        
        try:
            env = Environment(
                name="mini-xdr-training-env",
                description="Mini-XDR threat detection training environment",
                conda_file=str(conda_file),
                image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04"
            )
            
            env = self.ml_client.environments.create_or_update(env)
            logger.info(f"   ✅ Environment created: {env.name}")
            return env.name
            
        except Exception as e:
            logger.warning(f"   ⚠️  Failed to create environment: {e}")
            return None
    
    def upload_training_data(self, data_path="/Users/chasemad/Desktop/mini-xdr/datasets/real_datasets"):
        """Upload training data to Azure ML datastore"""
        logger.info(f"\n📤 Uploading Training Data")
        logger.info(f"   Source: {data_path}")
        
        data_path = Path(data_path)
        if not data_path.exists():
            logger.error(f"   ❌ Data path not found: {data_path}")
            return None
        
        try:
            # Create data asset
            data_asset = Data(
                name="mini-xdr-training-data",
                description="Mini-XDR cybersecurity training dataset (4M+ events)",
                path=str(data_path),
                type="uri_folder"
            )
            
            data = self.ml_client.data.create_or_update(data_asset)
            logger.info(f"   ✅ Data uploaded: {data.name} (v{data.version})")
            return data
            
        except Exception as e:
            logger.warning(f"   ⚠️  Failed to upload data: {e}")
            logger.info("   Data will be uploaded during training job submission")
            return None
    
    def save_config(self):
        """Save workspace configuration"""
        config = {
            'subscription_id': self.subscription_id,
            'resource_group': self.resource_group,
            'workspace_name': self.workspace_name,
            'location': self.location
        }
        
        config_file = Path("/Users/chasemad/Desktop/mini-xdr/scripts/azure-ml/workspace_config.json")
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"\n💾 Configuration saved to: {config_file}")
        return config_file
    
    def setup_complete_workspace(self):
        """Complete workspace setup"""
        logger.info("🚀 Azure ML Workspace Setup for Mini-XDR")
        logger.info("=" * 70)
        
        # Create workspace
        self.create_workspace()
        
        # Create compute
        gpu_clusters = self.create_gpu_compute()
        
        # Create environment
        env_name = self.create_training_environment()
        
        # Upload data (optional - can be done during training)
        # self.upload_training_data()
        
        # Save config
        config_file = self.save_config()
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("✅ AZURE ML SETUP COMPLETE!")
        logger.info("=" * 70)
        logger.info(f"📍 Workspace: {self.workspace_name}")
        logger.info(f"🖥️  GPU Clusters: {', '.join(gpu_clusters) if gpu_clusters else 'None (using CPU)'}")
        logger.info(f"📦 Environment: {env_name if env_name else 'Default'}")
        logger.info(f"💾 Config: {config_file}")
        
        logger.info("\n📝 Next steps:")
        logger.info("   1. Run: python3 scripts/azure-ml/launch_azure_training.py")
        logger.info("   2. This will start GPU-accelerated training on your 4M+ events")
        logger.info("   3. Training will be 10-50x faster than local CPU training")
        
        logger.info("\n💰 Cost Estimate:")
        logger.info("   V100 GPU: ~$3/hour (finishes in 30-60 min = $1.50-3)")
        logger.info("   T4 GPU (low priority): ~$0.30/hour (finishes in 1-2 hours = $0.30-0.60)")
        
        return {
            'workspace': self.workspace_name,
            'resource_group': self.resource_group,
            'gpu_clusters': gpu_clusters,
            'environment': env_name,
            'config_file': str(config_file)
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup Azure ML Workspace')
    parser.add_argument('--subscription-id', type=str, help='Azure subscription ID')
    parser.add_argument('--resource-group', type=str, default='mini-xdr-ml-rg')
    parser.add_argument('--workspace', type=str, default='mini-xdr-ml-workspace')
    parser.add_argument('--location', type=str, default='eastus')
    
    args = parser.parse_args()
    
    try:
        setup = AzureMLWorkspaceSetup(
            subscription_id=args.subscription_id,
            resource_group=args.resource_group,
            workspace_name=args.workspace,
            location=args.location
        )
        
        result = setup.setup_complete_workspace()
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Setup interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

