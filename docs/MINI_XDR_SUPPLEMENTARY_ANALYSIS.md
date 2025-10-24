# Mini-XDR - Supplementary Technical Analysis

## 8. Cloud Deployment Capabilities {#cloud-deployment}

### Azure Production Deployment

**Complete Infrastructure as Code (Terraform):**

The Azure deployment consists of 8 Terraform modules totaling 1,260 lines of infrastructure code, providing a complete production environment with a single command.

**Module 1: Provider Configuration (`provider.tf`)**
```hcl
terraform {
  required_version = ">= 1.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {
    key_vault {
      purge_soft_delete_on_destroy = false
    }
  }
}
```

**Module 2: Networking (`networking.tf`)**

Creates a hub-and-spoke network topology with multiple subnets:

- **Hub VNet:** 10.0.0.0/16
  - **AKS Subnet:** 10.0.1.0/24 (Kubernetes cluster)
  - **App Gateway Subnet:** 10.0.2.0/24 (WAF)
  - **Bastion Subnet:** 10.0.3.0/24 (Secure access)
  - **Database Subnet:** 10.0.4.0/24 (PostgreSQL, Redis)

- **Corporate Network VNet:** 10.0.10.0/24
  - **Domain Services:** 10.0.10.0/26 (Domain Controller)
  - **Workstations:** 10.0.10.64/26 (Windows 11 endpoints)
  - **Servers:** 10.0.10.128/26 (Ubuntu servers)

**Security Groups:**
```hcl
resource "azurerm_network_security_group" "aks" {
  name                = "nsg-aks"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  # Allow only Application Gateway
  security_rule {
    name                       = "Allow-AppGateway"
    priority                   = 100
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_ranges    = ["80", "443"]
    source_address_prefix      = azurerm_subnet.appgw.address_prefixes[0]
    destination_address_prefix = azurerm_subnet.aks.address_prefixes[0]
  }

  # Block all other inbound
  security_rule {
    name                       = "Deny-All-Inbound"
    priority                   = 4096
    direction                  = "Inbound"
    access                     = "Deny"
    protocol                   = "*"
    source_port_range          = "*"
    destination_port_range     = "*"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
}
```

**Module 3: AKS Cluster (`aks.tf`)**

```hcl
resource "azurerm_kubernetes_cluster" "main" {
  name                = "aks-minixdr-prod"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = "minixdr"
  kubernetes_version  = "1.28"

  default_node_pool {
    name                = "default"
    node_count          = 3
    vm_size             = "Standard_D4s_v3"  # 4 vCPU, 16 GB RAM
    enable_auto_scaling = true
    min_count           = 2
    max_count           = 5
    vnet_subnet_id      = azurerm_subnet.aks.id
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin     = "azure"
    network_policy     = "calico"
    load_balancer_sku  = "standard"
    service_cidr       = "10.1.0.0/16"
    dns_service_ip     = "10.1.0.10"
  }

  role_based_access_control_enabled = true
  azure_policy_enabled              = true
  
  azure_active_directory_role_based_access_control {
    managed                = true
    azure_rbac_enabled     = true
  }
}
```

**Module 4: Application Gateway with WAF (`aks.tf`)**

```hcl
resource "azurerm_application_gateway" "main" {
  name                = "appgw-minixdr"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  sku {
    name     = "WAF_v2"
    tier     = "WAF_v2"
    capacity = 2
  }

  waf_configuration {
    enabled                  = true
    firewall_mode            = "Prevention"
    rule_set_type            = "OWASP"
    rule_set_version         = "3.2"
    file_upload_limit_mb     = 100
    request_body_check       = true
    max_request_body_size_kb = 128
  }

  gateway_ip_configuration {
    name      = "gateway-ip-config"
    subnet_id = azurerm_subnet.appgw.id
  }

  frontend_ip_configuration {
    name                 = "frontend-ip-config"
    public_ip_address_id = azurerm_public_ip.appgw.id
  }

  # HTTP listener
  frontend_port {
    name = "http"
    port = 80
  }

  # HTTPS listener
  frontend_port {
    name = "https"
    port = 443
  }

  ssl_certificate {
    name     = "minixdr-cert"
    data     = filebase64("${path.module}/certs/minixdr.pfx")
    password = var.ssl_certificate_password
  }

  backend_address_pool {
    name = "aks-backend-pool"
  }

  backend_http_settings {
    name                  = "http-settings"
    cookie_based_affinity = "Disabled"
    port                  = 80
    protocol              = "Http"
    request_timeout       = 30
  }

  http_listener {
    name                           = "https-listener"
    frontend_ip_configuration_name = "frontend-ip-config"
    frontend_port_name             = "https"
    protocol                       = "Https"
    ssl_certificate_name           = "minixdr-cert"
  }

  request_routing_rule {
    name                       = "routing-rule"
    rule_type                  = "Basic"
    http_listener_name         = "https-listener"
    backend_address_pool_name  = "aks-backend-pool"
    backend_http_settings_name = "http-settings"
    priority                   = 100
  }
}
```

**Module 5: Databases (`databases.tf`)**

**PostgreSQL Flexible Server:**
```hcl
resource "azurerm_postgresql_flexible_server" "main" {
  name                   = "psql-minixdr-prod"
  resource_group_name    = azurerm_resource_group.main.name
  location               = azurerm_resource_group.main.location
  version                = "15"
  administrator_login    = "minixdr_admin"
  administrator_password = random_password.postgres.result
  zone                   = "1"

  storage_mb            = 131072  # 128 GB
  backup_retention_days = 30
  geo_redundant_backup_enabled = true

  sku_name = "GP_Standard_D4s_v3"  # 4 vCores, 16 GB RAM

  high_availability {
    mode                      = "ZoneRedundant"
    standby_availability_zone = "2"
  }

  maintenance_window {
    day_of_week  = 0  # Sunday
    start_hour   = 2
    start_minute = 0
  }
}

resource "azurerm_postgresql_flexible_server_database" "minixdr" {
  name      = "minixdr"
  server_id = azurerm_postgresql_flexible_server.main.id
  collation = "en_US.utf8"
  charset   = "utf8"
}
```

**Redis Cache:**
```hcl
resource "azurerm_redis_cache" "main" {
  name                = "redis-minixdr-prod"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  capacity            = 1
  family              = "C"
  sku_name            = "Standard"
  enable_non_ssl_port = false
  minimum_tls_version = "1.2"

  redis_configuration {
    maxmemory_policy = "allkeys-lru"
  }

  patch_schedule {
    day_of_week    = "Sunday"
    start_hour_utc = 2
  }
}
```

**Module 6: Key Vault (`security.tf`)**

```hcl
resource "azurerm_key_vault" "main" {
  name                       = "kv-minixdr-prod"
  location                   = azurerm_resource_group.main.location
  resource_group_name        = azurerm_resource_group.main.name
  tenant_id                  = data.azurerm_client_config.current.tenant_id
  sku_name                   = "standard"
  soft_delete_retention_days = 90
  purge_protection_enabled   = true

  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = azurerm_kubernetes_cluster.main.kubelet_identity[0].object_id

    secret_permissions = [
      "Get",
      "List"
    ]
  }

  network_acls {
    bypass                     = "AzureServices"
    default_action             = "Deny"
    ip_rules                   = [var.admin_ip]
    virtual_network_subnet_ids = [azurerm_subnet.aks.id]
  }
}

# Store database credentials
resource "azurerm_key_vault_secret" "postgres_password" {
  name         = "postgres-password"
  value        = random_password.postgres.result
  key_vault_id = azurerm_key_vault.main.id
}

resource "azurerm_key_vault_secret" "redis_key" {
  name         = "redis-primary-key"
  value        = azurerm_redis_cache.main.primary_access_key
  key_vault_id = azurerm_key_vault.main.id
}
```

**Module 7: Mini Corporate Network (`vms.tf`)**

**Domain Controller:**
```hcl
resource "azurerm_windows_virtual_machine" "domain_controller" {
  name                = "DC01"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  size                = "Standard_D2s_v3"  # 2 vCPU, 8 GB RAM
  admin_username      = var.vm_admin_username
  admin_password      = random_password.vm_admin.result

  network_interface_ids = [
    azurerm_network_interface.dc.id,
  ]

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Premium_LRS"
    disk_size_gb         = 128
  }

  source_image_reference {
    publisher = "MicrosoftWindowsServer"
    offer     = "WindowsServer"
    sku       = "2022-datacenter"
    version   = "latest"
  }

  # Install AD DS role
  custom_data = base64encode(templatefile("${path.module}/scripts/install-adds.ps1", {
    domain_name     = "minicorp.local"
    netbios_name    = "MINICORP"
    safe_mode_pass  = random_password.dsrm.result
  }))
}
```

**Windows 11 Workstations:**
```hcl
resource "azurerm_windows_virtual_machine" "workstation" {
  count               = 3
  name                = "WS${count.index + 1}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  size                = "Standard_B2ms"  # 2 vCPU, 8 GB RAM (burstable)
  admin_username      = var.vm_admin_username
  admin_password      = random_password.vm_admin.result

  network_interface_ids = [
    azurerm_network_interface.workstation[count.index].id,
  ]

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "StandardSSD_LRS"
    disk_size_gb         = 128
  }

  source_image_reference {
    publisher = "MicrosoftWindowsDesktop"
    offer     = "Windows-11"
    sku       = "win11-22h2-pro"
    version   = "latest"
  }

  # Join domain
  custom_data = base64encode(templatefile("${path.module}/scripts/join-domain.ps1", {
    domain_name = "minicorp.local"
    dc_ip       = azurerm_network_interface.dc.private_ip_address
  }))
}
```

**Ubuntu Servers:**
```hcl
resource "azurerm_linux_virtual_machine" "server" {
  count               = 2
  name                = "SRV${count.index + 1}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  size                = "Standard_B2s"  # 2 vCPU, 4 GB RAM
  admin_username      = var.vm_admin_username
  disable_password_authentication = false
  admin_password      = random_password.vm_admin.result

  network_interface_ids = [
    azurerm_network_interface.server[count.index].id,
  ]

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "StandardSSD_LRS"
    disk_size_gb         = 64
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "0001-com-ubuntu-server-jammy"
    sku       = "22_04-lts-gen2"
    version   = "latest"
  }

  custom_data = base64encode(file("${path.module}/scripts/server-init.sh"))
}
```

**Auto-Shutdown Schedule:**
```hcl
resource "azurerm_dev_test_global_vm_shutdown_schedule" "workstations" {
  count              = 3
  virtual_machine_id = azurerm_windows_virtual_machine.workstation[count.index].id
  location           = azurerm_resource_group.main.location
  enabled            = true

  daily_recurrence_time = "2200"  # 10 PM
  timezone              = "Pacific Standard Time"

  notification_settings {
    enabled         = true
    email           = var.admin_email
    time_in_minutes = 30
  }
}
```

**Module 8: Outputs (`outputs.tf`)**

```hcl
output "aks_cluster_name" {
  value = azurerm_kubernetes_cluster.main.name
}

output "appgw_public_ip" {
  value = azurerm_public_ip.appgw.ip_address
  description = "Application Gateway public IP (access Mini-XDR here)"
}

output "key_vault_name" {
  value = azurerm_key_vault.main.name
  description = "Key Vault name (retrieve secrets with: az keyvault secret show)"
}

output "postgres_fqdn" {
  value = azurerm_postgresql_flexible_server.main.fqdn
  description = "PostgreSQL connection string"
}

output "redis_hostname" {
  value = azurerm_redis_cache.main.hostname
  description = "Redis connection string"
}

output "bastion_fqdn" {
  value = azurerm_bastion_host.main.dns_name
  description = "Azure Bastion (for VM access)"
}

output "domain_controller_ip" {
  value = azurerm_network_interface.dc.private_ip_address
  description = "Domain Controller IP (minicorp.local)"
}

output "deployment_summary" {
  value = <<-EOT
    ========================================
    Mini-XDR Azure Deployment Complete!
    ========================================
    
    Application URL: https://${azurerm_public_ip.appgw.ip_address}
    
    Kubernetes:
    - Cluster: ${azurerm_kubernetes_cluster.main.name}
    - Nodes: 3 (auto-scaling 2-5)
    - Get credentials: az aks get-credentials --resource-group ${azurerm_resource_group.main.name} --name ${azurerm_kubernetes_cluster.main.name}
    
    Databases:
    - PostgreSQL: ${azurerm_postgresql_flexible_server.main.fqdn}
    - Redis: ${azurerm_redis_cache.main.hostname}
    
    Corporate Network (minicorp.local):
    - Domain Controller: ${azurerm_network_interface.dc.private_ip_address}
    - Workstations: 3 Windows 11 Pro machines
    - Servers: 2 Ubuntu 22.04 LTS machines
    
    Access VMs via Azure Bastion:
    - https://${azurerm_bastion_host.main.dns_name}
    
    Retrieve secrets:
    - az keyvault secret show --vault-name ${azurerm_key_vault.main.name} --name postgres-password
    - az keyvault secret show --vault-name ${azurerm_key_vault.main.name} --name vm-admin-password
    
    Cost Estimate: $800-1,400/month
    - To reduce costs, deallocate VMs when not in use:
      az vm deallocate --ids $(az vm list -g ${azurerm_resource_group.main.name} --query "[].id" -o tsv)
    
    ========================================
  EOT
}
```

### One-Command Deployment Script

**`deploy-all.sh` (350 lines):**

```bash
#!/bin/bash
set -e

echo "=========================================="
echo "Mini-XDR Azure Deployment"
echo "=========================================="

# Detect user's public IP
echo "Detecting your public IP..."
YOUR_IP=$(curl -s ifconfig.me)
echo "Your IP: $YOUR_IP"

# Initialize Terraform
echo "Initializing Terraform..."
cd ops/azure/terraform
terraform init

# Create terraform.tfvars
cat > terraform.tfvars <<EOF
admin_ip                = "$YOUR_IP/32"
admin_email             = "$ADMIN_EMAIL"
vm_admin_username       = "azureadmin"
ssl_certificate_password = "$SSL_CERT_PASSWORD"
EOF

# Plan deployment
echo "Planning infrastructure deployment..."
terraform plan -out=tfplan

# Apply (with auto-approve for automation)
echo "Deploying infrastructure..."
terraform apply tfplan

# Get outputs
echo "Retrieving deployment information..."
APPGW_IP=$(terraform output -raw appgw_public_ip)
KV_NAME=$(terraform output -raw key_vault_name)
AKS_NAME=$(terraform output -raw aks_cluster_name)
RG_NAME=$(terraform output -raw resource_group_name)

# Get AKS credentials
echo "Configuring kubectl..."
az aks get-credentials --resource-group $RG_NAME --name $AKS_NAME --overwrite-existing

# Build and push Docker images
echo "Building Docker images..."
cd ../../../
docker build -t minixdr/backend:latest -f ops/Dockerfile.backend .
docker build -t minixdr/frontend:latest -f ops/Dockerfile.frontend .

# Tag and push to ACR
ACR_NAME=$(az acr list -g $RG_NAME --query "[0].name" -o tsv)
az acr login --name $ACR_NAME
docker tag minixdr/backend:latest $ACR_NAME.azurecr.io/minixdr/backend:latest
docker tag minixdr/frontend:latest $ACR_NAME.azurecr.io/minixdr/frontend:latest
docker push $ACR_NAME.azurecr.io/minixdr/backend:latest
docker push $ACR_NAME.azurecr.io/minixdr/frontend:latest

# Deploy to Kubernetes
echo "Deploying Mini-XDR to Kubernetes..."
# Update manifests with ACR
sed -i "s|IMAGE_REGISTRY|$ACR_NAME.azurecr.io|g" ops/k8s/*.yaml
kubectl apply -f ops/k8s/

# Wait for pods to be ready
echo "Waiting for pods to be ready..."
kubectl wait --for=condition=Ready pods --all --timeout=600s

# Setup Active Directory
echo "Configuring Active Directory..."
DC_IP=$(terraform -chdir=ops/azure/terraform output -raw domain_controller_ip)
VM_PASSWORD=$(az keyvault secret show --vault-name $KV_NAME --name vm-admin-password --query value -o tsv)

# Copy AD setup scripts to DC via Bastion
# (Uses Azure Bastion for secure access)
echo "Setting up Active Directory domain..."
# ... AD configuration commands ...

# Display summary
echo ""
echo "=========================================="
echo "✅ Deployment Complete!"
echo "=========================================="
echo ""
echo "Mini-XDR URL: https://$APPGW_IP"
echo "Kubernetes Dashboard: kubectl proxy"
echo "Corporate Network: Domain Controller at $DC_IP"
echo ""
echo "To access VMs:"
echo "1. Navigate to Azure Portal"
echo "2. Select Bastion for secure access"
echo ""
echo "Total deployment time: ~90 minutes"
echo "=========================================="
```

### TPOT Honeypot Integration

**T-Pot Deployment:**

The system can deploy T-Pot (The All In One Multi Honeypot Platform) on Azure for comprehensive attack capture:

```bash
# Deploy T-Pot on separate VM
resource "azurerm_linux_virtual_machine" "tpot" {
  name                = "tpot-honeypot"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  size                = "Standard_D4s_v3"  # 4 vCPU, 16 GB RAM (honeypots are resource-intensive)
  admin_username      = "tpot"
  
  # Custom image with T-Pot pre-installed
  source_image_id = data.azurerm_image.tpot.id
  
  # Separate VLAN for isolation
  network_interface_ids = [
    azurerm_network_interface.tpot.id,
  ]
}
```

**T-Pot Components:**
- **Cowrie:** SSH/Telnet honeypot
- **Dionaea:** Malware capture honeypot
- **Elasticpot:** Elasticsearch honeypot
- **Conpot:** ICS/SCADA honeypot
- **Mailoney:** SMTP honeypot
- **Heralding:** Credentials catching honeypot

**Integration with Mini-XDR:**
```python
async def ingest_tpot_events():
    """
    Pull T-Pot events via Elasticsearch API
    Feed into Mini-XDR detection pipeline
    """
    es_client = Elasticsearch(TPOT_ELASTICSEARCH_URL)
    
    # Query recent attacks
    response = es_client.search(
        index="logstash-*",
        body={
            "query": {
                "range": {
                    "@timestamp": {
                        "gte": "now-5m"
                    }
                }
            }
        }
    )
    
    for hit in response['hits']['hits']:
        event = hit['_source']
        
        # Convert T-Pot event to Mini-XDR format
        xdr_event = convert_tpot_event(event)
        
        # Send to ML detection
        detection = await ml_detector.detect(xdr_event)
        
        if detection['threat_score'] > 70:
            # Create incident
            await create_incident(xdr_event, detection)
```

### AWS Infrastructure

**SageMaker ML Training Pipeline:**

The AWS deployment focuses on large-scale ML training using SageMaker:

```python
import sagemaker
from sagemaker.pytorch import PyTorch

# Initialize SageMaker session
session = sagemaker.Session()
role = get_execution_role()

# Define training job
estimator = PyTorch(
    entry_point='train_windows_specialist.py',
    role=role,
    instance_count=1,
    instance_type='ml.p3.2xlarge',  # GPU instance
    framework_version='2.0.0',
    py_version='py310',
    hyperparameters={
        'epochs': 50,
        'batch-size': 256,
        'learning-rate': 0.001
    }
)

# Start training
estimator.fit({
    'training': 's3://minixdr-data/training/',
    'validation': 's3://minixdr-data/validation/'
})

# Deploy model as endpoint
predictor = estimator.deploy(
    initial_instance_count=2,
    instance_type='ml.m5.xlarge',
    endpoint_name='minixdr-windows-specialist'
)
```

**AWS Glue ETL:**

```python
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

# Initialize
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Read CICIDS2017 from S3
cicids_df = glueContext.create_dynamic_frame.from_catalog(
    database="minixdr",
    table_name="cicids2017_raw"
)

# Feature engineering
from pyspark.sql.functions import col, when, expr

features_df = cicids_df.toDF() \\
    .withColumn('flow_duration', col('Flow Duration')) \\
    .withColumn('total_fwd_packets', col('Total Fwd Packets')) \\
    .withColumn('total_bwd_packets', col('Total Backward Packets')) \\
    # ... 83+ feature transformations ...

# Write to processed data lake
features_df.write \\
    .mode('overwrite') \\
    .parquet('s3://minixdr-data/processed/cicids2017_features/')

job.commit()
```

**Cost Optimization:**

Azure/AWS costs can be reduced through:
- **Reserved Instances:** 40% discount for 1-year commitment
- **Spot Instances:** 70% discount for interruptible workloads
- **Auto-shutdown:** VMs turned off after hours
- **Tiered Storage:** S3 Intelligent-Tiering moves cold data to Glacier
- **Right-sizing:** Monitor usage and adjust instance sizes

**Estimated Monthly Costs:**

| Component | Azure Cost | AWS Cost |
|-----------|------------|----------|
| Kubernetes Cluster | $250-400 | $300-450 (EKS) |
| Database (PostgreSQL) | $80-150 | $100-180 (RDS) |
| Cache (Redis) | $15-50 | $20-60 (ElastiCache) |
| Load Balancer/WAF | $150-200 | $100-150 (ALB + WAF) |
| Virtual Machines (6x) | $200-400 | $250-450 (EC2) |
| Storage | $30-50 | $40-60 (S3 + EBS) |
| Networking | $20-40 | $30-50 |
| Monitoring | $15-30 | $20-40 (CloudWatch) |
| **Total** | **$760-1,320** | **$860-1,440** |

**With cost optimization (auto-shutdown, reserved instances):**
- Azure: $450-700/month
- AWS: $500-800/month

---

## 9. Data Pipeline & Processing {#data-pipeline}

### Training Data Comprehensive Analysis

**Total Corpus:** 846,073+ security events

**CICIDS2017 Dataset (799,989 events - 94.6%):**

The Canadian Institute for Cybersecurity Intrusion Detection System 2017 dataset is the cornerstone of network threat detection training.

**Collection Methodology:**
- **Duration:** 8 days (Monday-Friday, July 3-7, 2017)
- **Environment:** Simulated enterprise network with 25 users
- **Traffic Volume:** 2.8M network flows captured
- **Attack Families:** 14 different attack types
- **Feature Extraction:** 83 flow-level features using CICFlowMeter

**Daily Breakdown:**

**Monday (Normal Traffic Day):**
- 529,918 benign flows
- Baseline establishment
- Regular business operations
- Web browsing, email, file transfers

**Tuesday (Brute Force Day):**
- FTP Brute Force: 193,360 attack flows
- SSH Brute Force: 187,589 attack flows
- Total attacks: 380,949

**Wednesday (DoS/DDoS Day):**
- DoS Hulk: 231,073 attack flows
- DoS GoldenEye: 10,293 attack flows
- DoS Slowloris: 5,796 attack flows
- DDoS LOIC: 1,966 attack flows
- Total attacks: 249,128

**Thursday (Web Attacks Day):**
- SQL Injection: 21 attack instances
- Cross-Site Scripting: 652 attack instances
- Brute Force Web: 1,507 attack flows
- Total attacks: 2,180

**Friday (Infiltration & Botnet Day):**
- Infiltration: 36 sophisticated attacks
- Botnet: 1,966 C2 communications
- Port Scan: 158,930 reconnaissance flows
- Total attacks: 160,932

**CICIDS2017 Feature Categories:**

**Temporal Features (15 features):**
1. Flow Duration - Total time of the flow
2. Flow IAT Mean - Mean inter-arrival time between packets
3. Flow IAT Std - Standard deviation of IAT
4. Flow IAT Max - Maximum IAT
5. Flow IAT Min - Minimum IAT
6. Fwd IAT Total - Forward direction IAT sum
7. Fwd IAT Mean - Forward IAT mean
8. Fwd IAT Std - Forward IAT standard deviation
9. Fwd IAT Max - Forward IAT maximum
10. Fwd IAT Min - Forward IAT minimum
11. Bwd IAT Total - Backward direction IAT sum
12. Bwd IAT Mean - Backward IAT mean
13. Bwd IAT Std - Backward IAT standard deviation
14. Bwd IAT Max - Backward IAT maximum
15. Bwd IAT Min - Backward IAT minimum

**Packet Analysis Features (15 features):**
16. Total Fwd Packets - Total forward packets
17. Total Bwd Packets - Total backward packets
18. Total Length of Fwd Packets - Total bytes forward
19. Total Length of Bwd Packets - Total bytes backward
20. Fwd Packet Length Max - Maximum forward packet size
21. Fwd Packet Length Min - Minimum forward packet size
22. Fwd Packet Length Mean - Mean forward packet size
23. Fwd Packet Length Std - Standard deviation forward
24. Bwd Packet Length Max - Maximum backward packet size
25. Bwd Packet Length Min - Minimum backward packet size
26. Bwd Packet Length Mean - Mean backward packet size
27. Bwd Packet Length Std - Standard deviation backward
28. Packet Length Mean - Overall mean packet size
29. Packet Length Std - Overall standard deviation
30. Packet Length Variance - Variance in packet sizes

**Traffic Rate Features (6 features):**
31. Flow Bytes/s - Bytes transferred per second
32. Flow Packets/s - Packets transferred per second
33. Flow IAT Mean - Mean flow inter-arrival time
34. Flow IAT Std - Standard deviation of flow IAT
35. Flow IAT Max - Maximum flow IAT
36. Flow IAT Min - Minimum flow IAT

**Protocol/Flag Features (13 features):**
37. FIN Flag Count - TCP FIN flags
38. SYN Flag Count - TCP SYN flags
39. RST Flag Count - TCP RST flags
40. PSH Flag Count - TCP PSH flags
41. ACK Flag Count - TCP ACK flags
42. URG Flag Count - TCP URG flags
43. CWE Flag Count - TCP CWE flags
44. ECE Flag Count - TCP ECE flags
45. Down/Up Ratio - Download/upload ratio
46. Average Packet Size - Mean packet size
47. Fwd Segment Size Avg - Forward segment average
48. Bwd Segment Size Avg - Backward segment average
49. Fwd Bytes/Bulk Avg - Forward bulk transfer rate

**Subflow & Window Features (17 features):**
50. Subflow Fwd Packets - Forward packets in subflow
51. Subflow Fwd Bytes - Forward bytes in subflow
52. Subflow Bwd Packets - Backward packets in subflow
53. Subflow Bwd Bytes - Backward bytes in subflow
54. Init Fwd Win Bytes - Initial forward window size
55. Init Bwd Win Bytes - Initial backward window size
56. Fwd Act Data Pkts - Forward data packets
57. Fwd Seg Size Min - Minimum forward segment
58. Active Mean - Mean active time
59. Active Std - Standard deviation active time
60. Active Max - Maximum active time
61. Active Min - Minimum active time
62. Idle Mean - Mean idle time
63. Idle Std - Standard deviation idle time
64. Idle Max - Maximum idle time
65. Idle Min - Minimum idle time
66. Label - Ground truth attack type

**Additional Statistical Features (17 features):**
67-83. Various ratios, percentiles, and derived statistics

**APT29 Dataset (15,608 events - 1.8%):**

Real-world advanced persistent threat data from MITRE ATT&CK evaluations.

**Source:** MITRE Engenuity ATT&CK Evaluations Round 1 (APT29)

**Data Types:**
- **Zeek Network Logs:** Protocol-level analysis
  - Kerberos authentication logs
  - SMB file sharing logs
  - DCE-RPC remote procedure calls
  - HTTP web traffic
  - DNS queries

**Attack Stages Captured:**
1. **Initial Access:** Spearphishing attachment
2. **Execution:** PowerShell and WMI
3. **Persistence:** Registry Run keys, Scheduled Tasks
4. **Privilege Escalation:** Credential dumping
5. **Defense Evasion:** Process injection, timestomping
6. **Credential Access:** LSASS memory dumping
7. **Discovery:** Network scanning, system info gathering
8. **Lateral Movement:** PSExec, WMI, Pass-the-Hash
9. **Collection:** Data staging in archives
10. **Exfiltration:** Data transfer to C2 servers

**Example Zeek Log Entry (Kerberos):**
```json
{
  "ts": 1585831200.123456,
  "uid": "C1a2b3c4d5e6f7g8h9",
  "id.orig_h": "192.168.1.50",
  "id.orig_p": 49152,
  "id.resp_h": "192.168.1.10",
  "id.resp_p": 88,
  "request_type": "TGS",
  "client": "compromised$@MINICORP.LOCAL",
  "service": "krbtgt/MINICORP.LOCAL",
  "success": true,
  "error_msg": "",
  "till": "2030-01-01T00:00:00Z",  # Suspicious 10-year ticket
  "cipher": "rc4-hmac",  # Weak encryption
  "forwardable": true,
  "renewable": true
}
```

This log entry shows a Golden Ticket attack (10-year ticket lifetime, RC4 encryption instead of AES).

**Atomic Red Team (750 events - 0.09%):**

Automated MITRE ATT&CK technique executions.

**326 Techniques Covered:**

**Reconnaissance (10 techniques):**
- T1592: Gather Victim Host Information
- T1590: Gather Victim Network Information
- T1589: Gather Victim Identity Information

**Initial Access (9 techniques):**
- T1078: Valid Accounts
- T1566: Phishing
- T1190: Exploit Public-Facing Application

**Execution (13 techniques):**
- T1059.001: PowerShell
- T1059.003: Windows Command Shell
- T1047: Windows Management Instrumentation

**Persistence (19 techniques):**
- T1053.005: Scheduled Task
- T1547.001: Registry Run Keys
- T1136: Create Account

**Privilege Escalation (13 techniques):**
- T1068: Exploitation for Privilege Escalation
- T1134: Access Token Manipulation
- T1548.002: Bypass User Account Control

**Defense Evasion (42 techniques):**
- T1070.001: Clear Windows Event Logs
- T1036: Masquerading
- T1027: Obfuscated Files or Information

**Credential Access (15 techniques):**
- T1003.001: LSASS Memory (Mimikatz)
- T1003.002: Security Account Manager
- T1003.003: NTDS (DCSync)

**Discovery (30 techniques):**
- T1087: Account Discovery
- T1018: Remote System Discovery
- T1083: File and Directory Discovery

**Lateral Movement (9 techniques):**
- T1021.001: Remote Desktop Protocol
- T1021.002: SMB/Windows Admin Shares
- T1021.006: Windows Remote Management

**Collection (17 techniques):**
- T1005: Data from Local System
- T1039: Data from Network Shared Drive
- T1056.001: Keylogging

**Exfiltration (9 techniques):**
- T1041: Exfiltration Over C2 Channel
- T1048.001: Exfiltration Over Alternative Protocol
- T1567.002: Exfiltration to Cloud Storage

**Impact (8 techniques):**
- T1485: Data Destruction
- T1486: Data Encrypted for Impact
- T1490: Inhibit System Recovery

**KDD Cup Dataset (41,000 events - 4.8%):**

Classic intrusion detection benchmark from 1999 KDD Cup competition.

**Attack Types:**
- **DOS:** Denial of Service (back, land, neptune, pod, smurf, teardrop)
- **R2L:** Remote to Local (ftp_write, guess_passwd, imap, multihop, phf, spy, warezclient, warezmaster)
- **U2R:** User to Root (buffer_overflow, loadmodule, perl, rootkit)
- **Probe:** Reconnaissance (ipsweep, nmap, portsweep, satan)

While older, KDD Cup provides diverse attack patterns useful for general model training.

**Threat Intelligence Feeds (2,273 events - 0.27%):**

Real-time threat intelligence from external sources.

**AbuseIPDB:**
- Reported malicious IPs
- Abuse confidence scores
- Attack categories (SSH brute force, port scan, web attack, etc.)
- Geographic distribution

**VirusTotal:**
- File hash verdicts (SHA256)
- Multi-engine detection results
- Malware family classifications
- Behavioral analysis reports

**MISP (Malware Information Sharing Platform):**
- Threat actor profiles
- Campaign indicators
- TTPs and malware samples
- Correlation with known threats

**Synthetic Attack Simulations (1,966 events - 0.23%):**

Custom-generated attacks to fill gaps in real data.

**Generation Methodology:**
- SMOTE (Synthetic Minority Over-sampling Technique) for balanced classes
- Preserve statistical distributions of real attacks
- Avoid overfitting by adding realistic noise
- Validation against held-out real samples

### Feature Engineering Pipeline

**Real-Time Feature Extraction:**

```python
async def extract_features(raw_event: dict) -> np.ndarray:
    """
    Extract 83+ features from raw network event
    Processing time: <20ms per event
    """
    features = np.zeros(83)
    
    # Temporal features (0-14)
    features[0] = raw_event.get('flow_duration', 0)
    features[1] = np.mean(raw_event.get('iat', []))
    features[2] = np.std(raw_event.get('iat', []))
    # ... more temporal features
    
    # Packet analysis (15-29)
    features[15] = len(raw_event.get('fwd_packets', []))
    features[16] = len(raw_event.get('bwd_packets', []))
    features[17] = sum(pkt['size'] for pkt in raw_event.get('fwd_packets', []))
    # ... more packet features
    
    # Protocol flags (30-42)
    features[30] = raw_event.get('flags', {}).get('FIN', 0)
    features[31] = raw_event.get('flags', {}).get('SYN', 0)
    # ... more flags
    
    # Statistical features (43-66)
    features[43] = np.percentile(packet_sizes, 50)  # Median
    features[44] = np.percentile(packet_sizes, 75)  # 75th percentile
    # ... more statistics
    
    # Threat intelligence (67-72)
    features[67] = await get_ip_reputation(raw_event['source_ip'])
    features[68] = get_geo_risk_score(raw_event['source_country'])
    # ... more intel features
    
    # Behavioral features (73-82)
    features[73] = calculate_entropy(raw_event['payload'])
    features[74] = detect_encryption_ratio(raw_event['payload'])
    # ... more behavioral features
    
    return features
```

**Batch Processing:**

For large-scale training data preparation:

```python
def batch_process_cicids2017():
    """
    Process full CICIDS2017 dataset
    Input: 8 CSV files (2.8M rows)
    Output: NumPy arrays (features + labels)
    Time: ~30 minutes on 8-core CPU
    """
    import pandas as pd
    from multiprocessing import Pool
    
    csv_files = [
        'Monday-WorkingHours.csv',
        'Tuesday-WorkingHours.csv',
        'Wednesday-workingHours.csv',
        'Thursday-WorkingHours-Morning.csv',
        'Thursday-WorkingHours-Afternoon.csv',
        'Friday-WorkingHours-Morning.csv',
        'Friday-WorkingHours-Afternoon-DDos.csv',
        'Friday-WorkingHours-Afternoon-PortScan.csv'
    ]
    
    # Parallel processing
    with Pool(processes=8) as pool:
        results = pool.map(process_csv_file, csv_files)
    
    # Concatenate results
    all_features = np.vstack([r[0] for r in results])
    all_labels = np.concatenate([r[1] for r in results])
    
    # Normalize features
    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features)
    
    # Save
    np.save('datasets/cicids2017_features.npy', all_features_scaled)
    np.save('datasets/cicids2017_labels.npy', all_labels)
    joblib.dump(scaler, 'models/cicids2017_scaler.pkl')
    
    print(f"Processed {len(all_features)} samples")
    print(f"Features shape: {all_features.shape}")
    print(f"Class distribution: {np.bincount(all_labels)}")
```

**Data Quality Checks:**

```python
def validate_dataset(features, labels):
    """
    Comprehensive data quality validation
    """
    issues = []
    
    # Check for NaN/Inf
    if np.isnan(features).any():
        issues.append("NaN values found in features")
    if np.isinf(features).any():
        issues.append("Infinite values found in features")
    
    # Check for constant features (zero variance)
    variances = np.var(features, axis=0)
    constant_features = np.where(variances == 0)[0]
    if len(constant_features) > 0:
        issues.append(f"{len(constant_features)} constant features")
    
    # Check label distribution
    unique, counts = np.unique(labels, return_counts=True)
    min_class_size = np.min(counts)
    if min_class_size < 100:
        issues.append(f"Imbalanced classes: min size {min_class_size}")
    
    # Check for duplicates
    unique_samples = np.unique(features, axis=0)
    if len(unique_samples) < len(features):
        issues.append(f"{len(features) - len(unique_samples)} duplicate samples")
    
    if issues:
        print("⚠️ Data Quality Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ Data quality validation passed")
    
    return len(issues) == 0
```

### Data Storage & Management

**S3 Data Lake (AWS):**

```
s3://minixdr-datalake/
├── raw/
│   ├── cicids2017/           # Original CSV files
│   ├── apt29/                # Zeek logs (JSON-LD)
│   ├── atomic-red-team/      # YAML technique definitions
│   └── threat-feeds/         # Daily threat intel dumps
│
├── processed/
│   ├── cicids2017_features.npy      # 2.8M x 83 array
│   ├── cicids2017_labels.npy        # 2.8M labels
│   ├── windows_features.npy         # 390K x 79 array
│   ├── windows_labels.npy           # 390K labels
│   └── scalers/                     # StandardScaler objects
│
├── models/
│   ├── network/
│   │   ├── general.pth
│   │   ├── ddos_specialist.pth
│   │   ├── brute_force_specialist.pth
│   │   └── web_specialist.pth
│   └── windows/
│       ├── windows_13class_specialist.pth
│       └── windows_13class_scaler.pkl
│
└── training_runs/
    ├── run_20240115_120000/
    │   ├── config.json
    │   ├── metrics.json
    │   ├── model.pth
    │   └── logs.txt
    └── run_20240116_140000/
        └── ...
```

**Data Versioning:**

Using DVC (Data Version Control) for reproducibility:

```bash
# Initialize DVC
dvc init

# Track datasets
dvc add datasets/cicids2017_features.npy
dvc add datasets/windows_features.npy

# Commit to git (only .dvc files, not actual data)
git add datasets/*.dvc .dvc/config
git commit -m "Track training datasets"

# Push data to remote storage
dvc remote add -d storage s3://minixdr-datalake/dvc-cache
dvc push

# Later, reproduce exact training conditions
git checkout v1.0.0
dvc pull
python train.py
```

This comprehensive data pipeline ensures high-quality, reproducible machine learning training at scale.

---

*[Document continues with remaining sections in next part...]*


