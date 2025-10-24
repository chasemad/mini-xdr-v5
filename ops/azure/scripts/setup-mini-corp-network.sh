#!/bin/bash
# ============================================================================
# Mini Corporate Network Setup Script
# ============================================================================
# Configures the mini corporate network including:
# - Domain Controller promotion
# - Domain join for endpoints
# - Agent deployment
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TERRAFORM_DIR="$PROJECT_ROOT/ops/azure/terraform"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       Mini Corporate Network Setup                             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Get Terraform outputs
if [ ! -f "$TERRAFORM_DIR/terraform.tfstate" ]; then
    echo -e "${RED}❌ Terraform state not found. Run infrastructure deployment first.${NC}"
    exit 1
fi

RESOURCE_GROUP=$(terraform -chdir="$TERRAFORM_DIR" output -raw resource_group_name)
DC_NAME="mini-corp-dc01"
KEY_VAULT_NAME=$(terraform -chdir="$TERRAFORM_DIR" output -raw key_vault_name)
VM_ADMIN_PASSWORD=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "vm-admin-password" --query value -o tsv)
DC_RESTORE_PASSWORD=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "dc-restore-mode-password" --query value -o tsv)
ADMIN_USERNAME=$(terraform -chdir="$TERRAFORM_DIR" output -raw vm_admin_username)

echo "Configuration:"
echo "  • Resource Group: $RESOURCE_GROUP"
echo "  • Domain Controller: $DC_NAME"
echo "  • Admin Username: $ADMIN_USERNAME"
echo ""

# Step 1: Configure Domain Controller
echo -e "${YELLOW}Step 1/4: Configuring Domain Controller${NC}"
echo "This will promote the VM to a Domain Controller..."

# Create PowerShell script for AD DS configuration
DC_CONFIG_SCRIPT=$(cat <<'PSEOF'
# Install AD DS Role
Write-Host "Installing AD DS role..."
Install-WindowsFeature -Name AD-Domain-Services -IncludeManagementTools

# Promote to Domain Controller
Write-Host "Promoting to Domain Controller..."
$SecureRestorePassword = ConvertTo-SecureString "__RESTORE_PASSWORD__" -AsPlainText -Force

Install-ADDSForest `
    -DomainName "minicorp.local" `
    -DomainNetbiosName "MINICORP" `
    -ForestMode "WinThreshold" `
    -DomainMode "WinThreshold" `
    -InstallDns `
    -SafeModeAdministratorPassword $SecureRestorePassword `
    -Force

Write-Host "Domain Controller configuration complete. System will reboot..."
PSEOF
)

# Replace password placeholder
DC_CONFIG_SCRIPT="${DC_CONFIG_SCRIPT//__RESTORE_PASSWORD__/$DC_RESTORE_PASSWORD}"

# Upload and execute script
echo "$DC_CONFIG_SCRIPT" > /tmp/configure-dc.ps1

az vm run-command invoke \
    --resource-group "$RESOURCE_GROUP" \
    --name "$DC_NAME" \
    --command-id RunPowerShellScript \
    --scripts @/tmp/configure-dc.ps1

rm /tmp/configure-dc.ps1

echo -e "${GREEN}✅ Domain Controller configured${NC}"
echo -e "${YELLOW}⚠️  DC will reboot. Waiting 5 minutes...${NC}"
sleep 300

# Step 2: Create OUs and Security Groups
echo -e "${YELLOW}Step 2/4: Creating OUs and Security Groups${NC}"

CREATE_OU_SCRIPT=$(cat <<'PSEOF'
Import-Module ActiveDirectory

# Create OUs
New-ADOrganizationalUnit -Name "MiniCorp" -Path "DC=minicorp,DC=local" -ProtectedFromAccidentalDeletion $false
New-ADOrganizationalUnit -Name "Users" -Path "OU=MiniCorp,DC=minicorp,DC=local" -ProtectedFromAccidentalDeletion $false
New-ADOrganizationalUnit -Name "Workstations" -Path "OU=MiniCorp,DC=minicorp,DC=local" -ProtectedFromAccidentalDeletion $false
New-ADOrganizationalUnit -Name "Servers" -Path "OU=MiniCorp,DC=minicorp,DC=local" -ProtectedFromAccidentalDeletion $false
New-ADOrganizationalUnit -Name "Quarantine" -Path "OU=MiniCorp,DC=minicorp,DC=local" -ProtectedFromAccidentalDeletion $false

# Create Security Groups
New-ADGroup -Name "IT Administrators" -GroupScope Global -Path "OU=Users,OU=MiniCorp,DC=minicorp,DC=local"
New-ADGroup -Name "Finance Users" -GroupScope Global -Path "OU=Users,OU=MiniCorp,DC=minicorp,DC=local"
New-ADGroup -Name "HR Users" -GroupScope Global -Path "OU=Users,OU=MiniCorp,DC=minicorp,DC=local"
New-ADGroup -Name "Developers" -GroupScope Global -Path "OU=Users,OU=MiniCorp,DC=minicorp,DC=local"

# Create test users
$Password = ConvertTo-SecureString "P@ssw0rd123!" -AsPlainText -Force

New-ADUser -Name "John Smith" -SamAccountName "john.smith" -UserPrincipalName "john.smith@minicorp.local" `
    -Path "OU=Users,OU=MiniCorp,DC=minicorp,DC=local" -AccountPassword $Password -Enabled $true
New-ADUser -Name "Jane Doe" -SamAccountName "jane.doe" -UserPrincipalName "jane.doe@minicorp.local" `
    -Path "OU=Users,OU=MiniCorp,DC=minicorp,DC=local" -AccountPassword $Password -Enabled $true
New-ADUser -Name "Bob Johnson" -SamAccountName "bob.johnson" -UserPrincipalName "bob.johnson@minicorp.local" `
    -Path "OU=Users,OU=MiniCorp,DC=minicorp,DC=local" -AccountPassword $Password -Enabled $true

# Add users to groups
Add-ADGroupMember -Identity "IT Administrators" -Members "john.smith"
Add-ADGroupMember -Identity "Finance Users" -Members "jane.doe"
Add-ADGroupMember -Identity "Developers" -Members "bob.johnson"

Write-Host "OUs, groups, and users created successfully"
PSEOF
)

echo "$CREATE_OU_SCRIPT" > /tmp/create-ou.ps1

az vm run-command invoke \
    --resource-group "$RESOURCE_GROUP" \
    --name "$DC_NAME" \
    --command-id RunPowerShellScript \
    --scripts @/tmp/create-ou.ps1

rm /tmp/create-ou.ps1

echo -e "${GREEN}✅ OUs and security groups created${NC}"

# Step 3: Configure DNS for endpoints
echo -e "${YELLOW}Step 3/4: Configuring DNS for Windows Endpoints${NC}"

DC_IP=$(terraform -chdir="$TERRAFORM_DIR" output -raw domain_controller_private_ip)
ENDPOINT_COUNT=$(terraform -chdir="$TERRAFORM_DIR" output -json windows_endpoint_private_ips | jq '. | length')

for i in $(seq 1 $ENDPOINT_COUNT); do
    WS_NAME="mini-corp-ws$(printf '%02d' $i)"
    echo "Configuring $WS_NAME..."
    
    DNS_CONFIG_SCRIPT="Set-DnsClientServerAddress -InterfaceAlias 'Ethernet' -ServerAddresses $DC_IP"
    
    az vm run-command invoke \
        --resource-group "$RESOURCE_GROUP" \
        --name "$WS_NAME" \
        --command-id RunPowerShellScript \
        --scripts "$DNS_CONFIG_SCRIPT" \
        > /dev/null 2>&1
    
    echo "  ✓ DNS configured for $WS_NAME"
done

echo -e "${GREEN}✅ DNS configured for all endpoints${NC}"

# Step 4: Display summary
echo -e "${YELLOW}Step 4/4: Deployment Summary${NC}"

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║        Mini Corporate Network Setup Complete!                  ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Domain Information:"
echo "  • Domain Name: minicorp.local"
echo "  • NetBIOS Name: MINICORP"
echo "  • Domain Controller IP: $DC_IP"
echo ""
echo "Organizational Units:"
echo "  • OU=MiniCorp,DC=minicorp,DC=local"
echo "    ├── Users"
echo "    ├── Workstations"
echo "    ├── Servers"
echo "    └── Quarantine"
echo ""
echo "Test Users (Password: P@ssw0rd123!):"
echo "  • john.smith@minicorp.local (IT Administrators)"
echo "  • jane.doe@minicorp.local (Finance Users)"
echo "  • bob.johnson@minicorp.local (Developers)"
echo ""
echo "Next Steps:"
echo "  1. Join Windows endpoints to domain:"
echo "     ./ops/azure/scripts/join-endpoints-to-domain.sh"
echo "  2. Install agents on all VMs:"
echo "     ./ops/azure/scripts/deploy-agents-to-corp.sh"
echo "  3. Run attack simulations:"
echo "     ./ops/azure/attacks/run-all-tests.sh"
echo ""

