# ============================================================================
# Configure Active Directory Domain Services
# ============================================================================
# This script promotes a Windows Server to a Domain Controller and
# configures the mini corporate Active Directory environment
# ============================================================================

param(
    [Parameter(Mandatory=$true)]
    [string]$DomainName = "minicorp.local",
    
    [Parameter(Mandatory=$true)]
    [string]$NetBiosName = "MINICORP",
    
    [Parameter(Mandatory=$true)]
    [SecureString]$SafeModePassword
)

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "  Mini-XDR Active Directory Configuration" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Install AD DS Role
Write-Host "[1/5] Installing AD DS role and management tools..." -ForegroundColor Yellow
try {
    $result = Install-WindowsFeature -Name AD-Domain-Services -IncludeManagementTools
    if ($result.Success) {
        Write-Host "✓ AD DS role installed successfully" -ForegroundColor Green
    } else {
        Write-Host "✗ Failed to install AD DS role" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "✗ Error installing AD DS: $_" -ForegroundColor Red
    exit 1
}

# Step 2: Promote to Domain Controller
Write-Host "[2/5] Promoting server to Domain Controller..." -ForegroundColor Yellow
Write-Host "  Domain: $DomainName" -ForegroundColor Gray
Write-Host "  NetBIOS: $NetBiosName" -ForegroundColor Gray

try {
    Install-ADDSForest `
        -DomainName $DomainName `
        -DomainNetbiosName $NetBiosName `
        -ForestMode "WinThreshold" `
        -DomainMode "WinThreshold" `
        -InstallDns `
        -SafeModeAdministratorPassword $SafeModePassword `
        -Force `
        -NoRebootOnCompletion:$false
    
    Write-Host "✓ Domain Controller promotion initiated" -ForegroundColor Green
    Write-Host "  Server will reboot automatically..." -ForegroundColor Yellow
} catch {
    Write-Host "✗ Error promoting to DC: $_" -ForegroundColor Red
    exit 1
}

# Note: Remaining steps will be executed after reboot via a separate script
Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "  Phase 1 Complete - Server will reboot" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

