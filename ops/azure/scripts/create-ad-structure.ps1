# ============================================================================
# Create Active Directory Organizational Structure
# ============================================================================
# Creates OUs, security groups, and test users for the mini corporate network
# Run this AFTER the domain controller has been promoted and rebooted
# ============================================================================

Import-Module ActiveDirectory

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "  Creating Active Directory Structure" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

$DomainDN = "DC=minicorp,DC=local"

# Step 1: Create Organizational Units
Write-Host "[1/4] Creating Organizational Units..." -ForegroundColor Yellow

$OUs = @(
    @{Name="MiniCorp"; Path=$DomainDN; Description="Root OU for Mini Corporate Network"},
    @{Name="Users"; Path="OU=MiniCorp,$DomainDN"; Description="User accounts"},
    @{Name="Workstations"; Path="OU=MiniCorp,$DomainDN"; Description="Workstation computers"},
    @{Name="Servers"; Path="OU=MiniCorp,$DomainDN"; Description="Server computers"},
    @{Name="Quarantine"; Path="OU=MiniCorp,$DomainDN"; Description="Quarantined accounts for IAM agent"},
    @{Name="ServiceAccounts"; Path="OU=MiniCorp,$DomainDN"; Description="Service accounts"},
    @{Name="Groups"; Path="OU=MiniCorp,$DomainDN"; Description="Security and distribution groups"}
)

foreach ($OU in $OUs) {
    try {
        New-ADOrganizationalUnit `
            -Name $OU.Name `
            -Path $OU.Path `
            -Description $OU.Description `
            -ProtectedFromAccidentalDeletion $false `
            -ErrorAction Stop
        Write-Host "  ✓ Created OU: $($OU.Name)" -ForegroundColor Green
    } catch {
        if ($_.Exception.Message -like "*already exists*") {
            Write-Host "  - OU already exists: $($OU.Name)" -ForegroundColor Gray
        } else {
            Write-Host "  ✗ Error creating OU $($OU.Name): $_" -ForegroundColor Red
        }
    }
}

# Step 2: Create Security Groups
Write-Host ""
Write-Host "[2/4] Creating Security Groups..." -ForegroundColor Yellow

$GroupsPath = "OU=Groups,OU=MiniCorp,$DomainDN"

$Groups = @(
    @{Name="IT Administrators"; Scope="Global"; Category="Security"; Description="IT administrative staff"},
    @{Name="Finance Users"; Scope="Global"; Category="Security"; Description="Finance department users"},
    @{Name="HR Users"; Scope="Global"; Category="Security"; Description="Human Resources users"},
    @{Name="Developers"; Scope="Global"; Category="Security"; Description="Software developers"},
    @{Name="Executives"; Scope="Global"; Category="Security"; Description="Executive leadership"},
    @{Name="Remote Workers"; Scope="Global"; Category="Security"; Description="Remote/VPN access users"},
    @{Name="Service Accounts Access"; Scope="Global"; Category="Security"; Description="Service account permissions"}
)

foreach ($Group in $Groups) {
    try {
        New-ADGroup `
            -Name $Group.Name `
            -GroupScope $Group.Scope `
            -GroupCategory $Group.Category `
            -Path $GroupsPath `
            -Description $Group.Description `
            -ErrorAction Stop
        Write-Host "  ✓ Created group: $($Group.Name)" -ForegroundColor Green
    } catch {
        if ($_.Exception.Message -like "*already exists*") {
            Write-Host "  - Group already exists: $($Group.Name)" -ForegroundColor Gray
        } else {
            Write-Host "  ✗ Error creating group $($Group.Name): $_" -ForegroundColor Red
        }
    }
}

# Step 3: Create Test Users
Write-Host ""
Write-Host "[3/4] Creating Test Users..." -ForegroundColor Yellow

$UsersPath = "OU=Users,OU=MiniCorp,$DomainDN"
$DefaultPassword = ConvertTo-SecureString "P@ssw0rd123!" -AsPlainText -Force

$Users = @(
    @{FirstName="John"; LastName="Smith"; SamAccountName="john.smith"; Groups=@("IT Administrators"); Title="IT Administrator"},
    @{FirstName="Jane"; LastName="Doe"; SamAccountName="jane.doe"; Groups=@("Finance Users"); Title="Financial Analyst"},
    @{FirstName="Bob"; LastName="Johnson"; SamAccountName="bob.johnson"; Groups=@("Developers"); Title="Senior Developer"},
    @{FirstName="Alice"; LastName="Williams"; SamAccountName="alice.williams"; Groups=@("HR Users"); Title="HR Manager"},
    @{FirstName="Charlie"; LastName="Brown"; SamAccountName="charlie.brown"; Groups=@("Executives"); Title="CEO"},
    @{FirstName="Diana"; LastName="Prince"; SamAccountName="diana.prince"; Groups=@("Developers"); Title="Junior Developer"},
    @{FirstName="Eve"; LastName="Davis"; SamAccountName="eve.davis"; Groups=@("Finance Users"); Title="Accountant"},
    @{FirstName="Frank"; LastName="Miller"; SamAccountName="frank.miller"; Groups=@("Remote Workers"); Title="Sales Representative"}
)

foreach ($User in $Users) {
    $DisplayName = "$($User.FirstName) $($User.LastName)"
    $UPN = "$($User.SamAccountName)@minicorp.local"
    
    try {
        New-ADUser `
            -Name $DisplayName `
            -GivenName $User.FirstName `
            -Surname $User.LastName `
            -SamAccountName $User.SamAccountName `
            -UserPrincipalName $UPN `
            -EmailAddress "$($User.SamAccountName)@minicorp.local" `
            -Title $User.Title `
            -Path $UsersPath `
            -AccountPassword $DefaultPassword `
            -Enabled $true `
            -ChangePasswordAtLogon $false `
            -PasswordNeverExpires $true `
            -ErrorAction Stop
        
        Write-Host "  ✓ Created user: $DisplayName ($($User.SamAccountName))" -ForegroundColor Green
        
        # Add to groups
        foreach ($GroupName in $User.Groups) {
            try {
                Add-ADGroupMember -Identity $GroupName -Members $User.SamAccountName -ErrorAction Stop
                Write-Host "    → Added to group: $GroupName" -ForegroundColor Gray
            } catch {
                Write-Host "    ✗ Error adding to group $GroupName: $_" -ForegroundColor Red
            }
        }
    } catch {
        if ($_.Exception.Message -like "*already exists*") {
            Write-Host "  - User already exists: $DisplayName" -ForegroundColor Gray
        } else {
            Write-Host "  ✗ Error creating user $DisplayName: $_" -ForegroundColor Red
        }
    }
}

# Step 4: Create Service Accounts
Write-Host ""
Write-Host "[4/4] Creating Service Accounts..." -ForegroundColor Yellow

$ServiceAccountsPath = "OU=ServiceAccounts,OU=MiniCorp,$DomainDN"
$ServicePassword = ConvertTo-SecureString "Svc@Acct2024!" -AsPlainText -Force

$ServiceAccounts = @(
    @{Name="SQL Service"; SamAccountName="svc_sql"; Description="SQL Server service account"},
    @{Name="Backup Service"; SamAccountName="svc_backup"; Description="Backup service account"},
    @{Name="Monitoring Service"; SamAccountName="svc_monitoring"; Description="System monitoring service"}
)

foreach ($SvcAcct in $ServiceAccounts) {
    try {
        New-ADUser `
            -Name $SvcAcct.Name `
            -SamAccountName $SvcAcct.SamAccountName `
            -UserPrincipalName "$($SvcAcct.SamAccountName)@minicorp.local" `
            -Description $SvcAcct.Description `
            -Path $ServiceAccountsPath `
            -AccountPassword $ServicePassword `
            -Enabled $true `
            -PasswordNeverExpires $true `
            -CannotChangePassword $true `
            -ErrorAction Stop
        
        Write-Host "  ✓ Created service account: $($SvcAcct.Name)" -ForegroundColor Green
    } catch {
        if ($_.Exception.Message -like "*already exists*") {
            Write-Host "  - Service account already exists: $($SvcAcct.Name)" -ForegroundColor Gray
        } else {
            Write-Host "  ✗ Error creating service account: $_" -ForegroundColor Red
        }
    }
}

# Display summary
Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "  Active Directory Structure Created Successfully!" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Summary:" -ForegroundColor Green
Write-Host "  • Domain: minicorp.local" -ForegroundColor Gray
Write-Host "  • OUs: $($OUs.Count) organizational units" -ForegroundColor Gray
Write-Host "  • Security Groups: $($Groups.Count) groups" -ForegroundColor Gray
Write-Host "  • User Accounts: $($Users.Count) users" -ForegroundColor Gray
Write-Host "  • Service Accounts: $($ServiceAccounts.Count) accounts" -ForegroundColor Gray
Write-Host ""
Write-Host "Default Passwords:" -ForegroundColor Yellow
Write-Host "  • Users: P@ssw0rd123!" -ForegroundColor Gray
Write-Host "  • Service Accounts: Svc@Acct2024!" -ForegroundColor Gray
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Join Windows workstations to the domain" -ForegroundColor Gray
Write-Host "  2. Deploy Mini-XDR agents to all endpoints" -ForegroundColor Gray
Write-Host "  3. Configure GPO policies for agent deployment" -ForegroundColor Gray
Write-Host ""

