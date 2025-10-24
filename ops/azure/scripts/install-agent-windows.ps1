# ============================================================================
# Mini-XDR Windows Agent Installer
# ============================================================================
# Installs and configures the Mini-XDR agent on Windows systems
# ============================================================================

param(
    [Parameter(Mandatory=$true)]
    [string]$BackendUrl,
    
    [Parameter(Mandatory=$true)]
    [string]$ApiKey,
    
    [Parameter(Mandatory=$false)]
    [string]$AgentType = "endpoint",
    
    [Parameter(Mandatory=$false)]
    [string]$InstallPath = "C:\Program Files\MiniXDR"
)

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "  Mini-XDR Windows Agent Installer" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as Administrator
if (-NOT ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "✗ This script must be run as Administrator!" -ForegroundColor Red
    exit 1
}

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Backend URL: $BackendUrl" -ForegroundColor Gray
Write-Host "  Agent Type: $AgentType" -ForegroundColor Gray
Write-Host "  Install Path: $InstallPath" -ForegroundColor Gray
Write-Host ""

# Step 1: Create installation directory
Write-Host "[1/7] Creating installation directory..." -ForegroundColor Yellow
try {
    if (-not (Test-Path $InstallPath)) {
        New-Item -Path $InstallPath -ItemType Directory -Force | Out-Null
        Write-Host "  ✓ Created: $InstallPath" -ForegroundColor Green
    } else {
        Write-Host "  - Directory already exists" -ForegroundColor Gray
    }
} catch {
    Write-Host "  ✗ Error creating directory: $_" -ForegroundColor Red
    exit 1
}

# Step 2: Download agent from ACR or build locally
Write-Host "[2/7] Downloading agent binaries..." -ForegroundColor Yellow

# For now, create a Python-based agent
$AgentScript = @"
import os
import sys
import time
import json
import requests
import logging
from datetime import datetime
import psutil
import socket

# Configure logging
logging.basicConfig(
    filename=r'$InstallPath\agent.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class MiniXDRAgent:
    def __init__(self):
        self.backend_url = '$BackendUrl'
        self.api_key = '$ApiKey'
        self.hostname = socket.gethostname()
        self.agent_type = '$AgentType'
        
    def collect_metrics(self):
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'hostname': self.hostname,
                'agent_type': self.agent_type,
                'timestamp': datetime.utcnow().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available': memory.available,
                'disk_percent': disk.percent,
                'disk_free': disk.free
            }
        except Exception as e:
            logging.error(f'Error collecting metrics: {e}')
            return None
            
    def send_heartbeat(self):
        try:
            metrics = self.collect_metrics()
            if metrics:
                response = requests.post(
                    f'{self.backend_url}/api/agents/heartbeat',
                    json=metrics,
                    headers={'X-API-Key': self.api_key},
                    timeout=10
                )
                if response.status_code == 200:
                    logging.info('Heartbeat sent successfully')
                    return True
                else:
                    logging.error(f'Heartbeat failed: {response.status_code}')
        except Exception as e:
            logging.error(f'Error sending heartbeat: {e}')
        return False
        
    def run(self):
        logging.info(f'Mini-XDR Agent started - {self.hostname}')
        print(f'Mini-XDR Agent running on {self.hostname}')
        
        while True:
            try:
                self.send_heartbeat()
                time.sleep(60)  # Send heartbeat every minute
            except KeyboardInterrupt:
                logging.info('Agent stopped by user')
                break
            except Exception as e:
                logging.error(f'Agent error: {e}')
                time.sleep(60)

if __name__ == '__main__':
    agent = MiniXDRAgent()
    agent.run()
"@

try {
    $AgentScript | Out-File -FilePath "$InstallPath\agent.py" -Encoding UTF8
    Write-Host "  ✓ Agent script created" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Error creating agent script: $_" -ForegroundColor Red
    exit 1
}

# Step 3: Install Python if not present
Write-Host "[3/7] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    if ($?) {
        Write-Host "  ✓ Python already installed: $pythonVersion" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ Python not found. Installing..." -ForegroundColor Yellow
        # Download and install Python
        $pythonInstaller = "$env:TEMP\python-installer.exe"
        Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.11.0/python-3.11.0-amd64.exe" -OutFile $pythonInstaller
        Start-Process -FilePath $pythonInstaller -Args "/quiet InstallAllUsers=1 PrependPath=1" -Wait
        Write-Host "  ✓ Python installed" -ForegroundColor Green
        Remove-Item $pythonInstaller
    }
} catch {
    Write-Host "  ✗ Error checking/installing Python: $_" -ForegroundColor Red
}

# Step 4: Install required Python packages
Write-Host "[4/7] Installing Python dependencies..." -ForegroundColor Yellow
try {
    python -m pip install --quiet requests psutil 2>&1 | Out-Null
    Write-Host "  ✓ Dependencies installed" -ForegroundColor Green
} catch {
    Write-Host "  ⚠ Warning: Could not install dependencies" -ForegroundColor Yellow
}

# Step 5: Create Windows Service
Write-Host "[5/7] Creating Windows Service..." -ForegroundColor Yellow

$serviceName = "MiniXDRAgent"
$serviceDisplayName = "Mini-XDR Security Agent"
$serviceDescription = "Mini-XDR endpoint security monitoring agent"

# Create service wrapper script
$ServiceWrapper = @"
pythonw.exe "$InstallPath\agent.py"
"@

$ServiceWrapper | Out-File -FilePath "$InstallPath\start-agent.bat" -Encoding ASCII

# Create NSSM service or use sc.exe
try {
    # Check if service exists
    $existingService = Get-Service -Name $serviceName -ErrorAction SilentlyContinue
    
    if ($existingService) {
        Write-Host "  - Service already exists, stopping..." -ForegroundColor Gray
        Stop-Service -Name $serviceName -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
        sc.exe delete $serviceName | Out-Null
        Start-Sleep -Seconds 2
    }
    
    # Create new service
    New-Service `
        -Name $serviceName `
        -BinaryPathName "cmd.exe /c `"$InstallPath\start-agent.bat`"" `
        -DisplayName $serviceDisplayName `
        -Description $serviceDescription `
        -StartupType Automatic `
        -ErrorAction Stop | Out-Null
    
    Write-Host "  ✓ Service created successfully" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Error creating service: $_" -ForegroundColor Red
}

# Step 6: Configure Windows Firewall
Write-Host "[6/7] Configuring Windows Firewall..." -ForegroundColor Yellow
try {
    New-NetFirewallRule `
        -DisplayName "Mini-XDR Agent" `
        -Direction Outbound `
        -Action Allow `
        -Protocol TCP `
        -RemotePort 443,8000 `
        -ErrorAction SilentlyContinue | Out-Null
    Write-Host "  ✓ Firewall rule created" -ForegroundColor Green
} catch {
    Write-Host "  - Firewall rule may already exist" -ForegroundColor Gray
}

# Step 7: Start the service
Write-Host "[7/7] Starting the agent service..." -ForegroundColor Yellow
try {
    Start-Service -Name $serviceName -ErrorAction Stop
    Start-Sleep -Seconds 3
    $service = Get-Service -Name $serviceName
    if ($service.Status -eq 'Running') {
        Write-Host "  ✓ Service started successfully" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ Service status: $($service.Status)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  ✗ Error starting service: $_" -ForegroundColor Red
}

# Display summary
Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "  Mini-XDR Agent Installation Complete!" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Installation Details:" -ForegroundColor Green
Write-Host "  • Install Path: $InstallPath" -ForegroundColor Gray
Write-Host "  • Service Name: $serviceName" -ForegroundColor Gray
Write-Host "  • Backend URL: $BackendUrl" -ForegroundColor Gray
Write-Host "  • Agent Type: $AgentType" -ForegroundColor Gray
Write-Host ""
Write-Host "Useful Commands:" -ForegroundColor Cyan
Write-Host "  • Check status:  Get-Service $serviceName" -ForegroundColor Gray
Write-Host "  • View logs:     Get-Content '$InstallPath\agent.log' -Tail 50" -ForegroundColor Gray
Write-Host "  • Restart:       Restart-Service $serviceName" -ForegroundColor Gray
Write-Host "  • Stop:          Stop-Service $serviceName" -ForegroundColor Gray
Write-Host ""

