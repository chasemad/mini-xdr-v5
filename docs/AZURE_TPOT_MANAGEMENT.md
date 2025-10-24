# 🚀 Azure TPOT Management Guide

Easy-to-use scripts for managing your Azure TPOT honeypot VM.

---

## 📋 Quick Reference

### Start TPOT
```bash
./scripts/azure-tpot-start.sh
```
- Starts the Azure VM
- Waits for TPOT services to initialize
- Shows connection information
- **Time:** ~1-2 minutes

### Stop TPOT
```bash
./scripts/azure-tpot-stop.sh
```
- Gracefully stops TPOT services
- Deallocates VM to stop charges
- **Saves:** ~$40-60/month when stopped

### Check Status
```bash
./scripts/azure-tpot-status.sh
```
- Shows VM and service status
- Displays running containers
- Shows resource usage
- Provides connection info

### Restart TPOT
```bash
./scripts/azure-tpot-restart.sh
```
- Stops then starts TPOT
- Useful for troubleshooting
- **Time:** ~2-3 minutes

---

## 🎯 Common Scenarios

### Starting Your Work Session
```bash
# 1. Start TPOT
./scripts/azure-tpot-start.sh

# 2. Wait for services (2-3 minutes)
# The script will tell you when ready

# 3. Access TPOT
# Web UI: https://74.235.242.205:64297
# Username: tsec
# Password: minixdrtpot2025
```

### Stopping When Done
```bash
# Stop TPOT to save costs
./scripts/azure-tpot-stop.sh

# This deallocates the VM - you won't be charged for compute
# Only storage costs remain (~$3-5/month)
```

### Checking if TPOT is Running
```bash
./scripts/azure-tpot-status.sh
```

**Output shows:**
- VM status (running/stopped)
- IP address
- TPOT services status
- Container count
- System resources (CPU, memory, disk)
- Cost estimate
- Quick action commands

---

## 💰 Cost Management

### When Running
- **Compute:** ~$0.05-0.10/hour
- **Daily:** ~$1.20-2.40
- **Monthly:** ~$40-65

### When Stopped
- **Compute:** $0 (deallocated)
- **Storage:** ~$0.10-0.15/day
- **Monthly:** ~$3-5

### Cost Saving Tips
1. **Stop TPOT when not using it**
   ```bash
   ./scripts/azure-tpot-stop.sh
   ```

2. **Check status regularly**
   ```bash
   ./scripts/azure-tpot-status.sh
   ```

3. **Use scheduled start/stop** (optional)
   - Start: 9 AM when you start work
   - Stop: 6 PM when you finish

---

## 🔧 Troubleshooting

### TPOT Won't Start
```bash
# 1. Check status
./scripts/azure-tpot-status.sh

# 2. Try restart
./scripts/azure-tpot-restart.sh

# 3. Check Azure Portal
# https://portal.azure.com
```

### Services Not Running
```bash
# Wait 2-3 minutes after VM start
# TPOT containers take time to initialize

# Check status
./scripts/azure-tpot-status.sh

# If still not running, connect via SSH
ssh -i ~/.ssh/mini-xdr-tpot-azure azureuser@74.235.242.205 -p 64295

# Check Docker
sudo docker ps -a

# Restart Docker if needed
sudo systemctl restart docker
```

### Can't Connect to Web UI
```bash
# 1. Check VM is running
./scripts/azure-tpot-status.sh

# 2. Check your IP is authorized
curl ifconfig.me

# 3. Update NSG rules if IP changed
./scripts/secure-azure-tpot-testing.sh

# 4. Wait 2-3 minutes after start
# Web UI takes time to initialize
```

### SSH Connection Fails
```bash
# 1. Check VM is running
./scripts/azure-tpot-status.sh

# 2. Verify SSH key exists
ls -la ~/.ssh/mini-xdr-tpot-azure

# 3. Test connection
ssh -i ~/.ssh/mini-xdr-tpot-azure azureuser@74.235.242.205 -p 64295

# 4. Check NSG rules
az network nsg rule list \
  --resource-group mini-xdr-rg \
  --nsg-name mini-xdr-tpotNSG \
  --output table
```

---

## 📊 Script Details

### azure-tpot-start.sh
**What it does:**
1. Checks Azure CLI authentication
2. Checks current VM status
3. Starts VM if stopped
4. Waits for VM to be ready
5. Tests SSH connectivity
6. Checks TPOT services
7. Shows connection information

**When to use:**
- Start of work day
- After stopping TPOT
- After VM restart

### azure-tpot-stop.sh
**What it does:**
1. Checks VM status
2. Attempts graceful shutdown of services
3. Stops and deallocates VM
4. Shows cost savings information

**When to use:**
- End of work day
- When stepping away for extended period
- To save costs when not testing

### azure-tpot-status.sh
**What it does:**
1. Shows VM power state
2. Shows IP address (if running)
3. Checks TPOT installation
4. Lists running containers
5. Shows system resources
6. Displays cost estimate
7. Provides quick action commands

**When to use:**
- Check if TPOT is running
- See resource usage
- Get connection information
- Verify services are healthy

### azure-tpot-restart.sh
**What it does:**
1. Runs stop script
2. Waits 10 seconds
3. Runs start script

**When to use:**
- Troubleshooting issues
- After configuration changes
- Services not responding

---

## 🔐 Security Notes

### Safe Operations
All scripts are designed to be safe:
- ✅ No data deletion
- ✅ Graceful shutdowns
- ✅ Proper error handling
- ✅ Confirmation prompts (restart only)

### What Gets Stopped
When you run `azure-tpot-stop.sh`:
- ✅ VM is deallocated (saved)
- ✅ All data is preserved
- ✅ Configurations remain intact
- ✅ Disk contents unchanged
- ✅ IP address may change on next start

### What Doesn't Change
- NSG rules (security settings)
- Disk data (all TPOT data preserved)
- Resource group
- VM size/configuration

---

## 📅 Recommended Workflow

### Daily Use Pattern
```bash
# Morning - Start work
./scripts/azure-tpot-start.sh

# Check it's ready
./scripts/azure-tpot-status.sh

# Work with TPOT all day...

# Evening - Stop to save costs
./scripts/azure-tpot-stop.sh
```

### Weekly Pattern
```bash
# Start of week
./scripts/azure-tpot-start.sh

# Keep running all week...

# Check status occasionally
./scripts/azure-tpot-status.sh

# End of week
./scripts/azure-tpot-stop.sh
```

### Production Use
```bash
# Keep TPOT running 24/7
./scripts/azure-tpot-start.sh

# Monitor with status checks
./scripts/azure-tpot-status.sh

# Only stop for maintenance
./scripts/azure-tpot-stop.sh
```

---

## 🎓 Examples

### Check Status
```bash
$ ./scripts/azure-tpot-status.sh

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VM STATUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Status: ● Running
  Name: mini-xdr-tpot
  Size: Standard_B2s
  Resource Group: mini-xdr-rg
  IP Address: 74.235.242.205

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TPOT SERVICES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  SSH Connection: ✅ Connected
  TPOT Installation: ✅ Installed
  Docker: ✅ Installed
  Containers Running: ✅ 36/36

  Active Containers:
    • cowrie (Up 2 hours)
    • dionaea (Up 2 hours)
    • honeytrap (Up 2 hours)
    • mailoney (Up 2 hours)
    • ...
```

### Start TPOT
```bash
$ ./scripts/azure-tpot-start.sh

╔════════════════════════════════════════════════════════════════╗
║           Azure TPOT Honeypot - START                          ║
╚════════════════════════════════════════════════════════════════╝

✅ Azure CLI authenticated

[1/4] Checking VM status...
  Current status: VM stopped
[2/4] Starting VM...
  This may take 30-60 seconds...
  Waiting for VM to start...
  ✅ VM started successfully
  Waiting for networking (15 seconds)...

[3/4] Getting VM IP address...
  ✅ VM IP: 74.235.242.205

[4/4] Checking SSH connectivity...
  Testing SSH connection...
  ✅ SSH connection successful

╔════════════════════════════════════════════════════════════════╗
║                    TPOT STARTED SUCCESSFULLY                    ║
╚════════════════════════════════════════════════════════════════╝

📋 Connection Information:
  • VM Status: Running
  • IP Address: 74.235.242.205
  • SSH Port: 64295

🔗 Access TPOT:
  • Web UI: https://74.235.242.205:64297
  • SSH: ssh -i ~/.ssh/mini-xdr-tpot-azure azureuser@74.235.242.205 -p 64295
```

---

## 💡 Pro Tips

1. **Add to PATH for easy access**
   ```bash
   echo 'export PATH="$PATH:$HOME/Desktop/mini-xdr/scripts"' >> ~/.zshrc
   source ~/.zshrc
   
   # Now you can run from anywhere:
   azure-tpot-start.sh
   azure-tpot-status.sh
   azure-tpot-stop.sh
   ```

2. **Create aliases**
   ```bash
   alias tpot-start='~/Desktop/mini-xdr/scripts/azure-tpot-start.sh'
   alias tpot-stop='~/Desktop/mini-xdr/scripts/azure-tpot-stop.sh'
   alias tpot-status='~/Desktop/mini-xdr/scripts/azure-tpot-status.sh'
   ```

3. **Schedule automatic start/stop** (macOS)
   ```bash
   # Edit crontab
   crontab -e
   
   # Start at 9 AM weekdays
   0 9 * * 1-5 ~/Desktop/mini-xdr/scripts/azure-tpot-start.sh
   
   # Stop at 6 PM weekdays
   0 18 * * 1-5 ~/Desktop/mini-xdr/scripts/azure-tpot-stop.sh
   ```

4. **Quick status check**
   ```bash
   watch -n 30 ./scripts/azure-tpot-status.sh
   ```

---

## 📞 Support

### If Scripts Fail
1. Check Azure CLI: `az --version`
2. Check authentication: `az account show`
3. Check resource group exists: `az group show --name mini-xdr-rg`
4. Check VM exists: `az vm show --resource-group mini-xdr-rg --name mini-xdr-tpot`

### Manual Operations
If scripts don't work, you can use Azure CLI directly:

```bash
# Start VM
az vm start --resource-group mini-xdr-rg --name mini-xdr-tpot

# Stop VM
az vm deallocate --resource-group mini-xdr-rg --name mini-xdr-tpot

# Get status
az vm get-instance-view --resource-group mini-xdr-rg --name mini-xdr-tpot

# Get IP
az vm show -d --resource-group mini-xdr-rg --name mini-xdr-tpot --query publicIps -o tsv
```

---

## 🎉 Summary

You now have complete control over your Azure TPOT honeypot:

✅ **Easy start/stop** - Save money when not in use  
✅ **Status monitoring** - Know what's running  
✅ **Cost tracking** - See estimated charges  
✅ **Graceful operations** - No data loss  
✅ **Quick troubleshooting** - Built-in diagnostics  

**Start saving money today by stopping TPOT when you're not using it!** 💰

---

*Created: October 5, 2025*  
*For: Mini-XDR Azure TPOT Deployment*

