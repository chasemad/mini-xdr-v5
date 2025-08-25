# Mini-XDR Management Scripts

This directory contains powerful management scripts for the Mini-XDR system that provide complete lifecycle management with health checks and monitoring.

## 🚀 Scripts Overview

### **`start-all.sh`** - Complete System Startup
**The main script you'll use** - Handles complete system lifecycle with health verification.

**Features:**
- ✅ **Smart Cleanup**: Automatically kills existing backend/frontend processes
- ✅ **Prerequisites Check**: Verifies Python venv, Node modules, config files
- ✅ **Health Monitoring**: Waits for services to start and validates they're working
- ✅ **Comprehensive Testing**: Tests all API endpoints and system components
- ✅ **Error Handling**: Fails fast with clear error messages and logs
- ✅ **Status Display**: Shows complete system status after startup

**Usage:**
```bash
# Start the complete Mini-XDR system
./scripts/start-all.sh

# The script will:
# 1. Kill any existing processes on ports 8000, 3000, 3001
# 2. Check all prerequisites (venv, node_modules, configs)
# 3. Start backend with health monitoring
# 4. Start frontend with connectivity testing  
# 5. Start MCP server (if available)
# 6. Perform comprehensive health checks
# 7. Display system status and access URLs
```

**Output Example:**
```
=== 🛡️  Mini-XDR Complete System Startup ===

[23:30:15] Cleaning up existing Mini-XDR services...
✅ Service cleanup completed

[23:30:17] Checking system prerequisites...
✅ Prerequisites check completed

[23:30:19] Starting all services...
[23:30:19] Starting backend server...
[23:30:19] Backend starting (PID: 12345)...
✅ Backend server ready on port 8000

[23:30:22] Starting frontend server...
[23:30:22] Frontend starting (PID: 12346)...
✅ Frontend server ready on port 3000

[23:30:25] Starting MCP server...
✅ MCP server started

[23:30:28] Performing system health checks...

🔍 Testing Backend API...
✅ Backend API responding
   Response: {"status":"healthy","timestamp":"...","auto_contain":false}

🔍 Testing Incidents API...
✅ Incidents API responding (5 incidents)

🔍 Testing Frontend...
✅ Frontend responding

🔍 Testing Auto-contain API...
✅ Auto-contain API responding
   Setting: {"enabled":false}

🔍 Testing Database...
✅ Database file exists

🔍 Checking Configuration...
✅ LLM configuration detected

✅ Health checks completed!

✅ 🎉 Mini-XDR System Successfully Started!

=== 🚀 Mini-XDR System Status ===

📊 Services:
   • Frontend:  http://localhost:3000
   • Backend:   http://localhost:8000
   • API Docs:  http://localhost:8000/docs

📋 Process IDs:
   • Backend PID:  12345
   • Frontend PID: 12346
   • MCP PID:      12347

📝 Logs:
   • Backend:  /path/to/mini-xdr/backend/backend.log
   • Frontend: /path/to/mini-xdr/frontend/frontend.log
   • MCP:      /path/to/mini-xdr/backend/mcp.log

🎮 Controls:
   • Dashboard: Open http://localhost:3000
   • Stop All:  Press Ctrl+C
   • Restart:   Run this script again

Press Ctrl+C to stop all services
```

### **`stop-all.sh`** - Clean System Shutdown
Gracefully stops all Mini-XDR services.

**Features:**
- ✅ **Graceful Shutdown**: Sends TERM signals first, then force kills if needed
- ✅ **Port-based Cleanup**: Finds and stops processes by port numbers
- ✅ **Process Pattern Cleanup**: Kills processes by command patterns
- ✅ **Verification**: Confirms all processes are stopped

**Usage:**
```bash
# Stop all Mini-XDR services
./scripts/stop-all.sh
```

### **`system-status.sh`** - Real-time Status Check
Provides detailed status of all system components without starting/stopping anything.

**Features:**
- ✅ **Port Status**: Checks which services are running on expected ports
- ✅ **API Testing**: Tests backend API endpoints for functionality
- ✅ **Process Information**: Shows process IDs for running services
- ✅ **Database Status**: Checks database file existence and size
- ✅ **Configuration Check**: Verifies all required files are present

**Usage:**
```bash
# Check current system status
./scripts/system-status.sh
```

## 🔧 Configuration

The scripts automatically detect and use these configuration files:

- **Backend**: `backend/.env` - API keys, database settings, honeypot config
- **Frontend**: `frontend/env.local` - API endpoints and keys
- **Python Environment**: `backend/.venv/` - Python virtual environment
- **Node Dependencies**: `frontend/node_modules/` - Frontend dependencies

## 📊 Ports Used

| Service | Port | Purpose |
|---------|------|---------|
| Backend API | 8000 | FastAPI server with XDR endpoints |
| Frontend | 3000 | Next.js development server |
| MCP Server | 3001 | LLM integration server |

## 🔍 Health Checks Performed

The startup script performs these comprehensive checks:

1. **Backend Health**: `/health` endpoint responding
2. **Incidents API**: `/incidents` endpoint with data count
3. **Frontend**: HTTP connectivity test
4. **Auto-contain**: `/settings/auto_contain` endpoint
5. **Database**: SQLite file existence and accessibility
6. **Configuration**: Environment variables and API keys
7. **Dependencies**: Python venv and Node modules

## 🚨 Troubleshooting

### **"Port already in use" errors**
The scripts automatically handle this by killing existing processes before starting.

### **"Prerequisites check failed"**
Install missing dependencies:
```bash
# Backend dependencies
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Frontend dependencies  
cd frontend
npm install
```

### **"Health checks failed"**
Check the log files for detailed error information:
- Backend: `backend/backend.log`
- Frontend: `frontend/frontend.log`
- MCP: `backend/mcp.log`

### **Services won't start**
Run the stop script first, then try starting again:
```bash
./scripts/stop-all.sh
./scripts/start-all.sh
```

## 📝 Log Files

All services create log files for debugging:

- **Backend Log**: `backend/backend.log` - FastAPI server logs
- **Frontend Log**: `frontend/frontend.log` - Next.js development logs  
- **MCP Log**: `backend/mcp.log` - MCP server logs

## 🎯 Quick Start

1. **First Time Setup**:
   ```bash
   # Install all dependencies first
   cd backend && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
   cd ../frontend && npm install
   
   # Start the system
   ./scripts/start-all.sh
   ```

2. **Daily Usage**:
   ```bash
   # Start everything
   ./scripts/start-all.sh
   
   # Check status anytime
   ./scripts/system-status.sh
   
   # Stop everything
   ./scripts/stop-all.sh
   ```

3. **Troubleshooting**:
   ```bash
   # Force clean restart
   ./scripts/stop-all.sh
   ./scripts/start-all.sh
   
   # Check what's running
   ./scripts/system-status.sh
   ```

## 🎮 Integration with IDE

These scripts work great with your development workflow:

- **VS Code**: Add tasks in `.vscode/tasks.json` to run scripts from Command Palette
- **Terminal**: Run scripts from any terminal in the project root
- **CI/CD**: Use in automation pipelines for testing and deployment

The enhanced `start-all.sh` script ensures your Mini-XDR system starts reliably every time with full health verification!
