#!/bin/bash
# Restart both frontend and backend servers to apply all changes

set -e

echo "=========================================="
echo "Restarting Mini-XDR Servers"
echo "=========================================="
echo

# Get the project root
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

echo "Project root: $PROJECT_ROOT"
echo

# Step 1: Kill existing servers
echo "Step 1: Stopping existing servers..."
echo "--------------------------------------"

# Kill backend (uvicorn on port 8000)
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "Stopping backend server on port 8000..."
    kill $(lsof -Pi :8000 -sTCP:LISTEN -t) 2>/dev/null || true
    sleep 2
    echo "✅ Backend stopped"
else
    echo "ℹ️  Backend not running"
fi

# Kill frontend (Next.js on port 3000)
if lsof -Pi :3000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "Stopping frontend server on port 3000..."
    kill $(lsof -Pi :3000 -sTCP:LISTEN -t) 2>/dev/null || true
    sleep 2
    echo "✅ Frontend stopped"
else
    echo "ℹ️  Frontend not running"
fi

echo

# Step 2: Start Backend
echo "Step 2: Starting Backend..."
echo "--------------------------------------"
cd "$BACKEND_DIR"

if [ ! -d "venv" ]; then
    echo "❌ Error: Backend venv not found at $BACKEND_DIR/venv"
    echo "Please create it first: python -m venv venv"
    exit 1
fi

echo "Starting backend server..."
source venv/bin/activate
nohup uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 > "$BACKEND_DIR/logs/server.log" 2>&1 &
BACKEND_PID=$!

echo "Backend PID: $BACKEND_PID"
echo "Waiting for backend to start..."
sleep 3

# Check if backend is running
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "✅ Backend running on http://localhost:8000"
    echo "   Logs: tail -f $BACKEND_DIR/logs/server.log"
else
    echo "❌ Backend failed to start. Check logs:"
    echo "   tail -f $BACKEND_DIR/logs/server.log"
    exit 1
fi

echo

# Step 3: Start Frontend
echo "Step 3: Starting Frontend..."
echo "--------------------------------------"
cd "$FRONTEND_DIR"

if [ ! -d "node_modules" ]; then
    echo "❌ Error: Frontend node_modules not found"
    echo "Please install dependencies first: npm install"
    exit 1
fi

echo "Starting frontend server..."
nohup npm run dev > "$FRONTEND_DIR/logs/server.log" 2>&1 &
FRONTEND_PID=$!

echo "Frontend PID: $FRONTEND_PID"
echo "Waiting for frontend to start..."
sleep 5

# Check if frontend is running
if lsof -Pi :3000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "✅ Frontend running on http://localhost:3000"
    echo "   Logs: tail -f $FRONTEND_DIR/logs/server.log"
else
    echo "❌ Frontend failed to start. Check logs:"
    echo "   tail -f $FRONTEND_DIR/logs/server.log"
    exit 1
fi

echo
echo "=========================================="
echo "✅ All Servers Started Successfully!"
echo "=========================================="
echo
echo "URLs:"
echo "  Frontend: http://localhost:3000"
echo "  Backend:  http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo
echo "PIDs:"
echo "  Backend:  $BACKEND_PID"
echo "  Frontend: $FRONTEND_PID"
echo
echo "Logs:"
echo "  Backend:  tail -f $BACKEND_DIR/logs/server.log"
echo "  Frontend: tail -f $FRONTEND_DIR/logs/server.log"
echo
echo "To stop servers:"
echo "  kill $BACKEND_PID $FRONTEND_PID"
echo
echo "Next steps:"
echo "  1. Go to http://localhost:3000/incidents/incident/3"
echo "  2. Hard refresh (Cmd+Shift+R)"
echo "  3. Try executing an action"
echo
