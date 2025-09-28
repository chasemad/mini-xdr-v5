#!/bin/bash
# Mini-XDR complete setup script

set -e

echo "=== Mini-XDR Setup Script ==="
echo ""

# Check prerequisites
command -v python3 >/dev/null 2>&1 || { echo "Python 3 is required but not installed. Aborting." >&2; exit 1; }
command -v node >/dev/null 2>&1 || { echo "Node.js is required but not installed. Aborting." >&2; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "npm is required but not installed. Aborting." >&2; exit 1; }

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Setting up Mini-XDR in: $PROJECT_ROOT"
echo ""

# Backend setup
echo "=== Setting up Backend ==="
cd backend

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install Node.js dependencies for MCP server
echo "Installing Node.js dependencies..."
npm install

# Setup environment file
if [ ! -f ".env" ]; then
    echo "Creating backend environment file..."
    cp env.example .env
    echo "Please edit backend/.env with your configuration"
fi

# Initialize database
echo "Initializing database..."
python -c "
import asyncio
from app.db import init_db
asyncio.run(init_db())
print('Database initialized successfully')
"

cd ..

# Frontend setup
echo ""
echo "=== Setting up Frontend ==="
cd frontend

# Install dependencies
echo "Installing frontend dependencies..."
npm install

# Setup environment file
if [ ! -f ".env.local" ]; then
    echo "Creating frontend environment file..."
    cp env.local .env.local
    echo "Please edit frontend/.env.local with your configuration"
fi

cd ..

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Edit backend/.env with your honeypot configuration"
echo "2. Edit frontend/.env.local with your API settings"
echo "3. Start the backend: cd backend && source .venv/bin/activate && uvicorn app.main:app --reload"
echo "4. Start the frontend: cd frontend && npm run dev"
echo "5. (Optional) Start MCP server: cd backend && npm run mcp"
echo ""
echo "Access the UI at: http://localhost:3000"
echo "API documentation at: http://localhost:8000/docs"
