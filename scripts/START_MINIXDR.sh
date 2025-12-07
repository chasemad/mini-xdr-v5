#!/bin/bash
# Mini-XDR Local Startup Script

set -e

cd "$(dirname "$0")"

echo "🛡️  Starting Mini-XDR..."
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "🐳 Docker is not running. Starting Docker Desktop..."
    open -a Docker
    echo "⏳ Waiting for Docker to start..."
    for i in {1..30}; do
        if docker info > /dev/null 2>&1; then
            echo "✅ Docker is ready!"
            break
        fi
        sleep 2
        echo -n "."
    done
    echo ""
fi

# Start services
echo ""
echo "🚀 Starting all services..."
docker-compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready (30 seconds)..."
sleep 30

# Check status
echo ""
echo "📊 Service Status:"
docker-compose ps

# Test backend
echo ""
echo "🏥 Backend Health Check:"
curl -s http://localhost:8000/health | python3 -m json.tool 2>/dev/null || curl http://localhost:8000/health

echo ""
echo ""
echo "🎉 Mini-XDR is ready!"
echo ""
echo "🌐 Access Points:"
echo "   • Dashboard:   http://localhost:3000"
echo "   • Login:       http://localhost:3000/login"
echo "   • API Docs:    http://localhost:8000/docs"
echo ""
echo "🔑 Your Credentials:"
echo "   Email:    admin@example.com"
echo "   Password: demo-tpot-api-key"
echo ""
echo "📋 To view logs:    docker-compose logs -f"
echo "⏹️  To stop:         docker-compose down"
echo ""

# Open browser
echo "🌐 Opening browser..."
open http://localhost:3000/login

echo ""
echo "✨ All set! Happy hunting! 🎯🛡️"
