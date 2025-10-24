#!/bin/bash
#
# GET YOUR MINI-XDR DEMO LIVE IN 2 MINUTES
# This bypasses the broken Azure LoadBalancer
#

set -e

echo "===================================================================="
echo "   Mini-XDR Demo Setup - Public URL in 2 Minutes"
echo "===================================================================="
echo ""
echo "This script will:"
echo "  1. Forward your frontend service to localhost"
echo "  2. Create a public ngrok URL you can share with recruiters"
echo ""
echo "Prerequisites:"
echo "  - Install ngrok: brew install ngrok"
echo "  - Sign up at https://ngrok.com (free)"
echo "  - Get your authtoken from: https://dashboard.ngrok.com/get-started/your-authtoken"
echo ""

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo "❌ ngrok is not installed."
    echo ""
    echo "Install it with:"
    echo "  brew install ngrok"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo "✅ ngrok is installed"
echo ""

# Check if ngrok is authenticated
if ! ngrok config check &> /dev/null; then
    echo "⚠️  ngrok is not authenticated."
    echo ""
    echo "1. Sign up at: https://ngrok.com"
    echo "2. Get your authtoken from: https://dashboard.ngrok.com/get-started/your-authtoken"
    echo "3. Run: ngrok config add-authtoken YOUR_TOKEN"
    echo ""
    read -p "Have you done this? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Please authenticate ngrok first, then run this script again."
        exit 1
    fi
fi

echo "✅ ngrok is configured"
echo ""

echo "Starting port-forward in background..."
kubectl port-forward -n mini-xdr svc/mini-xdr-frontend-service 3000:3000 > /tmp/mini-xdr-port-forward.log 2>&1 &
PORT_FORWARD_PID=$!
echo "✅ Port-forward started (PID: $PORT_FORWARD_PID)"
echo ""

# Wait for port-forward to be ready
echo "Waiting for port-forward to be ready..."
sleep 3

# Test localhost
if curl -s -o /dev/null -w "%{http_code}" http://localhost:3000 | grep -q "200"; then
    echo "✅ Frontend is accessible on localhost:3000"
else
    echo "⚠️  Frontend might not be ready yet, but continuing..."
fi
echo ""

echo "Starting ngrok to create public URL..."
echo ""
echo "===================================================================="
echo "   🚀 Starting ngrok - DO NOT CLOSE THIS WINDOW"
echo "===================================================================="
echo ""
echo "Your public demo URL will appear below."
echo "Share this URL with recruiters!"
echo ""
echo "To stop the demo:"
echo "  1. Press Ctrl+C in this window"
echo "  2. Run: kill $PORT_FORWARD_PID"
echo ""
echo "-------------------------------------------------------------------"
echo ""

# Start ngrok
ngrok http 3000

# Cleanup when ngrok exits
echo ""
echo "Cleaning up..."
kill $PORT_FORWARD_PID 2>/dev/null || true
echo "✅ Demo stopped"


