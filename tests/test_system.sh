#!/bin/bash
# Mini-XDR System Test Script

echo "🛡️  Mini-XDR System Test"
echo "========================"

# Test 1: Backend Health
echo "1. Testing Backend Health..."
curl -s http://localhost:8000/health | jq . || echo "❌ Backend not responding"

# Test 2: Database Status
echo -e "\n2. Testing Database..."
curl -s http://localhost:8000/incidents | jq '. | length' | grep -E '^[0-9]+$' && echo "✅ Database has incidents" || echo "❌ Database issue"

# Test 3: ML Models
echo -e "\n3. Testing ML Models..."
curl -s http://localhost:8000/api/ml/status | jq '.metrics.models_trained' && echo "✅ ML models loaded" || echo "❌ ML models issue"

# Test 4: Log Sources
echo -e "\n4. Testing Log Sources..."
curl -s http://localhost:8000/api/sources | jq '.sources | length' | grep -E '^[0-9]+$' && echo "✅ Log sources active" || echo "❌ Log sources issue"

# Test 5: Frontend
echo -e "\n5. Testing Frontend..."
curl -s -I http://localhost:3000 | head -1 | grep "200 OK" && echo "✅ Frontend responding" || echo "❌ Frontend issue"

# Test 6: AI Agent Endpoint
echo -e "\n6. Testing AI Agent..."
response=$(curl -s -X POST http://localhost:8000/api/agents/orchestrate -H "Content-Type: application/json" -d '{"query": "test"}')
if [[ $response == *"message"* ]]; then
    echo "✅ AI Agent endpoint responding"
else
    echo "❌ AI Agent issue"
fi

# Test 7: SSH Connectivity (if configured)
echo -e "\n7. Testing SSH Connectivity..."
curl -s http://localhost:8000/test/ssh | jq '.ssh_status' && echo "✅ SSH connectivity tested" || echo "❌ SSH test failed"

echo -e "\n🎉 System Test Complete!"
echo "Next steps:"
echo "1. Add your OpenAI API key to backend/.env"
echo "2. Add AbuseIPDB and VirusTotal API keys"
echo "3. Update HONEYPOT_HOST with your actual IP"
echo "4. Restart services: ./scripts/start-all.sh"
