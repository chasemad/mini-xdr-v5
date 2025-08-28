#!/bin/bash
# Mini-XDR AI Agent Test Script

echo "🤖 Mini-XDR AI Agent Test"
echo "=========================="

API_KEY="xdr-secure-api-key-2024"

echo "1. Testing Containment Agent..."
response1=$(curl -s -X POST http://localhost:8000/api/agents/orchestrate \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d '{"query": "analyze incident 14 and recommend containment actions"}')

echo "$response1" | jq .

echo -e "\n2. Testing IP Analysis..."
response2=$(curl -s -X POST http://localhost:8000/api/agents/orchestrate \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d '{"query": "analyze IP 192.168.168.132"}')

echo "$response2" | jq .

echo -e "\n3. Testing System Status Query..."
response3=$(curl -s -X POST http://localhost:8000/api/agents/orchestrate \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d '{"query": "what is the current threat level and system status?"}')

echo "$response3" | jq .

echo -e "\n4. Testing ML Model Retrain..."
retrain_response=$(curl -s -X POST http://localhost:8000/api/ml/retrain \
  -H "Content-Type: application/json" \
  -d '{"model_type": "ensemble"}')

echo "$retrain_response" | jq .

echo -e "\n🎯 AI Agent Test Complete!"

# Check if responses contain meaningful AI analysis
if [[ $response1 == *"confidence"* ]] && [[ $response1 != *"0.0"* ]]; then
    echo "✅ AI agents are fully operational with OpenAI integration!"
else
    echo "⚠️  AI agents responding but may need OpenAI API key for full functionality"
fi
