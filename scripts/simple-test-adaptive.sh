#!/bin/bash
# Simple Adaptive Detection Test
# Tests the adaptive detection system with clean JSON

BASE_URL="http://localhost:8000"

echo "🧠 Simple Adaptive Detection Test"
echo "================================="

# Test 1: Single web attack
echo "1. Testing single web attack..."
response=$(curl -s -X POST "$BASE_URL/ingest/multi" \
    -H 'Content-Type: application/json' \
    -H 'Authorization: Bearer test-api-key' \
    -d '{
        "source_type": "webhoneypot",
        "hostname": "test-server",
        "events": [
            {
                "eventid": "webhoneypot.request",
                "src_ip": "192.168.100.99",
                "dst_port": 80,
                "message": "GET /admin",
                "raw": {
                    "path": "/admin",
                    "status_code": 404,
                    "user_agent": "AttackBot/1.0",
                    "attack_indicators": ["admin_scan"]
                }
            }
        ]
    }')

echo "Response: $response"
processed=$(echo "$response" | jq -r '.processed' 2>/dev/null)
incidents=$(echo "$response" | jq -r '.incidents_detected' 2>/dev/null)
echo "✅ Processed: $processed, Incidents: $incidents"

# Test 2: Rapid attack sequence
echo ""
echo "2. Testing rapid attack sequence..."
for i in {1..5}; do
    path="/admin$i"
    response=$(curl -s -X POST "$BASE_URL/ingest/multi" \
        -H 'Content-Type: application/json' \
        -H 'Authorization: Bearer test-api-key' \
        -d '{
            "source_type": "webhoneypot", 
            "hostname": "test-server",
            "events": [
                {
                    "eventid": "webhoneypot.request",
                    "src_ip": "192.168.100.77", 
                    "dst_port": 80,
                    "message": "GET '"$path"'",
                    "raw": {
                        "path": "'"$path"'",
                        "status_code": 404,
                        "user_agent": "RapidBot/1.0",
                        "attack_indicators": ["admin_scan", "rapid_enumeration"]
                    }
                }
            ]
        }')
    echo -n "."
    sleep 0.3
done
echo ""

# Check incidents
echo ""
echo "3. Checking for incidents..."
incidents=$(curl -s "$BASE_URL/incidents")
count=$(echo "$incidents" | jq 'length' 2>/dev/null)
echo "✅ Total incidents: $count"

if [ "$count" -gt 0 ]; then
    echo "Latest incidents:"
    echo "$incidents" | jq '.[0:3]' 2>/dev/null
fi

# Test learning pipeline
echo ""
echo "4. Testing learning pipeline..."
learning=$(curl -s -X POST "$BASE_URL/api/adaptive/force_learning")
echo "Learning response: $learning"

echo ""
echo "✅ Simple test completed!"
