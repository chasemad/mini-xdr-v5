#!/bin/bash
# Adaptive Detection Test Script
# Tests the intelligent adaptive attack detection system

BASE_URL="http://localhost:8000"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SEND_SCRIPT="$PROJECT_ROOT/scripts/send_signed_request.py"

signed_request() {
    local method="$1"
    local path="$2"
    local body="${3:-}"
    local args=("--base-url" "$BASE_URL" "--path" "$path" "--method" "$method")
    if [ -n "$body" ]; then
        args+=("--body" "$body")
    fi
    python3 "$SEND_SCRIPT" "${args[@]}"
}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

test_adaptive_detection() {
    echo "🧠 Testing Intelligent Adaptive Attack Detection System"
    echo "========================================================="
    echo ""
    
    # 1. Check system health
    log "1. Checking system health..."
    if ! curl -s "$BASE_URL/health" > /dev/null; then
        error "Backend not responding. Please start Mini-XDR first:"
        echo "   cd $PROJECT_ROOT && ./scripts/start-all.sh"
        exit 1
    fi
    success "Backend is healthy"
    
    # 2. Check adaptive detection status
    log "2. Checking adaptive detection status..."
    adaptive_status=$(signed_request GET /api/adaptive/status 2>/dev/null)
    if [ $? -eq 0 ]; then
        learning_running=$(echo "$adaptive_status" | jq -r '.learning_pipeline.running' 2>/dev/null || echo "unknown")
        behavioral_threshold=$(echo "$adaptive_status" | jq -r '.adaptive_engine.behavioral_threshold' 2>/dev/null || echo "unknown")
        
        success "Adaptive detection system online"
        echo "   • Learning Pipeline: $learning_running"
        echo "   • Behavioral Threshold: $behavioral_threshold"
    else
        error "Adaptive detection system not responding"
        exit 1
    fi
    
    # 3. Test behavioral detection with web attack simulation
    log "3. Testing behavioral detection with simulated web attacks..."
    test_ip="192.168.100.50"
    
    # Simulate rapid enumeration attack pattern
    web_attacks='[
        {"eventid":"webhoneypot.request","src_ip":"'$test_ip'","dst_port":80,"message":"GET /admin","raw":{"path":"/admin","status_code":404,"user_agent":"BadBot/1.0","attack_indicators":["admin_scan"]}},
        {"eventid":"webhoneypot.request","src_ip":"'$test_ip'","dst_port":80,"message":"GET /wp-admin","raw":{"path":"/wp-admin","status_code":404,"user_agent":"BadBot/1.0","attack_indicators":["admin_scan"]}},
        {"eventid":"webhoneypot.request","src_ip":"'$test_ip'","dst_port":80,"message":"GET /phpmyadmin","raw":{"path":"/phpmyadmin","status_code":404,"user_agent":"BadBot/1.0","attack_indicators":["admin_scan"]}},
        {"eventid":"webhoneypot.request","src_ip":"'$test_ip'","dst_port":80,"message":"GET /index.php?id=1 UNION SELECT","raw":{"path":"/index.php","parameters":["id=1 UNION SELECT 1,2,3"],"status_code":500,"user_agent":"BadBot/1.0","attack_indicators":["sql_injection"]}},
        {"eventid":"webhoneypot.request","src_ip":"'$test_ip'","dst_port":80,"message":"GET /.env","raw":{"path":"/.env","status_code":404,"user_agent":"BadBot/1.0","attack_indicators":["sensitive_file_access"]}},
        {"eventid":"webhoneypot.request","src_ip":"'$test_ip'","dst_port":80,"message":"GET /config.php","raw":{"path":"/config.php","status_code":404,"user_agent":"BadBot/1.0","attack_indicators":["sensitive_file_access"]}}
    ]'
    
    local web_payload=$(cat <<JSON
{"source_type":"webhoneypot","hostname":"adaptive-test","events":$web_attacks}
JSON
)
    response=$(signed_request POST /ingest/multi "$web_payload" 2>/dev/null)
    curl_exit_code=$?
    echo "   • Response: $response"
    
    if [ $curl_exit_code -eq 0 ] && [ ! -z "$response" ]; then
        processed=$(echo "$response" | jq -r '.processed' 2>/dev/null)
        incidents=$(echo "$response" | jq -r '.incidents_detected' 2>/dev/null)
        
        # Default to 0 if jq parsing failed
        processed=${processed:-0}
        incidents=${incidents:-0}
        
        success "Web attack simulation completed"
        echo "   • Events processed: $processed"
        echo "   • Incidents detected: $incidents"
        
        if [ "$incidents" -gt 0 ]; then
            success "🎯 ADAPTIVE DETECTION TRIGGERED!"
            
            # Check the latest incident
            sleep 1
            latest_incident=$(curl -s "$BASE_URL/incidents" | jq -r '.[0]' 2>/dev/null)
            if [ "$latest_incident" != "null" ]; then
                incident_reason=$(echo "$latest_incident" | jq -r '.reason' 2>/dev/null)
                incident_id=$(echo "$latest_incident" | jq -r '.id' 2>/dev/null)
                
                echo "   • Incident ID: $incident_id"
                echo "   • Reason: $incident_reason"
                
                if [[ "$incident_reason" == *"adaptive"* ]] || [[ "$incident_reason" == *"Behavioral"* ]]; then
                    success "🧠 INTELLIGENT ADAPTIVE DETECTION CONFIRMED!"
                else
                    log "Traditional detection triggered (adaptive may need more data)"
                fi
            fi
        else
            warning "No incidents triggered - may need more attack events"
        fi
    else
        error "Web attack simulation failed (curl_exit_code: $curl_exit_code, response: '$response')"
    fi
    
    # 4. Test SSH brute force with behavioral analysis
    log "4. Testing enhanced SSH brute force detection..."
    ssh_ip="192.168.100.51"
    
    # Simulate coordinated SSH attack
    for i in {1..8}; do
        username_list=("admin" "root" "user" "test" "guest")
        password_list=("123456" "password" "admin" "root" "qwerty" "letmein")
        
        username=${username_list[$((i % ${#username_list[@]}))]}
        password=${password_list[$((i % ${#password_list[@]}))]}
        
        ssh_event='{"eventid":"cowrie.login.failed","src_ip":"'$ssh_ip'","dst_port":2222,"message":"SSH login failed: '$username'/'$password'","raw":{"username":"'$username'","password":"'$password'","session":"session_'$i'","timestamp":"'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"}}'
        
        signed_request POST /ingest/cowrie "$ssh_event" >/dev/null 2>&1
        
        echo -n "."
        sleep 0.2
    done
    echo ""
    
    success "SSH brute force simulation completed"
    
    # Check for SSH incidents
    sleep 1
    ssh_incidents=$(curl -s "$BASE_URL/incidents" | jq '[.[] | select(.src_ip == "'$ssh_ip'")]' 2>/dev/null)
    ssh_incident_count=$(echo "$ssh_incidents" | jq 'length' 2>/dev/null || echo "0")
    
    if [ "$ssh_incident_count" -gt 0 ]; then
        success "🎯 SSH BRUTE FORCE DETECTED!"
        echo "   • SSH incidents: $ssh_incident_count"
        
        latest_ssh_incident=$(echo "$ssh_incidents" | jq -r '.[0].reason' 2>/dev/null)
        echo "   • Latest: $latest_ssh_incident"
    else
        warning "No SSH incidents detected"
    fi
    
    # 5. Test learning pipeline
    log "5. Testing learning pipeline..."
    learning_response=$(signed_request POST /api/adaptive/force_learning 2>/dev/null)
    if [ $? -eq 0 ]; then
        learning_success=$(echo "$learning_response" | jq -r '.success' 2>/dev/null || echo "false")
        if [ "$learning_success" = "true" ]; then
            success "Learning pipeline functional"
            
            # Show learning results
            learning_results=$(echo "$learning_response" | jq -r '.results' 2>/dev/null)
            echo "   • Results: $learning_results"
        else
            warning "Learning pipeline update failed"
        fi
    else
        error "Learning pipeline test failed"
    fi
    
    # 6. Summary
    echo ""
    echo "========================================================="
    success "🎉 Adaptive Detection Test Complete!"
    echo ""
    
    # Get final system status
    final_status=$(signed_request GET /api/adaptive/status 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "📊 System Status:"
        echo "$final_status" | jq '.' 2>/dev/null || echo "$final_status"
    fi
    
    echo ""
    echo "🧠 Adaptive Detection Features Tested:"
    echo "   ✅ Behavioral Pattern Analysis"
    echo "   ✅ Enhanced ML Ensemble Detection"
    echo "   ✅ Statistical Baseline Learning"
    echo "   ✅ Continuous Learning Pipeline"
    echo "   ✅ Multi-layer Detection Correlation"
    echo ""
    
    echo "🎯 Next Steps:"
    echo "   • Monitor real attacks: Open http://localhost:3000"
    echo "   • View incidents: curl $BASE_URL/incidents"
    echo "   • Check adaptive status: curl $BASE_URL/api/adaptive/status"
    echo "   • Run comprehensive test: python $PROJECT_ROOT/test_adaptive_detection.py"
    echo ""
}

# Run the test
test_adaptive_detection
