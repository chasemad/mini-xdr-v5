#!/bin/bash

# Enhanced MCP Server Demonstration Script
# This script demonstrates the new AI-powered capabilities

echo "🚀 Enhanced Mini-XDR MCP Server Demo"
echo "======================================"
echo ""

# Check if backend is running
echo "📋 Checking system status..."
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "❌ Backend server not running. Please start it first:"
    echo "   cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"
    exit 1
fi

echo "✅ Backend server is running"
echo ""

# Demo 1: Get incidents with enhanced filtering
echo "🔍 Demo 1: Enhanced Incident Listing"
echo "------------------------------------"
curl -s -X GET "http://localhost:8000/incidents?status=new&limit=5" | jq '.[:3] | .[] | {id, src_ip, status, reason}' 2>/dev/null || echo "No incidents found or jq not available"
echo ""

# Demo 2: Test orchestrator status
echo "🤖 Demo 2: Orchestrator Status"
echo "------------------------------"
curl -s http://localhost:8000/api/orchestrator/status | jq '.' 2>/dev/null || echo "Orchestrator status not available"
echo ""

# Demo 3: Show available MCP tools
echo "🛠️  Demo 3: Available MCP Tools"
echo "-------------------------------"
echo "The enhanced MCP server now provides the following tools:"
echo ""

tools=(
    "get_incidents - Enhanced incident listing with filtering"
    "analyze_incident_deep - AI-powered deep incident analysis"
    "threat_hunt - Execute AI-powered threat hunting"
    "orchestrate_response - Trigger orchestrated multi-agent response"
    "threat_intel_lookup - Comprehensive threat intelligence"
    "attribution_analysis - Threat actor attribution"
    "start_incident_stream - Real-time incident monitoring"
    "query_threat_patterns - Advanced threat pattern analysis"
    "correlation_analysis - Multi-incident correlation analysis"
    "forensic_investigation - Comprehensive forensic analysis"
    "get_orchestrator_status - Real-time orchestrator health"
    "get_workflow_status - Individual workflow tracking"
)

for i in "${!tools[@]}"; do
    echo "$((i+1)). ${tools[$i]}"
done

echo ""
echo "📚 Demo 4: Usage Examples"
echo "-------------------------"
echo "# Deep Incident Analysis:"
echo "analyze_incident_deep({"
echo "  incident_id: 123,"
echo "  workflow_type: 'comprehensive',"
echo "  include_threat_intel: true"
echo "})"
echo ""

echo "# Threat Hunting:"
echo "threat_hunt({"
echo "  query: 'brute force authentication attempts',"
echo "  hours_back: 24,"
echo "  threat_types: ['brute_force', 'reconnaissance']"
echo "})"
echo ""

echo "# Orchestrated Response:"
echo "orchestrate_response({"
echo "  incident_id: 123,"
echo "  workflow_type: 'comprehensive',"
echo "  priority: 'critical'"
echo "})"
echo ""

echo "🎯 Demo 5: Test Script Execution"
echo "---------------------------------"
echo "To test all enhanced features, run:"
echo "python test_enhanced_mcp.py"
echo ""

echo "📖 Demo 6: Documentation"
echo "------------------------"
echo "Complete documentation available in:"
echo "ENHANCED_MCP_GUIDE.md"
echo ""

echo "🔧 Demo 7: Environment Setup"
echo "-----------------------------"
echo "To enable all features, ensure these environment variables:"
echo "export ENABLE_STREAMING=true"
echo "export STREAMING_INTERVAL=5000"
echo "export API_BASE=http://localhost:8000"
echo ""

echo "✅ Enhanced MCP Server Demo Complete!"
echo "======================================"
echo ""
echo "🎉 Your Mini-XDR system now has enterprise-grade AI-powered"
echo "   incident response and orchestration capabilities!"
echo ""
echo "Next Steps:"
echo "1. Review ENHANCED_MCP_GUIDE.md for detailed usage"
echo "2. Run test_enhanced_mcp.py to verify all features"
echo "3. Integrate with your SOC tools and workflows"
echo "4. Configure real threat intelligence API keys"
echo ""
