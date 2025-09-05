# Enhanced MCP Server Guide

## Overview

The Enhanced MCP (Model Context Protocol) Server provides advanced AI-powered incident response and orchestration capabilities for the Mini-XDR system. This guide covers the new features and how to use them effectively.

## 🚀 New Capabilities

### 1. AI-Powered Incident Analysis
- **Deep Analysis**: Comprehensive incident analysis using all available AI agents
- **Threat Hunting**: Proactive threat hunting with AI assistance
- **Forensic Investigation**: Automated evidence collection and analysis
- **Attribution Analysis**: Threat actor attribution with confidence scoring

### 2. Orchestration Framework
- **Multi-Agent Coordination**: Intelligent agent communication and decision fusion
- **Workflow Management**: Track and manage complex incident response workflows
- **Real-time Monitoring**: Live incident streaming with filtering
- **Priority-based Response**: Intelligent prioritization and escalation

### 3. Threat Intelligence Integration
- **Comprehensive Lookups**: Multi-source threat intelligence queries
- **Reputation Analysis**: Automated reputation scoring and analysis
- **Geolocation Intelligence**: IP geolocation and network analysis
- **Campaign Correlation**: Link incidents to known threat campaigns

### 4. Advanced Analytics
- **Pattern Recognition**: Identify threat patterns across incidents
- **Correlation Analysis**: Multi-incident correlation and analysis
- **Risk Assessment**: Automated risk scoring and evaluation
- **Trend Analysis**: Long-term threat trend identification

## 🛠️ Available MCP Tools

### Basic Incident Management (Enhanced)

#### `get_incidents`
Enhanced incident listing with advanced filtering options.

**Parameters:**
- `status`: Filter by incident status (new, contained, open, closed)
- `limit`: Maximum number of incidents to return (1-100)
- `hours_back`: Look back time window in hours (1-168)

**Example:**
```typescript
// Get recent high-priority incidents
get_incidents({
  status: "new",
  limit: 50,
  hours_back: 24
})
```

### AI-Powered Analysis Tools

#### `analyze_incident_deep`
Perform comprehensive AI-powered incident analysis using all available agents.

**Parameters:**
- `incident_id`: Incident ID to analyze (required)
- `workflow_type`: Analysis type (basic, rapid, comprehensive)
- `include_threat_intel`: Include threat intelligence enrichment

**Example:**
```typescript
analyze_incident_deep({
  incident_id: 123,
  workflow_type: "comprehensive",
  include_threat_intel: true
})
```

#### `threat_hunt`
Execute AI-powered threat hunting queries across incident data.

**Parameters:**
- `query`: Threat hunting query or pattern (required)
- `hours_back`: Time window for hunting (1-168 hours)
- `threat_types`: Specific threat types to hunt for

**Example:**
```typescript
threat_hunt({
  query: "brute force authentication attempts",
  hours_back: 24,
  threat_types: ["brute_force", "reconnaissance"]
})
```

#### `forensic_investigation`
Initiate comprehensive forensic investigation with evidence collection.

**Parameters:**
- `incident_id`: Incident ID to investigate (required)
- `evidence_types`: Types of evidence to collect
- `include_network_capture`: Include network traffic capture

**Example:**
```typescript
forensic_investigation({
  incident_id: 123,
  evidence_types: ["event_logs", "network_artifacts", "file_artifacts"],
  include_network_capture: true
})
```

### Orchestration & Workflow Management

#### `orchestrate_response`
Trigger orchestrated multi-agent incident response.

**Parameters:**
- `incident_id`: Incident ID to orchestrate response for (required)
- `workflow_type`: Orchestration workflow type
- `priority`: Response priority level

**Example:**
```typescript
orchestrate_response({
  incident_id: 123,
  workflow_type: "comprehensive",
  priority: "critical"
})
```

#### `get_orchestrator_status`
Get comprehensive orchestrator status and active workflows.

**Parameters:**
- None required

**Example:**
```typescript
get_orchestrator_status()
```

#### `get_workflow_status`
Get status of a specific orchestration workflow.

**Parameters:**
- `workflow_id`: Workflow ID to check (required)

**Example:**
```typescript
get_workflow_status({
  workflow_id: "workflow-123-456"
})
```

### Threat Intelligence Tools

#### `threat_intel_lookup`
Perform comprehensive threat intelligence lookup.

**Parameters:**
- `ip_address`: IP address to analyze (required)
- `include_reputation`: Include reputation analysis
- `include_geolocation`: Include geolocation data
- `sources`: Specific intelligence sources to query

**Example:**
```typescript
threat_intel_lookup({
  ip_address: "192.168.1.100",
  include_reputation: true,
  include_geolocation: true,
  sources: ["virustotal", "abuseipdb", "alienvault"]
})
```

#### `attribution_analysis`
Perform threat actor attribution analysis.

**Parameters:**
- `incident_id`: Incident ID to attribute (required)
- `include_campaign_analysis`: Include campaign correlation
- `confidence_threshold`: Minimum confidence threshold

**Example:**
```typescript
attribution_analysis({
  incident_id: 123,
  include_campaign_analysis: true,
  confidence_threshold: 0.7
})
```

### Real-Time Monitoring

#### `start_incident_stream`
Start real-time incident monitoring stream.

**Parameters:**
- `client_id`: Unique client identifier (required)
- `filters`: Optional filtering criteria

**Example:**
```typescript
start_incident_stream({
  client_id: "soc_monitor_001",
  filters: {
    severity: ["high", "critical"],
    threat_categories: ["malware", "intrusion"]
  }
})
```

#### `stop_incident_stream`
Stop real-time incident monitoring stream.

**Parameters:**
- `client_id`: Client identifier to stop streaming for (required)

**Example:**
```typescript
stop_incident_stream({
  client_id: "soc_monitor_001"
})
```

### Advanced Analytics

#### `query_threat_patterns`
Query for specific threat patterns and behaviors.

**Parameters:**
- `pattern_type`: Type of threat pattern to query (required)
- `time_range`: Time window for analysis
- `min_confidence`: Minimum confidence threshold

**Example:**
```typescript
query_threat_patterns({
  pattern_type: "lateral_movement",
  time_range: { hours_back: 72 },
  min_confidence: 0.8
})
```

#### `correlation_analysis`
Analyze correlations between incidents and events.

**Parameters:**
- `correlation_type`: Type of correlation analysis (required)
- `incidents`: Incident IDs to correlate
- `time_window_hours`: Time window for correlation

**Example:**
```typescript
correlation_analysis({
  correlation_type: "temporal",
  incidents: [123, 124, 125],
  time_window_hours: 48
})
```

## 🔧 Configuration

### Environment Variables

```bash
# Enable streaming capabilities
ENABLE_STREAMING=true
STREAMING_INTERVAL=5000  # 5 seconds

# API Configuration
API_BASE=http://localhost:8000
API_KEY=your-api-key-here
```

### MCP Server Setup

1. **Install Dependencies:**
```bash
cd backend
npm install
```

2. **Configure Environment:**
```bash
export API_BASE="http://localhost:8000"
export ENABLE_STREAMING=true
```

3. **Start MCP Server:**
```bash
node app/mcp_server.ts
```

## 📊 Response Formats

### Orchestration Results
```json
{
  "orchestration_result": {
    "agents_involved": ["attribution", "forensics", "containment"],
    "execution_time": 2.34,
    "coordination": {
      "confidence_levels": {
        "overall": 0.85,
        "attribution": 0.78,
        "containment": 0.92
      },
      "recommended_actions": [
        "Block source IP 192.168.1.100",
        "Escalate to SOC lead",
        "Collect additional evidence"
      ]
    }
  }
}
```

### Threat Intelligence
```json
{
  "reputation_score": 85,
  "geolocation": {
    "country": "China",
    "city": "Beijing"
  },
  "threat_categories": ["malware", "botnet"],
  "sources": {
    "virustotal": "available",
    "abuseipdb": "available"
  },
  "summary": "High-risk IP associated with known malicious activity"
}
```

### Attribution Analysis
```json
{
  "confidence_score": 0.76,
  "threat_category": "APT",
  "attributed_actors": [
    {
      "name": "APT28",
      "confidence": 0.73
    }
  ],
  "infrastructure_analysis": {
    "infrastructure_clusters": [
      {
        "cluster_type": "command_and_control",
        "ips": ["192.168.1.100", "192.168.1.101"],
        "confidence_score": 0.89
      }
    ]
  }
}
```

## 🚀 Usage Scenarios

### Scenario 1: Rapid Incident Triage
```typescript
// Get recent high-severity incidents
const incidents = await get_incidents({
  status: "new",
  hours_back: 4
});

// Deep analysis of most critical incident
const analysis = await analyze_incident_deep({
  incident_id: incidents[0].id,
  workflow_type: "rapid",
  include_threat_intel: true
});
```

### Scenario 2: Threat Hunting Campaign
```typescript
// Start threat hunting for lateral movement
const huntResults = await threat_hunt({
  query: "lateral movement indicators",
  hours_back: 72,
  threat_types: ["lateral_movement", "privilege_escalation"]
});

// Check correlations between suspicious incidents
const correlations = await correlation_analysis({
  correlation_type: "behavioral",
  incidents: huntResults.suspicious_incidents,
  time_window_hours: 24
});
```

### Scenario 3: Real-Time SOC Monitoring
```typescript
// Start real-time monitoring for critical incidents
await start_incident_stream({
  client_id: "soc_monitor_main",
  filters: {
    severity: ["high", "critical"],
    threat_categories: ["intrusion", "malware"]
  }
});

// Get orchestrator status for dashboard
const status = await get_orchestrator_status();
```

### Scenario 4: Forensic Investigation
```typescript
// Initiate comprehensive forensic investigation
const forensics = await forensic_investigation({
  incident_id: 123,
  evidence_types: ["event_logs", "network_artifacts", "memory_dump"],
  include_network_capture: true
});

// Perform threat attribution
const attribution = await attribution_analysis({
  incident_id: 123,
  include_campaign_analysis: true,
  confidence_threshold: 0.8
});
```

## 📈 Performance Considerations

### Optimization Tips
1. **Use Appropriate Workflow Types**: Choose 'basic' for simple analysis, 'comprehensive' for thorough investigation
2. **Set Reasonable Time Windows**: Balance analysis depth with performance
3. **Configure Streaming Intervals**: Adjust based on incident volume and monitoring needs
4. **Use Filtering**: Reduce data processing by filtering irrelevant incidents

### Resource Management
- **Concurrent Workflows**: Limit to 5-10 simultaneous orchestrations
- **Streaming Clients**: Monitor active streams to prevent resource exhaustion
- **Cache Intelligence**: Implement caching for frequently queried threat intelligence
- **Batch Processing**: Group similar analysis requests for efficiency

## 🔒 Security Best Practices

1. **API Key Management**: Rotate API keys regularly
2. **Access Control**: Implement proper MCP client authentication
3. **Data Encryption**: Ensure encrypted communication channels
4. **Audit Logging**: Enable comprehensive audit logging for all operations
5. **Rate Limiting**: Implement rate limiting to prevent abuse

## 📚 Troubleshooting

### Common Issues

**Streaming Not Working:**
```bash
# Check environment variables
echo $ENABLE_STREAMING
# Should be 'true'
```

**Orchestrator Not Responding:**
```bash
# Check orchestrator status
curl http://localhost:8000/api/orchestrator/status
```

**Analysis Taking Too Long:**
- Reduce time windows in queries
- Use 'basic' workflow type for faster results
- Check agent health status

**Memory Issues:**
- Reduce concurrent workflows
- Implement result caching
- Monitor streaming client connections

## 🎯 Next Steps

1. **Integrate with SOC Tools**: Connect MCP server to existing SOC platforms
2. **Custom Workflows**: Develop organization-specific orchestration workflows
3. **Advanced Analytics**: Implement machine learning-based pattern recognition
4. **Automated Response**: Enable fully automated incident response capabilities
5. **Dashboard Integration**: Build real-time dashboards using streaming data

## 📞 Support

For additional assistance:
- Review the test script: `test_enhanced_mcp.py`
- Check orchestrator logs: `backend/backend.log`
- Monitor MCP server output for debugging information

---

*This enhanced MCP server represents a significant advancement in AI-powered incident response capabilities, providing SOC teams with powerful automation and orchestration tools.*
