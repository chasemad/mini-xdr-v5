# 🚀 Mini-XDR: API Reference Documentation

**Comprehensive API reference for Mini-XDR Extended Detection & Response Platform**

> **API Version**: v1.0
> **Base URL**: `https://your-domain.com/api` (Production) | `http://localhost:8000/api` (Development)
> **Authentication**: HMAC-SHA256 with nonce replay protection
> **Content-Type**: `application/json`

---

## 📋 Table of Contents

1. [Authentication](#authentication)
2. [Core Endpoints](#core-endpoints)
3. [AI Agent Orchestration](#ai-agent-orchestration)
4. [Machine Learning](#machine-learning)
5. [Incident Management](#incident-management)
6. [Event Ingestion](#event-ingestion)
7. [Response Actions](#response-actions)
8. [Threat Intelligence](#threat-intelligence)
9. [System Management](#system-management)
10. [WebSocket Connections](#websocket-connections)
11. [Error Handling](#error-handling)
12. [Rate Limiting](#rate-limiting)

---

## 🔐 Authentication

### HMAC Authentication

Mini-XDR uses HMAC-SHA256 authentication with nonce-based replay protection for secure API access.

#### Required Headers

```http
X-Device-ID: your-device-id
X-Timestamp: 1640995200
X-Nonce: unique-request-nonce
X-Signature: computed-hmac-signature
Content-Type: application/json
```

#### Signature Generation

```python
import hashlib
import hmac
import time
import uuid

def generate_signature(method, path, body, timestamp, nonce, secret_key):
    canonical_message = f"{method}|{path}|{body}|{timestamp}|{nonce}"
    return hmac.new(
        secret_key.encode(),
        canonical_message.encode(),
        hashlib.sha256
    ).hexdigest()

# Example usage
device_id = "your-device-id"
secret_key = "your-hmac-secret-key"
method = "POST"
path = "/api/incidents"
body = '{"src_ip": "192.168.1.100"}'
timestamp = str(int(time.time()))
nonce = str(uuid.uuid4())

signature = generate_signature(method, path, body, timestamp, nonce, secret_key)
```

#### Simple API Key Authentication

Some endpoints support simple API key authentication for easier integration:

```http
Authorization: Bearer your-api-key
```

**Endpoints supporting simple auth:**
- `/api/response/*` - Response system endpoints
- `/api/intelligence/*` - Visualization endpoints
- `/api/incidents/*` - Incident endpoints with AI analysis
- `/api/ml/*` - ML and SageMaker endpoints

---

## 🏗️ Core Endpoints

### System Health

#### GET `/health`

Check system health and status.

**Authentication**: None required

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-09-29T12:00:00Z",
  "version": "1.0.0",
  "components": {
    "database": "healthy",
    "ai_agents": "healthy",
    "ml_models": "healthy",
    "secrets_manager": "healthy"
  },
  "metrics": {
    "uptime_seconds": 86400,
    "total_incidents": 156,
    "active_agents": 6,
    "ml_models_loaded": 4
  }
}
```

#### GET `/api/system/status`

Detailed system status with performance metrics.

**Authentication**: API Key

**Response**:
```json
{
  "system": {
    "status": "operational",
    "load": {
      "cpu_percent": 45.2,
      "memory_percent": 62.1,
      "disk_usage_percent": 34.8
    },
    "database": {
      "status": "connected",
      "pool_size": 10,
      "active_connections": 3,
      "query_time_avg_ms": 12.4
    },
    "ml_engine": {
      "status": "ready",
      "models_loaded": 4,
      "inference_time_avg_ms": 248,
      "accuracy_current": 0.94
    }
  }
}
```

---

## 🤖 AI Agent Orchestration

### Agent Management

#### GET `/api/agents/health`

Get health status of all AI agents.

**Authentication**: HMAC or API Key

**Response**:
```json
{
  "agents": {
    "containment_agent": {
      "status": "active",
      "last_heartbeat": "2024-09-29T12:00:00Z",
      "actions_executed": 23,
      "success_rate": 0.96,
      "average_response_time_ms": 1850
    },
    "attribution_agent": {
      "status": "active",
      "last_heartbeat": "2024-09-29T12:00:00Z",
      "analysis_count": 45,
      "confidence_avg": 0.89,
      "threat_intel_lookups": 127
    },
    "forensics_agent": {
      "status": "active",
      "last_heartbeat": "2024-09-29T12:00:00Z",
      "evidence_collected": 67,
      "timeline_reconstructions": 12
    }
  }
}
```

#### POST `/api/agents/orchestrate`

Coordinate AI agents for incident analysis and response.

**Authentication**: HMAC or API Key

**Request**:
```json
{
  "message": "Analyze incident #123 for potential threats",
  "incident_id": 123,
  "context": {
    "incident_data": {
      "src_ip": "192.168.1.100",
      "reason": "Multiple failed SSH login attempts",
      "risk_score": 0.8
    },
    "priority": "high",
    "requested_agents": ["containment", "attribution", "forensics"]
  }
}
```

**Response**:
```json
{
  "orchestration_id": "orch-uuid-123",
  "status": "in_progress",
  "agents_coordinated": ["containment_agent", "attribution_agent", "forensics_agent"],
  "analysis": {
    "threat_assessment": {
      "confidence": 0.92,
      "threat_type": "brute_force_attack",
      "severity": "high",
      "recommended_actions": ["block_ip", "reset_passwords"]
    },
    "attribution": {
      "likely_source": "automated_botnet",
      "campaign_correlation": "similar_attacks_detected",
      "threat_actor_profile": "script_kiddie"
    },
    "forensics": {
      "evidence_summary": "Login attempts from multiple source IPs",
      "timeline": "2024-09-29T11:45:00Z - 2024-09-29T12:00:00Z",
      "affected_accounts": ["admin", "root", "ubuntu"]
    }
  },
  "recommendations": [
    {
      "action": "block_ip",
      "confidence": 0.95,
      "reasoning": "Clear brute force pattern detected"
    }
  ],
  "message": "Analysis complete. High-confidence brute force attack detected from 192.168.1.100. Recommend immediate IP blocking."
}
```

---

## 🧠 Machine Learning

### Model Management

#### GET `/api/ml/status`

Get ML model status and performance metrics.

**Authentication**: API Key

**Response**:
```json
{
  "models": {
    "isolation_forest": {
      "status": "loaded",
      "accuracy": 0.94,
      "last_trained": "2024-09-25T10:30:00Z",
      "predictions_count": 1547,
      "avg_inference_time_ms": 45
    },
    "lstm_autoencoder": {
      "status": "loaded",
      "reconstruction_error_threshold": 0.1,
      "sequence_length": 10,
      "predictions_count": 892,
      "avg_inference_time_ms": 120
    },
    "xgboost_classifier": {
      "status": "loaded",
      "accuracy": 0.96,
      "feature_importance": {
        "src_ip_reputation": 0.25,
        "request_frequency": 0.22,
        "time_of_day": 0.18
      }
    }
  },
  "ensemble": {
    "status": "active",
    "combined_accuracy": 0.97,
    "voting_strategy": "weighted_average",
    "total_predictions": 3521
  }
}
```

#### POST `/api/ml/predict`

Get ML-based threat prediction for event data.

**Authentication**: HMAC or API Key

**Request**:
```json
{
  "events": [
    {
      "src_ip": "192.168.1.100",
      "dst_port": 22,
      "eventid": "ssh_login_failed",
      "timestamp": "2024-09-29T12:00:00Z",
      "features": {
        "request_frequency": 15,
        "time_since_last": 2,
        "geographic_distance": 5000
      }
    }
  ],
  "models": ["isolation_forest", "xgboost_classifier"]
}
```

**Response**:
```json
{
  "predictions": [
    {
      "event_id": 0,
      "anomaly_score": 0.89,
      "threat_probability": 0.92,
      "confidence": 0.94,
      "model_predictions": {
        "isolation_forest": {
          "anomaly_score": 0.85,
          "is_outlier": true
        },
        "xgboost_classifier": {
          "threat_probability": 0.94,
          "predicted_class": "malicious"
        }
      },
      "feature_contributions": {
        "request_frequency": 0.35,
        "src_ip_reputation": -0.12,
        "time_of_day": 0.08
      }
    }
  ],
  "ensemble_result": {
    "final_score": 0.92,
    "risk_level": "high",
    "recommended_action": "immediate_investigation"
  }
}
```

### SageMaker Integration

#### GET `/api/ml/sagemaker/status`

Get SageMaker endpoint status and health.

**Authentication**: API Key

**Response**:
```json
{
  "endpoint": {
    "name": "mini-xdr-ml-endpoint-20240929",
    "status": "InService",
    "instance_type": "ml.t2.medium",
    "instance_count": 1,
    "creation_time": "2024-09-29T08:00:00Z"
  },
  "health": {
    "last_health_check": "2024-09-29T12:00:00Z",
    "response_time_ms": 245,
    "success_rate": 0.998,
    "error_rate": 0.002
  }
}
```

---

## 📊 Incident Management

### Incident Operations

#### GET `/api/incidents`

Retrieve incidents with filtering and pagination.

**Authentication**: API Key

**Query Parameters**:
- `limit` (integer): Maximum number of incidents to return (default: 50)
- `offset` (integer): Offset for pagination (default: 0)
- `status` (string): Filter by status (open, contained, dismissed)
- `severity` (string): Filter by severity (low, medium, high, critical)
- `src_ip` (string): Filter by source IP address
- `since` (string): ISO timestamp for incidents since date

**Response**:
```json
{
  "incidents": [
    {
      "id": 123,
      "created_at": "2024-09-29T12:00:00Z",
      "src_ip": "192.168.1.100",
      "reason": "Multiple failed SSH login attempts detected",
      "status": "open",
      "auto_contained": false,
      "risk_score": 0.85,
      "escalation_level": "high",
      "threat_category": "brute_force",
      "containment_confidence": 0.92,
      "agent_confidence": 0.89,
      "triage_note": {
        "summary": "Brute force SSH attack from suspicious IP",
        "severity": "high",
        "recommendation": "Block source IP and reset affected passwords",
        "rationale": [
          "15 failed login attempts in 5 minutes",
          "IP not in whitelist",
          "No legitimate user activity from this IP"
        ]
      },
      "ml_features": {
        "request_frequency": 15,
        "geographic_anomaly": 0.8,
        "time_anomaly": 0.3
      }
    }
  ],
  "pagination": {
    "total": 156,
    "limit": 50,
    "offset": 0,
    "has_next": true
  }
}
```

#### GET `/api/incidents/{incident_id}`

Get detailed information about specific incident.

**Authentication**: API Key

**Response**:
```json
{
  "incident": {
    "id": 123,
    "created_at": "2024-09-29T12:00:00Z",
    "updated_at": "2024-09-29T12:05:00Z",
    "src_ip": "192.168.1.100",
    "dst_ip": "10.0.1.50",
    "dst_port": 22,
    "reason": "Multiple failed SSH login attempts detected",
    "status": "contained",
    "auto_contained": true,
    "containment_actions": [
      {
        "action": "block_ip",
        "executed_at": "2024-09-29T12:02:00Z",
        "agent_id": "containment_agent",
        "result": "success",
        "duration": 3600
      }
    ],
    "events": [
      {
        "id": 1001,
        "timestamp": "2024-09-29T11:58:00Z",
        "eventid": "ssh_login_failed",
        "message": "Failed login attempt for user 'root'",
        "raw": {
          "user": "root",
          "source": "192.168.1.100",
          "port": 22
        }
      }
    ],
    "analysis": {
      "threat_classification": "brute_force_attack",
      "attack_vector": "ssh_bruteforce",
      "target_assets": ["10.0.1.50"],
      "potential_impact": "unauthorized_access",
      "mitigation_status": "contained"
    }
  }
}
```

#### POST `/api/incidents/{incident_id}/analyze`

Trigger AI analysis for specific incident.

**Authentication**: API Key

**Request**:
```json
{
  "analysis_type": "comprehensive",
  "include_agents": ["attribution", "forensics"],
  "priority": "high"
}
```

**Response**:
```json
{
  "analysis_id": "analysis-uuid-456",
  "status": "completed",
  "results": {
    "threat_assessment": {
      "confidence": 0.94,
      "threat_type": "automated_attack",
      "sophistication": "low",
      "persistence": false
    },
    "attribution": {
      "source_country": "Unknown",
      "organization": "likely_botnet",
      "attack_campaign": "mass_ssh_scanning"
    },
    "impact_analysis": {
      "affected_systems": 1,
      "data_exposure_risk": "low",
      "business_impact": "minimal"
    },
    "recommendations": [
      "Maintain IP block for 24 hours",
      "Monitor for similar patterns from different IPs",
      "Review SSH hardening configuration"
    ]
  }
}
```

---

## 📥 Event Ingestion

### Event Processing

#### POST `/ingest/events`

Ingest security events from various sources.

**Authentication**: HMAC Required

**Request**:
```json
{
  "source_type": "cowrie",
  "hostname": "honeypot-01",
  "events": [
    {
      "timestamp": "2024-09-29T12:00:00Z",
      "src_ip": "192.168.1.100",
      "dst_ip": "10.0.1.50",
      "dst_port": 22,
      "eventid": "ssh_login_failed",
      "message": "Failed login attempt for user 'admin'",
      "raw": {
        "user": "admin",
        "password": "password123",
        "session": "session-uuid-789"
      },
      "signature": "event-integrity-hash"
    }
  ]
}
```

**Response**:
```json
{
  "status": "success",
  "events_processed": 1,
  "events_created": 1,
  "processing_time_ms": 156,
  "incidents_triggered": [
    {
      "incident_id": 124,
      "reason": "Failed login threshold exceeded",
      "auto_contained": true
    }
  ]
}
```

#### POST `/ingest/bulk`

Bulk event ingestion for high-volume sources.

**Authentication**: HMAC Required

**Request Headers**:
```http
Content-Encoding: gzip
X-Batch-Size: 1000
```

**Request**: Gzipped JSON array of events

**Response**:
```json
{
  "status": "success",
  "batch_size": 1000,
  "events_processed": 1000,
  "events_created": 847,
  "duplicates_skipped": 153,
  "processing_time_ms": 2340,
  "incidents_triggered": 3,
  "ml_predictions_generated": 1000
}
```

---

## 🛡️ Response Actions

### Automated Response

#### POST `/api/response/block-ip`

Block IP address using automated containment.

**Authentication**: API Key

**Request**:
```json
{
  "incident_id": 123,
  "ip_address": "192.168.1.100",
  "duration_seconds": 3600,
  "reason": "Brute force attack detected",
  "auto_unblock": true
}
```

**Response**:
```json
{
  "action_id": "action-uuid-101",
  "status": "success",
  "ip_address": "192.168.1.100",
  "blocked_at": "2024-09-29T12:00:00Z",
  "expires_at": "2024-09-29T13:00:00Z",
  "method": "iptables_drop",
  "agent_confidence": 0.95,
  "rollback_available": true
}
```

#### POST `/api/response/isolate-host`

Isolate compromised host from network.

**Authentication**: API Key

**Request**:
```json
{
  "incident_id": 123,
  "hostname": "workstation-01",
  "ip_address": "10.0.1.100",
  "isolation_level": "quarantine",
  "preserve_evidence": true
}
```

**Response**:
```json
{
  "action_id": "action-uuid-102",
  "status": "success",
  "hostname": "workstation-01",
  "isolated_at": "2024-09-29T12:00:00Z",
  "isolation_method": "network_acl",
  "evidence_preserved": true,
  "recovery_instructions": "Contact IT for manual review and cleanup"
}
```

#### POST `/api/response/reset-passwords`

Reset passwords for affected accounts.

**Authentication**: API Key

**Request**:
```json
{
  "incident_id": 123,
  "accounts": ["admin", "root", "service-account"],
  "force_mfa_setup": true,
  "notify_users": true
}
```

**Response**:
```json
{
  "action_id": "action-uuid-103",
  "status": "success",
  "accounts_reset": 3,
  "temporary_passwords_generated": true,
  "mfa_enforcement": "enabled",
  "notifications_sent": true,
  "password_policy": {
    "min_length": 12,
    "complexity": "high",
    "expiry_days": 90
  }
}
```

### Action Management

#### GET `/api/response/actions`

List all response actions with status.

**Authentication**: API Key

**Response**:
```json
{
  "actions": [
    {
      "id": "action-uuid-101",
      "type": "block_ip",
      "status": "active",
      "created_at": "2024-09-29T12:00:00Z",
      "expires_at": "2024-09-29T13:00:00Z",
      "target": "192.168.1.100",
      "incident_id": 123,
      "agent_id": "containment_agent",
      "rollback_available": true
    }
  ],
  "summary": {
    "active_actions": 5,
    "completed_actions": 28,
    "failed_actions": 1
  }
}
```

#### POST `/api/response/actions/{action_id}/rollback`

Rollback a specific response action.

**Authentication**: API Key

**Request**:
```json
{
  "reason": "False positive - legitimate traffic",
  "approve_rollback": true
}
```

**Response**:
```json
{
  "rollback_id": "rollback-uuid-201",
  "status": "success",
  "original_action": "block_ip",
  "target": "192.168.1.100",
  "rolled_back_at": "2024-09-29T12:30:00Z",
  "verification": {
    "connectivity_restored": true,
    "no_side_effects": true
  }
}
```

---

## 🔍 Threat Intelligence

### Intelligence Lookup

#### POST `/api/intelligence/lookup`

Lookup threat intelligence for IP addresses, domains, or hashes.

**Authentication**: API Key

**Request**:
```json
{
  "indicators": [
    {
      "type": "ip",
      "value": "192.168.1.100"
    },
    {
      "type": "domain",
      "value": "malicious-domain.com"
    },
    {
      "type": "hash",
      "value": "d41d8cd98f00b204e9800998ecf8427e"
    }
  ],
  "sources": ["abuseipdb", "virustotal"]
}
```

**Response**:
```json
{
  "results": [
    {
      "indicator": "192.168.1.100",
      "type": "ip",
      "threat_level": "high",
      "confidence": 0.89,
      "sources": {
        "abuseipdb": {
          "abuse_confidence": 75,
          "country": "Unknown",
          "usage_type": "hosting",
          "reports_count": 15
        },
        "virustotal": {
          "malicious_votes": 8,
          "total_votes": 12,
          "last_analysis": "2024-09-29T10:00:00Z"
        }
      },
      "tags": ["bruteforce", "scanner", "malware_c2"],
      "first_seen": "2024-09-20T14:30:00Z",
      "last_seen": "2024-09-29T11:45:00Z"
    }
  ],
  "summary": {
    "total_indicators": 1,
    "malicious_count": 1,
    "suspicious_count": 0,
    "clean_count": 0
  }
}
```

#### GET `/api/intelligence/feeds`

Get current threat intelligence feed status.

**Authentication**: API Key

**Response**:
```json
{
  "feeds": [
    {
      "name": "abuseipdb",
      "status": "active",
      "last_update": "2024-09-29T11:00:00Z",
      "next_update": "2024-09-29T15:00:00Z",
      "indicators_count": 15420,
      "quality_score": 0.94
    },
    {
      "name": "virustotal",
      "status": "active",
      "last_update": "2024-09-29T11:30:00Z",
      "next_update": "2024-09-29T15:30:00Z",
      "indicators_count": 8976,
      "quality_score": 0.97
    }
  ]
}
```

---

## 🔧 System Management

### Configuration

#### GET `/api/system/config`

Get current system configuration.

**Authentication**: API Key

**Response**:
```json
{
  "detection": {
    "fail_threshold": 6,
    "fail_window_seconds": 60,
    "auto_contain": false,
    "ml_threshold": 0.8
  },
  "agents": {
    "orchestration_enabled": true,
    "active_agents": 6,
    "coordination_timeout": 30
  },
  "integrations": {
    "honeypot_enabled": true,
    "sagemaker_enabled": true,
    "threat_intel_enabled": true
  }
}
```

#### PUT `/api/system/config`

Update system configuration.

**Authentication**: HMAC Required

**Request**:
```json
{
  "detection": {
    "fail_threshold": 8,
    "auto_contain": true,
    "ml_threshold": 0.85
  }
}
```

**Response**:
```json
{
  "status": "success",
  "updated_fields": ["fail_threshold", "auto_contain", "ml_threshold"],
  "restart_required": false
}
```

### Metrics and Analytics

#### GET `/api/analytics/metrics`

Get system performance and security metrics.

**Authentication**: API Key

**Query Parameters**:
- `timerange` (string): Time range for metrics (1h, 24h, 7d, 30d)
- `granularity` (string): Data granularity (minute, hour, day)

**Response**:
```json
{
  "timerange": "24h",
  "metrics": {
    "incidents": {
      "total": 45,
      "by_severity": {
        "critical": 2,
        "high": 8,
        "medium": 20,
        "low": 15
      },
      "by_status": {
        "open": 12,
        "contained": 28,
        "dismissed": 5
      }
    },
    "events": {
      "total": 8756,
      "by_source": {
        "cowrie": 4523,
        "suricata": 2134,
        "osquery": 1890,
        "custom": 209
      }
    },
    "ml_performance": {
      "predictions": 8756,
      "accuracy": 0.94,
      "false_positives": 124,
      "false_negatives": 43
    },
    "response_actions": {
      "total": 35,
      "successful": 33,
      "failed": 2,
      "rolled_back": 3
    }
  }
}
```

---

## 🔌 WebSocket Connections

### Real-time Event Streaming

#### WebSocket `/ws/events`

Real-time event streaming for dashboards and monitoring.

**Authentication**: Query parameter `token=your-api-key`

**Connection URL**: `wss://your-domain.com/ws/events?token=your-api-key`

**Message Types**:

**Event Stream**:
```json
{
  "type": "event",
  "data": {
    "id": 1001,
    "timestamp": "2024-09-29T12:00:00Z",
    "src_ip": "192.168.1.100",
    "eventid": "ssh_login_failed",
    "severity": "medium"
  }
}
```

**Incident Alert**:
```json
{
  "type": "incident",
  "data": {
    "id": 123,
    "status": "new",
    "severity": "high",
    "src_ip": "192.168.1.100",
    "reason": "Brute force attack detected"
  }
}
```

**Agent Status**:
```json
{
  "type": "agent_status",
  "data": {
    "agent_id": "containment_agent",
    "status": "active",
    "action": "blocking_ip",
    "target": "192.168.1.100"
  }
}
```

**System Metrics**:
```json
{
  "type": "metrics",
  "data": {
    "timestamp": "2024-09-29T12:00:00Z",
    "cpu_percent": 45.2,
    "memory_percent": 62.1,
    "active_incidents": 12,
    "events_per_second": 15.7
  }
}
```

---

## ⚠️ Error Handling

### Standard Error Response Format

All API endpoints return errors in a consistent format:

```json
{
  "error": {
    "code": "AUTHENTICATION_FAILED",
    "message": "Invalid HMAC signature",
    "details": "Signature verification failed for request",
    "timestamp": "2024-09-29T12:00:00Z",
    "request_id": "req-uuid-456"
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `AUTHENTICATION_FAILED` | 401 | HMAC signature invalid or missing |
| `AUTHORIZATION_DENIED` | 403 | Insufficient permissions for resource |
| `RESOURCE_NOT_FOUND` | 404 | Requested resource does not exist |
| `RATE_LIMIT_EXCEEDED` | 429 | Rate limit threshold exceeded |
| `VALIDATION_ERROR` | 422 | Request validation failed |
| `INTERNAL_ERROR` | 500 | Internal server error occurred |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

### Error Response Examples

**Authentication Error**:
```json
{
  "error": {
    "code": "AUTHENTICATION_FAILED",
    "message": "Missing required authentication headers",
    "details": "X-Device-ID, X-Timestamp, X-Nonce, and X-Signature headers are required",
    "timestamp": "2024-09-29T12:00:00Z",
    "request_id": "req-uuid-456"
  }
}
```

**Validation Error**:
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": {
      "src_ip": "Invalid IP address format",
      "severity": "Must be one of: low, medium, high, critical"
    },
    "timestamp": "2024-09-29T12:00:00Z",
    "request_id": "req-uuid-789"
  }
}
```

---

## 🚦 Rate Limiting

### Rate Limit Headers

All responses include rate limiting information:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
X-RateLimit-Window: 3600
```

### Rate Limit Policies

| Endpoint Category | Requests per Hour | Burst Limit |
|------------------|-------------------|-------------|
| Authentication | 100 | 10 |
| Event Ingestion | 10,000 | 100 |
| Incident Management | 1,000 | 50 |
| AI Agent Orchestration | 500 | 20 |
| Response Actions | 200 | 10 |
| System Management | 100 | 5 |

### Rate Limit Exceeded Response

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded for endpoint",
    "details": "Maximum 1000 requests per hour allowed",
    "retry_after": 3600,
    "timestamp": "2024-09-29T12:00:00Z",
    "request_id": "req-uuid-999"
  }
}
```

---

## 📚 SDKs and Libraries

### Python SDK Example

```python
import requests
import hashlib
import hmac
import time
import uuid

class MiniXDRClient:
    def __init__(self, base_url, device_id, secret_key):
        self.base_url = base_url
        self.device_id = device_id
        self.secret_key = secret_key

    def _generate_signature(self, method, path, body, timestamp, nonce):
        canonical_message = f"{method}|{path}|{body}|{timestamp}|{nonce}"
        return hmac.new(
            self.secret_key.encode(),
            canonical_message.encode(),
            hashlib.sha256
        ).hexdigest()

    def _make_request(self, method, path, data=None):
        timestamp = str(int(time.time()))
        nonce = str(uuid.uuid4())
        body = json.dumps(data) if data else ""

        signature = self._generate_signature(method, path, body, timestamp, nonce)

        headers = {
            'X-Device-ID': self.device_id,
            'X-Timestamp': timestamp,
            'X-Nonce': nonce,
            'X-Signature': signature,
            'Content-Type': 'application/json'
        }

        url = f"{self.base_url}{path}"
        return requests.request(method, url, headers=headers, data=body)

    def get_incidents(self, limit=50, status=None):
        params = {'limit': limit}
        if status:
            params['status'] = status

        path = f"/api/incidents?{urllib.parse.urlencode(params)}"
        return self._make_request('GET', path)

    def orchestrate_agents(self, message, incident_id=None):
        data = {'message': message}
        if incident_id:
            data['incident_id'] = incident_id

        return self._make_request('POST', '/api/agents/orchestrate', data)

# Usage example
client = MiniXDRClient(
    base_url="https://your-domain.com",
    device_id="your-device-id",
    secret_key="your-hmac-secret"
)

# Get recent incidents
incidents = client.get_incidents(limit=10, status='open')

# Orchestrate AI analysis
analysis = client.orchestrate_agents(
    message="Analyze recent brute force attempts",
    incident_id=123
)
```

---

## 🔄 Webhook Integration

### Webhook Configuration

Configure webhooks to receive real-time notifications:

#### POST `/api/webhooks`

**Request**:
```json
{
  "url": "https://your-system.com/webhooks/mini-xdr",
  "events": ["incident.created", "incident.contained", "response.executed"],
  "secret": "your-webhook-secret",
  "active": true
}
```

### Webhook Payloads

**Incident Created**:
```json
{
  "event": "incident.created",
  "timestamp": "2024-09-29T12:00:00Z",
  "data": {
    "incident": {
      "id": 123,
      "src_ip": "192.168.1.100",
      "severity": "high",
      "reason": "Brute force attack detected"
    }
  },
  "signature": "webhook-hmac-signature"
}
```

---

**API Reference Complete! 📚**

This comprehensive API documentation covers all endpoints and functionality for the Mini-XDR platform. Use the interactive documentation at `/docs` for testing and detailed schema information.

For additional support:
- **GitHub Issues**: Report bugs and request features
- **Documentation**: Complete guides at `/docs/`
- **Support**: Contact support@your-domain.com

*Version: 1.0 | Last Updated: 2024-09-29*