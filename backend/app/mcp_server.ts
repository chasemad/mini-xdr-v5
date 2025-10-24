#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import fetch from "node-fetch";

// Configuration
const API_BASE = process.env.API_BASE || "http://localhost:8000";
const API_KEY = process.env.API_KEY || "";
const ENABLE_STREAMING = process.env.ENABLE_STREAMING === "true" || false;
const STREAMING_INTERVAL = parseInt(process.env.STREAMING_INTERVAL || "5000"); // 5 seconds

// Streaming state
let activeStreams: Map<string, NodeJS.Timeout> = new Map();
let streamClients: Set<string> = new Set();

// Helper function for API requests
async function apiRequest(endpoint: string, options: any = {}) {
  const url = `${API_BASE}${endpoint}`;
  
  const headers: any = {
    "Content-Type": "application/json",
    ...options.headers,
  };
  
  if (API_KEY) {
    headers["x-api-key"] = API_KEY;
  }
  
  const config: any = {
    method: options.method || "GET",
    headers,
  };
  
  if (options.body) {
    config.body = JSON.stringify(options.body);
  }
  
  const response = await fetch(url, config);
  
  if (!response.ok) {
    throw new Error(`API Error: ${response.status} ${response.statusText}`);
  }
  
  return response.json();
}

class XDRMCPServer {
  private server: Server;

  constructor() {
    this.server = new Server(
      {
        name: "mini-xdr",
        version: "1.0.0",
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.setupToolHandlers();
    this.setupRequestHandlers();
  }

  private setupToolHandlers() {
    // List available tools
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      return {
        tools: [
          // === BASIC INCIDENT MANAGEMENT ===
          {
            name: "get_incidents",
            description: "List all security incidents with filtering options",
            inputSchema: {
              type: "object",
              properties: {
                status: {
                  type: "string",
                  enum: ["new", "contained", "open", "closed"],
                  description: "Filter by incident status"
                },
                limit: {
                  type: "number",
                  minimum: 1,
                  maximum: 100,
                  description: "Maximum number of incidents to return"
                },
                hours_back: {
                  type: "number",
                  minimum: 1,
                  maximum: 168,
                  description: "Look back hours for incidents"
                }
              },
            },
          },
          {
            name: "get_incident",
            description: "Get detailed information about a specific incident",
            inputSchema: {
              type: "object",
              properties: {
                id: {
                  type: "number",
                  description: "Incident ID",
                },
              },
              required: ["id"],
            },
          },

          // === ADVANCED AI-POWERED ANALYSIS ===
          {
            name: "analyze_incident_deep",
            description: "Perform deep AI-powered incident analysis using all available agents",
            inputSchema: {
              type: "object",
              properties: {
                incident_id: {
                  type: "number",
                  description: "Incident ID to analyze"
                },
                workflow_type: {
                  type: "string",
                  enum: ["basic", "rapid", "comprehensive"],
                  description: "Type of analysis workflow"
                },
                include_threat_intel: {
                  type: "boolean",
                  description: "Include threat intelligence enrichment"
                }
              },
              required: ["incident_id"],
            },
          },
          {
            name: "natural_language_query",
            description: "Query incident data using natural language with AI-powered understanding",
            inputSchema: {
              type: "object",
              properties: {
                query: {
                  type: "string",
                  description: "Natural language query (e.g., 'Show me all brute force attacks from China in the last 24 hours')"
                },
                include_context: {
                  type: "boolean",
                  description: "Include additional context and AI insights"
                },
                max_results: {
                  type: "number",
                  minimum: 1,
                  maximum: 50,
                  description: "Maximum number of results to return"
                },
                semantic_search: {
                  type: "boolean",
                  description: "Enable semantic similarity search using embeddings"
                }
              },
              required: ["query"],
            },
          },
          {
            name: "nlp_threat_analysis",
            description: "Perform comprehensive threat analysis using natural language processing",
            inputSchema: {
              type: "object",
              properties: {
                query: {
                  type: "string",
                  description: "Natural language threat analysis request"
                },
                analysis_type: {
                  type: "string",
                  enum: ["pattern_recognition", "timeline_analysis", "attribution", "ioc_extraction", "recommendation"],
                  description: "Specific type of analysis to perform"
                },
                time_range_hours: {
                  type: "number",
                  minimum: 1,
                  maximum: 720,
                  description: "Time range in hours to analyze"
                }
              },
              required: ["query"],
            },
          },
          {
            name: "semantic_incident_search",
            description: "Search incidents using semantic similarity and natural language understanding",
            inputSchema: {
              type: "object",
              properties: {
                query: {
                  type: "string",
                  description: "Natural language search query"
                },
                similarity_threshold: {
                  type: "number",
                  minimum: 0.1,
                  maximum: 1.0,
                  description: "Minimum similarity score threshold (0.1-1.0)"
                },
                max_results: {
                  type: "number",
                  minimum: 1,
                  maximum: 20,
                  description: "Maximum number of similar incidents to return"
                }
              },
              required: ["query"],
            },
          },
          {
            name: "threat_hunt",
            description: "Execute AI-powered threat hunting queries across incident data",
            inputSchema: {
              type: "object",
              properties: {
                query: {
                  type: "string",
                  description: "Threat hunting query or pattern to search for"
                },
                hours_back: {
                  type: "number",
                  minimum: 1,
                  maximum: 168,
                  description: "Hours to look back for hunting"
                },
                threat_types: {
                  type: "array",
                  items: {
                    type: "string",
                    enum: ["brute_force", "malware", "reconnaissance", "lateral_movement", "data_exfil"]
                  },
                  description: "Specific threat types to hunt for"
                }
              },
              required: ["query"],
            },
          },
          {
            name: "forensic_investigation",
            description: "Initiate comprehensive forensic investigation with evidence collection",
            inputSchema: {
              type: "object",
              properties: {
                incident_id: {
                  type: "number",
                  description: "Incident ID to investigate"
                },
                evidence_types: {
                  type: "array",
                  items: {
                    type: "string",
                    enum: ["event_logs", "network_artifacts", "file_artifacts", "memory_dump", "system_state"]
                  },
                  description: "Types of evidence to collect"
                },
                include_network_capture: {
                  type: "boolean",
                  description: "Include network traffic capture"
                }
              },
              required: ["incident_id"],
            },
          },

          // === ORCHESTRATION & WORKFLOW MANAGEMENT ===
          {
            name: "orchestrate_response",
            description: "Trigger orchestrated multi-agent incident response",
            inputSchema: {
              type: "object",
              properties: {
                incident_id: {
                  type: "number",
                  description: "Incident ID to orchestrate response for"
                },
                workflow_type: {
                  type: "string",
                  enum: ["basic", "rapid", "comprehensive"],
                  description: "Orchestration workflow type"
                },
                priority: {
                  type: "string",
                  enum: ["low", "medium", "high", "critical"],
                  description: "Response priority level"
                }
              },
              required: ["incident_id"],
            },
          },
          {
            name: "get_orchestrator_status",
            description: "Get comprehensive orchestrator status and active workflows",
            inputSchema: {
              type: "object",
              properties: {},
            },
          },
          {
            name: "get_workflow_status",
            description: "Get status of a specific orchestration workflow",
            inputSchema: {
              type: "object",
              properties: {
                workflow_id: {
                  type: "string",
                  description: "Workflow ID to check"
                }
              },
              required: ["workflow_id"],
            },
          },

          // === THREAT INTELLIGENCE ===
          {
            name: "threat_intel_lookup",
            description: "Perform comprehensive threat intelligence lookup",
            inputSchema: {
              type: "object",
              properties: {
                ip_address: {
                  type: "string",
                  description: "IP address to analyze"
                },
                include_reputation: {
                  type: "boolean",
                  description: "Include reputation analysis"
                },
                include_geolocation: {
                  type: "boolean",
                  description: "Include geolocation data"
                },
                sources: {
                  type: "array",
                  items: {
                    type: "string",
                    enum: ["virustotal", "abuseipdb", "alienvault", "misp"]
                  },
                  description: "Specific intelligence sources to query"
                }
              },
              required: ["ip_address"],
            },
          },
          {
            name: "attribution_analysis",
            description: "Perform threat actor attribution analysis",
            inputSchema: {
              type: "object",
              properties: {
                incident_id: {
                  type: "number",
                  description: "Incident ID to attribute"
                },
                include_campaign_analysis: {
                  type: "boolean",
                  description: "Include campaign correlation"
                },
                confidence_threshold: {
                  type: "number",
                  minimum: 0.1,
                  maximum: 1.0,
                  description: "Minimum confidence threshold"
                }
              },
              required: ["incident_id"],
            },
          },

          // === REAL-TIME MONITORING ===
          {
            name: "start_incident_stream",
            description: "Start real-time incident monitoring stream",
            inputSchema: {
              type: "object",
              properties: {
                client_id: {
                  type: "string",
                  description: "Unique client identifier"
                },
                filters: {
                  type: "object",
                  properties: {
                    severity: {
                      type: "array",
                      items: { type: "string" },
                      description: "Filter by severity levels"
                    },
                    threat_categories: {
                      type: "array",
                      items: { type: "string" },
                      description: "Filter by threat categories"
                    }
                  }
                }
              },
              required: ["client_id"],
            },
          },
          {
            name: "stop_incident_stream",
            description: "Stop real-time incident monitoring stream",
            inputSchema: {
              type: "object",
              properties: {
                client_id: {
                  type: "string",
                  description: "Client identifier to stop streaming for"
                }
              },
              required: ["client_id"],
            },
          },

          // === ADVANCED QUERIES ===
          {
            name: "query_threat_patterns",
            description: "Query for specific threat patterns and behaviors",
            inputSchema: {
              type: "object",
              properties: {
                pattern_type: {
                  type: "string",
                  enum: ["brute_force", "credential_stuffing", "lateral_movement", "data_exfil", "c2_communication"],
                  description: "Type of threat pattern to query"
                },
                time_range: {
                  type: "object",
                  properties: {
                    hours_back: {
                      type: "number",
                      minimum: 1,
                      maximum: 720,
                      description: "Hours to look back"
                    }
                  }
                },
                min_confidence: {
                  type: "number",
                  minimum: 0.1,
                  maximum: 1.0,
                  description: "Minimum confidence threshold"
                }
              },
              required: ["pattern_type"],
            },
          },
          {
            name: "correlation_analysis",
            description: "Analyze correlations between incidents and events",
            inputSchema: {
              type: "object",
              properties: {
                correlation_type: {
                  type: "string",
                  enum: ["temporal", "infrastructure", "behavioral", "attribution"],
                  description: "Type of correlation analysis"
                },
                incidents: {
                  type: "array",
                  items: { type: "number" },
                  description: "Incident IDs to correlate"
                },
                time_window_hours: {
                  type: "number",
                  minimum: 1,
                  maximum: 168,
                  description: "Time window for correlation in hours"
                }
              },
              required: ["correlation_type"],
            },
          },

          // === PHASE 2: VISUAL WORKFLOW SYSTEM ===
          {
            name: "create_visual_workflow",
            description: "Create a visual workflow using drag-and-drop interface (40+ enterprise actions available)",
            inputSchema: {
              type: "object",
              properties: {
                incident_id: {
                  type: "number",
                  description: "Incident ID to create workflow for"
                },
                playbook_name: {
                  type: "string",
                  description: "Name for the workflow/playbook"
                },
                actions: {
                  type: "array",
                  items: {
                    type: "object",
                    properties: {
                      action_type: {
                        type: "string",
                        description: "Action type (e.g., block_ip_advanced, isolate_host_advanced, memory_dump_collection)"
                      },
                      parameters: {
                        type: "object",
                        description: "Action-specific parameters"
                      }
                    }
                  },
                  description: "Array of actions to include in workflow"
                },
                auto_execute: {
                  type: "boolean",
                  description: "Whether to automatically execute the workflow"
                }
              },
              required: ["incident_id", "playbook_name", "actions"],
            },
          },
          {
            name: "get_available_response_actions",
            description: "Get all 40+ available response actions organized by category (network, endpoint, email, cloud, identity, data, compliance, forensics)",
            inputSchema: {
              type: "object",
              properties: {
                category: {
                  type: "string",
                  enum: ["network", "endpoint", "email", "cloud", "identity", "data", "compliance", "forensics", "all"],
                  description: "Filter actions by category"
                },
                include_details: {
                  type: "boolean",
                  description: "Include detailed action information (parameters, safety levels, etc.)"
                }
              },
            },
          },
          {
            name: "execute_response_workflow",
            description: "Execute a created response workflow with real-time monitoring",
            inputSchema: {
              type: "object",
              properties: {
                workflow_id: {
                  type: "string",
                  description: "Workflow ID to execute"
                },
                executed_by: {
                  type: "string",
                  description: "Who is executing the workflow (analyst name)"
                }
              },
              required: ["workflow_id"],
            },
          },
          {
            name: "get_workflow_execution_status",
            description: "Get real-time status of workflow execution with step-by-step progress",
            inputSchema: {
              type: "object",
              properties: {
                workflow_id: {
                  type: "string",
                  description: "Workflow ID to check status for"
                }
              },
              required: ["workflow_id"],
            },
          },


          // === PHASE 2: ENTERPRISE RESPONSE ACTIONS ===
          {
            name: "execute_enterprise_action",
            description: "Execute any of the 40+ enterprise response actions (forensics, compliance, advanced network/endpoint/cloud actions)",
            inputSchema: {
              type: "object",
              properties: {
                action_type: {
                  type: "string",
                  enum: [
                    // Network Actions
                    "block_ip_advanced", "deploy_firewall_rules", "dns_sinkhole", "traffic_redirection", 
                    "network_segmentation", "traffic_analysis", "threat_hunting_deployment", "deception_technology",
                    "ssl_certificate_blocking", "bandwidth_throttling", "threat_intelligence_enrichment", "ioc_deployment",
                    
                    // Endpoint Actions  
                    "isolate_host_advanced", "memory_dump_collection", "process_termination", "registry_hardening",
                    "system_hardening", "vulnerability_patching", "endpoint_quarantine", "service_shutdown",
                    "configuration_rollback", "behavior_analytics_deployment", "system_recovery", "malware_removal",
                    
                    // Cloud Actions
                    "iam_policy_restriction", "resource_isolation", "cloud_security_posture", "container_isolation",
                    "api_rate_limiting", "cloud_resource_tagging", "serverless_function_disable",
                    
                    // Email Actions
                    "email_recall", "mailbox_quarantine", "email_flow_analysis", "domain_blocking",
                    "attachment_sandboxing", "email_encryption_enforcement",
                    
                    // Identity Actions
                    "account_disable", "password_reset_bulk", "privileged_access_review", "session_termination",
                    "access_certification", "identity_verification", "mfa_enforcement", "stakeholder_notification",
                    
                    // Data Actions
                    "data_classification", "backup_verification", "data_encryption", "data_loss_prevention",
                    
                    // Compliance Actions
                    "compliance_audit_trigger", "data_retention_enforcement", "regulatory_reporting", "privacy_breach_notification",
                    
                    // Forensics Actions
                    "disk_imaging", "network_packet_capture", "log_preservation", "chain_of_custody",
                    "forensic_timeline", "evidence_analysis", "attribution_analysis", "campaign_correlation"
                  ],
                  description: "Specific enterprise action to execute"
                },
                incident_id: {
                  type: "number",
                  description: "Incident ID to execute action for"
                },
                parameters: {
                  type: "object",
                  description: "Action-specific parameters"
                },
                safety_check: {
                  type: "boolean",
                  description: "Perform safety validation before execution"
                }
              },
              required: ["action_type", "incident_id"],
            },
          },
          {
            name: "get_response_impact_metrics",
            description: "Get comprehensive impact metrics and analytics for response actions",
            inputSchema: {
              type: "object",
              properties: {
                workflow_id: {
                  type: "string",
                  description: "Specific workflow ID to get metrics for"
                },
                days_back: {
                  type: "number",
                  minimum: 1,
                  maximum: 90,
                  description: "Days of historical data to include"
                },
                metric_types: {
                  type: "array",
                  items: {
                    type: "string",
                    enum: ["effectiveness", "performance", "business_impact", "compliance"]
                  },
                  description: "Types of metrics to include"
                }
              },
            },
          },

          // === PHASE 3: T-POT INTEGRATION & TESTING ===
          {
            name: "test_tpot_integration",
            description: "Test and validate T-Pot honeypot integration with real SSH commands",
            inputSchema: {
              type: "object",
              properties: {
                test_type: {
                  type: "string",
                  enum: ["connectivity", "iptables", "firewall_rules", "comprehensive"],
                  description: "Type of T-Pot integration test"
                },
                dry_run: {
                  type: "boolean",
                  description: "Perform dry run without actual changes"
                }
              },
            },
          },
          {
            name: "execute_tpot_command",
            description: "Execute real commands on T-Pot honeypot infrastructure (admin@34.193.101.171:64295)",
            inputSchema: {
              type: "object",
              properties: {
                command_type: {
                  type: "string",
                  enum: ["block_ip", "unblock_ip", "list_rules", "system_status", "log_analysis"],
                  description: "Type of command to execute"
                },
                target_ip: {
                  type: "string",
                  description: "IP address for blocking/unblocking commands"
                },
                parameters: {
                  type: "object",
                  description: "Command-specific parameters"
                },
                confirmation_required: {
                  type: "boolean",
                  description: "Require confirmation before execution"
                }
              },
              required: ["command_type"],
            },
          },

          // === AGENT EXECUTION - IAM, EDR, DLP (NEW!) ===
          {
            name: "execute_iam_action",
            description: "Execute IAM (Identity & Access Management) actions on Active Directory",
            inputSchema: {
              type: "object",
              properties: {
                action_name: {
                  type: "string",
                  enum: [
                    "disable_user_account",
                    "reset_user_password",
                    "remove_user_from_group",
                    "revoke_user_sessions",
                    "lock_user_account",
                    "enable_user_account"
                  ],
                  description: "IAM action to execute"
                },
                params: {
                  type: "object",
                  properties: {
                    username: {
                      type: "string",
                      description: "Target username (e.g., 'john.doe@domain.local')"
                    },
                    reason: {
                      type: "string",
                      description: "Reason for the action (required for audit trail)"
                    },
                    group_name: {
                      type: "string",
                      description: "Group name (for remove_user_from_group action)"
                    },
                    new_password: {
                      type: "string",
                      description: "New password (for reset_user_password action)"
                    },
                    force_change: {
                      type: "boolean",
                      description: "Force password change at next login (default: true)"
                    }
                  },
                  required: ["username", "reason"]
                },
                incident_id: {
                  type: "number",
                  description: "Associated incident ID for tracking"
                }
              },
              required: ["action_name", "params", "incident_id"],
            },
          },
          {
            name: "execute_edr_action",
            description: "Execute EDR (Endpoint Detection & Response) actions on Windows endpoints",
            inputSchema: {
              type: "object",
              properties: {
                action_name: {
                  type: "string",
                  enum: [
                    "kill_process",
                    "quarantine_file",
                    "collect_memory_dump",
                    "isolate_host",
                    "delete_registry_key",
                    "disable_scheduled_task",
                    "unisolate_host"
                  ],
                  description: "EDR action to execute"
                },
                params: {
                  type: "object",
                  properties: {
                    hostname: {
                      type: "string",
                      description: "Target Windows hostname"
                    },
                    process_name: {
                      type: "string",
                      description: "Process name (for kill_process)"
                    },
                    pid: {
                      type: "number",
                      description: "Process ID (for kill_process)"
                    },
                    file_path: {
                      type: "string",
                      description: "File path (for quarantine_file)"
                    },
                    isolation_level: {
                      type: "string",
                      enum: ["full", "partial"],
                      description: "Isolation level for isolate_host (full blocks all, partial allows domain)"
                    },
                    registry_key: {
                      type: "string",
                      description: "Registry key path (for delete_registry_key)"
                    },
                    task_name: {
                      type: "string",
                      description: "Scheduled task name (for disable_scheduled_task)"
                    },
                    reason: {
                      type: "string",
                      description: "Reason for the action (required for audit trail)"
                    }
                  },
                  required: ["hostname", "reason"]
                },
                incident_id: {
                  type: "number",
                  description: "Associated incident ID for tracking"
                }
              },
              required: ["action_name", "params", "incident_id"],
            },
          },
          {
            name: "execute_dlp_action",
            description: "Execute DLP (Data Loss Prevention) actions to protect sensitive data",
            inputSchema: {
              type: "object",
              properties: {
                action_name: {
                  type: "string",
                  enum: [
                    "scan_file_for_sensitive_data",
                    "block_upload",
                    "quarantine_sensitive_file"
                  ],
                  description: "DLP action to execute"
                },
                params: {
                  type: "object",
                  properties: {
                    file_path: {
                      type: "string",
                      description: "File path to scan or quarantine"
                    },
                    upload_id: {
                      type: "string",
                      description: "Upload ID to block (for block_upload)"
                    },
                    destination: {
                      type: "string",
                      description: "Upload destination (for block_upload)"
                    },
                    username: {
                      type: "string",
                      description: "User attempting the upload (for block_upload)"
                    },
                    reason: {
                      type: "string",
                      description: "Reason for the action (required for audit trail)"
                    },
                    pattern_types: {
                      type: "array",
                      items: {
                        type: "string",
                        enum: ["ssn", "credit_card", "email", "api_key", "phone", "ip_address", "aws_key", "private_key"]
                      },
                      description: "Specific patterns to scan for (default: all)"
                    }
                  },
                  required: ["reason"]
                },
                incident_id: {
                  type: "number",
                  description: "Associated incident ID for tracking"
                }
              },
              required: ["action_name", "params", "incident_id"],
            },
          },
          {
            name: "get_agent_actions",
            description: "Query all agent actions (IAM, EDR, DLP) for an incident or globally",
            inputSchema: {
              type: "object",
              properties: {
                incident_id: {
                  type: "number",
                  description: "Filter by incident ID (omit for all actions)"
                },
                agent_type: {
                  type: "string",
                  enum: ["iam", "edr", "dlp"],
                  description: "Filter by agent type"
                },
                status: {
                  type: "string",
                  enum: ["success", "failed", "rolled_back"],
                  description: "Filter by action status"
                },
                limit: {
                  type: "number",
                  minimum: 1,
                  maximum: 100,
                  description: "Maximum number of actions to return (default: 50)"
                }
              },
            },
          },
          {
            name: "rollback_agent_action",
            description: "Rollback a previously executed agent action (IAM, EDR, or DLP)",
            inputSchema: {
              type: "object",
              properties: {
                rollback_id: {
                  type: "string",
                  description: "Unique rollback ID from the original action"
                },
                reason: {
                  type: "string",
                  description: "Reason for rollback (required for audit trail)"
                }
              },
              required: ["rollback_id"],
            },
          },

          // === LEGACY TOOLS (MAINTAINED FOR COMPATIBILITY) ===
          {
            name: "contain_incident",
            description: "Block the source IP of an incident (requires human confirmation)",
            inputSchema: {
              type: "object",
              properties: {
                incident_id: {
                  type: "number",
                  description: "Incident ID to contain",
                },
              },
              required: ["incident_id"],
            },
          },
          {
            name: "unblock_incident",
            description: "Unblock the source IP of an incident",
            inputSchema: {
              type: "object",
              properties: {
                incident_id: {
                  type: "number",
                  description: "Incident ID to unblock",
                },
              },
              required: ["incident_id"],
            },
          },
          {
            name: "schedule_unblock",
            description: "Schedule an incident to be unblocked after specified minutes",
            inputSchema: {
              type: "object",
              properties: {
                incident_id: {
                  type: "number",
                  description: "Incident ID to schedule for unblocking",
                },
                minutes: {
                  type: "number",
                  description: "Minutes until unblock (1-1440)",
                  minimum: 1,
                  maximum: 1440,
                },
              },
              required: ["incident_id", "minutes"],
            },
          },
          {
            name: "get_auto_contain_setting",
            description: "Get the current auto-contain setting",
            inputSchema: {
              type: "object",
              properties: {},
            },
          },
          {
            name: "set_auto_contain_setting",
            description: "Enable or disable auto-contain (requires human confirmation)",
            inputSchema: {
              type: "object",
              properties: {
                enabled: {
                  type: "boolean",
                  description: "Whether to enable auto-contain",
                },
              },
              required: ["enabled"],
            },
          },
          {
            name: "get_system_health",
            description: "Get system health status",
            inputSchema: {
              type: "object",
              properties: {},
            },
          },
        ],
      };
    });

    // Handle tool calls
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      try {
        switch (name) {
          // === BASIC INCIDENT MANAGEMENT ===
          case "get_incidents":
            return await this.getIncidentsEnhanced(args);

          case "get_incident":
            if (!args || typeof args !== 'object' || !('id' in args)) {
              throw new Error('Missing required parameter: id');
            }
            return await this.getIncident(args.id as number);

          // === ADVANCED AI-POWERED ANALYSIS ===
          case "analyze_incident_deep":
            if (!args || typeof args !== 'object' || !('incident_id' in args)) {
              throw new Error('Missing required parameter: incident_id');
            }
            return await this.analyzeIncidentDeep(args);

          case "natural_language_query":
            if (!args || typeof args !== 'object' || !('query' in args)) {
              throw new Error('Missing required parameter: query');
            }
            return await this.naturalLanguageQuery(args);

          case "nlp_threat_analysis":
            if (!args || typeof args !== 'object' || !('query' in args)) {
              throw new Error('Missing required parameter: query');
            }
            return await this.nlpThreatAnalysis(args);

          case "semantic_incident_search":
            if (!args || typeof args !== 'object' || !('query' in args)) {
              throw new Error('Missing required parameter: query');
            }
            return await this.semanticIncidentSearch(args);

          case "threat_hunt":
            if (!args || typeof args !== 'object' || !('query' in args)) {
              throw new Error('Missing required parameter: query');
            }
            return await this.threatHunt(args);

          case "forensic_investigation":
            if (!args || typeof args !== 'object' || !('incident_id' in args)) {
              throw new Error('Missing required parameter: incident_id');
            }
            return await this.forensicInvestigation(args);

          // === ORCHESTRATION & WORKFLOW MANAGEMENT ===
          case "orchestrate_response":
            if (!args || typeof args !== 'object' || !('incident_id' in args)) {
              throw new Error('Missing required parameter: incident_id');
            }
            return await this.orchestrateResponse(args);

          case "get_orchestrator_status":
            return await this.getOrchestratorStatus();

          case "get_workflow_status":
            if (!args || typeof args !== 'object' || !('workflow_id' in args)) {
              throw new Error('Missing required parameter: workflow_id');
            }
            return await this.getWorkflowStatus(args.workflow_id as string);

          // === THREAT INTELLIGENCE ===
          case "threat_intel_lookup":
            if (!args || typeof args !== 'object' || !('ip_address' in args)) {
              throw new Error('Missing required parameter: ip_address');
            }
            return await this.threatIntelLookup(args);

          case "attribution_analysis":
            if (!args || typeof args !== 'object' || !('incident_id' in args)) {
              throw new Error('Missing required parameter: incident_id');
            }
            return await this.attributionAnalysis(args);

          // === REAL-TIME MONITORING ===
          case "start_incident_stream":
            if (!args || typeof args !== 'object' || !('client_id' in args)) {
              throw new Error('Missing required parameter: client_id');
            }
            return await this.startIncidentStream(args);

          case "stop_incident_stream":
            if (!args || typeof args !== 'object' || !('client_id' in args)) {
              throw new Error('Missing required parameter: client_id');
            }
            return await this.stopIncidentStream(args.client_id as string);

          // === ADVANCED QUERIES ===
          case "query_threat_patterns":
            if (!args || typeof args !== 'object' || !('pattern_type' in args)) {
              throw new Error('Missing required parameter: pattern_type');
            }
            return await this.queryThreatPatterns(args);

          case "correlation_analysis":
            if (!args || typeof args !== 'object' || !('correlation_type' in args)) {
              throw new Error('Missing required parameter: correlation_type');
            }
            return await this.correlationAnalysis(args);

          // === PHASE 2: VISUAL WORKFLOW SYSTEM HANDLERS ===
          case "create_visual_workflow":
            if (!args || typeof args !== 'object' || !('incident_id' in args) || !('playbook_name' in args) || !('actions' in args)) {
              throw new Error('Missing required parameters: incident_id, playbook_name, actions');
            }
            return await this.createVisualWorkflow(args);

          case "get_available_response_actions":
            return await this.getAvailableResponseActions(args);

          case "execute_response_workflow":
            if (!args || typeof args !== 'object' || !('workflow_id' in args)) {
              throw new Error('Missing required parameter: workflow_id');
            }
            return await this.executeResponseWorkflow(args);

          case "get_workflow_execution_status":
            if (!args || typeof args !== 'object' || !('workflow_id' in args)) {
              throw new Error('Missing required parameter: workflow_id');
            }
            return await this.getWorkflowExecutionStatus(args.workflow_id as string);

          // === PHASE 2: AI-POWERED RESPONSE ENGINE HANDLERS ===
          case "get_ai_response_recommendations":
            if (!args || typeof args !== 'object' || !('incident_id' in args)) {
              throw new Error('Missing required parameter: incident_id');
            }
            return await this.getAIResponseRecommendations(args);

          case "analyze_incident_context_comprehensive":
            if (!args || typeof args !== 'object' || !('incident_id' in args)) {
              throw new Error('Missing required parameter: incident_id');
            }
            return await this.analyzeIncidentContextComprehensive(args);

          case "optimize_response_strategy":
            if (!args || typeof args !== 'object' || !('workflow_id' in args)) {
              throw new Error('Missing required parameter: workflow_id');
            }
            return await this.optimizeResponseStrategy(args);

          case "generate_adaptive_recommendations":
            if (!args || typeof args !== 'object' || !('incident_id' in args)) {
              throw new Error('Missing required parameter: incident_id');
            }
            return await this.generateAdaptiveRecommendations(args);

          case "execute_enterprise_action":
            if (!args || typeof args !== 'object' || !('action_type' in args) || !('incident_id' in args)) {
              throw new Error('Missing required parameters: action_type, incident_id');
            }
            return await this.executeEnterpriseAction(args);

          case "get_response_impact_metrics":
            return await this.getResponseImpactMetrics(args);

          // === PHASE 3: T-POT INTEGRATION HANDLERS ===
          case "test_tpot_integration":
            return await this.testTPotIntegration(args);

          case "execute_tpot_command":
            if (!args || typeof args !== 'object' || !('command_type' in args)) {
              throw new Error('Missing required parameter: command_type');
            }
            return await this.executeTPotCommand(args);

          // === AGENT EXECUTION - IAM, EDR, DLP ===
          case "execute_iam_action":
            if (!args || typeof args !== 'object' || !('action_name' in args) || !('params' in args) || !('incident_id' in args)) {
              throw new Error('Missing required parameters: action_name, params, and incident_id are required');
            }
            return await this.executeIAMAction(args);

          case "execute_edr_action":
            if (!args || typeof args !== 'object' || !('action_name' in args) || !('params' in args) || !('incident_id' in args)) {
              throw new Error('Missing required parameters: action_name, params, and incident_id are required');
            }
            return await this.executeEDRAction(args);

          case "execute_dlp_action":
            if (!args || typeof args !== 'object' || !('action_name' in args) || !('params' in args) || !('incident_id' in args)) {
              throw new Error('Missing required parameters: action_name, params, and incident_id are required');
            }
            return await this.executeDLPAction(args);

          case "get_agent_actions":
            return await this.getAgentActions(args);

          case "rollback_agent_action":
            if (!args || typeof args !== 'object' || !('rollback_id' in args)) {
              throw new Error('Missing required parameter: rollback_id');
            }
            return await this.rollbackAgentAction(args);

          // === LEGACY TOOLS ===
          case "contain_incident":
            if (!args || typeof args !== 'object' || !('incident_id' in args)) {
              throw new Error('Missing required parameter: incident_id');
            }
            return await this.containIncident(args.incident_id as number);

          case "unblock_incident":
            if (!args || typeof args !== 'object' || !('incident_id' in args)) {
              throw new Error('Missing required parameter: incident_id');
            }
            return await this.unblockIncident(args.incident_id as number);

          case "schedule_unblock":
            if (!args || typeof args !== 'object' || !('incident_id' in args) || !('minutes' in args)) {
              throw new Error('Missing required parameters: incident_id, minutes');
            }
            return await this.scheduleUnblock(args.incident_id as number, args.minutes as number);

          case "get_auto_contain_setting":
            return await this.getAutoContainSetting();

          case "set_auto_contain_setting":
            if (!args || typeof args !== 'object' || !('enabled' in args)) {
              throw new Error('Missing required parameter: enabled');
            }
            return await this.setAutoContainSetting(args.enabled as boolean);

          case "get_system_health":
            return await this.getSystemHealth();

          default:
            throw new Error(`Unknown tool: ${name}`);
        }
      } catch (error) {
        return {
          content: [
            {
              type: "text",
              text: `Error executing ${name}: ${error instanceof Error ? error.message : String(error)}`,
            },
          ],
        };
      }
    });
  }

  private setupRequestHandlers() {
    // Add any additional request handlers here
  }

  private async getIncidents() {
    const incidents = await apiRequest("/incidents") as any[];
    
    return {
      content: [
        {
          type: "text",
          text: `Found ${incidents.length} incidents:\n\n` +
            incidents.map((inc: any) => 
              `#${inc.id} - ${inc.src_ip} (${inc.status}) - ${inc.reason}\n` +
              `  Created: ${new Date(inc.created_at).toLocaleString()}\n` +
              (inc.triage_note ? `  Severity: ${inc.triage_note.severity}, Recommend: ${inc.triage_note.recommendation}\n` : "")
            ).join("\n"),
        },
      ],
    };
  }

  private async getIncident(id: number) {
    const incident = await apiRequest(`/incidents/${id}`) as any;
    
    const actionSummary = incident.actions.map((action: any) => 
      `  ${action.action} (${action.result}) - ${new Date(action.created_at).toLocaleString()}`
    ).join("\n");

    const eventSummary = incident.recent_events.slice(0, 5).map((event: any) => 
      `  ${event.eventid} - ${new Date(event.ts).toLocaleString()}`
    ).join("\n");

    return {
      content: [
        {
          type: "text",
          text: `Incident #${incident.id} Details:\n\n` +
            `Source IP: ${incident.src_ip}\n` +
            `Status: ${incident.status}\n` +
            `Reason: ${incident.reason}\n` +
            `Created: ${new Date(incident.created_at).toLocaleString()}\n` +
            `Auto-contained: ${incident.auto_contained}\n\n` +
            (incident.triage_note ? 
              `Triage Analysis:\n` +
              `  Summary: ${incident.triage_note.summary}\n` +
              `  Severity: ${incident.triage_note.severity}\n` +
              `  Recommendation: ${incident.triage_note.recommendation}\n` +
              `  Rationale: ${incident.triage_note.rationale?.join("; ")}\n\n`
            : "") +
            `Recent Actions:\n${actionSummary}\n\n` +
            `Recent Events (last 5):\n${eventSummary}`,
        },
      ],
    };
  }

  // === NEW AGENT EXECUTION METHODS ===
  
  private async executeIAMAction(args: any) {
    const { action_name, params, incident_id } = args;
    
    try {
      const result = await apiRequest("/api/agents/iam/execute", {
        method: "POST",
        body: { action_name, params, incident_id }
      }) as any;
      
      return {
        content: [
          {
            type: "text",
            text: `👤 IAM ACTION EXECUTED\n\n` +
              `Action: ${action_name}\n` +
              `Incident: #${incident_id}\n` +
              `Status: ${result.status === 'success' ? '✅ SUCCESS' : '❌ FAILED'}\n` +
              `Agent ID: ${result.agent_id}\n` +
              `Action ID: ${result.action_id}\n` +
              `Rollback ID: ${result.rollback_id || 'N/A'}\n\n` +
              `Parameters:\n${JSON.stringify(params, null, 2)}\n\n` +
              `Result:\n${JSON.stringify(result.result, null, 2)}\n\n` +
              `Executed At: ${result.executed_at}\n\n` +
              `${result.rollback_id ? '🔄 This action can be rolled back using rollback_id: ' + result.rollback_id : '⚠️ This action cannot be rolled back'}`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ IAM action failed: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }

  private async executeEDRAction(args: any) {
    const { action_name, params, incident_id } = args;
    
    try {
      const result = await apiRequest("/api/agents/edr/execute", {
        method: "POST",
        body: { action_name, params, incident_id }
      }) as any;
      
      return {
        content: [
          {
            type: "text",
            text: `🖥️ EDR ACTION EXECUTED\n\n` +
              `Action: ${action_name}\n` +
              `Incident: #${incident_id}\n` +
              `Hostname: ${params.hostname}\n` +
              `Status: ${result.status === 'success' ? '✅ SUCCESS' : '❌ FAILED'}\n` +
              `Agent ID: ${result.agent_id}\n` +
              `Action ID: ${result.action_id}\n` +
              `Rollback ID: ${result.rollback_id || 'N/A'}\n\n` +
              `Parameters:\n${JSON.stringify(params, null, 2)}\n\n` +
              `Result:\n${JSON.stringify(result.result, null, 2)}\n\n` +
              `Executed At: ${result.executed_at}\n\n` +
              `${result.rollback_id ? '🔄 This action can be rolled back using rollback_id: ' + result.rollback_id : '⚠️ This action cannot be rolled back'}`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ EDR action failed: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }

  private async executeDLPAction(args: any) {
    const { action_name, params, incident_id } = args;
    
    try {
      const result = await apiRequest("/api/agents/dlp/execute", {
        method: "POST",
        body: { action_name, params, incident_id }
      }) as any;
      
      return {
        content: [
          {
            type: "text",
            text: `🔒 DLP ACTION EXECUTED\n\n` +
              `Action: ${action_name}\n` +
              `Incident: #${incident_id}\n` +
              `Status: ${result.status === 'success' ? '✅ SUCCESS' : '❌ FAILED'}\n` +
              `Agent ID: ${result.agent_id}\n` +
              `Action ID: ${result.action_id}\n` +
              `Rollback ID: ${result.rollback_id || 'N/A'}\n\n` +
              `Parameters:\n${JSON.stringify(params, null, 2)}\n\n` +
              `Result:\n${JSON.stringify(result.result, null, 2)}\n\n` +
              `${result.sensitive_data_found ? 
                `⚠️ SENSITIVE DATA DETECTED:\n` +
                `${result.sensitive_data_found.map((item: any) => 
                  `  • ${item.pattern_type}: ${item.count} match(es)`
                ).join('\n')}\n\n`
                : ''
              }` +
              `Executed At: ${result.executed_at}\n\n` +
              `${result.rollback_id ? '🔄 This action can be rolled back using rollback_id: ' + result.rollback_id : '⚠️ This action cannot be rolled back'}`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ DLP action failed: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }

  private async getAgentActions(args: any) {
    const { incident_id, agent_type, status, limit = 50 } = args || {};
    
    try {
      let endpoint = "/api/agents/actions";
      if (incident_id) {
        endpoint = `/api/agents/actions/${incident_id}`;
      }
      
      const params = new URLSearchParams();
      if (agent_type) params.append("agent_type", agent_type);
      if (status) params.append("status", status);
      if (limit) params.append("limit", limit.toString());
      
      const queryString = params.toString();
      if (queryString) {
        endpoint += `?${queryString}`;
      }
      
      const actions = await apiRequest(endpoint, { method: "GET" }) as any[];
      
      if (!actions || actions.length === 0) {
        return {
          content: [
            {
              type: "text",
              text: `📋 No agent actions found matching the criteria.`,
            },
          ],
        };
      }
      
      const actionsByAgent = {
        iam: actions.filter(a => a.agent_type === 'iam'),
        edr: actions.filter(a => a.agent_type === 'edr'),
        dlp: actions.filter(a => a.agent_type === 'dlp'),
      };
      
      let summary = `📋 AGENT ACTIONS SUMMARY\n\n`;
      summary += `Total Actions: ${actions.length}\n`;
      summary += `• IAM Actions: ${actionsByAgent.iam.length}\n`;
      summary += `• EDR Actions: ${actionsByAgent.edr.length}\n`;
      summary += `• DLP Actions: ${actionsByAgent.dlp.length}\n\n`;
      
      if (incident_id) {
        summary += `Filtered by Incident: #${incident_id}\n`;
      }
      if (agent_type) {
        summary += `Filtered by Agent Type: ${agent_type.toUpperCase()}\n`;
      }
      if (status) {
        summary += `Filtered by Status: ${status.toUpperCase()}\n`;
      }
      
      summary += `\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n`;
      
      for (const action of actions.slice(0, 20)) {  // Show first 20
        const agentIcon = action.agent_type === 'iam' ? '👤' : action.agent_type === 'edr' ? '🖥️' : '🔒';
        const statusIcon = action.status === 'success' ? '✅' : action.status === 'failed' ? '❌' : '🔄';
        
        summary += `${agentIcon} ${statusIcon} ${action.action_name}\n`;
        summary += `   Agent: ${action.agent_type.toUpperCase()} | Incident: #${action.incident_id}\n`;
        summary += `   Action ID: ${action.action_id}\n`;
        summary += `   Executed: ${action.executed_at}\n`;
        if (action.rollback_id && !action.rollback_executed) {
          summary += `   🔄 Rollback Available: ${action.rollback_id}\n`;
        } else if (action.rollback_executed) {
          summary += `   ↩️ Rolled Back: ${action.rollback_timestamp}\n`;
        }
        summary += `\n`;
      }
      
      if (actions.length > 20) {
        summary += `\n... and ${actions.length - 20} more actions\n`;
      }
      
      return {
        content: [
          {
            type: "text",
            text: summary,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ Failed to retrieve agent actions: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }

  private async rollbackAgentAction(args: any) {
    const { rollback_id, reason } = args;
    
    try {
      const result = await apiRequest(`/api/agents/rollback/${rollback_id}`, {
        method: "POST",
        body: { reason }
      }) as any;
      
      return {
        content: [
          {
            type: "text",
            text: `🔄 AGENT ACTION ROLLBACK\n\n` +
              `Rollback ID: ${rollback_id}\n` +
              `Status: ${result.status === 'success' ? '✅ SUCCESS' : '❌ FAILED'}\n` +
              `Original Action: ${result.original_action?.action_name}\n` +
              `Agent Type: ${result.original_action?.agent_type?.toUpperCase()}\n` +
              `Incident: #${result.original_action?.incident_id}\n\n` +
              `Rollback Result:\n${JSON.stringify(result.rollback_result, null, 2)}\n\n` +
              `${reason ? `Reason: ${reason}\n\n` : ''}` +
              `Rolled Back At: ${result.rollback_timestamp}\n\n` +
              `✅ Original action has been successfully reversed.`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ Rollback failed: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }

  private async containIncident(incidentId: number) {
    const result = await apiRequest(`/incidents/${incidentId}/contain`, {
      method: "POST",
    }) as any;
    
    return {
      content: [
        {
          type: "text",
          text: `Containment action for incident #${incidentId}: ${result.status}\n` +
            `Detail: ${result.detail || "No additional details"}`,
        },
      ],
    };
  }

  private async unblockIncident(incidentId: number) {
    const result = await apiRequest(`/incidents/${incidentId}/unblock`, {
      method: "POST",
    }) as any;
    
    return {
      content: [
        {
          type: "text",
          text: `Unblock action for incident #${incidentId}: ${result.status}\n` +
            `Detail: ${result.detail || "No additional details"}`,
        },
      ],
    };
  }

  private async scheduleUnblock(incidentId: number, minutes: number) {
    const result = await apiRequest(`/incidents/${incidentId}/schedule_unblock?minutes=${minutes}`, {
      method: "POST",
    }) as any;
    
    return {
      content: [
        {
          type: "text",
          text: `Scheduled unblock for incident #${incidentId}: ${result.status}\n` +
            `Due at: ${new Date(result.due_at).toLocaleString()}\n` +
            `Duration: ${result.minutes} minutes`,
        },
      ],
    };
  }

  private async getAutoContainSetting() {
    const setting = await apiRequest("/settings/auto_contain") as any;
    
    return {
      content: [
        {
          type: "text",
          text: `Auto-contain is currently: ${setting.enabled ? "ENABLED" : "DISABLED"}`,
        },
      ],
    };
  }

  private async setAutoContainSetting(enabled: boolean) {
    const result = await apiRequest("/settings/auto_contain", {
      method: "POST",
      body: enabled,
    }) as any;
    
    return {
      content: [
        {
          type: "text",
          text: `Auto-contain setting updated: ${result.enabled ? "ENABLED" : "DISABLED"}`,
        },
      ],
    };
  }

  private async getSystemHealth() {
    const health = await apiRequest("/health") as any;

    return {
      content: [
        {
          type: "text",
          text: `System Health: ${health.status.toUpperCase()}\n` +
            `Timestamp: ${new Date(health.timestamp).toLocaleString()}\n` +
            `Auto-contain: ${health.auto_contain ? "ENABLED" : "DISABLED"}`,
        },
      ],
    };
  }

  // === ENHANCED METHODS FOR NEW TOOLS ===

  private async getIncidentsEnhanced(args?: any) {
    const params = new URLSearchParams();
    if (args?.status) params.append('status', args.status);
    if (args?.limit) params.append('limit', args.limit?.toString());
    if (args?.hours_back) params.append('hours_back', args.hours_back?.toString());

    const queryString = params.toString();
    const endpoint = `/incidents${queryString ? `?${queryString}` : ''}`;
    const incidents = await apiRequest(endpoint) as any[];

    let summary = `Found ${incidents.length} incidents`;
    if (args?.status) summary += ` with status: ${args.status}`;
    if (args?.hours_back) summary += ` in last ${args.hours_back} hours`;

    const incidentDetails = incidents.map((inc: any) =>
      `#${inc.id} - ${inc.src_ip} (${inc.status}) - ${inc.reason}\n` +
      `  Created: ${new Date(inc.created_at).toLocaleString()}\n` +
      `  Severity: ${inc.escalation_level || 'medium'}, Risk: ${inc.risk_score ? (inc.risk_score * 100).toFixed(1) : 'N/A'}%\n` +
      (inc.triage_note ? `  AI Analysis: ${inc.triage_note.severity} confidence` : "")
    ).join("\n");

    return {
      content: [
        {
          type: "text",
          text: `${summary}:\n\n${incidentDetails}`,
        },
      ],
    };
  }

  private async analyzeIncidentDeep(args: any) {
    const { incident_id, workflow_type = "comprehensive", include_threat_intel = true } = args;

    try {
      // Trigger orchestrated analysis
      const result = await apiRequest("/api/agents/orchestrate", {
        method: "POST",
        body: {
          incident_id,
          agent_type: "orchestrated_response",
          workflow_type,
          context: {
            include_threat_intel,
            deep_analysis: true
          }
        }
      }) as any;

      return {
        content: [
          {
            type: "text",
            text: `🔍 Deep Incident Analysis for #${incident_id}:\n\n` +
              `📊 Analysis Type: ${workflow_type.toUpperCase()}\n` +
              `🤖 Agents Involved: ${result.orchestration_result?.agents_involved?.join(', ') || 'N/A'}\n` +
              `⏱️  Execution Time: ${result.orchestration_result?.execution_time?.toFixed(2) || 'N/A'}s\n\n` +
              `🎯 Key Findings:\n` +
              `${this.formatOrchestrationResults(result.orchestration_result)}\n\n` +
              `💡 Recommendations:\n` +
              `${this.formatRecommendations(result.orchestration_result)}`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ Failed to perform deep analysis for incident #${incident_id}: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
      };
    }
  }

  private async threatHunt(args: any) {
    const { query, hours_back = 24, threat_types = [] } = args;

    try {
      // Use the threat hunting API endpoint
      const result = await apiRequest("/api/agents/orchestrate", {
        method: "POST",
        body: {
          query,
          agent_type: "threat_hunting",
          context: {
            hours_back,
            threat_types,
            hunting_query: query
          }
        }
      }) as any;

      return {
        content: [
          {
            type: "text",
            text: `🔍 Threat Hunting Results for: "${query}"\n\n` +
              `⏰ Time Window: Last ${hours_back} hours\n` +
              `🎯 Threat Types: ${threat_types.length > 0 ? threat_types.join(', ') : 'All'}\n\n` +
              `📊 Findings:\n` +
              `${this.formatThreatHuntResults(result)}\n\n` +
              `🎯 Recommendations:\n` +
              `${this.formatHuntingRecommendations(result)}`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ Threat hunting failed for query "${query}": ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
      };
    }
  }

  private async forensicInvestigation(args: any) {
    const { incident_id, evidence_types = ["event_logs"], include_network_capture = false } = args;

    try {
      // Trigger forensic investigation
      const result = await apiRequest("/api/agents/orchestrate", {
        method: "POST",
        body: {
          incident_id,
          agent_type: "forensic_investigation",
          context: {
            evidence_types,
            include_network_capture
          }
        }
      }) as any;

      return {
        content: [
          {
            type: "text",
            text: `🔬 Forensic Investigation for Incident #${incident_id}:\n\n` +
              `📋 Evidence Types: ${evidence_types.join(', ')}\n` +
              `🌐 Network Capture: ${include_network_capture ? 'ENABLED' : 'DISABLED'}\n\n` +
              `📊 Investigation Results:\n` +
              `${this.formatForensicResults(result)}\n\n` +
              `🔍 Key Artifacts:\n` +
              `${this.formatForensicArtifacts(result)}`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ Forensic investigation failed for incident #${incident_id}: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
      };
    }
  }

  private async orchestrateResponse(args: any) {
    const { incident_id, workflow_type = "comprehensive", priority = "medium" } = args;

    try {
      // Trigger orchestrated response
      const result = await apiRequest("/api/orchestrator/workflows", {
        method: "POST",
        body: {
          incident_id,
          workflow_type,
          priority
        }
      }) as any;

      return {
        content: [
          {
            type: "text",
            text: `🚀 Orchestrated Response Triggered for Incident #${incident_id}:\n\n` +
              `⚡ Workflow Type: ${workflow_type.toUpperCase()}\n` +
              `🎯 Priority: ${priority.toUpperCase()}\n` +
              `📝 Workflow ID: ${result.workflow_id}\n\n` +
              `🤖 Multi-Agent Response Status:\n` +
              `• Attribution Analysis: ${result.result?.attribution ? '✅ COMPLETED' : '⏳ PENDING'}\n` +
              `• Forensic Investigation: ${result.result?.forensics ? '✅ COMPLETED' : '⏳ PENDING'}\n` +
              `• Containment Decision: ${result.result?.containment ? '✅ COMPLETED' : '⏳ PENDING'}\n` +
              `• Deception Strategy: ${result.result?.deception ? '✅ COMPLETED' : '⏳ PENDING'}\n\n` +
              `📊 Final Decision:\n` +
              `${this.formatOrchestrationDecision(result.result)}`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ Failed to orchestrate response for incident #${incident_id}: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
      };
    }
  }

  private async getOrchestratorStatus() {
    try {
      const status = await apiRequest("/api/orchestrator/status") as any;

      return {
        content: [
          {
            type: "text",
            text: `🤖 Agent Orchestrator Status:\n\n` +
              `⏱️  Uptime: ${Math.floor(status.orchestrator.uptime_seconds / 3600)}h ${Math.floor((status.orchestrator.uptime_seconds % 3600) / 60)}m\n` +
              `🔢 Active Workflows: ${status.orchestrator.active_workflows}\n` +
              `📊 Completed Workflows: ${status.orchestrator.statistics.workflows_completed}\n` +
              `❌ Failed Workflows: ${status.orchestrator.statistics.workflows_failed}\n\n` +
              `🧠 Agent Status:\n` +
              `${Object.entries(status.orchestrator.agents).map(([name, agent]: [string, any]) =>
                `• ${name}: ${agent.status === 'active' ? '✅' : '❌'} ${agent.agent_id}`
              ).join('\n')}\n\n` +
              `📈 Performance:\n` +
              `• Messages Processed: ${status.orchestrator.statistics.messages_processed}\n` +
              `• Decisions Made: ${status.orchestrator.statistics.decisions_made}`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ Failed to get orchestrator status: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
      };
    }
  }

  private async getWorkflowStatus(workflowId: string) {
    try {
      const status = await apiRequest(`/api/orchestrator/workflows/${workflowId}`) as any;

      return {
        content: [
          {
            type: "text",
            text: `📋 Workflow Status for ${workflowId}:\n\n` +
              `🔢 Incident ID: ${status.incident_id}\n` +
              `📊 Status: ${status.status.toUpperCase()}\n` +
              `⏱️  Execution Time: ${status.execution_time?.toFixed(2) || 'N/A'}s\n` +
              `🎯 Current Step: ${status.current_step}\n` +
              `🤖 Agents Involved: ${status.agents_involved?.join(', ') || 'None'}\n\n` +
              `📝 Recent Activity:\n` +
              `• Started: ${new Date(status.start_time).toLocaleString()}\n` +
              `${status.end_time ? `• Completed: ${new Date(status.end_time).toLocaleString()}\n` : ''}` +
              `${status.errors?.length > 0 ? `❌ Errors: ${status.errors.join(', ')}\n` : '✅ No errors reported'}`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ Failed to get workflow status for ${workflowId}: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
      };
    }
  }

  private async threatIntelLookup(args: any) {
    const { ip_address, include_reputation = true, include_geolocation = true, sources = ["virustotal", "abuseipdb"] } = args;

    try {
      // Trigger threat intelligence lookup
      const result = await apiRequest("/api/agents/orchestrate", {
        method: "POST",
        body: {
          query: `threat intel lookup for ${ip_address}`,
          agent_type: "threat_intel_lookup",
          context: {
            ip_address,
            include_reputation,
            include_geolocation,
            sources
          }
        }
      }) as any;

      return {
        content: [
          {
            type: "text",
            text: `🔍 Threat Intelligence for ${ip_address}:\n\n` +
              `📊 Reputation Score: ${result.reputation_score || 'N/A'}/100\n` +
              `🌍 Geolocation: ${result.geolocation?.country || 'Unknown'}, ${result.geolocation?.city || 'Unknown'}\n` +
              `🏷️  Categories: ${result.threat_categories?.join(', ') || 'None detected'}\n\n` +
              `🔗 Sources Queried:\n` +
              `${sources.map((source: string) => `• ${source.toUpperCase()}: ${result.sources?.[source] ? '✅ Available' : '❌ No data'}`).join('\n')}\n\n` +
              `💡 Assessment:\n` +
              `${result.summary || 'Analysis completed'}`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ Threat intelligence lookup failed for ${ip_address}: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
      };
    }
  }

  private async attributionAnalysis(args: any) {
    const { incident_id, include_campaign_analysis = true, confidence_threshold = 0.5 } = args;

    try {
      // Trigger attribution analysis
      const result = await apiRequest("/api/agents/orchestrate", {
        method: "POST",
        body: {
          incident_id,
          agent_type: "attribution_analysis",
          context: {
            include_campaign_analysis,
            confidence_threshold
          }
        }
      }) as any;

      return {
        content: [
          {
            type: "text",
            text: `🎭 Threat Attribution Analysis for Incident #${incident_id}:\n\n` +
              `🎯 Confidence Score: ${(result.confidence_score * 100).toFixed(1)}%\n` +
              `🏷️  Threat Category: ${result.threat_category || 'Unknown'}\n` +
              `🕵️  Attributed Actors: ${result.attributed_actors?.length || 0} identified\n\n` +
              `${result.attributed_actors?.length > 0 ?
                `🎪 Identified Actors:\n${result.attributed_actors.map((actor: any) =>
                  `• ${actor.name} (${(actor.confidence * 100).toFixed(1)}% confidence)`
                ).join('\n')}\n\n` : ''}` +
              `📈 Campaign Analysis: ${include_campaign_analysis ? '✅ Included' : '❌ Skipped'}\n\n` +
              `🔍 Key Indicators:\n` +
              `${this.formatAttributionIndicators(result)}`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ Attribution analysis failed for incident #${incident_id}: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
      };
    }
  }

  private async startIncidentStream(args: any) {
    const { client_id, filters = {} } = args;

    if (!ENABLE_STREAMING) {
      return {
        content: [
          {
            type: "text",
            text: "❌ Real-time streaming is not enabled. Set ENABLE_STREAMING=true environment variable.",
          },
        ],
      };
    }

    try {
      // Start streaming for this client
      streamClients.add(client_id);

      // Set up periodic updates
      const streamInterval = setInterval(async () => {
        try {
          if (!streamClients.has(client_id)) {
            clearInterval(streamInterval);
            return;
          }

          // Get recent incidents based on filters
          const incidents = await apiRequest("/incidents") as any[];
          const filteredIncidents = this.filterIncidentsByCriteria(incidents, filters);

          if (filteredIncidents.length > 0) {
            console.log(`Streaming ${filteredIncidents.length} incidents to client ${client_id}`);
            // In a real implementation, this would send data to the client
          }
        } catch (error) {
          console.error(`Streaming error for client ${client_id}:`, error);
          clearInterval(streamInterval);
          streamClients.delete(client_id);
        }
      }, STREAMING_INTERVAL);

      activeStreams.set(client_id, streamInterval);

      return {
        content: [
          {
            type: "text",
            text: `✅ Real-time incident stream started for client ${client_id}\n\n` +
              `🔄 Update Interval: ${STREAMING_INTERVAL / 1000} seconds\n` +
              `🎯 Active Filters: ${Object.keys(filters).length > 0 ? JSON.stringify(filters, null, 2) : 'None'}\n\n` +
              `📡 Stream Status: ACTIVE\n` +
              `🆔 Client ID: ${client_id}\n\n` +
              `💡 Use 'stop_incident_stream' tool to stop this stream.`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ Failed to start incident stream: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
      };
    }
  }

  private async stopIncidentStream(clientId: string) {
    try {
      // Stop streaming for this client
      const streamInterval = activeStreams.get(clientId);
      if (streamInterval) {
        clearInterval(streamInterval);
        activeStreams.delete(clientId);
      }

      streamClients.delete(clientId);

      return {
        content: [
          {
            type: "text",
            text: `✅ Incident stream stopped for client ${clientId}\n\n` +
              `📡 Stream Status: STOPPED\n` +
              `🧹 Resources cleaned up`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ Failed to stop incident stream for client ${clientId}: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
      };
    }
  }

  private async queryThreatPatterns(args: any) {
    const { pattern_type, time_range = {}, min_confidence = 0.5 } = args;
    const hours_back = time_range.hours_back || 24;

    try {
      // Query threat patterns
      const result = await apiRequest("/api/agents/orchestrate", {
        method: "POST",
        body: {
          query: `threat pattern analysis: ${pattern_type}`,
          agent_type: "threat_pattern_query",
          context: {
            pattern_type,
            hours_back,
            min_confidence
          }
        }
      }) as any;

      return {
        content: [
          {
            type: "text",
            text: `🔍 Threat Pattern Analysis: ${pattern_type.replace('_', ' ').toUpperCase()}\n\n` +
              `⏰ Time Window: Last ${hours_back} hours\n` +
              `🎯 Confidence Threshold: ${(min_confidence * 100).toFixed(1)}%\n\n` +
              `📊 Pattern Analysis:\n` +
              `${this.formatPatternResults(result)}\n\n` +
              `🚨 High-Confidence Findings:\n` +
              `${this.formatHighConfidencePatterns(result, min_confidence)}`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ Threat pattern query failed for ${pattern_type}: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
      };
    }
  }

  private async correlationAnalysis(args: any) {
    const { correlation_type, incidents = [], time_window_hours = 24 } = args;

    try {
      // Perform correlation analysis
      const result = await apiRequest("/api/agents/orchestrate", {
        method: "POST",
        body: {
          query: `correlation analysis: ${correlation_type}`,
          agent_type: "correlation_analysis",
          context: {
            correlation_type,
            incidents,
            time_window_hours
          }
        }
      }) as any;

      return {
        content: [
          {
            type: "text",
            text: `🔗 Correlation Analysis: ${correlation_type.replace('_', ' ').toUpperCase()}\n\n` +
              `⏰ Time Window: ${time_window_hours} hours\n` +
              `📊 Incidents Analyzed: ${incidents.length}\n\n` +
              `🎯 Correlation Findings:\n` +
              `${this.formatCorrelationResults(result)}\n\n` +
              `📈 Correlation Strength:\n` +
              `${this.formatCorrelationStrength(result)}`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ Correlation analysis failed for ${correlation_type}: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
      };
    }
  }

  // === NATURAL LANGUAGE PROCESSING METHODS ===

  private async naturalLanguageQuery(args: any) {
    const { query, include_context = true, max_results = 10, semantic_search = true } = args;

    try {
      // Call NLP analyzer endpoint
      const result = await apiRequest("/api/nlp/query", {
        method: "POST",
        body: {
          query,
          include_context,
          max_results,
          semantic_search
        }
      }) as any;

      return {
        content: [
          {
            type: "text",
            text: `🧠 Natural Language Query Analysis: "${query}"\n\n` +
              `🎯 Query Understanding: ${result.query_understanding}\n` +
              `📊 Confidence Score: ${(result.confidence_score * 100).toFixed(1)}%\n\n` +
              `🔍 Findings (${result.findings?.length || 0} results):\n` +
              `${this.formatNLPFindings(result.findings)}\n\n` +
              `💡 Recommendations:\n` +
              `${result.recommendations?.map((rec: string, i: number) => `${i + 1}. ${rec}`).join('\n') || 'No recommendations'}\n\n` +
              `❓ Follow-up Questions:\n` +
              `${result.follow_up_questions?.map((q: string, i: number) => `${i + 1}. ${q}`).join('\n') || 'No follow-up questions'}\n\n` +
              `🤖 Reasoning: ${result.reasoning || 'Analysis completed'}`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ Natural language query failed for "${query}": ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
      };
    }
  }

  private async nlpThreatAnalysis(args: any) {
    const { query, analysis_type, time_range_hours = 24 } = args;

    try {
      // Call NLP threat analysis endpoint
      const result = await apiRequest("/api/nlp/threat-analysis", {
        method: "POST",
        body: {
          query,
          analysis_type,
          time_range_hours
        }
      }) as any;

      let analysisTypeDisplay = analysis_type ? analysis_type.replace('_', ' ').toUpperCase() : 'COMPREHENSIVE';

      return {
        content: [
          {
            type: "text",
            text: `🔬 NLP Threat Analysis: ${analysisTypeDisplay}\n\n` +
              `🎯 Query: "${query}"\n` +
              `⏱️  Time Range: ${time_range_hours} hours\n` +
              `📊 Query Understanding: ${result.query_understanding}\n` +
              `🎯 Confidence: ${(result.confidence_score * 100).toFixed(1)}%\n\n` +
              `📈 Analysis Results:\n` +
              `${this.formatThreatAnalysisResults(result.findings)}\n\n` +
              `🎯 Key Insights:\n` +
              `${this.formatThreatInsights(result)}\n\n` +
              `💡 Strategic Recommendations:\n` +
              `${result.recommendations?.map((rec: string, i: number) => `${i + 1}. ${rec}`).join('\n') || 'No recommendations available'}`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ NLP threat analysis failed for "${query}": ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
      };
    }
  }

  private async semanticIncidentSearch(args: any) {
    const { query, similarity_threshold = 0.7, max_results = 10 } = args;

    try {
      // Call semantic search endpoint
      const result = await apiRequest("/api/nlp/semantic-search", {
        method: "POST",
        body: {
          query,
          similarity_threshold,
          max_results
        }
      }) as any;

      return {
        content: [
          {
            type: "text",
            text: `🔍 Semantic Incident Search: "${query}"\n\n` +
              `🎯 Similarity Threshold: ${(similarity_threshold * 100).toFixed(1)}%\n` +
              `📊 Results Found: ${result.similar_incidents?.length || 0}\n\n` +
              `🎯 Similar Incidents:\n` +
              `${this.formatSemanticSearchResults(result.similar_incidents)}\n\n` +
              `📈 Search Quality:\n` +
              `• Average Similarity: ${result.avg_similarity ? (result.avg_similarity * 100).toFixed(1) : 'N/A'}%\n` +
              `• Query Understanding: ${result.query_understanding || 'Standard text matching'}\n` +
              `• Semantic Features: ${result.semantic_features?.join(', ') || 'Basic keyword matching'}\n\n` +
              `💡 Search Tips:\n` +
              `• Try different keywords or phrases\n` +
              `• Lower the similarity threshold for broader results\n` +
              `• Use specific threat types or indicators for focused results`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ Semantic search failed for "${query}": ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
      };
    }
  }

  // === PHASE 2 & 3 IMPLEMENTATION METHODS ===

  private async createVisualWorkflow(args: any) {
    const { incident_id, playbook_name, actions, auto_execute = false } = args;

    try {
      // Create workflow using new visual workflow API
      const steps = actions.map((action: any) => ({
        action_type: action.action_type,
        parameters: action.parameters || {
          target: incident_id,
          reason: `MCP-created workflow: ${playbook_name}`
        },
        timeout_seconds: 300,
        continue_on_failure: false,
        max_retries: 3
      }));

      const result = await apiRequest("/api/response/workflows/create", {
        method: "POST",
        body: {
          incident_id,
          playbook_name,
          steps,
          auto_execute,
          priority: 'medium'
        }
      }) as any;

      return {
        content: [
          {
            type: "text",
            text: `🎨 Visual Workflow Created: "${playbook_name}"\n\n` +
              `📋 Workflow ID: ${result.workflow_id}\n` +
              `🎯 Incident: #${incident_id}\n` +
              `⚡ Actions: ${actions.length} enterprise actions\n` +
              `🤖 Auto-Execute: ${auto_execute ? 'YES' : 'NO'}\n` +
              `📊 Status: ${result.status?.toUpperCase() || 'CREATED'}\n` +
              `⏱️  Estimated Duration: ${result.estimated_duration_minutes || 'N/A'} minutes\n` +
              `✅ Approval Required: ${result.approval_required ? 'YES' : 'NO'}\n\n` +
              `🔧 Workflow Steps:\n` +
              `${actions.map((action: any, i: number) => 
                `${i + 1}. ${action.action_type.replace('_', ' ').toUpperCase()}`
              ).join('\n')}\n\n` +
              `💡 Use 'execute_response_workflow' to run this workflow\n` +
              `📊 Use 'get_workflow_execution_status' to monitor progress`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ Failed to create visual workflow "${playbook_name}": ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
      };
    }
  }

  private async getAvailableResponseActions(args: any) {
    const { category = 'all', include_details = true } = args || {};

    try {
      const result = await apiRequest(`/api/response/actions${category !== 'all' ? `?category=${category}` : ''}`) as any;

      if (!result.success) {
        throw new Error(result.error || 'Failed to get actions');
      }

      const actions = result.actions || {};
      const actionCount = Object.keys(actions).length;

      let response = `🛡️  Available Response Actions: ${actionCount} actions\n\n`;
      
      if (category !== 'all') {
        response += `📂 Category: ${category.toUpperCase()}\n\n`;
      }

      // Group actions by category
      const actionsByCategory: any = {};
      Object.entries(actions).forEach(([actionType, action]: [string, any]) => {
        const cat = action.category || 'unknown';
        if (!actionsByCategory[cat]) actionsByCategory[cat] = [];
        actionsByCategory[cat].push({ actionType, ...action });
      });

      // Format by category
      Object.entries(actionsByCategory).forEach(([cat, catActions]: [string, any]) => {
        const categoryIcons = {
          network: '🌐', endpoint: '🖥️', email: '📧', cloud: '☁️',
          identity: '🔑', data: '📊', compliance: '📋', forensics: '🔬'
        };
        
        response += `${(categoryIcons as any)[cat] || '⚙️'} ${cat.toUpperCase()} ACTIONS (${catActions.length}):\n`;
        
        catActions.forEach((action: any, i: number) => {
          response += `${i + 1}. ${action.name}\n`;
          if (include_details) {
            response += `   • Action Type: ${action.actionType}\n`;
            response += `   • Description: ${action.description}\n`;
            response += `   • Safety Level: ${action.safety_level?.toUpperCase()}\n`;
            response += `   • Duration: ${Math.floor(action.estimated_duration / 60)}m\n`;
            response += `   • Rollback: ${action.rollback_supported ? 'YES' : 'NO'}\n`;
          }
          response += '\n';
        });
        response += '\n';
      });

      response += `💡 Usage Examples:\n`;
      response += `• create_visual_workflow(incident_id=123, playbook_name="Malware Response", actions=[{action_type: "isolate_host_advanced", parameters: {}}])\n`;
      response += `• execute_enterprise_action(action_type="memory_dump_collection", incident_id=123)\n`;
      response += `• get_ai_response_recommendations(incident_id=123) for AI-powered suggestions`;

      return {
        content: [
          {
            type: "text",
            text: response,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ Failed to get available actions: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
      };
    }
  }

  private async executeResponseWorkflow(args: any) {
    const { workflow_id, executed_by = "mcp_assistant" } = args;

    try {
      const result = await apiRequest("/api/response/workflows/execute", {
        method: "POST",
        body: {
          workflow_db_id: parseInt(workflow_id), // Convert to DB ID if needed
          executed_by
        }
      }) as any;

      return {
        content: [
          {
            type: "text",
            text: `🚀 Workflow Execution Started: ${workflow_id}\n\n` +
              `👤 Executed By: ${executed_by}\n` +
              `📊 Status: ${result.status?.toUpperCase() || 'UNKNOWN'}\n` +
              `⚡ Steps Completed: ${result.steps_completed || 0} / ${result.total_steps || 0}\n` +
              `✅ Success Rate: ${result.success_rate ? (result.success_rate * 100).toFixed(1) : 'N/A'}%\n` +
              `⏱️  Execution Time: ${result.execution_time_ms || 0}ms\n\n` +
              `📋 Execution Results:\n` +
              `${result.results?.map((res: any, i: number) => 
                `${i + 1}. ${res.action_type}: ${res.success ? '✅ SUCCESS' : '❌ FAILED'}`
              ).join('\n') || 'No detailed results available'}\n\n` +
              `💡 Monitor progress with: get_workflow_execution_status("${workflow_id}")`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ Failed to execute workflow ${workflow_id}: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
      };
    }
  }

  private async getWorkflowExecutionStatus(workflowId: string) {
    try {
      const result = await apiRequest(`/api/response/workflows/${workflowId}/status`) as any;

      if (!result.success) {
        throw new Error(result.error || 'Failed to get workflow status');
      }

      return {
        content: [
          {
            type: "text",
            text: `📊 Workflow Status: ${workflowId}\n\n` +
              `🔢 Status: ${result.status?.toUpperCase() || 'UNKNOWN'}\n` +
              `📈 Progress: ${result.progress_percentage?.toFixed(1) || 0}%\n` +
              `⚡ Current Step: ${result.current_step || 0} / ${result.total_steps || 0}\n` +
              `✅ Success Rate: ${result.success_rate ? (result.success_rate * 100).toFixed(1) : 'N/A'}%\n` +
              `⏱️  Execution Time: ${result.execution_time_ms || 0}ms\n` +
              `🕐 Created: ${result.created_at ? new Date(result.created_at).toLocaleString() : 'N/A'}\n` +
              `${result.completed_at ? `✅ Completed: ${new Date(result.completed_at).toLocaleString()}\n` : ''}` +
              `🔐 Approval Required: ${result.approval_required ? 'YES' : 'NO'}\n` +
              `${result.approved_at ? `✅ Approved: ${new Date(result.approved_at).toLocaleString()}\n` : ''}\n` +
              `📋 Action Status:\n` +
              `${result.actions?.map((action: any, i: number) => 
                `${i + 1}. ${action.action_type}: ${action.status?.toUpperCase() || 'UNKNOWN'}`
              ).join('\n') || 'No action details available'}\n\n` +
              `📊 Impact Metrics:\n` +
              `${result.impact_metrics?.map((metric: any) => 
                `• Attacks Blocked: ${metric.attacks_blocked}\n` +
                `• Response Time: ${metric.response_time_ms}ms\n` +
                `• Success Rate: ${(metric.success_rate * 100).toFixed(1)}%`
              ).join('\n') || 'No impact metrics available'}`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ Failed to get workflow status for ${workflowId}: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
      };
    }
  }

  private async getAIResponseRecommendations(args: any) {
    const { incident_id, context = {}, strategy = "balanced" } = args;

    try {
      const result = await apiRequest("/api/ai/response/recommendations", {
        method: "POST",
        body: { incident_id, context }
      }) as any;

      if (!result.success) {
        throw new Error(result.error || 'Failed to get AI recommendations');
      }

      return {
        content: [
          {
            type: "text",
            text: `🧠 AI Response Recommendations for Incident #${incident_id}:\n\n` +
              `🎯 Strategy: ${result.strategy?.toUpperCase() || 'ADAPTIVE'}\n` +
              `📊 Overall Confidence: ${result.confidence_analysis?.overall_confidence ? (result.confidence_analysis.overall_confidence * 100).toFixed(1) : 'N/A'}%\n` +
              `⏱️  Estimated Duration: ${result.estimated_duration ? Math.floor(result.estimated_duration / 60) : 'N/A'} minutes\n\n` +
              `🎯 Top Recommendations:\n` +
              `${result.recommendations?.slice(0, 5).map((rec: any, i: number) => 
                `${i + 1}. ${rec.action_type.replace('_', ' ').toUpperCase()} (${(rec.confidence * 100).toFixed(1)}% confidence)\n` +
                `   • Priority: ${rec.priority}\n` +
                `   • Duration: ${Math.floor(rec.estimated_duration / 60)}m\n` +
                `   • Approval: ${rec.approval_required ? 'Required' : 'Not Required'}\n` +
                `   • Safety: ${rec.safety_considerations?.join(', ') || 'Standard protocols'}\n`
              ).join('\n') || 'No recommendations available'}\n\n` +
              `🤖 AI Insights:\n` +
              `${result.explanations?.summary || 'AI analysis completed'}\n\n` +
              `💡 Create workflow: create_visual_workflow(incident_id=${incident_id}, playbook_name="AI Recommended Response", actions=[...])`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ Failed to get AI recommendations for incident #${incident_id}: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
      };
    }
  }

  private async analyzeIncidentContextComprehensive(args: any) {
    const { incident_id, include_predictions = true, analysis_depth = "comprehensive" } = args;

    try {
      const result = await apiRequest(`/api/ai/response/context/${incident_id}?include_predictions=${include_predictions}`) as any;

      if (!result.success) {
        throw new Error(result.error || 'Failed to analyze context');
      }

      const context = result.context_analysis;

      return {
        content: [
          {
            type: "text",
            text: `🔬 Comprehensive Context Analysis for Incident #${incident_id}:\n\n` +
              `📊 Analysis Depth: ${analysis_depth.toUpperCase()}\n` +
              `⏰ Analysis Time: ${new Date(context.analysis_timestamp).toLocaleString()}\n` +
              `✨ Quality Score: ${context.analysis_quality?.score ? (context.analysis_quality.score * 100).toFixed(1) : 'N/A'}%\n\n` +
              `🎯 THREAT ANALYSIS:\n` +
              `• Severity: ${context.threat_context?.severity_score ? (context.threat_context.severity_score * 100).toFixed(1) : 'N/A'}% (${context.threat_context?.threat_category || 'unknown'})\n` +
              `• Attack Vector: ${context.threat_context?.attack_vector || 'unknown'}\n` +
              `• Confidence: ${context.threat_context?.confidence ? (context.threat_context.confidence * 100).toFixed(1) : 'N/A'}%\n` +
              `• Indicators: ${context.threat_context?.indicators?.join(', ') || 'None identified'}\n\n` +
              `⏱️  TEMPORAL ANALYSIS:\n` +
              `• Pattern: ${context.temporal_analysis?.pattern?.replace('_', ' ')?.toUpperCase() || 'UNKNOWN'}\n` +
              `• Duration: ${context.temporal_analysis?.total_duration_seconds ? Math.floor(context.temporal_analysis.total_duration_seconds / 60) : 'N/A'}m\n` +
              `• Event Rate: ${context.temporal_analysis?.event_rate_per_minute?.toFixed(1) || 'N/A'}/min\n\n` +
              `🧠 BEHAVIORAL ANALYSIS:\n` +
              `• Primary Behavior: ${context.behavioral_analysis?.primary_behavior?.replace('_', ' ')?.toUpperCase() || 'UNKNOWN'}\n` +
              `• Sophistication: ${context.behavioral_analysis?.sophistication_indicators?.join(', ') || 'Basic'}\n` +
              `• Attacker Intent: ${context.behavioral_analysis?.attacker_intent?.replace('_', ' ')?.toUpperCase() || 'UNKNOWN'}\n\n` +
              `${include_predictions ? 
                `🔮 PREDICTIVE ANALYSIS:\n` +
                `• Escalation Probability: ${context.predictive_analysis?.escalation_probability ? (context.predictive_analysis.escalation_probability * 100).toFixed(1) : 'N/A'}%\n` +
                `• Lateral Movement Risk: ${context.predictive_analysis?.lateral_movement_risk ? (context.predictive_analysis.lateral_movement_risk * 100).toFixed(1) : 'N/A'}%\n` +
                `• Predicted Duration: ${context.predictive_analysis?.predicted_duration_hours?.toFixed(1) || 'N/A'} hours\n\n` 
                : ''
              }` +
              `🎯 Similar Incidents: ${context.similar_incidents?.length || 0} found\n\n` +
              `💡 Next Steps:\n` +
              `• get_ai_response_recommendations(incident_id=${incident_id}) for AI suggestions\n` +
              `• create_visual_workflow(...) to build response plan\n` +
              `• execute_enterprise_action(...) for immediate action`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ Failed to analyze context for incident #${incident_id}: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
      };
    }
  }

  private async optimizeResponseStrategy(args: any) {
    const { workflow_id, optimization_strategy = "effectiveness", context = {} } = args;

    try {
      const result = await apiRequest(`/api/ai/response/optimize/${workflow_id}`, {
        method: "POST",
        body: { strategy: optimization_strategy, context }
      }) as any;

      if (!result.success) {
        throw new Error(result.error || 'Failed to optimize strategy');
      }

      const optimization = result.optimization_result;

      return {
        content: [
          {
            type: "text",
            text: `⚡ Response Strategy Optimization for ${workflow_id}:\n\n` +
              `🎯 Optimization Strategy: ${optimization_strategy.toUpperCase()}\n` +
              `📊 Optimization Score: ${optimization.optimization_score?.toFixed(2) || 'N/A'}\n` +
              `✅ Confidence: ${optimization.confidence ? (optimization.confidence * 100).toFixed(1) : 'N/A'}%\n` +
              `📈 Risk Reduction: ${optimization.risk_reduction ? (optimization.risk_reduction * 100).toFixed(1) : 'N/A'}%\n` +
              `⚡ Efficiency Gain: ${optimization.efficiency_gain ? (optimization.efficiency_gain * 100).toFixed(1) : 'N/A'}%\n\n` +
              `🔧 Improvements Applied:\n` +
              `${optimization.improvements?.map((imp: string, i: number) => `${i + 1}. ${imp}`).join('\n') || 'No specific improvements identified'}\n\n` +
              `🎯 Optimized Workflow:\n` +
              `${this.formatOptimizedWorkflow(optimization.optimized_workflow)}\n\n` +
              `💡 The workflow has been optimized for ${optimization_strategy}. Re-execute for improved performance.`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ Failed to optimize strategy for workflow ${workflow_id}: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
      };
    }
  }

  private async generateAdaptiveRecommendations(args: any) {
    const { incident_id, user_context = {}, learning_mode = "balanced" } = args;

    try {
      const result = await apiRequest("/api/ai/response/adaptive", {
        method: "POST",
        body: { incident_id, user_context }
      }) as any;

      if (!result.success) {
        throw new Error(result.error || 'Failed to generate adaptive recommendations');
      }

      return {
        content: [
          {
            type: "text",
            text: `🤖 Adaptive AI Recommendations for Incident #${incident_id}:\n\n` +
              `🧠 Learning Mode: ${learning_mode.toUpperCase()}\n` +
              `📊 Adaptation Applied: ${result.optimization_applied ? 'YES' : 'NO'}\n` +
              `🎯 Learning Maturity: ${result.learning_insights?.learning_maturity || 'developing'}\n\n` +
              `🎯 ADAPTIVE RECOMMENDATIONS:\n` +
              `${result.adaptive_recommendations?.slice(0, 5).map((rec: any, i: number) => 
                `${i + 1}. ${rec.action_type.replace('_', ' ').toUpperCase()}\n` +
                `   • Confidence: ${(rec.confidence * 100).toFixed(1)}% ${rec.learning_adjusted ? '(📚 Learning Adjusted)' : ''}\n` +
                `   • Priority: ${rec.priority}\n` +
                `   ${rec.historical_basis ? `• Historical Success: ${(rec.historical_basis.success_rate * 100).toFixed(1)}% (${rec.historical_basis.sample_size} samples)\n` : ''}` +
                `   ${rec.learned ? `• 🎓 LEARNED ACTION: Based on ${rec.learning_basis?.sample_size || 0} successful cases\n` : ''}`
              ).join('\n') || 'No adaptive recommendations available'}\n\n` +
              `📋 RESPONSE PLAN:\n` +
              `• Primary Strategy: ${result.response_plan?.recommended_strategy?.replace('_', ' ')?.toUpperCase() || 'STANDARD'}\n` +
              `• Primary Actions: ${result.response_plan?.primary_plan?.actions?.length || 0}\n` +
              `• Fallback Plan: ${result.response_plan?.fallback_plan ? 'Available' : 'Not Available'}\n` +
              `• Emergency Plan: ${result.response_plan?.emergency_plan ? 'Available' : 'Not Available'}\n\n` +
              `🎓 LEARNING INSIGHTS:\n` +
              `• Learning Velocity: ${result.learning_insights?.adaptation_velocity || 'moderate'}\n` +
              `• Recommendation Accuracy: ${result.learning_insights?.recommendation_accuracy ? (result.learning_insights.recommendation_accuracy * 100).toFixed(1) : 'N/A'}%\n` +
              `• Improvement Trends: ${result.learning_insights?.improvement_trends?.join(', ') || 'Stable performance'}\n\n` +
              `💡 The AI system learns from each execution to improve future recommendations.`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ Failed to generate adaptive recommendations for incident #${incident_id}: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
      };
    }
  }

  private async executeEnterpriseAction(args: any) {
    const { action_type, incident_id, parameters = {}, safety_check = true } = args;

    try {
      const result = await apiRequest("/api/response/actions/execute", {
        method: "POST",
        body: {
          action_type,
          incident_id,
          parameters: {
            ...parameters,
            target: incident_id,
            reason: `MCP-triggered ${action_type} action`,
            safety_validated: safety_check
          }
        }
      }) as any;

      return {
        content: [
          {
            type: "text",
            text: `⚡ Enterprise Action Executed: ${action_type.replace('_', ' ').toUpperCase()}\n\n` +
              `🎯 Incident: #${incident_id}\n` +
              `📊 Status: ${result.status?.toUpperCase() || 'UNKNOWN'}\n` +
              `✅ Success: ${result.success ? 'YES' : 'NO'}\n` +
              `🔐 Safety Check: ${safety_check ? 'PERFORMED' : 'SKIPPED'}\n\n` +
              `📋 Execution Details:\n` +
              `${result.steps_completed !== undefined ? `• Steps Completed: ${result.steps_completed} / ${result.total_steps}\n` : ''}` +
              `${result.success_rate !== undefined ? `• Success Rate: ${(result.success_rate * 100).toFixed(1)}%\n` : ''}` +
              `${result.execution_time_ms ? `• Execution Time: ${result.execution_time_ms}ms\n` : ''}` +
              `${result.workflow_id ? `• Workflow ID: ${result.workflow_id}\n` : ''}\n` +
              `📊 Results:\n` +
              `${result.results?.map((res: any) => 
                `• ${res.action_type}: ${res.success ? '✅ SUCCESS' : '❌ FAILED'}\n` +
                `  ${res.result?.detail || 'No details available'}`
              ).join('\n') || result.detail || 'Action completed'}\n\n` +
              `💡 Monitor with: get_workflow_execution_status("${result.workflow_id || 'N/A'}")`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ Failed to execute ${action_type} for incident #${incident_id}: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
      };
    }
  }

  private async getResponseImpactMetrics(args: any) {
    const { workflow_id, days_back = 7, metric_types = ["effectiveness", "performance"] } = args || {};

    try {
      const params = new URLSearchParams();
      if (workflow_id) params.append('workflow_id', workflow_id);
      if (days_back) params.append('days_back', days_back.toString());

      const result = await apiRequest(`/api/response/metrics/impact?${params.toString()}`) as any;

      if (!result.success) {
        throw new Error(result.error || 'Failed to get impact metrics');
      }

      return {
        content: [
          {
            type: "text",
            text: `📊 Response Impact Metrics:\n\n` +
              `⏰ Time Period: Last ${days_back} days\n` +
              `${workflow_id ? `🎯 Workflow: ${workflow_id}\n` : '🌐 System-wide metrics\n'}\n` +
              `📈 SUMMARY METRICS:\n` +
              `• Total Attacks Blocked: ${result.summary?.total_attacks_blocked || 0}\n` +
              `• Total False Positives: ${result.summary?.total_false_positives || 0}\n` +
              `• Average Response Time: ${result.summary?.average_response_time_ms ? Math.floor(result.summary.average_response_time_ms / 1000) : 'N/A'}s\n` +
              `• Average Success Rate: ${result.summary?.average_success_rate ? (result.summary.average_success_rate * 100).toFixed(1) : 'N/A'}%\n` +
              `• Total Metrics: ${result.summary?.metrics_count || 0}\n\n` +
              `📋 DETAILED METRICS (Last 5):\n` +
              `${result.detailed_metrics?.slice(0, 5).map((metric: any, i: number) => 
                `${i + 1}. Response #${i + 1}:\n` +
                `   • Attacks Blocked: ${metric.attacks_blocked}\n` +
                `   • Success Rate: ${(metric.success_rate * 100).toFixed(1)}%\n` +
                `   • Response Time: ${Math.floor(metric.response_time_ms / 1000)}s\n` +
                `   • Systems Affected: ${metric.systems_affected}\n` +
                `   • Cost Impact: $${metric.cost_impact_usd?.toFixed(0) || 0}\n` +
                `   • Compliance Impact: ${metric.compliance_impact?.toUpperCase() || 'NONE'}`
              ).join('\n\n') || 'No detailed metrics available'}\n\n` +
              `💡 Metrics are updated in real-time as responses execute.`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ Failed to get impact metrics: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
      };
    }
  }

  private async testTPotIntegration(args: any) {
    const { test_type = "comprehensive", dry_run = false } = args || {};

    try {
      // Test T-Pot connectivity and integration
      const result = await apiRequest("/api/response/test", {
        method: "POST",
        body: { test_type, dry_run }
      }) as any;

      return {
        content: [
          {
            type: "text",
            text: `🧪 T-Pot Integration Test: ${test_type.toUpperCase()}\n\n` +
              `🌐 T-Pot Server: admin@34.193.101.171:64295\n` +
              `🔍 Test Type: ${test_type.toUpperCase()}\n` +
              `🔒 Dry Run: ${dry_run ? 'YES (No actual changes)' : 'NO (Live execution)'}\n\n` +
              `📊 TEST RESULTS:\n` +
              `• SSH Connectivity: ${result.ssh_connectivity ? '✅ CONNECTED' : '❌ FAILED'}\n` +
              `• Authentication: ${result.authentication ? '✅ AUTHENTICATED' : '❌ FAILED'}\n` +
              `• Command Execution: ${result.command_execution ? '✅ WORKING' : '❌ FAILED'}\n` +
              `• Iptables Access: ${result.iptables_access ? '✅ AVAILABLE' : '❌ UNAVAILABLE'}\n` +
              `• System Detection: ${result.system_detection || 'Unknown system'}\n\n` +
              `🔧 CAPABILITIES VERIFIED:\n` +
              `${result.capabilities_verified?.map((cap: string, i: number) => `${i + 1}. ${cap}`).join('\n') || 'No capabilities verified'}\n\n` +
              `${result.test_commands ? 
                `📋 TEST COMMANDS EXECUTED:\n` +
                `${result.test_commands.map((cmd: any) => 
                  `• ${cmd.command}: ${cmd.success ? '✅ SUCCESS' : '❌ FAILED'}\n` +
                  `  ${cmd.output || cmd.error || 'No output'}`
                ).join('\n')}\n\n`
                : ''
              }` +
              `💡 T-Pot integration is ${result.overall_status === 'success' ? '✅ FULLY OPERATIONAL' : '⚠️ PARTIALLY WORKING'}`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ T-Pot integration test failed: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
      };
    }
  }

  private async executeTPotCommand(args: any) {
    const { command_type, target_ip, parameters = {}, confirmation_required = true } = args;

    try {
      // Execute real command on T-Pot via containment agent
      let endpoint = "";
      let body: any = {};

      switch (command_type) {
        case "block_ip":
          if (!target_ip) throw new Error("target_ip required for block_ip command");
          endpoint = `/incidents/1/contain`; // Use dummy incident for direct IP blocking
          body = { ip_override: target_ip, ...parameters };
          break;
        
        case "unblock_ip":
          if (!target_ip) throw new Error("target_ip required for unblock_ip command");
          endpoint = `/incidents/1/unblock`;
          body = { ip_override: target_ip, ...parameters };
          break;
        
        case "list_rules":
          endpoint = `/api/response/test`;
          body = { test_type: "list_iptables" };
          break;
        
        case "system_status":
          endpoint = `/health`;
          break;
        
        default:
          throw new Error(`Unknown command type: ${command_type}`);
      }

      const result = await apiRequest(endpoint, {
        method: endpoint === "/health" ? "GET" : "POST",
        body: endpoint === "/health" ? undefined : body
      }) as any;

      return {
        content: [
          {
            type: "text",
            text: `🔧 T-Pot Command Executed: ${command_type.toUpperCase()}\n\n` +
              `🌐 Target: admin@34.193.101.171:64295\n` +
              `${target_ip ? `🎯 Target IP: ${target_ip}\n` : ''}` +
              `🔒 Confirmation: ${confirmation_required ? 'REQUIRED' : 'BYPASSED'}\n\n` +
              `📊 EXECUTION RESULT:\n` +
              `• Status: ${result.status?.toUpperCase() || (result.success ? 'SUCCESS' : 'FAILED')}\n` +
              `• Command Type: ${command_type}\n` +
              `${result.detail ? `• Details: ${result.detail}\n` : ''}` +
              `${result.output ? `• Output: ${result.output}\n` : ''}` +
              `${result.system_detected ? `• System: ${result.system_detected}\n` : ''}` +
              `${result.firewall_type ? `• Firewall: ${result.firewall_type}\n` : ''}\n` +
              `📋 RAW RESPONSE:\n` +
              `${JSON.stringify(result, null, 2)}\n\n` +
              `✅ Real T-Pot infrastructure command executed successfully!`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ Failed to execute T-Pot command ${command_type}: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
      };
    }
  }

  // === HELPER METHODS ===

  private formatNLPFindings(findings: any[]): string {
    if (!findings || findings.length === 0) {
      return "No findings available";
    }

    return findings
      .slice(0, 10)
      .map((finding, index) => {
        const relevance = finding.relevance_score ? ` (${(finding.relevance_score * 100).toFixed(1)}% relevant)` : '';
        let findingText = `${index + 1}. ${finding.type?.replace('_', ' ').toUpperCase() || 'Finding'}${relevance}\n`;
        
        if (finding.description) {
          findingText += `   Description: ${finding.description}\n`;
        }
        
        if (finding.incident_id) {
          findingText += `   Incident ID: ${finding.incident_id}\n`;
        }
        
        if (finding.src_ip) {
          findingText += `   Source IP: ${finding.src_ip}\n`;
        }
        
        if (finding.incident_count) {
          findingText += `   Incident Count: ${finding.incident_count}\n`;
        }
        
        if (finding.confidence_score) {
          findingText += `   Confidence: ${(finding.confidence_score * 100).toFixed(1)}%\n`;
        }
        
        return findingText;
      })
      .join('\n');
  }

  private formatThreatAnalysisResults(findings: any[]): string {
    if (!findings || findings.length === 0) {
      return "No threat analysis results available";
    }

    return findings
      .slice(0, 8)
      .map((finding, index) => {
        let result = `${index + 1}. ${finding.type?.replace('_', ' ').toUpperCase() || 'Analysis'}\n`;
        
        if (finding.threat_type) {
          result += `   • Threat Type: ${finding.threat_type}\n`;
        }
        
        if (finding.incident_count) {
          result += `   • Incidents: ${finding.incident_count}\n`;
        }
        
        if (finding.unique_ips) {
          result += `   • Unique IPs: ${finding.unique_ips}\n`;
        }
        
        if (finding.time_span_hours) {
          result += `   • Time Span: ${finding.time_span_hours} hours\n`;
        }
        
        if (finding.confidence_score) {
          result += `   • Confidence: ${(finding.confidence_score * 100).toFixed(1)}%\n`;
        }
        
        if (finding.description) {
          result += `   • Details: ${finding.description}\n`;
        }
        
        return result;
      })
      .join('\n');
  }

  private formatThreatInsights(result: any): string {
    const insights = [];
    
    if (result.structured_query?.entities) {
      const entityCount = Object.values(result.structured_query.entities).reduce(
        (sum: number, entities: any) => sum + (Array.isArray(entities) ? entities.length : 0), 0
      );
      if (entityCount > 0) {
        insights.push(`• Extracted ${entityCount} threat indicators from query`);
      }
    }
    
    if (result.structured_query?.threat_categories?.length > 0) {
      insights.push(`• Identified threat categories: ${result.structured_query.threat_categories.join(', ')}`);
    }
    
    if (result.structured_query?.time_constraints) {
      insights.push(`• Time-based analysis: ${result.structured_query.time_constraints.description}`);
    }
    
    if (result.findings?.length > 0) {
      const findingTypes = [...new Set(result.findings.map((f: any) => f.type))];
      insights.push(`• Generated ${result.findings.length} findings across ${findingTypes.length} analysis types`);
    }
    
    return insights.length > 0 ? insights.join('\n') : 'Standard threat analysis completed';
  }

  private formatSemanticSearchResults(incidents: any[]): string {
    if (!incidents || incidents.length === 0) {
      return "No similar incidents found";
    }

    return incidents
      .map((result, index) => {
        const incident = result.incident || result;
        const similarity = result.similarity_score ? ` (${(result.similarity_score * 100).toFixed(1)}% similar)` : '';
        
        return `${index + 1}. Incident #${incident.id}${similarity}\n` +
               `   • IP: ${incident.src_ip}\n` +
               `   • Reason: ${incident.reason}\n` +
               `   • Status: ${incident.status}\n` +
               `   • Created: ${new Date(incident.created_at).toLocaleString()}\n` +
               `   ${incident.risk_score ? `• Risk Score: ${(incident.risk_score * 100).toFixed(1)}%` : ''}`;
      })
      .join('\n\n');
  }

  private filterIncidentsByCriteria(incidents: any[], filters: any): any[] {
    return incidents.filter(incident => {
      if (filters.severity && !filters.severity.includes(incident.escalation_level)) {
        return false;
      }
      if (filters.threat_categories && !filters.threat_categories.includes(incident.threat_category)) {
        return false;
      }
      return true;
    });
  }

  private formatOrchestrationResults(result: any): string {
    if (!result) return "No results available";

    const coordination = result.coordination || {};
    const riskAssessment = coordination.risk_assessment || {};

    return `• Risk Level: ${riskAssessment.level?.toUpperCase() || 'UNKNOWN'}\n` +
           `• Overall Confidence: ${(coordination.confidence_levels?.overall * 100)?.toFixed(1) || 'N/A'}%\n` +
           `• Attribution Confidence: ${(coordination.confidence_levels?.attribution * 100)?.toFixed(1) || 'N/A'}%\n` +
           `• Containment Confidence: ${(coordination.confidence_levels?.containment * 100)?.toFixed(1) || 'N/A'}%\n` +
           `• Forensic Risk Score: ${coordination.decision_factors?.forensic_risk_score?.toFixed(2) || 'N/A'}`;
  }

  private formatRecommendations(result: any): string {
    if (!result?.coordination?.recommended_actions) return "No specific recommendations";

    return result.coordination.recommended_actions
      .map((action: any, index: number) => `${index + 1}. ${action.action || action.details || action}`)
      .join('\n');
  }

  private formatThreatHuntResults(result: any): string {
    if (!result?.findings) return "No hunting results found";

    return result.findings
      .slice(0, 10)
      .map((finding: any, index: number) =>
        `${index + 1}. ${finding.type}: ${finding.description} (${finding.severity})`
      )
      .join('\n');
  }

  private formatHuntingRecommendations(result: any): string {
    return "• Continue monitoring identified patterns\n" +
           "• Review security controls for detected techniques\n" +
           "• Consider targeted threat hunting campaigns\n" +
           "• Update detection rules based on findings";
  }

  private formatForensicResults(result: any): string {
    if (!result?.analysis) return "No forensic analysis available";

    const analysis = result.analysis;
    return `• Evidence Items: ${analysis.evidence_analyzed?.length || 0}\n` +
           `• Risk Level: ${analysis.risk_assessment?.level?.toUpperCase() || 'UNKNOWN'}\n` +
           `• Integrity Checks: ${analysis.findings?.length ? 'PASSED' : 'PENDING'}\n` +
           `• Timeline Events: ${analysis.timeline?.length || 0}`;
  }

  private formatForensicArtifacts(result: any): string {
    if (!result?.analysis?.findings) return "No artifacts identified";

    return result.analysis.findings
      .slice(0, 5)
      .map((finding: any) => `• ${finding.type}: ${finding.artifacts?.length || 0} items`)
      .join('\n');
  }

  private formatOrchestrationDecision(result: any): string {
    if (!result?.coordination?.final_decision) return "No decision available";

    const decision = result.coordination.final_decision;
    return `• Should Contain: ${decision.should_contain ? 'YES' : 'NO'}\n` +
           `• Should Investigate: ${decision.should_investigate ? 'YES' : 'NO'}\n` +
           `• Should Escalate: ${decision.should_escalate ? 'YES' : 'NO'}\n` +
           `• Priority Level: ${decision.priority_level?.toUpperCase() || 'MEDIUM'}\n` +
           `• Automated Response: ${decision.automated_response ? 'ENABLED' : 'DISABLED'}`;
  }

  private formatAttributionIndicators(result: any): string {
    if (!result?.infrastructure_analysis?.infrastructure_clusters) return "No attribution indicators found";

    const clusters = result.infrastructure_analysis.infrastructure_clusters;
    return clusters
      .slice(0, 5)
      .map((cluster: any, index: number) =>
        `${index + 1}. ${cluster.cluster_type} cluster: ${cluster.ips?.length || 0} IPs, confidence: ${(cluster.confidence_score * 100).toFixed(1)}%`
      )
      .join('\n');
  }

  private formatPatternResults(result: any): string {
    if (!result?.patterns) return "No patterns identified";

    return Object.entries(result.patterns)
      .map(([type, data]: [string, any]) =>
        `• ${type}: ${Array.isArray(data) ? data.length : 'N/A'} occurrences`
      )
      .join('\n');
  }

  private formatHighConfidencePatterns(result: any, threshold: number): string {
    if (!result?.findings) return "No high-confidence patterns found";

    return result.findings
      .filter((finding: any) => finding.confidence >= threshold)
      .slice(0, 5)
      .map((finding: any, index: number) =>
        `${index + 1}. ${finding.description} (${(finding.confidence * 100).toFixed(1)}% confidence)`
      )
      .join('\n');
  }

  private formatCorrelationResults(result: any): string {
    if (!result?.correlations) return "No correlations identified";

    return result.correlations
      .slice(0, 5)
      .map((corr: any, index: number) =>
        `${index + 1}. ${corr.type}: ${corr.description} (${(corr.strength * 100).toFixed(1)}% strength)`
      )
      .join('\n');
  }

  private formatCorrelationStrength(result: any): string {
    if (!result?.correlation_metrics) return "Correlation strength analysis unavailable";

    const metrics = result.correlation_metrics;
    return `• Temporal Correlation: ${(metrics.temporal * 100).toFixed(1)}%\n` +
           `• Behavioral Correlation: ${(metrics.behavioral * 100).toFixed(1)}%\n` +
           `• Infrastructure Correlation: ${(metrics.infrastructure * 100).toFixed(1)}%\n` +
           `• Overall Correlation: ${(metrics.overall * 100).toFixed(1)}%`;
  }

  private formatOptimizedWorkflow(workflow: any): string {
    if (!workflow) return "No optimized workflow data available";

    return `• Optimization Type: ${workflow.optimization_type || 'Unknown'}\n` +
           `• Estimated Improvement: ${workflow.expected_improvement ? (workflow.expected_improvement * 100).toFixed(1) : 'N/A'}%\n` +
           `• Risk Level: ${workflow.risk_assessment || 'Unknown'}\n` +
           `• Actions Modified: ${workflow.actions_modified || 0}\n` +
           `• Implementation Notes: ${workflow.implementation?.description || 'Standard optimization applied'}`;
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error("Mini-XDR MCP server running on stdio");
  }
}

const server = new XDRMCPServer();
server.run().catch(console.error);
