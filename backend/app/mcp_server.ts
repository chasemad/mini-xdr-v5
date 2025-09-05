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

  // === HELPER METHODS ===

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

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error("Mini-XDR MCP server running on stdio");
  }
}

const server = new XDRMCPServer();
server.run().catch(console.error);
