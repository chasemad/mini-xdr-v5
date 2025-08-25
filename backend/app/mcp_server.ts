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
          {
            name: "get_incidents",
            description: "List all security incidents",
            inputSchema: {
              type: "object",
              properties: {},
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
          case "get_incidents":
            return await this.getIncidents();

          case "get_incident":
            if (!args || typeof args !== 'object' || !('id' in args)) {
              throw new Error('Missing required parameter: id');
            }
            return await this.getIncident(args.id as number);

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

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error("Mini-XDR MCP server running on stdio");
  }
}

const server = new XDRMCPServer();
server.run().catch(console.error);
