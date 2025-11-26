import { apiUrl, createFetchOptions } from "@/app/utils/api";

export interface CoordinationHubStatus {
  active_agents: number;
  total_decisions_today: number;
  pending_coordinations: number;
  active_incidents: number;
  system_health: string;
  last_updated: string;
  performance_metrics: Record<string, any>;
}

export interface AIAgentStatus {
  agent_name: string;
  status: string;
  decisions_count: number;
  performance_metrics: Record<string, any>;
  last_active_timestamp: string;
}

export interface AgentDecision {
  id: string;
  agent_name: string;
  decision_type: string;
  timestamp: string;
  confidence: number;
  details: Record<string, any>;
  incident_id?: number;
}

export interface IncidentCoordination {
  incident_id: number;
  coordination_status: string;
  participating_agents: string[];
  agent_decisions: Record<string, any>;
  coordination_timeline: Array<{
    timestamp: string;
    event: string;
    details: string;
    verdict?: string;
    agents?: string[];
  }>;
  recommendations: string[];
}

export const agentApi = {
  getHubStatus: async (): Promise<CoordinationHubStatus> => {
    const response = await fetch(
      apiUrl('/api/agents/coordination-hub/status'),
      createFetchOptions()
    );
    if (!response.ok) throw new Error('Failed to fetch coordination hub status');
    return response.json();
  },

  getAgentStatus: async (agentName: string): Promise<AIAgentStatus> => {
    const response = await fetch(
      apiUrl(`/api/agents/ai/${agentName}/status`),
      createFetchOptions()
    );
    if (!response.ok) throw new Error(`Failed to fetch status for ${agentName}`);
    return response.json();
  },

  getAgentDecisions: async (agentName: string, limit: number = 50): Promise<AgentDecision[]> => {
    const response = await fetch(
      apiUrl(`/api/agents/ai/${agentName}/decisions?limit=${limit}`),
      createFetchOptions()
    );
    if (!response.ok) throw new Error(`Failed to fetch decisions for ${agentName}`);
    return response.json();
  },

  getIncidentCoordination: async (incidentId: number): Promise<IncidentCoordination> => {
    const response = await fetch(
      apiUrl(`/api/agents/incidents/${incidentId}/coordination`),
      createFetchOptions()
    );
    if (!response.ok) throw new Error('Failed to fetch incident coordination');
    return response.json();
  }
};
