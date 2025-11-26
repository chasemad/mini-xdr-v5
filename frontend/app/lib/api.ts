import { apiUrl, getApiKey, resolveApiBaseUrl } from "@/app/utils/api";

const API_KEY = getApiKey();
const BASE_OVERRIDE = resolveApiBaseUrl();

interface RequestOptions {
  method?: string;
  headers?: Record<string, string>;
  body?: unknown;
  skipAuth?: boolean;  // Option to skip JWT auth for login/register
  suppressLogoutOn401?: boolean; // Prevent auto-logout/redirect on 401 for background probes
}

const buildRequestUrl = (endpoint: string): string => {
  if (BASE_OVERRIDE !== null) {
    return apiUrl(endpoint, BASE_OVERRIDE);
  }
  return apiUrl(endpoint);
};

async function apiRequest(endpoint: string, options: RequestOptions = {}) {
  const url = buildRequestUrl(endpoint);

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...options.headers,
  };

  // Add JWT token if available and not skipped
  if (!options.skipAuth) {
    if (typeof window !== 'undefined') {
      const token = localStorage.getItem('access_token');
      if (token) {
        headers["Authorization"] = `Bearer ${token}`;
      }
    }
  }

  if (API_KEY) {
    headers["x-api-key"] = API_KEY;
  }

  const config: RequestInit = {
    method: options.method || "GET",
    headers,
    cache: "no-store",
  };

  if (options.body) {
    config.body = JSON.stringify(options.body);
  }

  try {
    const response = await fetch(url, config);

    // Handle 401 Unauthorized - redirect to login
    if (response.status === 401 && typeof window !== 'undefined') {
      if (options.suppressLogoutOn401) {
        // Return null quietly for background/optional probes
        return null as any;
      }
      console.warn('Authentication expired or invalid, redirecting to login...');
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');
      if (window.location.pathname !== '/login') {
        window.location.href = '/login';
      }
      throw new Error('Authentication required');
    }

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`API Error: ${response.status} ${response.statusText}`, errorText);
      throw new Error(`API Error: ${response.status} ${response.statusText} - ${errorText}`);
    }

    const data = await response.json();
    console.log(`✅ API Success: ${options.method || 'GET'} ${endpoint}`, data);
    return data;
  } catch (error) {
    console.error(`❌ API Request Failed: ${options.method || 'GET'} ${url}`, error);
    throw error;
  }
}

// ===== AUTHENTICATION API FUNCTIONS =====

export async function login(email: string, password: string) {
  return apiRequest("/api/auth/login", {
    method: "POST",
    body: { email, password },
    skipAuth: true
  });
}

export async function register(data: {
  organization_name: string;
  admin_email: string;
  admin_password: string;
  admin_name: string;
}) {
  return apiRequest("/api/auth/register", {
    method: "POST",
    body: data,
    skipAuth: true
  });
}

export async function getCurrentUser() {
  return apiRequest("/api/auth/me");
}

export async function logout() {
  return apiRequest("/api/auth/logout", {
    method: "POST"
  });
}

export async function inviteUser(data: {
  email: string;
  full_name: string;
  role: string;
}) {
  return apiRequest("/api/auth/invite", {
    method: "POST",
    body: data
  });
}

export async function getIncidents() {
  return apiRequest("/api/incidents");
}

export async function getIncident(id: number) {
  return apiRequest(`/api/incidents/${id}`);
}

export async function unblockIncident(id: number) {
  return apiRequest(`/api/incidents/${id}/unblock`, { method: "POST" });
}

export async function containIncident(id: number, durationSeconds?: number) {
  const url = durationSeconds
    ? `/api/incidents/${id}/contain?duration_seconds=${durationSeconds}`
    : `/api/incidents/${id}/contain`;
  return apiRequest(url, { method: "POST" });
}

export async function scheduleUnblock(id: number, minutes: number) {
  return apiRequest(`/api/incidents/${id}/schedule_unblock?minutes=${minutes}`, { method: "POST" });
}

export async function getAutoContainSetting() {
  return apiRequest("/settings/auto_contain");
}

export async function setAutoContainSetting(enabled: boolean) {
  return apiRequest("/settings/auto_contain", {
    method: "POST",
    body: enabled,
  });
}

export async function getHealth() {
  return apiRequest("/health");
}

// ===== SOC ACTION API FUNCTIONS =====

export async function socBlockIP(incidentId: number, durationSeconds?: number) {
  const body = durationSeconds ? { duration_seconds: durationSeconds } : {};
  return apiRequest(`/api/incidents/${incidentId}/actions/block-ip`, {
    method: "POST",
    body: body
  });
}

export async function socUnblockIP(incidentId: number) {
  return apiRequest(`/api/incidents/${incidentId}/actions/unblock-ip`, { method: "POST" });
}

export async function getBlockStatus(incidentId: number) {
  return apiRequest(`/api/incidents/${incidentId}/block-status`, { suppressLogoutOn401: true });
}

export async function socIsolateHost(incidentId: number, isolationLevel?: string, durationSeconds?: number) {
  const body: { isolation_level?: string; duration_seconds?: number } = {};
  if (isolationLevel) body.isolation_level = isolationLevel;
  if (durationSeconds) body.duration_seconds = durationSeconds;

  return apiRequest(`/api/incidents/${incidentId}/actions/isolate-host`, {
    method: "POST",
    body: Object.keys(body).length > 0 ? body : undefined
  });
}

export async function socUnIsolateHost(incidentId: number) {
  return apiRequest(`/api/incidents/${incidentId}/actions/un-isolate-host`, { method: "POST" });
}

export async function getIsolationStatus(incidentId: number) {
  return apiRequest(`/api/incidents/${incidentId}/isolation-status`);
}

export async function socResetPasswords(incidentId: number) {
  return apiRequest(`/api/incidents/${incidentId}/actions/reset-passwords`, { method: "POST" });
}

export async function socCheckDBIntegrity(incidentId: number) {
  return apiRequest(`/api/incidents/${incidentId}/actions/check-db-integrity`, { method: "POST" });
}

export async function socThreatIntelLookup(incidentId: number) {
  return apiRequest(`/api/incidents/${incidentId}/actions/threat-intel-lookup`, { method: "POST" });
}

export async function socDeployWAFRules(incidentId: number) {
  return apiRequest(`/api/incidents/${incidentId}/actions/deploy-waf-rules`, { method: "POST" });
}

export async function socCaptureTraffic(incidentId: number) {
  return apiRequest(`/api/incidents/${incidentId}/actions/capture-traffic`, { method: "POST" });
}

export async function socHuntSimilarAttacks(incidentId: number) {
  return apiRequest(`/api/incidents/${incidentId}/actions/hunt-similar-attacks`, { method: "POST" });
}

export async function socAlertAnalysts(incidentId: number) {
  return apiRequest(`/api/incidents/${incidentId}/actions/alert-analysts`, { method: "POST" });
}

export async function socCreateCase(incidentId: number) {
  return apiRequest(`/api/incidents/${incidentId}/actions/create-case`, { method: "POST" });
}

// ===== ADDITIONAL SOC ACTION API FUNCTIONS =====

export async function socDnsSinkhole(incidentId: number, domains?: string[]) {
  return apiRequest(`/api/incidents/${incidentId}/actions/dns-sinkhole`, {
    method: "POST",
    body: { domains }
  });
}

export async function socTrafficRedirection(incidentId: number, destination?: string) {
  return apiRequest(`/api/incidents/${incidentId}/actions/traffic-redirection`, {
    method: "POST",
    body: { destination: destination || "honeypot" }
  });
}

export async function socNetworkSegmentation(incidentId: number, segmentType?: string) {
  return apiRequest(`/api/incidents/${incidentId}/actions/network-segmentation`, {
    method: "POST",
    body: { segment_type: segmentType || "vlan" }
  });
}

export async function socMemoryDump(incidentId: number, targetHost?: string) {
  return apiRequest(`/api/incidents/${incidentId}/actions/memory-dump`, {
    method: "POST",
    body: { target_host: targetHost }
  });
}

export async function socKillProcess(incidentId: number, processCriteria?: Record<string, any>) {
  return apiRequest(`/api/incidents/${incidentId}/actions/kill-process`, {
    method: "POST",
    body: { process_criteria: processCriteria }
  });
}

export async function socMalwareRemoval(incidentId: number, targetHost?: string) {
  return apiRequest(`/api/incidents/${incidentId}/actions/malware-removal`, {
    method: "POST",
    body: { target_host: targetHost }
  });
}

export async function socEndpointScan(incidentId: number, targetHost?: string, scanType?: string) {
  return apiRequest(`/api/incidents/${incidentId}/actions/endpoint-scan`, {
    method: "POST",
    body: { target_host: targetHost, scan_type: scanType || "full" }
  });
}

export async function socBehaviorAnalysis(incidentId: number) {
  return apiRequest(`/api/incidents/${incidentId}/actions/behavior-analysis`, { method: "POST" });
}

export async function socCollectEvidence(incidentId: number, artifactTypes?: string[]) {
  return apiRequest(`/api/incidents/${incidentId}/actions/collect-evidence`, {
    method: "POST",
    body: { artifact_types: artifactTypes || ["logs", "memory", "network"] }
  });
}

export async function socAnalyzeLogs(incidentId: number, timeRange?: string) {
  return apiRequest(`/api/incidents/${incidentId}/actions/analyze-logs`, {
    method: "POST",
    body: { time_range: timeRange || "24h" }
  });
}

export async function socRevokeUserSessions(incidentId: number, userId?: string) {
  return apiRequest(`/api/incidents/${incidentId}/actions/revoke-sessions`, {
    method: "POST",
    body: { user_id: userId }
  });
}

export async function socDisableUserAccount(incidentId: number, userId?: string) {
  return apiRequest(`/api/incidents/${incidentId}/actions/disable-account`, {
    method: "POST",
    body: { user_id: userId }
  });
}

export async function socEnforceMFA(incidentId: number, userId?: string) {
  return apiRequest(`/api/incidents/${incidentId}/actions/enforce-mfa`, {
    method: "POST",
    body: { user_id: userId }
  });
}

export async function socEmergencyBackup(incidentId: number) {
  return apiRequest(`/api/incidents/${incidentId}/actions/emergency-backup`, { method: "POST" });
}

export async function socEnableDLP(incidentId: number, policyLevel?: string) {
  return apiRequest(`/api/incidents/${incidentId}/actions/enable-dlp`, {
    method: "POST",
    body: { policy_level: policyLevel || "strict" }
  });
}

export async function socNotifyStakeholders(incidentId: number, notificationLevel?: string) {
  return apiRequest(`/api/incidents/${incidentId}/actions/notify-stakeholders`, {
    method: "POST",
    body: { notification_level: notificationLevel || "executive" }
  });
}

// Agent Orchestration API
export async function agentOrchestrate(query: string, incident_id?: number, context?: Record<string, unknown>) {
  return apiRequest("/api/agents/orchestrate", {
    method: "POST",
    body: {
      query,
      incident_id,
      context,
      agent_type: "copilot" // Use new copilot handler
    }
  });
}

// Agent Action Confirmation API
export async function confirmAgentAction(
  pending_action_id: string,
  approved: boolean,
  incident_id?: number,
  context?: Record<string, unknown>
) {
  return apiRequest("/api/agents/confirm-action", {
    method: "POST",
    body: {
      pending_action_id,
      approved,
      incident_id,
      context
    }
  });
}

// ===== ADVANCED RESPONSE API FUNCTIONS =====

export async function getAvailableResponseActions(category?: string) {
  const params = new URLSearchParams();
  // Don't send category parameter if it's 'all' - let backend return all actions
  if (category && category !== 'all') {
    params.append('category', category);
  }

  return apiRequest(`/api/response/actions${params.toString() ? '?' + params.toString() : ''}`);
}

export async function createResponseWorkflow(workflowData: {
  incident_id: number;
  playbook_name: string;
  steps: Array<{
    action_type: string;
    parameters: Record<string, any>;
    timeout_seconds?: number;
    continue_on_failure?: boolean;
    max_retries?: number;
  }>;
  auto_execute?: boolean;
  priority?: string;
}) {
  return apiRequest("/api/response/workflows/create", {
    method: "POST",
    body: workflowData
  });
}

// ===== NLP WORKFLOW API FUNCTIONS =====

export async function parseNlpWorkflow(request: {
  text: string;
  incident_id?: number | null;
  auto_execute?: boolean;
}) {
  return apiRequest("/api/workflows/nlp/parse", {
    method: "POST",
    body: {
      text: request.text,
      incident_id: request.incident_id ?? null,
      auto_execute: request.auto_execute ?? false
    }
  });
}

export async function createNlpWorkflow(request: {
  text: string;
  incident_id?: number | null;
  auto_execute?: boolean;
}) {
  return apiRequest("/api/workflows/nlp/create", {
    method: "POST",
    body: {
      text: request.text,
      incident_id: request.incident_id ?? null,
      auto_execute: request.auto_execute ?? false
    }
  });
}

export async function executeResponseWorkflow(workflowDbId: number, executedBy?: string) {
  return apiRequest("/api/response/workflows/execute", {
    method: "POST",
    body: {
      workflow_db_id: workflowDbId,
      executed_by: executedBy || "analyst"
    }
  });
}

export async function getWorkflowStatus(workflowId: string) {
  return apiRequest(`/api/response/workflows/${workflowId}/status`);
}

export async function cancelWorkflow(workflowId: string, cancelledBy?: string) {
  return apiRequest(`/api/response/workflows/${workflowId}/cancel`, {
    method: "POST",
    body: { cancelled_by: cancelledBy || "analyst" }
  });
}

export async function listResponseWorkflows(filters?: {
  incident_id?: number;
  status?: string;
  limit?: number;
}) {
  const params = new URLSearchParams();
  if (filters?.incident_id) params.append('incident_id', filters.incident_id.toString());
  if (filters?.status) params.append('status', filters.status);
  if (filters?.limit) params.append('limit', filters.limit.toString());

  return apiRequest(`/api/response/workflows${params.toString() ? '?' + params.toString() : ''}`);
}

export async function getWorkflowActions(workflowId: string) {
  return apiRequest(`/api/response/workflows/${workflowId}/actions`);
}

export async function getResponseImpactMetrics(filters?: {
  workflow_id?: string;
  incident_id?: number;
  days_back?: number;
}) {
  const params = new URLSearchParams();
  if (filters?.workflow_id) params.append('workflow_id', filters.workflow_id);
  if (filters?.incident_id) params.append('incident_id', filters.incident_id.toString());
  if (filters?.days_back) params.append('days_back', filters.days_back.toString());

  return apiRequest(`/api/response/metrics/impact${params.toString() ? '?' + params.toString() : ''}`);
}

export async function executeSingleResponseAction(actionData: {
  action_type: string;
  incident_id: number;
  parameters: Record<string, any>;
}) {
  return apiRequest("/api/response/actions/execute", {
    method: "POST",
    body: actionData
  });
}

// Test endpoint for advanced response system
export async function testAdvancedResponseSystem() {
  return apiRequest("/api/response/test");
}

// ===== VISUAL WORKFLOW API FUNCTIONS =====

export async function getPlaybookTemplates(category?: string) {
  const params = new URLSearchParams();
  if (category && category !== 'all') {
    params.append('category', category);
  }

  return apiRequest(`/api/workflows/templates${params.toString() ? '?' + params.toString() : ''}`);
}

export async function createPlaybookTemplate(templateData: {
  name: string;
  description: string;
  category: string;
  steps: Array<{
    action_type: string;
    parameters: Record<string, any>;
  }>;
  threat_types?: string[];
  compliance_frameworks?: string[];
}) {
  return apiRequest("/api/workflows/templates/create", {
    method: "POST",
    body: templateData
  });
}

export async function validateWorkflow(workflowData: {
  steps: Array<{
    action_type: string;
    parameters: Record<string, any>;
  }>;
}) {
  return apiRequest("/api/workflows/visual/validate", {
    method: "POST",
    body: workflowData
  });
}

// ===== AI-POWERED RESPONSE API FUNCTIONS =====

export async function getAIRecommendations(incidentId: number, context?: Record<string, any>) {
  return apiRequest("/api/ai/response/recommendations", {
    method: "POST",
    body: { incident_id: incidentId, context }
  });
}

export async function analyzeIncidentContext(incidentId: number) {
  return apiRequest(`/api/ai/response/context/${incidentId}`);
}

export async function getIncidentContextForNLP(incidentId: number) {
  return apiRequest(`/api/incidents/${incidentId}/context`);
}

export async function optimizeResponseStrategy(workflowId: string) {
  return apiRequest(`/api/ai/response/optimize/${workflowId}`, {
    method: "POST"
  });
}

// ===== WORKFLOW TRIGGER API FUNCTIONS =====

export async function listWorkflowTriggers(filters?: {
  category?: string;
  enabled?: boolean;
}) {
  const params = new URLSearchParams();
  if (filters?.category) params.append('category', filters.category);
  if (filters?.enabled !== undefined) params.append('enabled', filters.enabled.toString());

  return apiRequest(`/api/triggers${params.toString() ? '?' + params.toString() : ''}`);
}

export async function getWorkflowTrigger(triggerId: number) {
  return apiRequest(`/api/triggers/${triggerId}`);
}

export async function createWorkflowTrigger(triggerData: {
  name: string;
  description?: string;
  category: string;
  enabled?: boolean;
  auto_execute?: boolean;
  priority?: string;
  conditions: Record<string, any>;
  playbook_name: string;
  workflow_steps: Array<Record<string, any>>;
  cooldown_seconds?: number;
  max_triggers_per_day?: number;
  tags?: string[];
}) {
  return apiRequest("/api/triggers", {
    method: "POST",
    body: triggerData
  });
}

export async function updateWorkflowTrigger(triggerId: number, triggerData: Partial<{
  name: string;
  description: string;
  category: string;
  enabled: boolean;
  auto_execute: boolean;
  priority: string;
  conditions: Record<string, any>;
  playbook_name: string;
  workflow_steps: Array<Record<string, any>>;
  cooldown_seconds: number;
  max_triggers_per_day: number;
  tags: string[];
}>) {
  return apiRequest(`/api/triggers/${triggerId}`, {
    method: "PUT",
    body: triggerData
  });
}

export async function deleteWorkflowTrigger(triggerId: number) {
  return apiRequest(`/api/triggers/${triggerId}`, {
    method: "DELETE"
  });
}

export async function enableWorkflowTrigger(triggerId: number) {
  return apiRequest(`/api/triggers/${triggerId}/enable`, {
    method: "POST"
  });
}

export async function disableWorkflowTrigger(triggerId: number) {
  return apiRequest(`/api/triggers/${triggerId}/disable`, {
    method: "POST"
  });
}

export async function getWorkflowTriggerStats() {
  return apiRequest("/api/triggers/stats/summary");
}
