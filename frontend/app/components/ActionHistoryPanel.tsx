"use client";

import { useMemo, useState, useEffect } from "react";
import { Shield, Clock, RefreshCw, User, HardDrive, Lock, Undo2 } from "lucide-react";
import { apiUrl, getApiKey } from "@/app/utils/api";

// Helper to get auth headers with JWT token
const getAuthHeaders = (): Record<string, string> => {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "x-api-key": getApiKey(),
  };

  if (typeof window !== 'undefined') {
    const token = localStorage.getItem('access_token');
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }
  }

  return headers;
};

interface Action {
  id: number;
  created_at: string;
  action: string;
  result: string;
  detail: string;
  params?: Record<string, unknown>;
  verified_on_tpot?: boolean;
  tpot_verification_details?: {
    verified: boolean;
    message: string;
    timestamp: string;
  };
}

interface AutomatedAction {
  id: number;
  action_id: string;
  workflow_db_id?: number;
  workflow_id?: string;
  workflow_name?: string;
  action_type: string;
  action_name: string;
  status: string;
  executed_by?: string;
  execution_method?: string;
  parameters?: Record<string, unknown>;
  result_data?: Record<string, unknown>;
  error_details?: Record<string, unknown>;
  created_at: string;
  completed_at?: string;
  rollback?: {
    action_type: string;
    label: string;
  } | null;
}

interface AgentAction {
  id: number;
  action_id: string;
  agent_id: string;
  agent_type: "iam" | "edr" | "dlp";
  action_name: string;
  incident_id: number;
  params: Record<string, any>;
  result: Record<string, any> | null;
  status: "success" | "failed" | "rolled_back";
  error: string | null;
  rollback_id: string | null;
  rollback_executed: boolean;
  rollback_timestamp: string | null;
  executed_at: string;
  created_at: string;
}

type ActionSource = "manual" | "workflow" | "agent";

interface RollbackMeta {
  actionType: string;
  label: string;
  source: ActionSource;
  originalId: number | string;
  rollbackId?: string;
}

interface UnifiedAction {
  id: string;
  originalId: number | string;
  createdAt: string;
  actionKey: string;
  displayName: string;
  source: ActionSource;
  sourceLabel: string;
  icon: string;
  status: string;
  detail?: string;
  params?: Record<string, unknown>;
  workflowName?: string;
  workflowId?: string;
  executedBy?: string;
  executionMethod?: string;
  verifiedOnTpot?: boolean;
  verificationDetails?: Action["tpot_verification_details"];
  rollback?: {
    action_type: string;
    label: string;
  };
  completedAt?: string;
  errorDetails?: Record<string, unknown>;
  // Agent-specific fields
  agentType?: "iam" | "edr" | "dlp";
  rollbackId?: string;
  rollbackExecuted?: boolean;
  rollbackTimestamp?: string;
  error?: string;
}

interface ActionHistoryPanelProps {
  incidentId: number;
  actions?: Action[];
  automatedActions?: AutomatedAction[];
  onRefresh?: () => void;
  onRollback?: (meta: RollbackMeta) => void;
  onActionClick?: (action: any) => void;
}

const statusColorMap: Record<string, string> = {
  success: "text-green-400 bg-green-500/20",
  completed: "text-green-400 bg-green-500/20",
  failed: "text-red-400 bg-red-500/20",
  error: "text-red-400 bg-red-500/20",
  pending: "text-yellow-400 bg-yellow-500/20",
  running: "text-blue-400 bg-blue-500/20",
  in_progress: "text-blue-400 bg-blue-500/20",
  awaiting_approval: "text-purple-300 bg-purple-500/20",
  cancelled: "text-orange-300 bg-orange-500/20",
  rolled_back: "text-orange-300 bg-orange-500/20",
};

const toTitle = (value: string) => value.replace(/_/g, " ");

const ACTION_NAME_MAP: Record<string, string> = {
  // IAM
  disable_user_account: "Disable User Account",
  quarantine_user: "Quarantine User",
  revoke_kerberos_tickets: "Revoke Kerberos Tickets",
  reset_password: "Reset Password",
  remove_from_group: "Remove from Group",
  enforce_mfa: "Enforce MFA",

  // EDR
  kill_process: "Kill Process",
  quarantine_file: "Quarantine File",
  collect_memory_dump: "Collect Memory Dump",
  isolate_host: "Isolate Host",
  delete_registry_key: "Delete Registry Key",
  disable_scheduled_task: "Disable Scheduled Task",

  // DLP
  scan_file: "Scan File",
  block_upload: "Block Upload",
  quarantine_sensitive_file: "Quarantine Sensitive File"
};

const getActionIcon = (action: string, agentType?: string) => {
  // Agent-specific icons
  if (agentType === "iam") return "👤";
  if (agentType === "edr") return "🖥️";
  if (agentType === "dlp") return "🔒";

  switch (action) {
    case "block":
    case "block_ip":
    case "block_ip_advanced":
      return "🛡️";
    case "unblock":
    case "unblock_ip":
      return "✅";
    case "isolate_host":
    case "isolate_host_advanced":
      return "🔒";
    case "un_isolate_host":
      return "🟢";
    case "notify":
    case "send_notification":
      return "📧";
    case "reset_passwords":
      return "🔑";
    case "deploy_firewall":
    case "deploy_waf_rules":
      return "🔥";
    case "invoke_ai_agent":
      return "🤖";
    case "create_incident":
      return "🗂️";
    default:
      return "⚡";
  }
};

const getStatusColor = (status: string) => {
  const normalized = (status || "").toLowerCase();
  return statusColorMap[normalized] || "text-gray-300 bg-gray-600/30";
};

const formatTimeAgo = (timestamp: string) => {
  if (!timestamp) return "";
  const date = new Date(timestamp);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);

  if (Number.isNaN(diffMins)) return "";
  if (diffMins < 1) return "just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours}h ago`;
  return `${Math.floor(diffHours / 24)}d ago`;
};

const extractDetail = (action: AutomatedAction): string | undefined => {
  if (!action.result_data) return undefined;
  const { result_data: result } = action;
  const candidateKeys = ["detail", "message", "summary", "reason"];
  for (const key of candidateKeys) {
    const value = (result as Record<string, unknown>)[key];
    if (typeof value === "string" && value.trim().length > 0) {
      return value;
    }
  }
  return undefined;
};

const formatValue = (value: unknown): string => {
  if (value === null || value === undefined) return "";
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
};

export default function ActionHistoryPanel({
  incidentId,
  actions = [],
  automatedActions = [],
  onRefresh,
  onRollback,
  onActionClick,
}: ActionHistoryPanelProps) {
  const [verifying, setVerifying] = useState(false);
  const [agentActions, setAgentActions] = useState<AgentAction[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const [rollingBackAgent, setRollingBackAgent] = useState<string | null>(null);
  const hasManualActions = actions.length > 0;

  // Fetch agent actions with smart change detection
  useEffect(() => {
    let previousDataSnapshot: string | null = null;

    const fetchAgentActions = async () => {
      try {
        const response = await fetch(apiUrl(`/api/agents/actions/${incidentId}`), {
          headers: getAuthHeaders()
        });
        if (response.ok) {
          const data = await response.json();

          // Only update state if data actually changed
          const currentSnapshot = JSON.stringify(data);
          if (currentSnapshot !== previousDataSnapshot) {
            previousDataSnapshot = currentSnapshot;
            setAgentActions(data);
          }
        }
      } catch (error) {
        console.error("Failed to fetch agent actions:", error);
      }
    };

    fetchAgentActions();
    // Auto-refresh every 10 seconds (reduced frequency, smart updates only)
    const interval = setInterval(fetchAgentActions, 10000);
    return () => clearInterval(interval);
  }, [incidentId]);

  const mergedActions = useMemo<UnifiedAction[]>(() => {
    const manualItems: UnifiedAction[] = actions.map((action) => ({
      id: `manual-${action.id}`,
      originalId: action.id,
      createdAt: action.created_at,
      actionKey: action.action,
      displayName: toTitle(action.action),
      source: "manual" as ActionSource,
      sourceLabel: "Manual Quick Action",
      icon: getActionIcon(action.action),
      status: action.result,
      detail: action.detail,
      params: action.params,
      verifiedOnTpot: action.verified_on_tpot,
      verificationDetails: action.tpot_verification_details ?? undefined,
    }));

    const workflowItems: UnifiedAction[] = automatedActions.map((action) => ({
      id: `workflow-${action.id}`,
      originalId: action.id,
      createdAt: action.created_at,
      actionKey: action.action_type,
      displayName: action.action_name || toTitle(action.action_type),
      source: "workflow" as ActionSource,
      sourceLabel: action.workflow_name ? `Workflow · ${action.workflow_name}` : "Automated Workflow",
      icon: getActionIcon(action.action_type),
      status: action.status,
      detail: extractDetail(action),
      params: action.parameters,
      workflowName: action.workflow_name,
      workflowId: action.workflow_id,
      executedBy: action.executed_by,
      executionMethod: action.execution_method,
      rollback: action.rollback ?? undefined,
      completedAt: action.completed_at,
      errorDetails: action.error_details ?? undefined,
    }));

    const agentItems: UnifiedAction[] = agentActions.map((action) => ({
      id: `agent-${action.id}`,
      originalId: action.id,
      createdAt: action.executed_at || action.created_at,
      actionKey: action.action_name,
      displayName: ACTION_NAME_MAP[action.action_name] || toTitle(action.action_name),
      source: "agent" as ActionSource,
      sourceLabel: `${action.agent_type.toUpperCase()} Agent`,
      icon: getActionIcon(action.action_name, action.agent_type),
      status: action.status,
      detail: action.result ? JSON.stringify(action.result) : undefined,
      params: action.params,
      agentType: action.agent_type,
      rollbackId: action.rollback_id ?? undefined,
      rollbackExecuted: action.rollback_executed,
      rollbackTimestamp: action.rollback_timestamp ?? undefined,
      error: action.error ?? undefined,
      errorDetails: action.error ? { error: action.error } : undefined,
    }));

    return [...manualItems, ...workflowItems, ...agentItems].sort((a, b) => {
      const aTime = new Date(a.createdAt).getTime();
      const bTime = new Date(b.createdAt).getTime();
      return bTime - aTime;
    });
  }, [actions, automatedActions, agentActions]);

  const verifyActions = async () => {
    if (!hasManualActions) return;
    setVerifying(true);
    try {
      const response = await fetch(apiUrl(`/api/incidents/${incidentId}/verify-actions`), {
        method: "POST",
        headers: getAuthHeaders(),
      });

      if (response.ok) {
        const data = await response.json();
        console.log("Verification results:", data);
        onRefresh?.();
      }
    } catch (error) {
      console.error("Verification failed:", error);
    } finally {
      setVerifying(false);
    }
  };

  const handleAgentRollback = async (action: UnifiedAction) => {
    if (!action.rollbackId || action.rollbackExecuted) return;

    if (!confirm(`Are you sure you want to rollback "${action.displayName}"?\n\nThis will restore the previous state.`)) {
      return;
    }

    setRollingBackAgent(action.rollbackId);

    try {
      const response = await fetch(apiUrl(`/api/agents/rollback/${action.rollbackId}`), {
        method: "POST",
        headers: getAuthHeaders()
      });

      if (response.ok) {
        const result = await response.json();
        console.log("Rollback result:", result);
        // Refresh agent actions
        const agentResponse = await fetch(apiUrl(`/api/agents/actions/${incidentId}`), {
          headers: getAuthHeaders()
        });
        if (agentResponse.ok) {
          const data = await agentResponse.json();
          setAgentActions(data);
        }
        onRefresh?.();
      } else {
        const error = await response.json();
        alert(`Rollback failed: ${error.detail || "Unknown error"}`);
      }
    } catch (error) {
      console.error("Rollback failed:", error);
      alert("Rollback failed: Network error");
    } finally {
      setRollingBackAgent(null);
    }
  };

  const handleRefresh = () => {
    setRefreshing(true);
    onRefresh?.();
    // Agent actions will be refreshed by useEffect
    setTimeout(() => setRefreshing(false), 1000);
  };

  if (mergedActions.length === 0) {
    return (
      <div className="bg-gray-800/30 border border-gray-700/50 rounded-xl p-6 text-center">
        <div className="text-gray-400 text-sm">
          <Clock className="w-8 h-8 mx-auto mb-2 opacity-50" />
          No response actions have been recorded yet
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800/30 border border-gray-700/50 rounded-xl">
      <div className="p-4 border-b border-gray-700/50 flex items-center justify-between">
        <h3 className="text-base font-semibold text-white flex items-center gap-2">
          <Shield className="w-5 h-5 text-blue-400" />
          Unified Response Actions
          <span className="text-xs text-gray-400 font-normal">
            ({mergedActions.length} total • {actions.length} manual • {automatedActions.length} workflow • {agentActions.length} agent)
          </span>
        </h3>
        <div className="flex items-center gap-2">
          <button
            onClick={handleRefresh}
            disabled={refreshing}
            className="flex items-center gap-2 px-3 py-1.5 text-xs bg-blue-600/20 hover:bg-blue-600/30 border border-blue-500/30 rounded-lg transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? "animate-spin" : ""}`} />
            Refresh
          </button>
          <button
            onClick={verifyActions}
            disabled={verifying || !hasManualActions}
            className="flex items-center gap-2 px-3 py-1.5 text-xs bg-green-600/20 hover:bg-green-600/30 border border-green-500/30 rounded-lg transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${verifying ? "animate-spin" : ""}`} />
            Verify
          </button>
        </div>
      </div>

      <div className="p-4 space-y-2 max-h-96 overflow-y-auto">
        {mergedActions.map((item) => {
          const isAgent = item.source === "agent";
          const canRollbackAgent = isAgent && item.rollbackId && !item.rollbackExecuted && item.status !== "failed";
          const isRollingBack = rollingBackAgent === item.rollbackId;

          // Agent color mapping
          const agentColors = {
            iam: { badge: "bg-blue-500/20 text-blue-300 border-blue-500/30", icon: "text-blue-400" },
            edr: { badge: "bg-purple-500/20 text-purple-300 border-purple-500/30", icon: "text-purple-400" },
            dlp: { badge: "bg-green-500/20 text-green-300 border-green-500/30", icon: "text-green-400" },
          };
          const agentColor = item.agentType ? agentColors[item.agentType] : null;

          return (
            <div
              key={item.id}
              onClick={() => onActionClick && onActionClick(item)}
              className="bg-gray-900/50 border border-gray-700/50 rounded-lg p-3 hover:border-gray-600/50 transition-colors cursor-pointer"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-start gap-3 flex-1">
                  <div className="text-2xl leading-none">{item.icon}</div>
                  <div className="flex-1 min-w-0 space-y-1">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-sm font-medium text-white capitalize">
                        {item.displayName}
                      </span>
                      <span className={`text-xs px-2 py-0.5 rounded-full ${getStatusColor(item.status)}`}>
                        {toTitle(item.status)}
                      </span>
                      {isAgent && agentColor ? (
                        <span className={`text-[10px] uppercase tracking-wide px-2 py-0.5 rounded border ${agentColor.badge}`}>
                          {item.sourceLabel}
                        </span>
                      ) : (
                        <span className="text-[10px] uppercase tracking-wide text-gray-300 bg-gray-700/50 px-2 py-0.5 rounded">
                          {item.sourceLabel}
                        </span>
                      )}
                      {item.verifiedOnTpot && (
                        <span className="text-[10px] uppercase px-2 py-0.5 rounded bg-green-500/20 text-green-400 border border-green-500/30">
                          Verified on T-Pot
                        </span>
                      )}
                    </div>

                    {item.workflowName && (
                      <div className="text-xs text-gray-400">
                        Workflow: <span className="text-gray-200">{item.workflowName}</span>
                        {item.workflowId ? ` (${item.workflowId})` : ""}
                      </div>
                    )}

                    {item.executedBy && (
                      <div className="text-xs text-gray-500">
                        Executed by {item.executedBy}
                        {item.executionMethod ? ` · ${toTitle(item.executionMethod)}` : ""}
                      </div>
                    )}

                    {item.params && Object.keys(item.params).length > 0 && (
                      <div className="text-xs text-gray-400">
                        {Object.entries(item.params).map(([key, value]) => (
                          <span key={key} className="mr-3">
                            <span className="text-gray-500">{key}:</span>{" "}
                            <span className="font-mono">{formatValue(value)}</span>
                          </span>
                        ))}
                      </div>
                    )}

                    {item.detail && (
                      <div className="text-xs text-gray-500 line-clamp-3">
                        {item.detail}
                      </div>
                    )}

                    {!item.detail && item.errorDetails && (
                      <div className="text-xs text-red-400 line-clamp-3">
                        ⚠️ {formatValue(item.errorDetails)}
                      </div>
                    )}

                    {item.verificationDetails && (
                      <div
                        className={`text-xs mt-1 px-2 py-1 rounded ${item.verificationDetails.verified
                            ? "bg-green-500/10 text-green-400"
                            : "bg-red-500/10 text-red-400"
                          }`}
                      >
                        {item.verificationDetails.verified ? "✓" : "✗"} {item.verificationDetails.message}
                      </div>
                    )}

                    {/* Rollback info for already rolled back agent actions */}
                    {isAgent && item.rollbackExecuted && item.rollbackTimestamp && (
                      <div className="text-xs text-orange-400 bg-orange-500/10 border border-orange-500/20 rounded px-2 py-1 flex items-center gap-1">
                        <Undo2 className="w-3 h-3" />
                        Rolled back {formatTimeAgo(item.rollbackTimestamp)}
                      </div>
                    )}

                    {/* Rollback button for agent actions */}
                    {canRollbackAgent && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleAgentRollback(item);
                        }}
                        disabled={isRollingBack}
                        className="mt-2 inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-orange-500/20 hover:bg-orange-500/30 border border-orange-500/30 text-orange-300 rounded-lg transition-colors disabled:opacity-50"
                      >
                        {isRollingBack ? (
                          <>
                            <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                            Rolling back...
                          </>
                        ) : (
                          <>
                            <Undo2 className="w-3.5 h-3.5" />
                            Rollback Action
                          </>
                        )}
                      </button>
                    )}

                    {/* Workflow rollback button */}
                    {item.rollback && item.source === "workflow" && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          if (item.rollback) {
                            onRollback?.({
                              actionType: item.rollback.action_type,
                              label: item.rollback.label,
                              source: item.source,
                              originalId: item.originalId,
                            });
                          }
                        }}
                        className="mt-2 inline-flex items-center gap-1 text-xs text-blue-300 hover:text-blue-200"
                      >
                        ⟲ {item.rollback?.label || 'Rollback'}
                      </button>
                    )}
                  </div>
                </div>

                <div className="text-right text-xs text-gray-500 whitespace-nowrap">
                  <div>{formatTimeAgo(item.createdAt)}</div>
                  {item.completedAt && (
                    <div className="text-[10px] text-gray-600">Completed {formatTimeAgo(item.completedAt)}</div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
