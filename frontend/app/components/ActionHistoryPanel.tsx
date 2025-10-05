"use client";

import { useMemo, useState } from "react";
import { Shield, Clock, RefreshCw } from "lucide-react";

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

type ActionSource = "manual" | "workflow";

interface RollbackMeta {
  actionType: string;
  label: string;
  source: ActionSource;
  originalId: number | string;
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
}

interface ActionHistoryPanelProps {
  incidentId: number;
  actions?: Action[];
  automatedActions?: AutomatedAction[];
  onRefresh?: () => void;
  onRollback?: (meta: RollbackMeta) => void;
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

const getActionIcon = (action: string) => {
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
}: ActionHistoryPanelProps) {
  const [verifying, setVerifying] = useState(false);
  const hasManualActions = actions.length > 0;

  const mergedActions = useMemo<UnifiedAction[]>(() => {
    const manualItems: UnifiedAction[] = actions.map((action) => ({
      id: `manual-${action.id}`,
      originalId: action.id,
      createdAt: action.created_at,
      actionKey: action.action,
      displayName: toTitle(action.action),
      source: "manual",
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
      source: "workflow",
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

    return [...manualItems, ...workflowItems].sort((a, b) => {
      const aTime = new Date(a.createdAt).getTime();
      const bTime = new Date(b.createdAt).getTime();
      return bTime - aTime;
    });
  }, [actions, automatedActions]);

  const verifyActions = async () => {
    if (!hasManualActions) return;
    setVerifying(true);
    try {
      const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "";
      const response = await fetch(`http://localhost:8000/api/incidents/${incidentId}/verify-actions`, {
        method: "POST",
        headers: {
          "x-api-key": API_KEY,
        },
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
          Action History
          <span className="text-xs text-gray-400 font-normal">
            ({mergedActions.length} total • {actions.length} manual / {automatedActions.length} automated)
          </span>
        </h3>
        <button
          onClick={verifyActions}
          disabled={verifying || !hasManualActions}
          className="flex items-center gap-2 px-3 py-1.5 text-xs bg-blue-600/20 hover:bg-blue-600/30 border border-blue-500/30 rounded-lg transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${verifying ? "animate-spin" : ""}`} />
          Verify Manual Actions
        </button>
      </div>

      <div className="p-4 space-y-2 max-h-96 overflow-y-auto">
        {mergedActions.map((item) => (
          <div
            key={item.id}
            className="bg-gray-900/50 border border-gray-700/50 rounded-lg p-3 hover:border-gray-600/50 transition-colors"
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
                    <span className="text-[10px] uppercase tracking-wide text-gray-300 bg-gray-700/50 px-2 py-0.5 rounded">
                      {item.sourceLabel}
                    </span>
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
                      className={`text-xs mt-1 px-2 py-1 rounded ${
                        item.verificationDetails.verified
                          ? "bg-green-500/10 text-green-400"
                          : "bg-red-500/10 text-red-400"
                      }`}
                    >
                      {item.verificationDetails.verified ? "✓" : "✗"} {item.verificationDetails.message}
                    </div>
                  )}

                  {item.rollback && item.source === "workflow" && (
                    <button
                      onClick={() =>
                        onRollback?.({
                          actionType: item.rollback.action_type,
                          label: item.rollback.label,
                          source: item.source,
                          originalId: item.originalId,
                        })
                      }
                      className="mt-2 inline-flex items-center gap-1 text-xs text-blue-300 hover:text-blue-200"
                    >
                      ⟲ {item.rollback.label}
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
        ))}
      </div>
    </div>
  );
}
