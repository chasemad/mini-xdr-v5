"use client";

import { useState, useEffect } from "react";
import { Shield, RefreshCw, Undo2, AlertTriangle, CheckCircle, Clock, User, HardDrive, Lock } from "lucide-react";

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

interface AgentActionsPanelProps {
  incidentId: number;
  onActionClick?: (action: AgentAction) => void;
}

const AGENT_CONFIG = {
  iam: {
    icon: User,
    color: "blue",
    label: "IAM",
    description: "Identity & Access Management"
  },
  edr: {
    icon: HardDrive,
    color: "purple",
    label: "EDR",
    description: "Endpoint Detection & Response"
  },
  dlp: {
    icon: Lock,
    color: "green",
    label: "DLP",
    description: "Data Loss Prevention"
  }
};

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

export default function AgentActionsPanel({ incidentId, onActionClick }: AgentActionsPanelProps) {
  const [actions, setActions] = useState<AgentAction[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [rollingBack, setRollingBack] = useState<string | null>(null);

  const fetchActions = async () => {
    try {
      const response = await fetch(`http://localhost:8000/api/agents/actions/${incidentId}`);
      if (response.ok) {
        const data = await response.json();
        setActions(data);
      }
    } catch (error) {
      console.error("Failed to fetch agent actions:", error);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchActions();
    // Auto-refresh every 5 seconds
    const interval = setInterval(fetchActions, 5000);
    return () => clearInterval(interval);
  }, [incidentId]);

  const handleRefresh = () => {
    setRefreshing(true);
    fetchActions();
  };

  const handleRollback = async (action: AgentAction) => {
    if (!action.rollback_id || action.rollback_executed) return;
    
    if (!confirm(`Are you sure you want to rollback "${ACTION_NAME_MAP[action.action_name] || action.action_name}"?\n\nThis will restore the previous state.`)) {
      return;
    }

    setRollingBack(action.rollback_id);
    
    try {
      const response = await fetch(`http://localhost:8000/api/agents/rollback/${action.rollback_id}`, {
        method: "POST"
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log("Rollback result:", result);
        // Refresh to show updated status
        await fetchActions();
      } else {
        const error = await response.json();
        alert(`Rollback failed: ${error.detail || "Unknown error"}`);
      }
    } catch (error) {
      console.error("Rollback failed:", error);
      alert("Rollback failed: Network error");
    } finally {
      setRollingBack(null);
    }
  };

  const formatTimeAgo = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    
    if (diffMins < 1) return "just now";
    if (diffMins < 60) return `${diffMins}m ago`;
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${Math.floor(diffHours / 24)}d ago`;
  };

  if (loading) {
    return (
      <div className="bg-gray-800/30 border border-gray-700/50 rounded-xl p-8 text-center">
        <RefreshCw className="w-8 h-8 text-gray-400 mx-auto mb-2 animate-spin" />
        <div className="text-gray-400 text-sm">Loading agent actions...</div>
      </div>
    );
  }

  if (actions.length === 0) {
    return (
      <div className="bg-gray-800/30 border border-gray-700/50 rounded-xl p-6 text-center">
        <Shield className="w-12 h-12 text-gray-600 mx-auto mb-3" />
        <div className="text-gray-400 text-sm mb-2">No agent actions yet</div>
        <div className="text-gray-500 text-xs">
          Agent actions (IAM, EDR, DLP) will appear here when executed
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800/30 border border-gray-700/50 rounded-xl">
      {/* Header */}
      <div className="p-4 border-b border-gray-700/50 flex items-center justify-between">
        <h3 className="text-base font-semibold text-white flex items-center gap-2">
          <Shield className="w-5 h-5 text-blue-400" />
          Agent Actions
          <span className="text-xs text-gray-400 font-normal">
            ({actions.length} total)
          </span>
        </h3>
        <button
          onClick={handleRefresh}
          disabled={refreshing}
          className="flex items-center gap-2 px-3 py-1.5 text-xs bg-blue-600/20 hover:bg-blue-600/30 border border-blue-500/30 rounded-lg transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? "animate-spin" : ""}`} />
          Refresh
        </button>
      </div>

      {/* Actions List */}
      <div className="p-4 space-y-3 max-h-96 overflow-y-auto">
        {actions.map((action) => {
          const agentConfig = AGENT_CONFIG[action.agent_type];
          const AgentIcon = agentConfig.icon;
          const canRollback = action.rollback_id && !action.rollback_executed && action.status !== "failed";
          const isRollingBack = rollingBack === action.rollback_id;

          return (
            <div
              key={action.id}
              className="bg-gray-900/50 border border-gray-700/50 rounded-lg p-4 hover:border-gray-600/50 transition-colors cursor-pointer"
              onClick={() => onActionClick?.(action)}
            >
              <div className="flex items-start justify-between gap-4">
                <div className="flex items-start gap-3 flex-1">
                  {/* Agent Icon */}
                  <div className={`p-2 rounded-lg bg-${agentConfig.color}-500/20 border border-${agentConfig.color}-500/30`}>
                    <AgentIcon className={`w-5 h-5 text-${agentConfig.color}-400`} />
                  </div>

                  {/* Action Details */}
                  <div className="flex-1 space-y-2">
                    {/* Title Row */}
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-sm font-medium text-white">
                        {ACTION_NAME_MAP[action.action_name] || action.action_name}
                      </span>
                      
                      {/* Status Badge */}
                      {action.status === "success" && (
                        <span className="text-xs px-2 py-0.5 rounded-full bg-green-500/20 text-green-300 flex items-center gap-1">
                          <CheckCircle className="w-3 h-3" />
                          Success
                        </span>
                      )}
                      {action.status === "failed" && (
                        <span className="text-xs px-2 py-0.5 rounded-full bg-red-500/20 text-red-300 flex items-center gap-1">
                          <AlertTriangle className="w-3 h-3" />
                          Failed
                        </span>
                      )}
                      {action.status === "rolled_back" && (
                        <span className="text-xs px-2 py-0.5 rounded-full bg-orange-500/20 text-orange-300 flex items-center gap-1">
                          <Undo2 className="w-3 h-3" />
                          Rolled Back
                        </span>
                      )}

                      {/* Agent Badge */}
                      <span className={`text-[10px] uppercase tracking-wide px-2 py-0.5 rounded bg-${agentConfig.color}-500/20 text-${agentConfig.color}-300 border border-${agentConfig.color}-500/30`}>
                        {agentConfig.label}
                      </span>
                    </div>

                    {/* Parameters */}
                    {action.params && Object.keys(action.params).length > 0 && (
                      <div className="text-xs text-gray-400">
                        {Object.entries(action.params).slice(0, 3).map(([key, value]) => (
                          <span key={key} className="mr-3">
                            <span className="text-gray-500">{key}:</span>{" "}
                            <span className="font-mono text-gray-300">
                              {typeof value === "string" ? value : JSON.stringify(value)}
                            </span>
                          </span>
                        ))}
                      </div>
                    )}

                    {/* Error Message */}
                    {action.error && (
                      <div className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded px-2 py-1">
                        ⚠️ {action.error}
                      </div>
                    )}

                    {/* Rollback Info */}
                    {action.rollback_executed && action.rollback_timestamp && (
                      <div className="text-xs text-orange-400 bg-orange-500/10 border border-orange-500/20 rounded px-2 py-1">
                        🔄 Rolled back {formatTimeAgo(action.rollback_timestamp)}
                      </div>
                    )}

                    {/* Rollback Button */}
                    {canRollback && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleRollback(action);
                        }}
                        disabled={isRollingBack}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-orange-500/20 hover:bg-orange-500/30 border border-orange-500/30 text-orange-300 rounded-lg transition-colors disabled:opacity-50"
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
                  </div>
                </div>

                {/* Timestamp */}
                <div className="text-right text-xs text-gray-500 whitespace-nowrap flex flex-col items-end gap-1">
                  <Clock className="w-3.5 h-3.5" />
                  <div>{formatTimeAgo(action.executed_at)}</div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

