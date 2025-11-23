"use client";

import React from 'react';
import {
  X, Shield, AlertCircle, CheckCircle, XCircle, Clock,
  Code, FileText, Activity, Database, Eye, Copy
} from 'lucide-react';

interface ActionDetail {
  id: number;
  action_id?: string;
  action: string;
  action_type?: string;
  action_name?: string;
  result?: string;
  status?: string;
  detail?: string;
  params?: Record<string, any>;
  parameters?: Record<string, any>;
  result_data?: Record<string, any>;
  error_details?: Record<string, any>;
  created_at: string;
  completed_at?: string;
  executed_at?: string;
  executed_by?: string;
  execution_method?: string;
  workflow_name?: string;
  confidence_score?: number;
  tpot_verification_details?: Record<string, any>;
  // Agent action fields
  agent_id?: string;
  agent_type?: "iam" | "edr" | "dlp";
  rollback_id?: string;
  rollback_executed?: boolean;
  rollback_timestamp?: string;
  error?: string;
}

interface ActionDetailModalProps {
  action: ActionDetail | null;
  isOpen: boolean;
  onClose: () => void;
  onRollback?: (rollbackId: string) => Promise<void>;
  incidentEvents?: Array<{
    id: number;
    ts: string;
    src_ip: string;
    dst_ip?: string;
    dst_port?: number;
    eventid: string;
    message: string;
    raw: Record<string, any>;
    source_type: string;
  }>;
}

export default function ActionDetailModal({
  action,
  isOpen,
  onClose,
  onRollback,
  incidentEvents = []
}: ActionDetailModalProps) {
  if (!isOpen || !action) return null;

  const actionName = action.action_name || action.action || action.action_type || 'Action';
  const status = action.status || action.result || 'unknown';
  const isSuccess = ['completed', 'success', 'done'].includes(status.toLowerCase());
  const isFailure = ['failed', 'error'].includes(status.toLowerCase());
  const isPending = ['pending', 'running'].includes(status.toLowerCase());
  const canRollback = action.rollback_id && !action.rollback_executed && !isFailure;

  const agentTypeConfig = {
    iam: { label: 'IAM', color: 'blue-400', bgColor: 'blue-500/10', borderColor: 'blue-500/20' },
    edr: { label: 'EDR', color: 'purple-400', bgColor: 'purple-500/10', borderColor: 'purple-500/20' },
    dlp: { label: 'DLP', color: 'green-400', bgColor: 'green-500/10', borderColor: 'green-500/20' },
  };

  // Format parameters for display
  const params = action.parameters || action.params || {};
  const resultData = action.result_data || {};
  const errorDetails = action.error_details || {};
  const verificationDetails = action.tpot_verification_details || {};

  // Get related events (events around the action execution time)
  const relatedEvents = incidentEvents.filter(event => {
    const eventTime = new Date(event.ts).getTime();
    const actionTime = new Date(action.created_at).getTime();
    const timeDiff = Math.abs(eventTime - actionTime);
    // Events within 5 minutes of action execution
    return timeDiff <= 5 * 60 * 1000;
  }).slice(0, 10);

  const formatJSON = (obj: Record<string, any>) => {
    return JSON.stringify(obj, null, 2);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const formatTimeAgo = (dateString: string) => {
    const now = new Date();
    const date = new Date(dateString);
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);

    if (diffMins < 1) return 'just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    const diffDays = Math.floor(diffHours / 24);
    return `${diffDays}d ago`;
  };

  const formatAbsoluteTime = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: true
    });
  };

  return (
    <div className="fixed inset-0 bg-black/90 backdrop-blur-md flex items-center justify-center z-50 p-4">
      <div className="glass-panel w-full max-w-4xl max-h-[90vh] overflow-hidden flex flex-col rounded-2xl border border-white/10 shadow-2xl relative">
        <div className="absolute top-0 right-0 w-64 h-64 bg-primary/10 rounded-full blur-3xl pointer-events-none -mr-32 -mt-32" />

        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-white/10 bg-white/5 relative z-10">
          <div className="flex items-center gap-4">
            <div className={`p-3 rounded-xl border ${
              isSuccess ? 'bg-green-500/10 border-green-500/20' :
              isFailure ? 'bg-red-500/10 border-red-500/20' :
              isPending ? 'bg-amber-500/10 border-amber-500/20' :
              'bg-blue-500/10 border-blue-500/20'
            }`}>
              {isSuccess && <CheckCircle className="w-6 h-6 text-green-500" />}
              {isFailure && <XCircle className="w-6 h-6 text-red-500" />}
              {isPending && <Clock className="w-6 h-6 text-amber-500 animate-pulse" />}
              {!isSuccess && !isFailure && !isPending && <Shield className="w-6 h-6 text-blue-500" />}
            </div>
            <div>
              <h2 className="text-xl font-bold font-heading text-white tracking-wide">{actionName}</h2>
              <div className="flex flex-wrap items-center gap-2 mt-1.5">
                <span className={`text-[10px] px-2 py-0.5 rounded font-bold font-mono uppercase tracking-wider border ${
                  isSuccess ? 'bg-green-500/10 text-green-400 border-green-500/20' :
                  isFailure ? 'bg-red-500/10 text-red-400 border-red-500/20' :
                  isPending ? 'bg-amber-500/10 text-amber-400 border-amber-500/20' :
                  'bg-blue-500/10 text-blue-400 border-blue-500/20'
                }`}>
                  {status}
                </span>
                {action.execution_method && (
                  <span className="text-[10px] px-2 py-0.5 rounded bg-purple-500/10 text-purple-400 border border-purple-500/20 font-bold font-mono uppercase tracking-wider">
                    {action.execution_method === 'automated' ? '🤖 AUTOMATED' : '⚡ ' + action.execution_method}
                  </span>
                )}
                {action.agent_type && agentTypeConfig[action.agent_type] && (
                  <span className={`text-[10px] px-2 py-0.5 rounded bg-${agentTypeConfig[action.agent_type].bgColor} text-${agentTypeConfig[action.agent_type].color} border border-${agentTypeConfig[action.agent_type].borderColor} font-bold font-mono uppercase tracking-wider`}>
                    {agentTypeConfig[action.agent_type].label} AGENT
                  </span>
                )}
                {action.workflow_name && (
                  <span className="text-[10px] text-gray-400 font-mono uppercase">
                    Workflow: <span className="text-gray-300">{action.workflow_name}</span>
                  </span>
                )}
                {action.rollback_executed && action.rollback_timestamp && (
                  <span className="text-[10px] px-2 py-0.5 rounded bg-orange-500/10 text-orange-400 border border-orange-500/20 font-bold font-mono uppercase tracking-wider">
                    🔄 ROLLED BACK
                  </span>
                )}
              </div>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-white/10 rounded-lg transition-colors group"
          >
            <X className="w-5 h-5 text-gray-400 group-hover:text-white" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6 custom-scrollbar relative z-10">
          {/* Execution Timeline */}
          <div className="bg-black/20 border border-white/5 rounded-xl p-4">
            <h3 className="text-sm font-bold text-white mb-4 flex items-center gap-2 font-heading uppercase tracking-wide">
              <Clock className="w-4 h-4 text-blue-500" />
              Execution Timeline
            </h3>
            <div className="grid grid-cols-2 gap-6 text-sm">
              <div>
                <div className="text-[10px] text-gray-500 uppercase font-bold font-mono mb-1">Started</div>
                <div className="text-white font-mono text-xs">
                  {formatAbsoluteTime(action.created_at)}
                </div>
                <div className="text-gray-500 text-[10px] font-mono mt-0.5">
                  {formatTimeAgo(action.created_at)}
                </div>
              </div>
              {action.completed_at && (
                <div>
                  <div className="text-[10px] text-gray-500 uppercase font-bold font-mono mb-1">Completed</div>
                  <div className="text-white font-mono text-xs">
                    {formatAbsoluteTime(action.completed_at)}
                  </div>
                  <div className="text-gray-500 text-[10px] font-mono mt-0.5">
                    {formatTimeAgo(action.completed_at)}
                  </div>
                </div>
              )}
              {action.executed_by && (
                <div>
                  <div className="text-[10px] text-gray-500 uppercase font-bold font-mono mb-1">Executed By</div>
                  <div className="text-white font-mono text-xs">{action.executed_by}</div>
                </div>
              )}
              {action.confidence_score !== undefined && (
                <div>
                  <div className="text-[10px] text-gray-500 uppercase font-bold font-mono mb-1">Confidence</div>
                  <div className="text-white font-mono text-xs">{Math.round(action.confidence_score * 100)}%</div>
                </div>
              )}
            </div>
          </div>

          {/* Action Description/Detail */}
          {action.detail && (
            <div className="bg-black/20 border border-white/5 rounded-xl p-4">
              <h3 className="text-sm font-bold text-white mb-3 flex items-center gap-2 font-heading uppercase tracking-wide">
                <FileText className="w-4 h-4 text-green-500" />
                Action Logs
              </h3>
              <p className="text-gray-300 text-xs font-mono leading-relaxed whitespace-pre-wrap opacity-90">
                {action.detail}
              </p>
            </div>
          )}

          {/* Parameters */}
          {Object.keys(params).length > 0 && (
            <div className="bg-black/20 border border-white/5 rounded-xl p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-bold text-white flex items-center gap-2 font-heading uppercase tracking-wide">
                  <Code className="w-4 h-4 text-purple-500" />
                  Input Parameters
                </h3>
                <button
                  onClick={() => copyToClipboard(formatJSON(params))}
                  className="text-gray-500 hover:text-white p-1 transition-colors"
                  title="Copy to clipboard"
                >
                  <Copy className="w-3 h-3" />
                </button>
              </div>
              <pre className="bg-black/40 border border-white/5 rounded-lg p-3 text-[10px] text-gray-300 font-mono overflow-x-auto custom-scrollbar">
                <code>{formatJSON(params)}</code>
              </pre>
            </div>
          )}

          {/* Result Data */}
          {Object.keys(resultData).length > 0 && (
            <div className="bg-green-500/5 border border-green-500/10 rounded-xl p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-bold text-green-400 flex items-center gap-2 font-heading uppercase tracking-wide">
                  <CheckCircle className="w-4 h-4" />
                  Execution Results
                </h3>
                <button
                  onClick={() => copyToClipboard(formatJSON(resultData))}
                  className="text-green-500/50 hover:text-green-400 p-1 transition-colors"
                  title="Copy to clipboard"
                >
                  <Copy className="w-3 h-3" />
                </button>
              </div>
              <pre className="bg-black/40 border border-green-500/10 rounded-lg p-3 text-[10px] text-green-200/90 font-mono overflow-x-auto custom-scrollbar">
                <code>{formatJSON(resultData)}</code>
              </pre>
            </div>
          )}

          {/* Error Details */}
          {Object.keys(errorDetails).length > 0 && (
            <div className="bg-red-500/5 border border-red-500/10 rounded-xl p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-bold text-red-400 flex items-center gap-2 font-heading uppercase tracking-wide">
                  <AlertCircle className="w-4 h-4" />
                  Error Details
                </h3>
                <button
                  onClick={() => copyToClipboard(formatJSON(errorDetails))}
                  className="text-red-500/50 hover:text-red-400 p-1 transition-colors"
                  title="Copy to clipboard"
                >
                  <Copy className="w-3 h-3" />
                </button>
              </div>
              <pre className="bg-black/40 border border-red-500/10 rounded-lg p-3 text-[10px] text-red-200/90 font-mono overflow-x-auto custom-scrollbar">
                <code>{formatJSON(errorDetails)}</code>
              </pre>
            </div>
          )}

          {/* Verification Details (T-Pot) */}
          {Object.keys(verificationDetails).length > 0 && (
            <div className="bg-blue-500/5 border border-blue-500/10 rounded-xl p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-bold text-blue-400 flex items-center gap-2 font-heading uppercase tracking-wide">
                  <Eye className="w-4 h-4" />
                  Verification Details
                </h3>
                <button
                  onClick={() => copyToClipboard(formatJSON(verificationDetails))}
                  className="text-blue-500/50 hover:text-blue-400 p-1 transition-colors"
                  title="Copy to clipboard"
                >
                  <Copy className="w-3 h-3" />
                </button>
              </div>
              <pre className="bg-black/40 border border-blue-500/10 rounded-lg p-3 text-[10px] text-blue-200/90 font-mono overflow-x-auto custom-scrollbar">
                <code>{formatJSON(verificationDetails)}</code>
              </pre>
            </div>
          )}

          {/* Related Events/Logs */}
          {relatedEvents.length > 0 && (
            <div className="bg-black/20 border border-white/5 rounded-xl p-4">
              <h3 className="text-sm font-bold text-white mb-3 flex items-center gap-2 font-heading uppercase tracking-wide">
                <Activity className="w-4 h-4 text-orange-500" />
                Related Events ({relatedEvents.length})
              </h3>
              <div className="text-[10px] text-gray-500 mb-3 font-mono uppercase tracking-wider">
                Events occurring within 5 minutes of action execution
              </div>
              <div className="space-y-2 max-h-64 overflow-y-auto custom-scrollbar">
                {relatedEvents.map((event) => (
                  <div
                    key={event.id}
                    className="bg-white/5 border border-white/5 rounded p-3 hover:bg-white/10 transition-colors"
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <Database className="w-3 h-3 text-blue-500" />
                        <span className="text-[10px] font-mono text-blue-400">{event.eventid}</span>
                        <span className="text-[10px] px-2 py-0.5 rounded bg-purple-500/10 text-purple-400 border border-purple-500/20 font-bold font-mono">
                          {event.source_type}
                        </span>
                      </div>
                      <span className="text-[10px] text-gray-500 font-mono">
                        {formatAbsoluteTime(event.ts)}
                      </span>
                    </div>
                    <div className="text-xs text-gray-300 mb-2 font-mono">
                      {event.message}
                    </div>
                    <div className="flex items-center gap-3 text-[10px] text-gray-500 font-mono">
                      <span>SRC: {event.src_ip}</span>
                      {event.dst_ip && <span>→ DST: {event.dst_ip}</span>}
                      {event.dst_port && <span>PORT: {event.dst_port}</span>}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* No Events Found */}
          {relatedEvents.length === 0 && incidentEvents.length > 0 && (
            <div className="bg-white/5 border border-dashed border-white/10 rounded-xl p-8 text-center">
              <Activity className="w-8 h-8 text-gray-700 mx-auto mb-3" />
              <div className="text-gray-500 font-heading font-medium text-sm">
                No correlated events found
              </div>
              <div className="text-xs text-gray-600 mt-1 font-mono">
                No system events detected within ±5 mins window
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-white/10 p-4 flex items-center justify-between bg-white/5 relative z-10">
          <div className="text-[10px] text-gray-500 font-mono">
            ID: {action.action_id || action.id}
            {action.rollback_id && (
              <span className="ml-2 pl-2 border-l border-white/10">
                RB_ID: {action.rollback_id}
              </span>
            )}
          </div>
          <div className="flex items-center gap-3">
            {canRollback && onRollback && (
              <button
                onClick={() => {
                  if (confirm(`Are you sure you want to rollback "${actionName}"?\n\nThis will restore the previous state.`)) {
                    onRollback(action.rollback_id!);
                  }
                }}
                className="px-4 py-2 bg-orange-600 hover:bg-orange-700 text-white rounded-lg text-xs font-bold font-mono transition-colors flex items-center gap-2 uppercase tracking-wide border border-orange-500/50"
              >
                🔄 Rollback Action
              </button>
            )}
            <button
              onClick={onClose}
              className="px-4 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg text-xs font-bold font-mono transition-colors uppercase tracking-wide border border-white/10"
            >
              Close Console
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
