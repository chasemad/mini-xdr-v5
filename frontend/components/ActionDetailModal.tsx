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
    iam: { label: 'IAM', color: 'blue-400', bgColor: 'blue-500/20', borderColor: 'blue-500/30' },
    edr: { label: 'EDR', color: 'purple-400', bgColor: 'purple-500/20', borderColor: 'purple-500/30' },
    dlp: { label: 'DLP', color: 'green-400', bgColor: 'green-500/20', borderColor: 'green-500/30' },
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
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-gray-900 border border-gray-700 rounded-xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-700">
          <div className="flex items-center gap-4">
            <div className={`p-3 rounded-lg ${
              isSuccess ? 'bg-green-500/20 border border-green-500/30' :
              isFailure ? 'bg-red-500/20 border border-red-500/30' :
              isPending ? 'bg-yellow-500/20 border border-yellow-500/30' :
              'bg-blue-500/20 border border-blue-500/30'
            }`}>
              {isSuccess && <CheckCircle className="w-6 h-6 text-green-400" />}
              {isFailure && <XCircle className="w-6 h-6 text-red-400" />}
              {isPending && <Clock className="w-6 h-6 text-yellow-400 animate-pulse" />}
              {!isSuccess && !isFailure && !isPending && <Shield className="w-6 h-6 text-blue-400" />}
            </div>
            <div>
              <h2 className="text-2xl font-bold text-white">{actionName}</h2>
              <div className="flex items-center gap-3 mt-1">
                <span className={`text-xs px-2 py-1 rounded font-semibold ${
                  isSuccess ? 'bg-green-500/20 text-green-300' :
                  isFailure ? 'bg-red-500/20 text-red-300' :
                  isPending ? 'bg-yellow-500/20 text-yellow-300' :
                  'bg-blue-500/20 text-blue-300'
                }`}>
                  {status.toUpperCase()}
                </span>
                {action.execution_method && (
                  <span className="text-xs px-2 py-1 rounded bg-purple-500/20 text-purple-300">
                    {action.execution_method === 'automated' ? '🤖 AUTOMATED' : '⚡ ' + action.execution_method.toUpperCase()}
                  </span>
                )}
                {action.agent_type && agentTypeConfig[action.agent_type] && (
                  <span className={`text-xs px-2 py-1 rounded bg-${agentTypeConfig[action.agent_type].bgColor} text-${agentTypeConfig[action.agent_type].color} border border-${agentTypeConfig[action.agent_type].borderColor}`}>
                    {agentTypeConfig[action.agent_type].label} AGENT
                  </span>
                )}
                {action.workflow_name && (
                  <span className="text-xs text-gray-400">
                    Workflow: {action.workflow_name}
                  </span>
                )}
                {action.rollback_executed && action.rollback_timestamp && (
                  <span className="text-xs px-2 py-1 rounded bg-orange-500/20 text-orange-300 border border-orange-500/30">
                    🔄 ROLLED BACK
                  </span>
                )}
              </div>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-gray-400" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Execution Timeline */}
          <div className="bg-gray-800/50 border border-gray-700/50 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
              <Clock className="w-5 h-5 text-blue-400" />
              Execution Timeline
            </h3>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <div className="text-gray-400 mb-1">Started</div>
                <div className="text-white font-mono">
                  {formatAbsoluteTime(action.created_at)}
                </div>
                <div className="text-gray-500 text-xs mt-1">
                  {formatTimeAgo(action.created_at)}
                </div>
              </div>
              {action.completed_at && (
                <div>
                  <div className="text-gray-400 mb-1">Completed</div>
                  <div className="text-white font-mono">
                    {formatAbsoluteTime(action.completed_at)}
                  </div>
                  <div className="text-gray-500 text-xs mt-1">
                    {formatTimeAgo(action.completed_at)}
                  </div>
                </div>
              )}
              {action.executed_by && (
                <div>
                  <div className="text-gray-400 mb-1">Executed By</div>
                  <div className="text-white">{action.executed_by}</div>
                </div>
              )}
              {action.confidence_score !== undefined && (
                <div>
                  <div className="text-gray-400 mb-1">Confidence Score</div>
                  <div className="text-white">{Math.round(action.confidence_score * 100)}%</div>
                </div>
              )}
            </div>
          </div>

          {/* Action Description/Detail */}
          {action.detail && (
            <div className="bg-gray-800/50 border border-gray-700/50 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                <FileText className="w-5 h-5 text-green-400" />
                Action Details
              </h3>
              <p className="text-gray-300 text-sm leading-relaxed whitespace-pre-wrap">
                {action.detail}
              </p>
            </div>
          )}

          {/* Parameters */}
          {Object.keys(params).length > 0 && (
            <div className="bg-gray-800/50 border border-gray-700/50 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                  <Code className="w-5 h-5 text-purple-400" />
                  Input Parameters
                </h3>
                <button
                  onClick={() => copyToClipboard(formatJSON(params))}
                  className="text-gray-400 hover:text-gray-300 p-1"
                  title="Copy to clipboard"
                >
                  <Copy className="w-4 h-4" />
                </button>
              </div>
              <pre className="bg-gray-900 border border-gray-700 rounded p-3 text-xs text-gray-300 overflow-x-auto">
                <code>{formatJSON(params)}</code>
              </pre>
            </div>
          )}

          {/* Result Data */}
          {Object.keys(resultData).length > 0 && (
            <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-lg font-semibold text-green-300 flex items-center gap-2">
                  <CheckCircle className="w-5 h-5" />
                  Execution Results
                </h3>
                <button
                  onClick={() => copyToClipboard(formatJSON(resultData))}
                  className="text-green-400 hover:text-green-300 p-1"
                  title="Copy to clipboard"
                >
                  <Copy className="w-4 h-4" />
                </button>
              </div>
              <pre className="bg-gray-900 border border-green-700/30 rounded p-3 text-xs text-green-200 overflow-x-auto">
                <code>{formatJSON(resultData)}</code>
              </pre>
            </div>
          )}

          {/* Error Details */}
          {Object.keys(errorDetails).length > 0 && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-lg font-semibold text-red-300 flex items-center gap-2">
                  <AlertCircle className="w-5 h-5" />
                  Error Details
                </h3>
                <button
                  onClick={() => copyToClipboard(formatJSON(errorDetails))}
                  className="text-red-400 hover:text-red-300 p-1"
                  title="Copy to clipboard"
                >
                  <Copy className="w-4 h-4" />
                </button>
              </div>
              <pre className="bg-gray-900 border border-red-700/30 rounded p-3 text-xs text-red-200 overflow-x-auto">
                <code>{formatJSON(errorDetails)}</code>
              </pre>
            </div>
          )}

          {/* Verification Details (T-Pot) */}
          {Object.keys(verificationDetails).length > 0 && (
            <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-lg font-semibold text-blue-300 flex items-center gap-2">
                  <Eye className="w-5 h-5" />
                  Verification Details
                </h3>
                <button
                  onClick={() => copyToClipboard(formatJSON(verificationDetails))}
                  className="text-blue-400 hover:text-blue-300 p-1"
                  title="Copy to clipboard"
                >
                  <Copy className="w-4 h-4" />
                </button>
              </div>
              <pre className="bg-gray-900 border border-blue-700/30 rounded p-3 text-xs text-blue-200 overflow-x-auto">
                <code>{formatJSON(verificationDetails)}</code>
              </pre>
            </div>
          )}

          {/* Related Events/Logs */}
          {relatedEvents.length > 0 && (
            <div className="bg-gray-800/50 border border-gray-700/50 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                <Activity className="w-5 h-5 text-orange-400" />
                Related Events ({relatedEvents.length})
              </h3>
              <div className="text-xs text-gray-400 mb-3">
                Events occurring within 5 minutes of action execution
              </div>
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {relatedEvents.map((event) => (
                  <div 
                    key={event.id}
                    className="bg-gray-900/50 border border-gray-700/30 rounded p-3 hover:border-gray-600/50 transition-colors"
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <Database className="w-3 h-3 text-blue-400" />
                        <span className="text-xs font-mono text-blue-300">{event.eventid}</span>
                        <span className="text-xs px-2 py-0.5 rounded bg-purple-500/20 text-purple-300">
                          {event.source_type}
                        </span>
                      </div>
                      <span className="text-xs text-gray-500">
                        {formatAbsoluteTime(event.ts)}
                      </span>
                    </div>
                    <div className="text-sm text-gray-300 mb-2">
                      {event.message}
                    </div>
                    <div className="flex items-center gap-3 text-xs text-gray-500">
                      <span>Source: {event.src_ip}</span>
                      {event.dst_ip && <span>→ Dest: {event.dst_ip}</span>}
                      {event.dst_port && <span>Port: {event.dst_port}</span>}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* No Events Found */}
          {relatedEvents.length === 0 && incidentEvents.length > 0 && (
            <div className="bg-gray-800/30 border border-gray-700/30 rounded-lg p-6 text-center">
              <Activity className="w-12 h-12 text-gray-600 mx-auto mb-3" />
              <div className="text-gray-400">
                No events found within 5 minutes of action execution
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-gray-700 p-4 flex items-center justify-between bg-gray-800/50">
          <div className="text-xs text-gray-400">
            Action ID: {action.action_id || action.id}
            {action.rollback_id && (
              <div className="text-xs text-gray-500 mt-1">
                Rollback ID: {action.rollback_id}
              </div>
            )}
          </div>
          <div className="flex items-center gap-2">
            {canRollback && onRollback && (
              <button
                onClick={() => {
                  if (confirm(`Are you sure you want to rollback "${actionName}"?\n\nThis will restore the previous state.`)) {
                    onRollback(action.rollback_id!);
                  }
                }}
                className="px-4 py-2 bg-orange-600 hover:bg-orange-700 text-white rounded-lg text-sm font-medium transition-colors flex items-center gap-2"
              >
                🔄 Rollback Action
              </button>
            )}
            <button
              onClick={onClose}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

