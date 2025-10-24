"use client";

import React, { useState } from 'react';
import { 
  ChevronDown, ChevronUp, Loader2, RotateCcw, Eye, Copy,
  Bot, Zap, User
} from 'lucide-react';
import { 
  getActionIcon, 
  getStatusColor, 
  getStatusIcon,
  formatTimeAgo,
  getActionDisplayName,
  formatJSON
} from '@/lib/actionFormatters';

export interface UnifiedAction {
  id: string;
  originalId: number | string;
  createdAt: string;
  completedAt?: string;
  actionKey: string;
  displayName: string;
  source: 'manual' | 'workflow' | 'agent';
  sourceLabel: string;
  icon: string;
  status: string;
  detail?: string;
  params?: Record<string, unknown>;
  result?: Record<string, unknown>;
  workflowName?: string;
  workflowId?: string;
  executedBy?: string;
  executionMethod?: string;
  agentType?: 'iam' | 'edr' | 'dlp';
  rollbackId?: string;
  rollbackExecuted?: boolean;
  rollbackTimestamp?: string;
  error?: string;
  errorDetails?: Record<string, unknown>;
}

interface ActionCardProps {
  action: UnifiedAction;
  onExpand?: () => void;
  onRollback?: () => void;
  onViewDetails?: () => void;
  isExpanded?: boolean;
}

export default function ActionCard({ 
  action, 
  onExpand, 
  onRollback,
  onViewDetails,
  isExpanded = false 
}: ActionCardProps) {
  const [localExpanded, setLocalExpanded] = useState(isExpanded);
  const [rolling, setRolling] = useState(false);

  const expanded = isExpanded || localExpanded;

  const statusColor = getStatusColor(action.status);
  const statusIcon = getStatusIcon(action.status);
  const iconConfig = getActionIcon(action.actionKey, action.agentType);
  const IconComponent = iconConfig.icon;

  const canRollback = action.rollbackId && !action.rollbackExecuted && 
                      !['failed', 'error'].includes(action.status.toLowerCase());

  const handleToggle = () => {
    if (onExpand) {
      onExpand();
    } else {
      setLocalExpanded(!expanded);
    }
  };

  const handleRollback = async () => {
    if (!onRollback || !canRollback) return;
    
    const confirmed = confirm(
      `Are you sure you want to rollback "${action.displayName}"?\n\nThis will restore the previous state.`
    );
    
    if (confirmed) {
      setRolling(true);
      try {
        await onRollback();
      } finally {
        setRolling(false);
      }
    }
  };

  const getSourceIcon = () => {
    switch (action.source) {
      case 'agent': return <Bot className="h-4 w-4" />;
      case 'workflow': return <Zap className="h-4 w-4" />;
      case 'manual': return <User className="h-4 w-4" />;
      default: return null;
    }
  };

  const getSourceColor = () => {
    switch (action.source) {
      case 'agent': return 'blue';
      case 'workflow': return 'purple';
      case 'manual': return 'gray';
      default: return 'gray';
    }
  };

  const sourceColor = getSourceColor();

  return (
    <div className={`
      bg-gray-800/50 border rounded-lg transition-all duration-200
      ${expanded ? 'border-gray-600' : 'border-gray-700/50'}
      hover:border-gray-600 hover:bg-gray-800/70
    `}>
      {/* Card Header */}
      <div 
        className="p-4 cursor-pointer"
        onClick={handleToggle}
      >
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-start gap-3 flex-1">
            {/* Icon */}
            <div className={`p-2 rounded-lg flex-shrink-0 bg-${iconConfig.bgColor} border border-${iconConfig.borderColor}`}>
              <IconComponent className={`w-5 h-5 text-${iconConfig.color}`} />
            </div>

            {/* Action Details */}
            <div className="flex-1 min-w-0">
              {/* Title Row */}
              <div className="flex items-center gap-2 mb-1">
                <div className={`flex items-center gap-1.5 px-2 py-0.5 rounded text-xs font-semibold bg-${sourceColor}-500/20 text-${sourceColor}-300`}>
                  {getSourceIcon()}
                  <span>{action.sourceLabel}</span>
                </div>
                <span className="text-xs text-gray-500">{formatTimeAgo(action.createdAt)}</span>
              </div>

              {/* Action Name */}
              <h3 className="text-white font-semibold text-base mb-2">
                {statusIcon} {action.displayName}
              </h3>

              {/* Quick Info */}
              <div className="flex items-center gap-3 text-xs">
                <span className={`px-2 py-0.5 rounded font-semibold bg-${statusColor}-500/20 text-${statusColor}-300`}>
                  {action.status.toUpperCase()}
                </span>
                
                {action.workflowName && (
                  <span className="text-gray-400">
                    Workflow: <span className="text-gray-300">{action.workflowName}</span>
                  </span>
                )}

                {action.executedBy && (
                  <span className="text-gray-400">
                    By: <span className="text-gray-300">{action.executedBy}</span>
                  </span>
                )}

                {action.rollbackExecuted && action.rollbackTimestamp && (
                  <span className="px-2 py-0.5 rounded bg-orange-500/20 text-orange-300 border border-orange-500/30">
                    🔄 Rolled Back
                  </span>
                )}
              </div>

              {/* Brief detail (if not expanded) */}
              {!expanded && action.detail && (
                <div className="mt-2 text-sm text-gray-400 line-clamp-2">
                  {action.detail}
                </div>
              )}
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center gap-2 flex-shrink-0">
            {expanded ? (
              <ChevronUp className="w-5 h-5 text-gray-400" />
            ) : (
              <ChevronDown className="w-5 h-5 text-gray-400" />
            )}
          </div>
        </div>
      </div>

      {/* Expanded Content */}
      {expanded && (
        <div className="px-4 pb-4 space-y-4 border-t border-gray-700/50 pt-4">
          {/* Full Details */}
          {action.detail && (
            <div>
              <h4 className="text-xs font-semibold text-gray-400 uppercase mb-2">Details</h4>
              <p className="text-sm text-gray-300 leading-relaxed whitespace-pre-wrap">
                {action.detail}
              </p>
            </div>
          )}

          {/* Parameters */}
          {action.params && Object.keys(action.params).length > 0 && (
            <div>
              <h4 className="text-xs font-semibold text-gray-400 uppercase mb-2">Parameters</h4>
              <div className="bg-gray-900/50 border border-gray-700 rounded p-3 text-xs space-y-1">
                {Object.entries(action.params).map(([key, value]) => (
                  <div key={key} className="flex gap-2">
                    <span className="text-purple-400 font-mono">{key}:</span>
                    <span className="text-gray-300 font-mono flex-1 break-all">
                      {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Result Data */}
          {action.result && Object.keys(action.result).length > 0 && (
            <div>
              <h4 className="text-xs font-semibold text-green-400 uppercase mb-2">Result</h4>
              <div className="bg-green-500/10 border border-green-500/30 rounded p-3 text-xs space-y-1">
                {Object.entries(action.result).map(([key, value]) => (
                  <div key={key} className="flex gap-2">
                    <span className="text-green-400 font-mono">{key}:</span>
                    <span className="text-green-200 font-mono flex-1 break-all">
                      {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Error Details */}
          {action.error && (
            <div>
              <h4 className="text-xs font-semibold text-red-400 uppercase mb-2">Error</h4>
              <div className="bg-red-500/10 border border-red-500/30 rounded p-3 text-xs text-red-200">
                {action.error}
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex items-center gap-2 pt-2">
            {onViewDetails && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onViewDetails();
                }}
                className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm font-medium transition-colors flex items-center gap-1.5"
              >
                <Eye className="w-4 h-4" />
                View Full Details
              </button>
            )}

            {canRollback && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleRollback();
                }}
                disabled={rolling}
                className="px-3 py-1.5 bg-orange-600 hover:bg-orange-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded text-sm font-medium transition-colors flex items-center gap-1.5"
              >
                {rolling ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <RotateCcw className="w-4 h-4" />
                )}
                Rollback
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

