"use client";

import React, { useState } from 'react';
import {
  ChevronDown, ChevronUp, Loader2, RotateCcw, Eye,
  Bot, Zap, User
} from 'lucide-react';
import {
  getActionIcon,
  getStatusColor,
  getStatusIcon,
  formatTimeAgo,
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
      case 'agent': return <Bot className="h-3 w-3" />;
      case 'workflow': return <Zap className="h-3 w-3" />;
      case 'manual': return <User className="h-3 w-3" />;
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
    <div className={`font-mono text-xs border-l-2 pl-3 py-2 hover:bg-muted/30 transition-all duration-200 ${
      action.status.toLowerCase().includes('success') || action.status.toLowerCase().includes('completed')
        ? 'border-green-500/50 bg-green-500/5'
        : action.status.toLowerCase().includes('failed') || action.status.toLowerCase().includes('error')
        ? 'border-red-500/50 bg-red-500/5'
        : 'border-amber-500/50 bg-amber-500/5'
    }`}>
      {/* Log Entry Header */}
      <div className="flex items-start gap-2">
        <span className={`font-bold min-w-[12px] ${
          action.status.toLowerCase().includes('success') || action.status.toLowerCase().includes('completed')
            ? 'text-green-400'
            : action.status.toLowerCase().includes('failed') || action.status.toLowerCase().includes('error')
            ? 'text-red-400'
            : 'text-amber-400'
        }`}>
          {statusIcon}
        </span>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 text-muted-foreground mb-1">
            <span className="text-primary/80 font-medium">{formatTimeAgo(action.createdAt)}</span>
            <span className="text-muted-foreground/60">•</span>
            <span className={`uppercase font-medium ${
              action.source === 'agent' ? 'text-blue-400' :
              action.source === 'workflow' ? 'text-purple-400' :
              'text-gray-400'
            }`}>
              {action.sourceLabel}
            </span>
            {action.executedBy && (
              <>
                <span className="text-muted-foreground/60">•</span>
                <span className="text-muted-foreground font-medium">{action.executedBy}</span>
              </>
            )}
          </div>

          <div className="flex items-center gap-2">
            <span className={`font-medium ${
              action.status.toLowerCase().includes('success') || action.status.toLowerCase().includes('completed')
                ? 'text-green-300'
                : action.status.toLowerCase().includes('failed') || action.status.toLowerCase().includes('error')
                ? 'text-red-300'
                : 'text-amber-300'
            }`}>
              {action.displayName}
            </span>
            <span className={`px-2 py-0.5 text-[10px] uppercase font-bold rounded border ${
              action.status.toLowerCase().includes('success') || action.status.toLowerCase().includes('completed')
                ? 'bg-green-500/20 text-green-300 border-green-500/30'
                : action.status.toLowerCase().includes('failed') || action.status.toLowerCase().includes('error')
                ? 'bg-red-500/20 text-red-300 border-red-500/30'
                : 'bg-amber-500/20 text-amber-300 border-amber-500/30'
            }`}>
              {action.status}
            </span>
            {action.rollbackExecuted && (
              <span className="px-2 py-0.5 text-[10px] uppercase font-bold bg-orange-500/20 text-orange-300 border border-orange-500/30 rounded">
                ROLLED BACK
              </span>
            )}
          </div>

          {/* Brief detail */}
          {action.detail && (
            <div className="text-muted-foreground mt-1 leading-relaxed">
              {expanded ? action.detail : action.detail.length > 100 ? `${action.detail.substring(0, 100)}...` : action.detail}
            </div>
          )}

          {/* Expanded Content with Animation */}
          <div className={`overflow-hidden transition-all duration-300 ease-in-out ${
            expanded ? 'max-h-96 opacity-100 mt-3' : 'max-h-0 opacity-0'
          }`}>
            {/* Parameters (if expanded) */}
            {action.params && Object.keys(action.params).length > 0 && (
              <div className="text-muted-foreground mb-3 p-3 bg-purple-500/5 border border-purple-500/20 rounded-lg">
                <div className="text-purple-400 font-medium mb-2 text-xs uppercase tracking-wide">Parameters:</div>
                {Object.entries(action.params).map(([key, value]) => (
                  <div key={key} className="ml-2 text-sm">
                    <span className="text-purple-300 font-medium">{key}:</span> <span className="text-purple-200/80">{typeof value === 'object' ? JSON.stringify(value) : String(value)}</span>
                  </div>
                ))}
              </div>
            )}

            {/* Result (if expanded) */}
            {action.result && Object.keys(action.result).length > 0 && (
              <div className="text-green-400 mb-3 p-3 bg-green-500/5 border border-green-500/20 rounded-lg">
                <div className="text-green-400 font-medium mb-2 text-xs uppercase tracking-wide">Result Output:</div>
                {Object.entries(action.result).map(([key, value]) => (
                  <div key={key} className="ml-2 text-sm text-green-200/90">
                    <span className="text-green-300 font-medium">{key}:</span> {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                  </div>
                ))}
              </div>
            )}

            {/* Error (if expanded) */}
            {action.error && (
              <div className="text-red-400 mb-3 p-3 bg-red-500/5 border border-red-500/20 rounded-lg">
                <div className="text-red-300 font-medium mb-2 text-xs uppercase tracking-wide">Error Details:</div>
                <div className="ml-2 text-sm text-red-200/90">{action.error}</div>
              </div>
            )}

            {/* Action buttons (if expanded) */}
            <div className="flex gap-2">
              {onViewDetails && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onViewDetails();
                  }}
                  className="px-3 py-1.5 bg-secondary hover:bg-secondary/80 text-secondary-foreground text-xs font-mono rounded transition-colors flex items-center gap-1"
                >
                  <Eye className="w-3 h-3" />
                  Full Logs
                </button>
              )}
              {canRollback && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleRollback();
                  }}
                  disabled={rolling}
                  className="px-3 py-1.5 bg-orange-600/80 hover:bg-orange-600 text-white text-xs font-mono rounded transition-colors disabled:opacity-50 flex items-center gap-1"
                >
                  <RotateCcw className="w-3 h-3" />
                  {rolling ? 'Rolling...' : 'Rollback'}
                </button>
              )}
              <button
                onClick={handleToggle}
                className="px-3 py-1.5 bg-muted hover:bg-muted/80 text-muted-foreground text-xs font-mono rounded transition-colors ml-auto flex items-center gap-1"
              >
                <ChevronUp className="w-3 h-3" />
                Collapse
              </button>
            </div>
          </div>

          {/* Expand toggle (if not expanded) */}
          {!expanded && (
            <button
              onClick={handleToggle}
              className="mt-2 text-primary/70 hover:text-primary text-xs font-mono transition-colors flex items-center gap-1 hover:bg-primary/10 px-2 py-1 rounded"
            >
              <ChevronDown className="w-3 h-3" />
              {action.detail && action.detail.length > 100 ? 'Show full details' : 'More details'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
