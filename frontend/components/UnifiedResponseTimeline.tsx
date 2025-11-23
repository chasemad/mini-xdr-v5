"use client";

import React, { useState, useEffect, useMemo } from 'react';
import {
  Activity, RefreshCw, Filter, TrendingUp, Bot, Zap, User,
  CheckCircle, XCircle, Clock, AlertCircle
} from 'lucide-react';
import ActionCard, { UnifiedAction } from './ActionCard';
import ActionDetailModal from './ActionDetailModal';
import { calculateActionSummary } from '@/lib/actionFormatters';
import { apiUrl } from '@/app/utils/api';

interface UnifiedResponseTimelineProps {
  incidentId: number;
  actions?: any[];
  automatedActions?: any[];
  agentActions?: any[];
  onRefresh?: () => void;
  onRollback?: (rollbackId: string) => Promise<void>;
  incidentEvents?: any[];
}

type ActionSource = 'all' | 'agent' | 'workflow' | 'manual';
type SortBy = 'newest' | 'oldest' | 'status';

export default function UnifiedResponseTimeline({
  incidentId,
  actions = [],
  automatedActions = [],
  agentActions: initialAgentActions = [],
  onRefresh,
  onRollback,
  incidentEvents = []
}: UnifiedResponseTimelineProps) {
  const [agentActions, setAgentActions] = useState<any[]>(initialAgentActions);
  const [refreshing, setRefreshing] = useState(false);
  const [filterSource, setFilterSource] = useState<ActionSource>('all');
  const [sortBy, setSortBy] = useState<SortBy>('newest');
  const [selectedAction, setSelectedAction] = useState<any | null>(null);
  const [showModal, setShowModal] = useState(false);

  useEffect(() => {
    fetchAgentActions();
    const interval = setInterval(fetchAgentActions, 5000);
    return () => clearInterval(interval);
  }, [incidentId]);

  const fetchAgentActions = async () => {
    try {
      const response = await fetch(apiUrl(`/api/agents/actions/${incidentId}`));
      if (response.ok) {
        const data = await response.json();
        setAgentActions(data);
      }
    } catch (error) {
      console.error('Failed to fetch agent actions:', error);
    }
  };

  const unifiedActions = useMemo<UnifiedAction[]>(() => {
    const manual: UnifiedAction[] = actions.map((a) => ({
      id: `manual-${a.id}`,
      originalId: a.id,
      createdAt: a.created_at,
      completedAt: a.due_at,
      actionKey: a.action,
      displayName: a.action,
      source: 'manual',
      sourceLabel: '⚡ MANUAL',
      icon: 'user',
      status: a.result || 'completed',
      detail: a.detail,
      params: a.params,
      executedBy: 'SOC Analyst'
    }));

    const workflow: UnifiedAction[] = automatedActions.map((a) => ({
      id: `workflow-${a.id}`,
      originalId: a.id,
      createdAt: a.created_at,
      completedAt: a.completed_at,
      actionKey: a.action_type,
      displayName: a.action_name,
      source: 'workflow',
      sourceLabel: '🤖 WORKFLOW',
      icon: 'workflow',
      status: a.status,
      detail: a.action_name,
      params: a.parameters,
      result: a.result_data,
      workflowName: a.workflow_name,
      workflowId: a.workflow_id,
      executedBy: a.executed_by,
      executionMethod: a.execution_method,
      errorDetails: a.error_details,
      rollbackId: a.rollback?.action_type
    }));

    const agent: UnifiedAction[] = agentActions.map((a) => ({
      id: `agent-${a.id}`,
      originalId: a.id,
      createdAt: a.executed_at || a.created_at,
      actionKey: a.action_name,
      displayName: a.action_name,
      source: 'agent',
      sourceLabel: `${a.agent_type.toUpperCase()} AGENT`,
      icon: a.agent_type,
      status: a.status,
      detail: a.result ? JSON.stringify(a.result) : undefined,
      params: a.params,
      result: a.result,
      agentType: a.agent_type,
      rollbackId: a.rollback_id,
      rollbackExecuted: a.rollback_executed,
      rollbackTimestamp: a.rollback_timestamp,
      error: a.error
    }));

    const merged = [...manual, ...workflow, ...agent];

    let filtered = merged;
    if (filterSource !== 'all') {
      filtered = merged.filter(a => a.source === filterSource);
    }

    filtered.sort((a, b) => {
      if (sortBy === 'newest') {
        return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
      } else if (sortBy === 'oldest') {
        return new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime();
      } else {
        const statusOrder: Record<string, number> = {
          pending: 1, running: 1, in_progress: 1,
          success: 2, completed: 2, done: 2,
          failed: 3, error: 3
        };
        const aOrder = statusOrder[a.status.toLowerCase()] || 4;
        const bOrder = statusOrder[b.status.toLowerCase()] || 4;
        return aOrder - bOrder;
      }
    });

    return filtered;
  }, [actions, automatedActions, agentActions, filterSource, sortBy]);

  const summary = useMemo(() => calculateActionSummary(unifiedActions), [unifiedActions]);

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await fetchAgentActions();
      onRefresh?.();
    } finally {
      setRefreshing(false);
    }
  };

  const handleActionClick = (action: UnifiedAction) => {
    const originalAction = {
      id: action.originalId,
      action: action.actionKey,
      action_name: action.displayName,
      action_type: action.actionKey,
      status: action.status,
      detail: action.detail,
      params: action.params,
      parameters: action.params,
      result_data: action.result,
      error_details: action.errorDetails,
      created_at: action.createdAt,
      completed_at: action.completedAt,
      executed_by: action.executedBy,
      execution_method: action.executionMethod,
      workflow_name: action.workflowName,
      agent_type: action.agentType,
      rollback_id: action.rollbackId,
      rollback_executed: action.rollbackExecuted,
      rollback_timestamp: action.rollbackTimestamp,
      error: action.error
    };

    setSelectedAction(originalAction);
    setShowModal(true);
  };

  const handleRollback = async (action: UnifiedAction) => {
    if (!onRollback || !action.rollbackId) return;

    try {
      await onRollback(action.rollbackId);
      await handleRefresh();
    } catch (error) {
      console.error('Rollback failed:', error);
    }
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2 font-heading">
            <Activity className="w-5 h-5 text-primary" />
            Response Activity
          </h2>
          <div className="flex items-center gap-3 mt-0.5 text-xs text-gray-500 font-mono">
            <span>{summary.totalActions} TOTAL</span>
            <span>•</span>
            <span>{summary.manualActions} MANUAL</span>
            <span>•</span>
            <span>{summary.workflowActions} WORKFLOW</span>
            <span>•</span>
            <span>{summary.agentActions} AGENT</span>
          </div>
        </div>

        <button
          onClick={handleRefresh}
          disabled={refreshing}
          className="px-3 py-1.5 bg-primary/10 hover:bg-primary/20 text-primary border border-primary/20 rounded-md text-xs flex items-center gap-2 transition-colors font-mono uppercase tracking-wider"
        >
          <RefreshCw className={`w-3 h-3 ${refreshing ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Enhanced Summary Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="bg-gradient-to-br from-green-500/10 to-green-600/5 border border-green-500/30 rounded-lg p-4 relative overflow-hidden group hover:shadow-lg hover:shadow-green-500/10 transition-all duration-300">
          <div className="absolute inset-0 bg-gradient-to-br from-green-500/5 to-green-600/10 opacity-0 group-hover:opacity-100 transition-opacity" />
          <div className="flex items-center justify-between relative z-10">
            <div>
              <div className="text-[10px] text-green-400/80 uppercase font-bold font-mono tracking-wider mb-1">Success</div>
              <div className="text-2xl font-bold text-green-300 font-heading">{summary.successCount}</div>
            </div>
            <div className="p-2 bg-green-500/20 rounded-lg">
              <CheckCircle className="w-6 h-6 text-green-400" />
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-br from-red-500/10 to-red-600/5 border border-red-500/30 rounded-lg p-4 relative overflow-hidden group hover:shadow-lg hover:shadow-red-500/10 transition-all duration-300">
          <div className="absolute inset-0 bg-gradient-to-br from-red-500/5 to-red-600/10 opacity-0 group-hover:opacity-100 transition-opacity" />
          <div className="flex items-center justify-between relative z-10">
            <div>
              <div className="text-[10px] text-red-400/80 uppercase font-bold font-mono tracking-wider mb-1">Failed</div>
              <div className="text-2xl font-bold text-red-300 font-heading">{summary.failureCount}</div>
            </div>
            <div className="p-2 bg-red-500/20 rounded-lg">
              <XCircle className="w-6 h-6 text-red-400" />
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-br from-amber-500/10 to-amber-600/5 border border-amber-500/30 rounded-lg p-4 relative overflow-hidden group hover:shadow-lg hover:shadow-amber-500/10 transition-all duration-300">
          <div className="absolute inset-0 bg-gradient-to-br from-amber-500/5 to-amber-600/10 opacity-0 group-hover:opacity-100 transition-opacity" />
          <div className="flex items-center justify-between relative z-10">
            <div>
              <div className="text-[10px] text-amber-400/80 uppercase font-bold font-mono tracking-wider mb-1">Pending</div>
              <div className="text-2xl font-bold text-amber-300 font-heading">{summary.pendingCount}</div>
            </div>
            <div className="p-2 bg-amber-500/20 rounded-lg">
              <Clock className="w-6 h-6 text-amber-400" />
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-br from-blue-500/10 to-blue-600/5 border border-blue-500/30 rounded-lg p-4 relative overflow-hidden group hover:shadow-lg hover:shadow-blue-500/10 transition-all duration-300">
          <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-blue-600/10 opacity-0 group-hover:opacity-100 transition-opacity" />
          <div className="flex items-center justify-between relative z-10">
            <div>
              <div className="text-[10px] text-blue-400/80 uppercase font-bold font-mono tracking-wider mb-1">Success Rate</div>
              <div className="text-2xl font-bold text-blue-300 font-heading">{summary.successRate}%</div>
            </div>
            <div className="p-2 bg-blue-500/20 rounded-lg">
              <TrendingUp className="w-6 h-6 text-blue-400" />
            </div>
          </div>
        </div>
      </div>

      {/* Filters and Sort */}
      <div className="glass-panel border-white/5 rounded-lg p-3 flex flex-col md:flex-row gap-4 items-center justify-between">
        <div className="flex items-center gap-2 w-full md:w-auto overflow-x-auto pb-2 md:pb-0">
          <Filter className="w-3 h-3 text-gray-500" />
          <div className="flex gap-1">
            {(['all', 'agent', 'workflow', 'manual'] as ActionSource[]).map((source) => (
              <button
                key={source}
                onClick={() => setFilterSource(source)}
                className={`px-2.5 py-1 rounded text-[10px] font-bold uppercase tracking-wider transition-all font-mono border ${
                  filterSource === source
                    ? 'bg-primary/20 text-primary border-primary/30'
                    : 'bg-transparent text-gray-500 border-transparent hover:bg-white/5 hover:text-gray-300'
                }`}
              >
                {source}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2 w-full md:w-auto">
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as SortBy)}
            className="w-full md:w-auto bg-black/40 text-gray-400 px-3 py-1 rounded text-xs border border-white/10 focus:outline-none focus:border-primary/50 font-mono uppercase"
          >
            <option value="newest">Newest First</option>
            <option value="oldest">Oldest First</option>
            <option value="status">By Status</option>
          </select>
        </div>
      </div>

      {/* Actions List */}
      <div className="space-y-2 max-h-[600px] overflow-y-auto pr-2 custom-scrollbar">
        {unifiedActions.length === 0 ? (
          <div className="glass-card border-dashed border-white/10 rounded-lg p-8 text-center">
            <AlertCircle className="w-8 h-8 text-gray-700 mx-auto mb-3" />
            <div className="text-gray-500 font-heading font-medium">No actions found</div>
            <div className="text-xs text-gray-600 mt-1 font-mono">
              {filterSource !== 'all'
                ? `No ${filterSource} actions logged`
                : 'Awaiting response initiation'}
            </div>
          </div>
        ) : (
          unifiedActions.map((action) => (
            <ActionCard
              key={action.id}
              action={action}
              onViewDetails={() => handleActionClick(action)}
              onRollback={() => handleRollback(action)}
            />
          ))
        )}
      </div>

      {/* Load More */}
      {unifiedActions.length > 20 && (
        <div className="text-center pt-2">
          <button className="px-4 py-2 bg-white/5 hover:bg-white/10 text-gray-400 rounded-lg text-xs font-bold font-mono transition-colors uppercase tracking-wider">
            Load Historical Actions
          </button>
        </div>
      )}

      <ActionDetailModal
        action={selectedAction}
        isOpen={showModal}
        onClose={() => {
          setShowModal(false);
          setSelectedAction(null);
        }}
        onRollback={onRollback}
        incidentEvents={incidentEvents}
      />
    </div>
  );
}
