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

  // Fetch agent actions
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

  // Merge all actions into unified format
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

    // Apply filter
    let filtered = merged;
    if (filterSource !== 'all') {
      filtered = merged.filter(a => a.source === filterSource);
    }

    // Apply sort
    filtered.sort((a, b) => {
      if (sortBy === 'newest') {
        return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
      } else if (sortBy === 'oldest') {
        return new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime();
      } else {
        // Sort by status (success, pending, failed)
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

  const summary = useMemo(() => calculateActionSummary(
    [...actions, ...automatedActions, ...agentActions]
  ), [actions, automatedActions, agentActions]);

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
    // Convert unified action back to original format for modal
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
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <Activity className="w-6 h-6 text-blue-400" />
            Unified Response Actions
          </h2>
          <div className="flex items-center gap-3 mt-1 text-sm text-gray-400">
            <span>{summary.totalActions} total</span>
            <span>•</span>
            <span>{summary.manualActions} manual</span>
            <span>•</span>
            <span>{summary.workflowActions} workflow</span>
            <span>•</span>
            <span>{summary.agentActions} agent</span>
          </div>
        </div>

        <button
          onClick={handleRefresh}
          disabled={refreshing}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white rounded-lg flex items-center gap-2 transition-colors"
        >
          <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
        <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-3">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs text-gray-400 uppercase font-semibold">Success</div>
              <div className="text-2xl font-bold text-green-300">{summary.successCount}</div>
            </div>
            <CheckCircle className="w-8 h-8 text-green-400" />
          </div>
        </div>

        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs text-gray-400 uppercase font-semibold">Failed</div>
              <div className="text-2xl font-bold text-red-300">{summary.failureCount}</div>
            </div>
            <XCircle className="w-8 h-8 text-red-400" />
          </div>
        </div>

        <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-3">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs text-gray-400 uppercase font-semibold">Pending</div>
              <div className="text-2xl font-bold text-yellow-300">{summary.pendingCount}</div>
            </div>
            <Clock className="w-8 h-8 text-yellow-400" />
          </div>
        </div>

        <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-3">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs text-gray-400 uppercase font-semibold">Success Rate</div>
              <div className="text-2xl font-bold text-purple-300">{summary.successRate}%</div>
            </div>
            <TrendingUp className="w-8 h-8 text-purple-400" />
          </div>
        </div>
      </div>

      {/* Filters and Sort */}
      <div className="flex items-center gap-4 bg-gray-800/50 border border-gray-700 rounded-lg p-4">
        {/* Filter */}
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-gray-400" />
          <span className="text-sm text-gray-400">Filter:</span>
          <div className="flex gap-1">
            {(['all', 'agent', 'workflow', 'manual'] as ActionSource[]).map((source) => (
              <button
                key={source}
                onClick={() => setFilterSource(source)}
                className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                  filterSource === source
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                {source === 'all' && 'All'}
                {source === 'agent' && <><Bot className="w-3 h-3 inline mr-1" />Agent</>}
                {source === 'workflow' && <><Zap className="w-3 h-3 inline mr-1" />Workflow</>}
                {source === 'manual' && <><User className="w-3 h-3 inline mr-1" />Manual</>}
              </button>
            ))}
          </div>
        </div>

        <div className="h-6 w-px bg-gray-600"></div>

        {/* Sort */}
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-400">Sort:</span>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as SortBy)}
            className="bg-gray-700 text-gray-300 px-3 py-1 rounded text-sm border border-gray-600 focus:outline-none focus:border-blue-500"
          >
            <option value="newest">Newest First</option>
            <option value="oldest">Oldest First</option>
            <option value="status">By Status</option>
          </select>
        </div>
      </div>

      {/* Actions List */}
      <div className="space-y-3 max-h-[600px] overflow-y-auto pr-2">
        {unifiedActions.length === 0 ? (
          <div className="bg-gray-800/30 border border-gray-700/50 rounded-lg p-12 text-center">
            <AlertCircle className="w-12 h-12 text-gray-600 mx-auto mb-3" />
            <div className="text-gray-400">No actions found</div>
            <div className="text-sm text-gray-500 mt-1">
              {filterSource !== 'all'
                ? `No ${filterSource} actions for this incident`
                : 'No actions have been taken yet'}
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

      {/* Load More (for future pagination) */}
      {unifiedActions.length > 20 && (
        <div className="text-center pt-4">
          <button className="px-6 py-2 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-lg text-sm font-medium transition-colors">
            Load More Actions
          </button>
        </div>
      )}

      {/* Action Detail Modal */}
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
