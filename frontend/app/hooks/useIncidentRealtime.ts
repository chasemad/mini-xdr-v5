/**
 * useIncidentRealtime Hook
 * Manages real-time updates for incident data via WebSocket
 */

import { useEffect, useState, useCallback } from 'react';
import { useIncidentWebSocket } from './useWebSocket';

interface IncidentUpdate {
  type: 'status_change' | 'new_action' | 'action_complete' | 'action_failed' | 'agent_action' | 'full_update';
  incidentId: number;
  data: any;
  timestamp: number;
}

interface UseIncidentRealtimeOptions {
  incidentId: number;
  onUpdate?: (incident: any) => void;
  onNewAction?: (action: any) => void;
  onActionComplete?: (action: any) => void;
  onStatusChange?: (status: string) => void;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

export function useIncidentRealtime({
  incidentId,
  onUpdate,
  onNewAction,
  onActionComplete,
  onStatusChange,
  autoRefresh = true,
  refreshInterval = 5000
}: UseIncidentRealtimeOptions) {
  const [incident, setIncident] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'connecting'>('disconnected');

  // WebSocket connection for real-time updates
  const { isConnected, sendMessage, error: wsError } = useIncidentWebSocket({
    onMessage: (message) => {
      handleWebSocketMessage(message);
    },
    onConnect: () => {
      setConnectionStatus('connected');
      // Subscribe to incident updates
      sendMessage({
        type: 'subscribe',
        incidentId: incidentId
      });
    },
    onDisconnect: () => {
      setConnectionStatus('disconnected');
    }
  });

  const handleWebSocketMessage = useCallback((message: any) => {
    if (!message.data || message.data.incidentId !== incidentId) return;

    const update: IncidentUpdate = {
      type: message.type,
      incidentId: message.data.incidentId,
      data: message.data,
      timestamp: message.timestamp || Date.now()
    };

    setLastUpdate(new Date(update.timestamp));

    switch (update.type) {
      case 'new_action':
        onNewAction?.(update.data.action);
        // Refresh full incident to get updated action list
        refreshIncident();
        break;

      case 'action_complete':
        onActionComplete?.(update.data.action);
        refreshIncident();
        break;

      case 'action_failed':
        console.error('Action failed:', update.data.action);
        refreshIncident();
        break;

      case 'agent_action':
        onNewAction?.(update.data.action);
        refreshIncident();
        break;

      case 'status_change':
        onStatusChange?.(update.data.status);
        refreshIncident();
        break;

      case 'full_update':
        setIncident(update.data.incident);
        onUpdate?.(update.data.incident);
        break;

      default:
        console.log('Unknown update type:', update.type);
    }
  }, [incidentId, onUpdate, onNewAction, onActionComplete, onStatusChange]);

  // Fetch incident data
  const fetchIncident = useCallback(async () => {
    try {
      setLoading(true);
      const response = await fetch(`http://localhost:8000/incidents/${incidentId}`, {
        headers: {
          'x-api-key': process.env.NEXT_PUBLIC_API_KEY || 'demo-minixdr-api-key'
        }
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch incident: ${response.statusText}`);
      }

      const data = await response.json();
      setIncident(data);
      setLastUpdate(new Date());
      onUpdate?.(data);
      return data;
    } catch (err) {
      console.error('Failed to fetch incident:', err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [incidentId, onUpdate]);

  // Refresh incident data
  const refreshIncident = useCallback(async () => {
    return await fetchIncident();
  }, [fetchIncident]);

  // Initial load
  useEffect(() => {
    fetchIncident();
  }, [incidentId]);

  // Auto-refresh polling (fallback if WebSocket fails)
  useEffect(() => {
    if (!autoRefresh || isConnected) return;

    const interval = setInterval(() => {
      fetchIncident();
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval, isConnected, fetchIncident]);

  // Update connection status
  useEffect(() => {
    if (isConnected) {
      setConnectionStatus('connected');
    } else if (wsError) {
      setConnectionStatus('disconnected');
    }
  }, [isConnected, wsError]);

  return {
    incident,
    loading,
    lastUpdate,
    connectionStatus,
    isConnected,
    refreshIncident,
    wsError
  };
}

/**
 * Hook for monitoring action status changes
 */
export function useActionRealtime(incidentId: number, actionId?: string) {
  const [actionStatus, setActionStatus] = useState<string | null>(null);
  const [actionResult, setActionResult] = useState<any>(null);

  const { sendMessage } = useIncidentWebSocket({
    onMessage: (message) => {
      if (message.type === 'action_complete' || message.type === 'action_failed') {
        if (!actionId || message.data.action?.id === actionId) {
          setActionStatus(message.data.action?.status);
          setActionResult(message.data.action?.result);
        }
      }
    }
  });

  useEffect(() => {
    if (actionId) {
      sendMessage({
        type: 'subscribe_action',
        incidentId,
        actionId
      });
    }
  }, [incidentId, actionId, sendMessage]);

  return {
    actionStatus,
    actionResult
  };
}

