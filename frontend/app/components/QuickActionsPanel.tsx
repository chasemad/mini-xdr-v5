"use client";

import { useState } from "react";
import {
  Ban, Shield, Key, Globe, Target, Loader2,
  AlertTriangle, CheckCircle, MessageSquare
} from "lucide-react";
import { socBlockIP, socIsolateHost, socResetPasswords, socThreatIntelLookup, socHuntSimilarAttacks } from "../lib/api";
import AIIncidentAnalysis from "./AIIncidentAnalysis";

interface Incident {
  id: number;
  created_at: string;
  src_ip: string;
  reason: string;
  status: string;
  auto_contained: boolean;
  triage_note?: {
    summary: string;
    severity: string;
    recommendation: string;
    rationale: string[];
  };
}

interface QuickActionsPanelProps {
  selectedIncident: Incident | null;
  onActionComplete?: () => void;
}

export default function QuickActionsPanel({ selectedIncident, onActionComplete }: QuickActionsPanelProps) {
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [showAIAnalysis, setShowAIAnalysis] = useState(false);
  const [toasts, setToasts] = useState<Array<{id: string, type: 'success' | 'error', message: string}>>([]);

  const showToast = (type: 'success' | 'error', message: string) => {
    const toast = { id: Date.now().toString(), type, message };
    setToasts(prev => [...prev, toast]);
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== toast.id));
    }, 5000);
  };

  const executeSOCAction = async (actionType: string, actionLabel: string, incidentId: number) => {
    setActionLoading(`${actionType}-${incidentId}`);
    try {
      switch (actionType) {
        case 'block_ip': await socBlockIP(incidentId); break;
        case 'isolate_host': await socIsolateHost(incidentId); break;
        case 'reset_passwords': await socResetPasswords(incidentId); break;
        case 'threat_intel_lookup': await socThreatIntelLookup(incidentId); break;
        case 'hunt_similar_attacks': await socHuntSimilarAttacks(incidentId); break;
        default: throw new Error(`Unknown action type: ${actionType}`);
      }

      showToast('success', `${actionLabel} completed successfully`);
      onActionComplete?.();

    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      showToast('error', `${actionLabel} failed: ${message}`);
    } finally {
      setActionLoading(null);
    }
  };

  if (!selectedIncident) {
    return (
      <div className="bg-surface-0 border border-border rounded-xl p-6">
        <div className="text-center text-text-muted">
          <AlertTriangle className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p className="text-sm">Select an incident to view actions</p>
        </div>
      </div>
    );
  }

  const getSeverityColor = (severity?: string) => {
    switch (severity) {
      case 'high': return 'text-red-400 bg-red-500/10';
      case 'medium': return 'text-orange-400 bg-orange-500/10';
      case 'low': return 'text-green-400 bg-green-500/10';
      default: return 'text-gray-400 bg-gray-500/10';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'open': return 'text-yellow-400 bg-yellow-500/10';
      case 'contained': return 'text-red-400 bg-red-500/10';
      case 'dismissed': return 'text-gray-400 bg-gray-500/10';
      default: return 'text-gray-400 bg-gray-500/10';
    }
  };

  return (
    <div className="space-y-6">
      {/* Toast Notifications */}
      <div className="fixed top-4 right-4 z-50 space-y-2 max-w-sm">
        {toasts.map(toast => (
          <div key={toast.id} className={`p-4 rounded-xl border shadow-lg animate-in slide-in-from-right duration-300 ${
            toast.type === 'success' ? 'bg-green-500/20 border-green-500 text-green-100' :
            'bg-red-500/20 border-red-500 text-red-100'
          }`}>
            <div className="text-sm font-medium">{toast.message}</div>
          </div>
        ))}
      </div>

      {/* Selected Incident Header */}
      <div className="bg-gradient-to-r from-danger/20 via-orange-600/20 to-blue-600/20 p-4 border border-border/50 rounded-xl">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-danger" />
            <span className="text-sm font-semibold text-text">INCIDENT #{selectedIncident.id}</span>
            <span className={`px-2 py-1 rounded-full text-xs font-semibold ${getSeverityColor(selectedIncident.triage_note?.severity)}`}>
              {selectedIncident.triage_note?.severity?.toUpperCase() || 'UNKNOWN'}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className={`px-2 py-1 rounded-full text-xs font-semibold ${getStatusColor(selectedIncident.status)}`}>
              {selectedIncident.status.toUpperCase()}
            </span>
            {selectedIncident.auto_contained && (
              <span className="px-2 py-1 rounded-full text-xs font-semibold bg-purple-500/20 text-purple-300">
                AUTO
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="bg-surface-0 border border-border rounded-xl p-6">
        <h3 className="text-lg font-semibold text-text mb-4">Quick Actions</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {[
            { action: 'block_ip', label: 'Block IP', icon: Ban, color: 'red' },
            { action: 'isolate_host', label: 'Isolate Host', icon: Shield, color: 'orange' },
            { action: 'reset_passwords', label: 'Reset Passwords', icon: Key, color: 'yellow' },
            { action: 'threat_intel_lookup', label: 'Threat Intel', icon: Globe, color: 'blue' },
            { action: 'hunt_similar_attacks', label: 'Hunt Similar', icon: Target, color: 'purple' }
          ].map(({ action, label, icon: Icon, color }) => (
            <button
              key={action}
              onClick={() => executeSOCAction(action, label, selectedIncident.id)}
              disabled={actionLoading === `${action}-${selectedIncident.id}`}
              className={`flex items-center gap-3 px-4 py-3 bg-${color}-600/20 hover:bg-${color}-600/30 border border-${color}-500/30 rounded-lg text-sm font-medium transition-all ${
                actionLoading === `${action}-${selectedIncident.id}` ? 'opacity-50 cursor-not-allowed' : 'hover:scale-105'
              }`}
            >
              {actionLoading === `${action}-${selectedIncident.id}` ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Icon className="w-4 h-4" />
              )}
              <span className="text-text">{label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* AI Analysis Toggle */}
      <div className="bg-surface-0 border border-border rounded-xl p-6">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-text">AI Analysis</h3>
            <p className="text-sm text-text-muted">Get AI-powered insights and recommendations</p>
          </div>
          <button
            onClick={() => setShowAIAnalysis(!showAIAnalysis)}
            className="flex items-center gap-2 px-4 py-2 bg-primary hover:bg-primary/90 text-bg rounded-lg text-sm font-medium transition-colors"
          >
            <MessageSquare className="w-4 h-4" />
            {showAIAnalysis ? 'Hide' : 'Analyze'}
          </button>
        </div>

        {showAIAnalysis && (
          <div className="mt-4">
            <AIIncidentAnalysis
              incident={selectedIncident}
              onRecommendationAction={(action) => executeSOCAction(action, `AI ${action}`, selectedIncident.id)}
            />
          </div>
        )}
      </div>
    </div>
  );
}
