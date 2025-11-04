"use client";

import React, { useEffect, useState, useCallback, use } from "react";
import { useRouter } from "next/navigation";
import { ArrowLeft, Loader2 } from "lucide-react";
import ThreatStatusBar from "@/components/ThreatStatusBar";
import EnhancedAIAnalysis from "@/components/EnhancedAIAnalysis";
import UnifiedResponseTimeline from "@/components/UnifiedResponseTimeline";
import TacticalDecisionCenter from "@/components/TacticalDecisionCenter";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useIncidentRealtime } from "@/app/hooks/useIncidentRealtime";
import { apiUrl } from "@/app/utils/api";

interface IncidentDetail {
  id: number;
  created_at: string;
  src_ip: string;
  reason: string;
  status: string;
  auto_contained: boolean;
  escalation_level?: string;
  risk_score?: number;
  threat_category?: string;
  containment_confidence?: number;
  containment_method?: string;
  agent_id?: string;
  agent_actions?: Array<{action: string; status: string}>;
  agent_confidence?: number;
  ml_features?: Record<string, unknown>;
  ensemble_scores?: Record<string, number>;
  triage_note?: {
    summary: string;
    severity: string;
    recommendation: string;
    rationale: string[];
  };
  actions: Array<any>;
  advanced_actions?: Array<any>;
  detailed_events: Array<any>;
  iocs: {
    ip_addresses: string[];
    domains: string[];
    hashes: string[];
  };
}

export default function EnterpriseIncidentPage({ params }: { params: Promise<{ id: string }> }) {
  const router = useRouter();
  // Unwrap params using React.use() for Next.js 15
  const { id } = use(params);
  const incidentId = parseInt(id);

  // Use real-time hook for incident data
  const {
    incident,
    loading: realtimeLoading,
    refreshIncident,
    connectionStatus,
    lastUpdate
  } = useIncidentRealtime({
    incidentId,
    autoRefresh: true,
    refreshInterval: 5000
  });

  const [executing, setExecuting] = useState(false);

  // Fetch initial incident data (hook will handle updates)
  const fetchIncident = useCallback(async () => {
    return await refreshIncident();
  }, [refreshIncident]);

  useEffect(() => {
    fetchIncident();
  }, [incidentId]);

  // Execute AI recommendation
  const handleExecuteRecommendation = async (action: string, params?: Record<string, any>) => {
    try {
      setExecuting(true);

      const response = await fetch(
        apiUrl(`/api/incidents/${incidentId}/execute-ai-recommendation`),
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'x-api-key': process.env.NEXT_PUBLIC_API_KEY || 'demo-minixdr-api-key'
          },
          body: JSON.stringify({
            action_type: action,
            parameters: params || {}
          })
        }
      );

      const data = await response.json();

      if (data.success) {
        // Refresh incident to show new action
        await fetchIncident();
        // Show success notification (you can add a toast here)
        console.log('Action executed successfully:', data);
      } else {
        throw new Error(data.error || 'Execution failed');
      }
    } catch (err) {
      console.error('Failed to execute recommendation:', err);
      alert(`Failed to execute action: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setExecuting(false);
    }
  };

  // Execute all AI recommendations
  const handleExecuteAllRecommendations = async () => {
    if (!confirm('Execute all AI-recommended priority actions?\n\nThis will create a workflow with multiple automated responses.')) {
      return;
    }

    try {
      setExecuting(true);

      const response = await fetch(
        apiUrl(`/api/incidents/${incidentId}/execute-ai-plan`),
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'x-api-key': process.env.NEXT_PUBLIC_API_KEY || 'demo-minixdr-api-key'
          }
        }
      );

      const data = await response.json();

      if (data.success) {
        await fetchIncident();
        alert(`AI Plan Executed!\n\n${data.successful_actions} actions succeeded\n${data.failed_actions} actions failed`);
      } else {
        throw new Error(data.error || 'Execution failed');
      }
    } catch (err) {
      console.error('Failed to execute AI plan:', err);
      alert(`Failed to execute AI plan: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setExecuting(false);
    }
  };

  // Tactical Decision Center handlers
  const handleContainNow = async () => {
    if (!incident) return;
    await handleExecuteRecommendation('block_ip', { ip: incident.src_ip, duration: 30 });
  };

  const handleHuntThreats = async () => {
    if (!incident) return;
    await handleExecuteRecommendation('hunt_similar_attacks', {});
  };

  const handleEscalate = async () => {
    alert('Escalation feature - SOC team will be notified');
    // Implement escalation logic
  };

  const handleCreatePlaybook = async () => {
    alert('Playbook creation feature - converting this response into a reusable playbook');
    // Implement playbook creation
  };

  const handleGenerateReport = async () => {
    alert('Report generation feature - comprehensive incident report will be generated');
    // Implement report generation
  };

  const handleAskAI = async () => {
    alert('AI Assistant feature - interactive AI chat about this incident');
    // Implement AI chat
  };

  // Rollback handler
  const handleRollback = async (rollbackId: string) => {
    try {
      const response = await fetch(
        apiUrl(`/api/agents/actions/rollback/${rollbackId}`),
        {
          method: 'POST',
          headers: {
            'x-api-key': process.env.NEXT_PUBLIC_API_KEY || 'demo-minixdr-api-key'
          }
        }
      );

      if (response.ok) {
        await fetchIncident();
      } else {
        throw new Error('Rollback failed');
      }
    } catch (err) {
      console.error('Rollback failed:', err);
      alert('Failed to rollback action');
    }
  };

  if (realtimeLoading && !incident) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-900 to-black flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 text-blue-400 animate-spin mx-auto mb-4" />
          <p className="text-gray-400">Loading incident details...</p>
        </div>
      </div>
    );
  }

  if (!incident) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-900 to-black flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-400 text-lg mb-4">Incident not found</p>
          <button
            onClick={() => router.push('/incidents')}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg"
          >
            Back to Incidents
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-900 to-black text-white">
      {/* Page Header */}
      <div className="border-b border-gray-800 bg-gray-900/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-[1600px] mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <button
                onClick={() => router.push('/incidents')}
                className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
              </button>
              <div>
                <h1 className="text-2xl font-bold">Incident #{incident.id}</h1>
                <div className="flex items-center gap-3 text-sm text-gray-400 mt-1">
                  <span className="font-mono">{incident.src_ip}</span>
                  <span>•</span>
                  <span>{new Date(incident.created_at).toLocaleString()}</span>
                  <span>•</span>
                  <span className={`px-2 py-0.5 rounded text-xs font-semibold ${
                    incident.status === 'open' ? 'bg-red-500/20 text-red-300' :
                    incident.status === 'investigating' ? 'bg-yellow-500/20 text-yellow-300' :
                    incident.status === 'contained' ? 'bg-green-500/20 text-green-300' :
                    'bg-gray-500/20 text-gray-300'
                  }`}>
                    {incident.status.toUpperCase()}
                  </span>
                </div>
              </div>
            </div>

            {/* Connection Status */}
            <div className="flex items-center gap-3 text-sm">
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${
                  connectionStatus === 'connected' ? 'bg-green-400 animate-pulse' :
                  connectionStatus === 'connecting' ? 'bg-yellow-400 animate-pulse' :
                  'bg-red-400'
                }`}></div>
                <span className="text-gray-400">
                  {connectionStatus === 'connected' ? 'Live Updates' :
                   connectionStatus === 'connecting' ? 'Connecting...' :
                   'Disconnected'}
                </span>
              </div>
              {lastUpdate && (
                <span className="text-gray-500 text-xs">
                  Updated {new Date(lastUpdate).toLocaleTimeString()}
                </span>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-[1600px] mx-auto px-6 py-6 space-y-6">
        {/* Threat Status Bar - Hero Section */}
        <ThreatStatusBar incident={incident} />

        {/* Main Content - 2 Column Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* LEFT COLUMN: AI Analysis */}
          <div className="space-y-6">
            <EnhancedAIAnalysis
              incident={incident}
              onExecuteRecommendation={handleExecuteRecommendation}
              onExecuteAllRecommendations={handleExecuteAllRecommendations}
            />
          </div>

          {/* RIGHT COLUMN: Response Timeline */}
          <div className="space-y-6">
            <UnifiedResponseTimeline
              incidentId={incident.id}
              actions={incident.actions}
              automatedActions={incident.advanced_actions}
              onRefresh={fetchIncident}
              onRollback={handleRollback}
              incidentEvents={incident.detailed_events}
            />
          </div>
        </div>

        {/* Tactical Decision Center */}
        <TacticalDecisionCenter
          incidentId={incident.id}
          onContainNow={handleContainNow}
          onHuntThreats={handleHuntThreats}
          onEscalate={handleEscalate}
          onCreatePlaybook={handleCreatePlaybook}
          onGenerateReport={handleGenerateReport}
          onAskAI={handleAskAI}
        />

        {/* Detailed Tabs Section */}
        <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-6">
          <Tabs defaultValue="timeline" className="w-full">
            <TabsList className="bg-gray-900 border-gray-700 mb-6">
              <TabsTrigger value="timeline">Attack Timeline</TabsTrigger>
              <TabsTrigger value="iocs">IOCs & Evidence</TabsTrigger>
              <TabsTrigger value="ml">ML Analysis</TabsTrigger>
              <TabsTrigger value="forensics">Forensics</TabsTrigger>
            </TabsList>

            <TabsContent value="timeline">
              <div className="space-y-3">
                <h3 className="text-lg font-semibold text-white mb-4">Event Timeline</h3>
                {incident.detailed_events && incident.detailed_events.length > 0 ? (
                  <div className="space-y-2 max-h-[400px] overflow-y-auto">
                    {incident.detailed_events.slice(0, 20).map((event: any, idx: number) => (
                      <div
                        key={idx}
                        className="bg-gray-900/50 border border-gray-700/50 rounded p-3 hover:border-gray-600/50 transition-colors"
                      >
                        <div className="flex items-start justify-between mb-2">
                          <span className="text-xs font-mono text-blue-300">{event.eventid}</span>
                          <span className="text-xs text-gray-500">
                            {new Date(event.ts).toLocaleString()}
                          </span>
                        </div>
                        <p className="text-sm text-gray-300 mb-2">{event.message}</p>
                        <div className="flex items-center gap-3 text-xs text-gray-500">
                          <span>Source: {event.src_ip}</span>
                          {event.dst_ip && <span>→ Dest: {event.dst_ip}</span>}
                          {event.dst_port && <span>Port: {event.dst_port}</span>}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    No detailed events available
                  </div>
                )}
              </div>
            </TabsContent>

            <TabsContent value="iocs">
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-white mb-4">Indicators of Compromise</h3>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
                    <h4 className="text-sm font-semibold text-red-300 mb-2">IP Addresses</h4>
                    <div className="space-y-1">
                      {incident.iocs?.ip_addresses?.length > 0 ? (
                        incident.iocs.ip_addresses.map((ip: string, idx: number) => (
                          <div key={idx} className="text-sm font-mono text-gray-300">{ip}</div>
                        ))
                      ) : (
                        <div className="text-sm text-gray-500">None detected</div>
                      )}
                    </div>
                  </div>

                  <div className="bg-orange-500/10 border border-orange-500/30 rounded-lg p-4">
                    <h4 className="text-sm font-semibold text-orange-300 mb-2">Domains</h4>
                    <div className="space-y-1">
                      {incident.iocs?.domains?.length > 0 ? (
                        incident.iocs.domains.map((domain: string, idx: number) => (
                          <div key={idx} className="text-sm font-mono text-gray-300">{domain}</div>
                        ))
                      ) : (
                        <div className="text-sm text-gray-500">None detected</div>
                      )}
                    </div>
                  </div>

                  <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                    <h4 className="text-sm font-semibold text-purple-300 mb-2">File Hashes</h4>
                    <div className="space-y-1">
                      {incident.iocs?.hashes?.length > 0 ? (
                        incident.iocs.hashes.map((hash: string, idx: number) => (
                          <div key={idx} className="text-xs font-mono text-gray-300 truncate" title={hash}>
                            {hash}
                          </div>
                        ))
                      ) : (
                        <div className="text-sm text-gray-500">None detected</div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="ml">
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-white mb-4">Machine Learning Analysis</h3>

                {incident.ensemble_scores && Object.keys(incident.ensemble_scores).length > 0 && (
                  <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                    <h4 className="text-sm font-semibold text-purple-300 mb-3">Ensemble Model Scores</h4>
                    <div className="space-y-2">
                      {Object.entries(incident.ensemble_scores).map(([model, score]: [string, any]) => (
                        <div key={model} className="flex items-center justify-between">
                          <span className="text-sm text-gray-300">{model}</span>
                          <span className="text-sm font-mono text-purple-300">
                            {(typeof score === 'number' ? score * 100 : 0).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {incident.ml_features && Object.keys(incident.ml_features).length > 0 && (
                  <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
                    <h4 className="text-sm font-semibold text-blue-300 mb-3">ML Features</h4>
                    <div className="space-y-1 text-xs font-mono text-gray-300">
                      {Object.entries(incident.ml_features).slice(0, 10).map(([key, value]: [string, any]) => (
                        <div key={key} className="flex justify-between">
                          <span>{key}:</span>
                          <span>{JSON.stringify(value)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </TabsContent>

            <TabsContent value="forensics">
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-white mb-4">Forensic Analysis</h3>
                <div className="text-center py-8 text-gray-500">
                  Forensic analysis tools coming soon
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </div>

      {/* Executing Overlay */}
      {executing && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-gray-800 border border-gray-700 rounded-xl p-8 text-center">
            <Loader2 className="w-12 h-12 text-blue-400 animate-spin mx-auto mb-4" />
            <p className="text-white font-semibold">Executing action...</p>
            <p className="text-sm text-gray-400 mt-2">Please wait</p>
          </div>
        </div>
      )}
    </div>
  );
}
