"use client";

import { useEffect, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { 
  getIncident, agentOrchestrate,
  socBlockIP, socIsolateHost, socResetPasswords, socCheckDBIntegrity,
  socThreatIntelLookup, socDeployWAFRules, socCaptureTraffic,
  socHuntSimilarAttacks, socAlertAnalysts, socCreateCase
} from "@/app/lib/api";
import { 
  ArrowLeft, Shield, AlertTriangle, TrendingUp, Bot, Zap, 
  Send, Copy, BarChart3, Target, Crosshair, Search,
  Globe, Ban, Key, History, Loader2, Network, Database, Bell,
  Flag, MapPin, Code, Clock, AlertCircle
} from "lucide-react";

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
  actions: Array<{
    id: number;
    created_at: string;
    action: string;
    result: string;
    detail: string;
    params: Record<string, unknown>;
    due_at?: string;
  }>;
  detailed_events: Array<{
    id: number;
    ts: string;
    src_ip: string;
    dst_ip?: string;
    dst_port?: number;
    eventid: string;
    message: string;
    raw: Record<string, unknown>;
    source_type: string;
    hostname?: string;
  }>;
  iocs: {
    ip_addresses: string[];
    domains: string[];
    urls: string[];
    file_hashes: string[];
    user_agents: string[];
    sql_injection_patterns: string[];
    command_patterns: string[];
    file_paths: string[];
    privilege_escalation_indicators: string[];
    data_exfiltration_indicators: string[];
    persistence_mechanisms: string[];
    lateral_movement_indicators: string[];
    database_access_patterns: string[];
    successful_auth_indicators: string[];
    reconnaissance_patterns: string[];
  };
  attack_timeline: Array<{
    timestamp: string;
    event_id: string;
    description: string;
    source_ip: string;
    event_type: string;
    raw_data: Record<string, unknown>;
    severity: string;
    attack_category: string;
  }>;
  event_summary: {
    total_events: number;
    event_types: string[];
    time_span_hours: number;
  };
}

interface ChatMessage {
  id: string;
  type: 'user' | 'ai';
  content: string;
  timestamp: Date;
  loading?: boolean;
}

interface ToastMessage {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message: string;
}

export default function AnalystIncidentDetail({ params }: { params: Promise<{ id: string }> }) {
  const router = useRouter();
  const [id, setId] = useState<number | null>(null);
  const [incident, setIncident] = useState<IncidentDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [toasts, setToasts] = useState<ToastMessage[]>([]);
  const [activeTab, setActiveTab] = useState('overview');
  // Removed unused expandedSections state

  // Chat state
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      type: 'ai',
      content: "I'm analyzing this incident for you. I can help explain the attack patterns, assess the risk, recommend actions, or answer any specific questions you have about this threat.",
      timestamp: new Date()
    }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);

  useEffect(() => {
    params.then((resolvedParams) => {
      setId(Number(resolvedParams.id));
    });
  }, [params]);

  const showToast = useCallback((type: ToastMessage['type'], title: string, message: string) => {
    const toast: ToastMessage = {
      id: Date.now().toString(),
      type,
      title,
      message
    };
    setToasts(prev => [...prev, toast]);
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== toast.id));
    }, 5000);
  }, []);

  const fetchIncident = useCallback(async () => {
    if (!id) return;
    
    try {
      const data = await getIncident(id);
      setIncident(data);
    } catch (error: unknown) {
      console.error("Failed to fetch incident:", error);
      showToast('error', 'Failed to Load', 'Could not fetch incident details');
    } finally {
      setLoading(false);
    }
  }, [id, showToast]);

  useEffect(() => {
    fetchIncident();
  }, [fetchIncident]);

  const executeSOCAction = async (actionType: string, actionLabel: string) => {
    if (!incident?.id) return;
    
    setActionLoading(actionType);
    try {
      let result;
      
      switch (actionType) {
        case 'block_ip': result = await socBlockIP(incident.id); break;
        case 'isolate_host': result = await socIsolateHost(incident.id); break;
        case 'reset_passwords': result = await socResetPasswords(incident.id); break;
        case 'check_db_integrity': result = await socCheckDBIntegrity(incident.id); break;
        case 'threat_intel_lookup': result = await socThreatIntelLookup(incident.id); break;
        case 'deploy_waf_rules': result = await socDeployWAFRules(incident.id); break;
        case 'capture_traffic': result = await socCaptureTraffic(incident.id); break;
        case 'hunt_similar_attacks': result = await socHuntSimilarAttacks(incident.id); break;
        case 'alert_analysts': result = await socAlertAnalysts(incident.id); break;
        case 'create_case': result = await socCreateCase(incident.id); break;
        default: throw new Error(`Unknown action type: ${actionType}`);
      }
      
      showToast('success', 'Action Completed', result.message || `${actionLabel} completed successfully`);
      await fetchIncident();
      
    } catch (error: unknown) {
      console.error(`SOC action ${actionType} failed:`, error);
      const message = error instanceof Error ? error.message : `${actionLabel} failed`;
      showToast('error', 'Action Failed', message);
    } finally {
      setActionLoading(null);
    }
  };

  const sendChatMessage = async () => {
    if (!chatInput.trim() || chatLoading) return;
    
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: chatInput.trim(),
      timestamp: new Date()
    };
    
    setChatMessages(prev => [...prev, userMessage]);
    setChatInput('');
    setChatLoading(true);
    
    const loadingMessage: ChatMessage = {
      id: (Date.now() + 1).toString(),
      type: 'ai',
      content: '',
      timestamp: new Date(),
      loading: true
    };
    
    setChatMessages(prev => [...prev, loadingMessage]);
    
    try {
      const response = await agentOrchestrate(userMessage.content, incident?.id, {
        incident_data: incident,
        chat_history: chatMessages.slice(-5)
      });
      
      const aiMessage: ChatMessage = {
        id: (Date.now() + 2).toString(),
        type: 'ai',
        content: response.message || response.analysis || "I've analyzed your query but need more context to provide a specific response.",
        timestamp: new Date()
      };
      
      setChatMessages(prev => prev.slice(0, -1).concat(aiMessage));
      
    } catch {
      setChatMessages(prev => prev.slice(0, -1));
      showToast('error', 'AI Error', 'Failed to get AI response');
    } finally {
      setChatLoading(false);
    }
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

  const getRiskColor = (score?: number) => {
    if (!score) return 'text-gray-400';
    if (score >= 0.8) return 'text-red-400';
    if (score >= 0.6) return 'text-orange-400';
    if (score >= 0.4) return 'text-yellow-400';
    return 'text-green-400';
  };

  const getCompromiseStatus = (incident: IncidentDetail) => {
    if (incident.iocs?.successful_auth_indicators?.length > 0) return 'CONFIRMED';
    if (incident.iocs?.database_access_patterns?.length > 0 || 
        incident.iocs?.privilege_escalation_indicators?.length > 0) return 'SUSPECTED';
    return 'UNLIKELY';
  };

  if (loading || !id) {
    return (
      <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-400">Loading incident analysis...</p>
        </div>
      </div>
    );
  }

  if (!incident) {
    return (
      <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
        <div className="text-center">
          <AlertTriangle className="w-16 h-16 text-red-400 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-white mb-2">Incident Not Found</h3>
          <p className="text-gray-400 mb-6">The requested incident could not be found.</p>
          <button
                          onClick={() => router.push("/")}
              className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-medium transition-colors"
          >
            Back to Dashboard
          </button>
        </div>
      </div>
    );
  }

  const compromiseStatus = getCompromiseStatus(incident);

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Toast Notifications */}
      <div className="fixed top-4 right-4 z-50 space-y-2 max-w-sm">
        {toasts.map(toast => (
          <div key={toast.id} className={`p-4 rounded-xl border shadow-lg animate-in slide-in-from-right duration-300 ${
            toast.type === 'success' ? 'bg-green-500/20 border-green-500 text-green-100' :
            toast.type === 'error' ? 'bg-red-500/20 border-red-500 text-red-100' :
            toast.type === 'warning' ? 'bg-orange-500/20 border-orange-500 text-orange-100' :
            'bg-blue-500/20 border-blue-500 text-blue-100'
          }`}>
            <div className="font-semibold text-sm">{toast.title}</div>
            <div className="text-sm opacity-90 mt-1">{toast.message}</div>
          </div>
        ))}
      </div>

      <div className="flex h-screen">
        {/* Main Content */}
        <div className="flex-1 flex flex-col">
          {/* Header */}
          <div className="bg-gray-800/50 border-b border-gray-700/50 p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <button
                  onClick={() => router.push("/")}
                  className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
                >
                  <ArrowLeft className="w-5 h-5 text-gray-400" />
                </button>
                <div>
                  <h1 className="text-2xl font-bold text-white">🚨 Incident #{incident.id}</h1>
                  <div className="flex items-center gap-3 text-gray-400 text-sm">
                    <span>Source: <code className="text-orange-400 font-mono">{incident.src_ip}</code></span>
                    <span>•</span>
                    <span>{formatTimeAgo(incident.created_at)}</span>
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                  incident.status === 'open' ? 'bg-yellow-500/20 text-yellow-300' :
                  incident.status === 'contained' ? 'bg-red-500/20 text-red-300' :
                  'bg-gray-700 text-gray-300'
                }`}>
                  {incident.status.toUpperCase()}
                </span>
                {incident.auto_contained && (
                  <span className="px-3 py-1 rounded-full text-xs font-semibold bg-purple-500/20 text-purple-300">
                    AUTO
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* Tab Navigation */}
          <div className="bg-gray-800/30 border-b border-gray-700/50 px-4">
            <nav className="flex space-x-8">
              {[
                { id: 'overview', label: 'Overview', icon: BarChart3 },
                { id: 'timeline', label: 'Attack Timeline', icon: Clock },
                { id: 'iocs', label: 'IOCs & Evidence', icon: Flag },
                { id: 'forensics', label: 'Digital Forensics', icon: Search },
                { id: 'actions', label: 'Response Actions', icon: Shield }
              ].map(({ id, label, icon: Icon }) => (
                <button
                  key={id}
                  onClick={() => setActiveTab(id)}
                  className={`flex items-center gap-2 px-4 py-3 border-b-2 text-sm font-medium transition-colors ${
                    activeTab === id 
                      ? 'border-blue-500 text-blue-300' 
                      : 'border-transparent text-gray-400 hover:text-gray-300'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  {label}
                </button>
              ))}
            </nav>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto">
            <div className="p-6">
              {activeTab === 'overview' && (
                <div className="space-y-6">
                  {/* Critical Metrics */}
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    <div className="bg-gradient-to-br from-red-500/10 to-red-600/20 border border-red-500/30 rounded-xl p-6">
                      <div className="flex items-center justify-between mb-4">
                        <AlertTriangle className="w-8 h-8 text-red-400" />
                        <div className="text-right">
                          <div className={`text-3xl font-bold ${getRiskColor(incident.risk_score)}`}>
                            {incident.risk_score ? `${Math.round(incident.risk_score * 100)}%` : 'N/A'}
                          </div>
                          <div className="text-xs text-red-400/70">Risk Score</div>
                        </div>
                      </div>
                      <div className="bg-gray-700/50 rounded-full h-2">
                        <div 
                          className="bg-gradient-to-r from-red-500 to-red-600 h-2 rounded-full transition-all duration-1000"
                          style={{ width: `${(incident.risk_score || 0) * 100}%` }}
                        />
                      </div>
                    </div>

                    <div className="bg-gradient-to-br from-blue-500/10 to-blue-600/20 border border-blue-500/30 rounded-xl p-6">
                      <div className="flex items-center justify-between mb-4">
                        <TrendingUp className="w-8 h-8 text-blue-400" />
                        <div className="text-right">
                          <div className="text-3xl font-bold text-blue-300">
                            {incident.agent_confidence ? `${Math.round(incident.agent_confidence * 100)}%` : 'N/A'}
                          </div>
                          <div className="text-xs text-blue-400/70">ML Confidence</div>
                        </div>
                      </div>
                      <div className="bg-gray-700/50 rounded-full h-2">
                        <div 
                          className="bg-gradient-to-r from-blue-500 to-blue-600 h-2 rounded-full transition-all duration-1000"
                          style={{ width: `${(incident.agent_confidence || 0) * 100}%` }}
                        />
                      </div>
                    </div>

                    <div className="bg-gradient-to-br from-purple-500/10 to-purple-600/20 border border-purple-500/30 rounded-xl p-6">
                      <div className="flex items-center justify-between mb-4">
                        <Zap className="w-8 h-8 text-purple-400" />
                        <div className="text-right">
                          <div className="text-2xl font-bold text-purple-300 capitalize">
                            {incident.escalation_level || 'High'}
                          </div>
                          <div className="text-xs text-purple-400/70">Escalation Level</div>
                        </div>
                      </div>
                    </div>

                    <div className="bg-gradient-to-br from-green-500/10 to-green-600/20 border border-green-500/30 rounded-xl p-6">
                      <div className="flex items-center justify-between mb-4">
                        <Bot className="w-8 h-8 text-green-400" />
                        <div className="text-right">
                          <div className="text-lg font-bold text-green-300 capitalize">
                            {incident.containment_method || 'ML-driven'}
                          </div>
                          <div className="text-xs text-green-400/70">Detection Engine</div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Compromise Assessment */}
                  <div className={`relative overflow-hidden p-6 rounded-xl border backdrop-blur-sm ${
                    compromiseStatus === 'CONFIRMED' ? 'bg-red-500/10 border-red-500/30' :
                    compromiseStatus === 'SUSPECTED' ? 'bg-orange-500/10 border-orange-500/30' :
                    'bg-green-500/10 border-green-500/30'
                  }`}>
                    <div className="flex items-center justify-between mb-6">
                      <div className="flex items-center gap-3">
                        <AlertTriangle className="w-6 h-6" />
                        <h3 className="text-xl font-bold">Compromise Assessment: {compromiseStatus}</h3>
                      </div>
                      <div className={`px-4 py-2 rounded-full text-sm font-bold ${
                        compromiseStatus === 'SUSPECTED' ? 'bg-orange-500/20 text-orange-200' :
                        compromiseStatus === 'CONFIRMED' ? 'bg-red-500/20 text-red-200' :
                        'bg-green-500/20 text-green-200'
                      }`}>
                        {compromiseStatus === 'SUSPECTED' ? 'INVESTIGATE IMMEDIATELY' :
                         compromiseStatus === 'CONFIRMED' ? 'CRITICAL - TAKE ACTION' :
                         'MONITOR CLOSELY'}
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      {[
                        { key: 'successful_auth_indicators', label: 'Authentication', icon: '🔑', critical: true },
                        { key: 'database_access_patterns', label: 'Database Access', icon: '🗄️', critical: true },
                        { key: 'privilege_escalation_indicators', label: 'Privilege Escalation', icon: '⬆️', critical: true },
                        { key: 'data_exfiltration_indicators', label: 'Data Exfiltration', icon: '📤', critical: true }
                      ].map(({ key, label, icon, critical }) => {
                        const count = (incident.iocs as Record<string, string[]>)?.[key]?.length || 0;
                        const isActive = count > 0;
                        return (
                          <div key={key} className={`p-4 rounded-xl border text-center ${
                            isActive && critical ? 'bg-red-500/20 border-red-500/50 animate-pulse' : 
                            isActive ? 'bg-orange-500/20 border-orange-500/50' : 
                            'bg-gray-700/20 border-gray-600/30'
                          }`}>
                            <div className={`text-2xl mb-2 ${isActive ? '' : 'opacity-50'}`}>{icon}</div>
                            <div className="text-sm font-semibold">{label}</div>
                            <div className={`text-xl font-bold mt-2 ${
                              isActive && critical ? 'text-red-200' : 
                              isActive ? 'text-orange-200' : 'text-gray-400'
                            }`}>
                              {count}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Attack Details */}
                  <div className="bg-gray-800/50 border border-gray-700/50 rounded-xl p-6">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                      <Target className="w-5 h-5 text-red-400" />
                      Attack Analysis
                    </h3>
                    <div className="grid md:grid-cols-2 gap-6">
                      <div>
                        <div className="space-y-3">
                          <div className="flex items-center gap-3">
                            <MapPin className="w-4 h-4 text-orange-400" />
                            <span className="text-sm font-medium text-gray-300">Source IP:</span>
                            <span className="text-sm font-mono bg-orange-500/20 border border-orange-500/30 text-orange-200 px-3 py-1 rounded-lg">
                              {incident.src_ip}
                            </span>
                          </div>
                          <div className="flex items-center gap-3">
                            <Flag className="w-4 h-4 text-purple-400" />
                            <span className="text-sm font-medium text-gray-300">Threat Category:</span>
                            <span className="text-sm text-purple-200 capitalize">
                              {incident.threat_category?.replace('_', ' ') || 'Multi-vector Attack'}
                            </span>
                          </div>
                          <div className="flex items-center gap-3">
                            <Clock className="w-4 h-4 text-blue-400" />
                            <span className="text-sm font-medium text-gray-300">Duration:</span>
                            <span className="text-sm text-blue-200">
                              {incident.event_summary?.time_span_hours || 'Unknown'} hours
                            </span>
                          </div>
                        </div>
                      </div>
                      <div>
                        <div className="bg-gray-700/50 border border-gray-600/50 rounded-lg p-4">
                          <div className="flex items-center gap-2 mb-2">
                            <AlertCircle className="w-4 h-4 text-yellow-400" />
                            <span className="text-sm font-medium text-gray-300">Attack Summary:</span>
                          </div>
                          <p className="text-sm text-gray-300 leading-relaxed">{incident.reason}</p>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Quick Actions */}
                  <div className="bg-gray-800/50 border border-gray-700/50 rounded-xl p-6">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                      <Shield className="w-5 h-5 text-blue-400" />
                      Quick Response Actions
                    </h3>
                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
                      {[
                        { action: 'block_ip', label: 'Block IP', icon: Ban, color: 'red' },
                        { action: 'isolate_host', label: 'Isolate Host', icon: Shield, color: 'orange' },
                        { action: 'reset_passwords', label: 'Reset Passwords', icon: Key, color: 'yellow' },
                        { action: 'threat_intel_lookup', label: 'Threat Intel', icon: Globe, color: 'blue' },
                        { action: 'hunt_similar_attacks', label: 'Hunt Similar', icon: Crosshair, color: 'purple' }
                      ].map(({ action, label, icon: Icon, color }) => (
                        <button
                          key={action}
                          onClick={() => executeSOCAction(action, label)}
                          disabled={actionLoading === action}
                          className={`flex flex-col items-center gap-2 p-4 bg-${color}-600/10 hover:bg-${color}-600/20 border border-${color}-500/30 rounded-lg transition-all`}
                        >
                          {actionLoading === action ? (
                            <Loader2 className="w-5 h-5 animate-spin" />
                          ) : (
                            <Icon className="w-5 h-5" />
                          )}
                          <span className="text-xs font-medium">{label}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'timeline' && (
                <div className="space-y-6">
                  <div className="bg-gray-800/50 border border-gray-700/50 rounded-xl p-6">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                      <Clock className="w-5 h-5 text-blue-400" />
                      Attack Timeline ({incident.attack_timeline?.length || 0} events)
                    </h3>
                    
                    {incident.attack_timeline && incident.attack_timeline.length > 0 ? (
                      <div className="space-y-4">
                        {incident.attack_timeline.map((event, idx) => (
                          <div key={idx} className="flex gap-4 border-l-2 border-gray-600 pl-4 pb-4">
                            <div className={`w-3 h-3 rounded-full mt-2 flex-shrink-0 ${
                              event.severity === 'critical' ? 'bg-red-500' :
                              event.severity === 'high' ? 'bg-red-400' :
                              event.severity === 'medium' ? 'bg-yellow-400' :
                              'bg-gray-400'
                            }`}></div>
                            
                            <div className="flex-1">
                              <div className="flex items-center gap-3 mb-2">
                                <span className={`px-2 py-1 text-xs rounded font-medium ${
                                  event.attack_category === 'web_attack' ? 'bg-red-500/20 text-red-300' :
                                  event.attack_category === 'authentication' ? 'bg-yellow-500/20 text-yellow-300' :
                                  event.attack_category === 'reconnaissance' ? 'bg-blue-500/20 text-blue-300' :
                                  'bg-gray-700 text-gray-300'
                                }`}>
                                  {event.attack_category.replace('_', ' ').toUpperCase()}
                                </span>
                                <span className="text-xs text-gray-400">
                                  {formatTimeAgo(event.timestamp)}
                                </span>
                                <span className={`text-xs px-2 py-1 rounded ${
                                  event.severity === 'critical' ? 'bg-red-500/20 text-red-300' :
                                  event.severity === 'high' ? 'bg-red-400/20 text-red-300' :
                                  event.severity === 'medium' ? 'bg-yellow-400/20 text-yellow-300' :
                                  'bg-gray-400/20 text-gray-300'
                                }`}>
                                  {event.severity.toUpperCase()}
                                </span>
                              </div>
                              
                              <div className="text-sm text-gray-300 mb-2">
                                {event.description || 'Attack event detected'}
                              </div>
                              
                              <div className="text-xs text-gray-500 font-mono">
                                Event ID: {event.event_id}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="text-center py-8 text-gray-500">
                        <Clock className="w-12 h-12 mx-auto mb-2 opacity-50" />
                        <p>No timeline events available</p>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {activeTab === 'iocs' && (
                <div className="space-y-6">
                  <div className="grid md:grid-cols-2 gap-6">
                    {Object.entries(incident.iocs || {}).map(([category, indicators]) => {
                      if (!Array.isArray(indicators) || indicators.length === 0) return null;
                      
                      return (
                        <div key={category} className="bg-gray-800/50 border border-gray-700/50 rounded-xl p-6">
                          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                            <Flag className="w-5 h-5 text-orange-400" />
                            {category.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                            <span className="text-sm text-gray-400">({indicators.length})</span>
                          </h3>
                          
                          <div className="space-y-2 max-h-64 overflow-y-auto">
                            {indicators.slice(0, 10).map((indicator, idx) => (
                              <div key={idx} className="flex items-center gap-2 p-2 bg-gray-700/30 rounded-lg">
                                <Code className="w-4 h-4 text-gray-400 flex-shrink-0" />
                                <span className="text-sm font-mono text-gray-300 break-all flex-1">
                                  {indicator}
                                </span>
                                <button
                                  onClick={() => navigator.clipboard.writeText(indicator)}
                                  className="p-1 hover:bg-gray-600 rounded transition-colors"
                                >
                                  <Copy className="w-3 h-3 text-gray-400" />
                                </button>
                              </div>
                            ))}
                            {indicators.length > 10 && (
                              <div className="text-center py-2 text-sm text-gray-400">
                                And {indicators.length - 10} more indicators...
                              </div>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {activeTab === 'actions' && (
                <div className="space-y-6">
                  {/* Response Actions Grid */}
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {/* Immediate Response */}
                    <div className="bg-gradient-to-br from-red-500/10 to-red-600/20 border border-red-500/30 rounded-xl p-6">
                      <h3 className="text-lg font-semibold text-red-300 mb-4 flex items-center gap-2">
                        <Ban className="w-5 h-5" />
                        Immediate Response
                      </h3>
                      <div className="space-y-3">
                        {[
                          { action: 'block_ip', label: 'Block Source IP', icon: Ban },
                          { action: 'isolate_host', label: 'Isolate Affected Host', icon: Shield },
                          { action: 'reset_passwords', label: 'Force Password Reset', icon: Key }
                        ].map(({ action, label, icon: Icon }) => (
                          <button
                            key={action}
                            onClick={() => executeSOCAction(action, label)}
                            disabled={actionLoading === action}
                            className="w-full flex items-center gap-3 p-3 bg-red-600/10 hover:bg-red-600/20 border border-red-500/30 rounded-lg transition-all text-left"
                          >
                            {actionLoading === action ? (
                              <Loader2 className="w-4 h-4 animate-spin text-red-400" />
                            ) : (
                              <Icon className="w-4 h-4 text-red-400" />
                            )}
                            <span className="text-red-200 text-sm">{label}</span>
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Investigation */}
                    <div className="bg-gradient-to-br from-blue-500/10 to-blue-600/20 border border-blue-500/30 rounded-xl p-6">
                      <h3 className="text-lg font-semibold text-blue-300 mb-4 flex items-center gap-2">
                        <Search className="w-5 h-5" />
                        Investigation
                      </h3>
                      <div className="space-y-3">
                        {[
                          { action: 'threat_intel_lookup', label: 'Threat Intelligence Lookup', icon: Globe },
                          { action: 'hunt_similar_attacks', label: 'Hunt Similar Attacks', icon: Crosshair },
                          { action: 'capture_traffic', label: 'Capture Network Traffic', icon: Network }
                        ].map(({ action, label, icon: Icon }) => (
                          <button
                            key={action}
                            onClick={() => executeSOCAction(action, label)}
                            disabled={actionLoading === action}
                            className="w-full flex items-center gap-3 p-3 bg-blue-600/10 hover:bg-blue-600/20 border border-blue-500/30 rounded-lg transition-all text-left"
                          >
                            {actionLoading === action ? (
                              <Loader2 className="w-4 h-4 animate-spin text-blue-400" />
                            ) : (
                              <Icon className="w-4 h-4 text-blue-400" />
                            )}
                            <span className="text-blue-200 text-sm">{label}</span>
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* System Hardening */}
                    <div className="bg-gradient-to-br from-purple-500/10 to-purple-600/20 border border-purple-500/30 rounded-xl p-6">
                      <h3 className="text-lg font-semibold text-purple-300 mb-4 flex items-center gap-2">
                        <Shield className="w-5 h-5" />
                        System Hardening
                      </h3>
                      <div className="space-y-3">
                        {[
                          { action: 'deploy_waf_rules', label: 'Deploy WAF Rules', icon: Shield },
                          { action: 'check_db_integrity', label: 'Check Database Integrity', icon: Database },
                          { action: 'alert_analysts', label: 'Alert Senior Analysts', icon: Bell }
                        ].map(({ action, label, icon: Icon }) => (
                          <button
                            key={action}
                            onClick={() => executeSOCAction(action, label)}
                            disabled={actionLoading === action}
                            className="w-full flex items-center gap-3 p-3 bg-purple-600/10 hover:bg-purple-600/20 border border-purple-500/30 rounded-lg transition-all text-left"
                          >
                            {actionLoading === action ? (
                              <Loader2 className="w-4 h-4 animate-spin text-purple-400" />
                            ) : (
                              <Icon className="w-4 h-4 text-purple-400" />
                            )}
                            <span className="text-purple-200 text-sm">{label}</span>
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Action History */}
                  <div className="bg-gray-800/50 border border-gray-700/50 rounded-xl p-6">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                      <History className="w-5 h-5 text-green-400" />
                      Action History
                    </h3>
                    
                    {incident.actions && incident.actions.length > 0 ? (
                      <div className="space-y-3">
                        {incident.actions.map((action) => (
                          <div key={action.id} className="flex items-center gap-4 p-3 bg-gray-700/30 rounded-lg">
                            <div className={`w-2 h-2 rounded-full ${
                              action.result === 'success' ? 'bg-green-500' :
                              action.result === 'pending' ? 'bg-yellow-500' :
                              'bg-red-500'
                            }`}></div>
                            <div className="flex-1">
                              <div className="flex items-center gap-2">
                                <span className="text-sm font-medium text-white">{action.action}</span>
                                <span className={`text-xs px-2 py-1 rounded ${
                                  action.result === 'success' ? 'bg-green-500/20 text-green-300' :
                                  action.result === 'pending' ? 'bg-yellow-500/20 text-yellow-300' :
                                  'bg-red-500/20 text-red-300'
                                }`}>
                                  {action.result}
                                </span>
                              </div>
                              <p className="text-xs text-gray-400">{action.detail}</p>
                            </div>
                            <div className="text-xs text-gray-500">
                              {formatTimeAgo(action.created_at)}
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="text-center py-8 text-gray-500">
                        <History className="w-12 h-12 mx-auto mb-2 opacity-50" />
                        <p>No actions taken yet</p>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* AI Chat Sidebar */}
        <div className="w-96 bg-gray-800/50 border-l border-gray-700/50 flex flex-col">
          <div className="p-4 border-b border-gray-700/50">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-purple-600/20 rounded-lg">
                <Bot className="w-5 h-5 text-purple-400" />
              </div>
              <div>
                <h3 className="text-sm font-semibold text-white">AI Security Analyst</h3>
                <p className="text-xs text-gray-400">Incident #{incident.id} Analysis</p>
              </div>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {chatMessages.map((message) => (
              <div
                key={message.id}
                className={`flex gap-3 ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                {message.type === 'ai' && (
                  <div className="w-8 h-8 bg-purple-600 rounded-full flex items-center justify-center flex-shrink-0">
                    <Bot className="w-4 h-4 text-white" />
                  </div>
                )}
                
                <div className={`max-w-[80%] p-3 rounded-lg ${
                  message.type === 'user'
                    ? 'bg-blue-600/20 border border-blue-500/30 text-blue-100'
                    : 'bg-purple-600/20 border border-purple-500/30 text-purple-100'
                }`}>
                  {message.loading ? (
                    <div className="flex items-center gap-2">
                      <div className="animate-spin w-4 h-4 border-2 border-purple-400 border-t-transparent rounded-full"></div>
                      <span>Analyzing incident...</span>
                    </div>
                  ) : (
                    <div className="text-sm whitespace-pre-wrap">{message.content}</div>
                  )}
                </div>
              </div>
            ))}
          </div>

          <div className="p-4 border-t border-gray-700/50">
            <div className="flex gap-2">
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && sendChatMessage()}
                placeholder="Ask about this incident..."
                className="flex-1 bg-gray-700/50 border border-gray-600/50 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-400 focus:outline-none focus:ring-1 focus:ring-purple-500/50"
                disabled={chatLoading}
              />
              <button
                onClick={sendChatMessage}
                disabled={!chatInput.trim() || chatLoading}
                className="px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-purple-800 disabled:opacity-50 text-white rounded-lg text-sm transition-colors"
              >
                {chatLoading ? (
                  <div className="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full"></div>
                ) : (
                  <Send className="w-4 h-4" />
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
