"use client";

import { useEffect, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { 
  getIncidents, agentOrchestrate,
  socBlockIP, socIsolateHost, socResetPasswords, socThreatIntelLookup, socHuntSimilarAttacks
} from "./lib/api";
import Link from "next/link";
import {
  Shield, AlertTriangle, Bot, Zap,
  Search, Filter, RefreshCw, Settings, Bell, User,
  ChevronDown, ChevronRight, Eye, MessageSquare,
  BarChart3, Activity, Target, Globe,
  ArrowUpRight, ArrowDownRight, Minus, Ban, Key, Loader2, Workflow, ArrowRight, Database, CheckCircle
} from "lucide-react";
import { useAuth } from "./contexts/AuthContext";
import { DashboardLayout } from "../components/DashboardLayout";
import { ActionButton } from "../components/ui/ActionButton";
import { StatusChip } from "../components/ui/StatusChip";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Incident {
  id: number;
  created_at: string;
  src_ip: string;
  reason: string;
  status: string;
  auto_contained: boolean;
  risk_score?: number;
  agent_confidence?: number;
  escalation_level?: string;
  containment_method?: string;
  threat_category?: string;
  triage_note?: {
    summary: string;
    severity: string;
    recommendation: string;
    rationale: string[];
  };
}

interface ChatMessage {
  id: string;
  type: 'user' | 'ai';
  content: string;
  timestamp: Date;
  loading?: boolean;
}

interface SystemMetrics {
  total_incidents: number;
  high_priority: number;
  contained: number;
  ml_detected: number;
  avg_response_time: number;
  threat_intel_hits: number;
}

export default function SOCAnalystDashboard() {
  // Auth and telemetry gating
  const router = useRouter();
  const { user, organization, loading: authLoading } = useAuth();
  const [telemetry, setTelemetry] = useState<{ hasLogs: boolean; lastEventAt?: string; assetsDiscovered?: number; agentsEnrolled?: number; incidents?: number } | null>(null);

  // Redirect to login if not authenticated
  useEffect(() => {
    if (!authLoading && !user) {
      console.log('⚠️ User not authenticated, redirecting to login...');
      router.push('/login');
    }
  }, [authLoading, user, router]);

  useEffect(() => {
    const fetchTelemetry = async () => {
      try {
        const token = typeof window !== 'undefined' ? localStorage.getItem('access_token') : null;
        if (!token) return;
        const res = await fetch(`${API_BASE}/api/telemetry/status`, { headers: { Authorization: `Bearer ${token}` } });
        if (res.ok) setTelemetry(await res.json());
      } catch {}
    };
    fetchTelemetry();
  }, []);
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [loading, setLoading] = useState(true);
  const [autoRefreshing, setAutoRefreshing] = useState(false);
  const [selectedIncident, setSelectedIncident] = useState<Incident | null>(null);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      type: 'ai',
      content: "Welcome to the SOC Command Center. I'm your AI analyst assistant. I can help you investigate incidents, analyze threats, and coordinate response actions. What would you like to explore?",
      timestamp: new Date()
    }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');
  const [filterSeverity, setFilterSeverity] = useState('all');
  const [filterStatus, setFilterStatus] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [toasts, setToasts] = useState<Array<{id: string, type: 'success' | 'error', message: string}>>([]);

  // Computed metrics
  const metrics: SystemMetrics = {
    total_incidents: incidents.length,
    high_priority: incidents.filter(i => i.triage_note?.severity === 'high' || (i.risk_score && i.risk_score > 0.7)).length,
    contained: incidents.filter(i => i.status === 'contained').length,
    ml_detected: incidents.filter(i => i.auto_contained).length,
    avg_response_time: 4.2, // minutes
    threat_intel_hits: incidents.filter(i => i.escalation_level === 'high').length
  };

  // Fetch incidents
  const fetchIncidents = useCallback(async () => {
    try {
      const data = await getIncidents();
      setIncidents(data);
    } catch (error) {
      console.error("Failed to fetch incidents:", error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchIncidents();

    // Auto-refresh every 10 seconds
    const interval = setInterval(async () => {
      if (!loading) {
        setAutoRefreshing(true);
        await fetchIncidents();
        setAutoRefreshing(false);
      }
    }, 10000);

    return () => clearInterval(interval);
  }, [fetchIncidents, loading]);

  // Toast notifications
  const showToast = useCallback((type: 'success' | 'error', message: string) => {
    const toast = { id: Date.now().toString(), type, message };
    setToasts(prev => [...prev, toast]);
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== toast.id));
    }, 5000);
  }, []);

  // SOC Actions
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
      await fetchIncidents(); // Refresh incidents
      
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      showToast('error', `${actionLabel} failed: ${message}`);
    } finally {
      setActionLoading(null);
    }
  };

  // Chat functionality
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
    
    // Add loading message
    const loadingMessage: ChatMessage = {
      id: (Date.now() + 1).toString(),
      type: 'ai',
      content: '',
      timestamp: new Date(),
      loading: true
    };
    
    setChatMessages(prev => [...prev, loadingMessage]);
    
    try {
      const response = await agentOrchestrate(userMessage.content, selectedIncident?.id, {
        incident_data: selectedIncident,
        chat_history: chatMessages.slice(-5)
      });
      
      const aiMessage: ChatMessage = {
        id: (Date.now() + 2).toString(),
        type: 'ai',
        content: response.message || response.analysis || "I've analyzed your query. How can I help further?",
        timestamp: new Date()
      };
      
      setChatMessages(prev => prev.slice(0, -1).concat(aiMessage));
      
    } catch (error) {
      setChatMessages(prev => prev.slice(0, -1));
      console.error('AI response failed:', error);
    } finally {
      setChatLoading(false);
    }
  };

  // Filter incidents
  const filteredIncidents = incidents.filter(incident => {
    if (filterSeverity !== 'all' && incident.triage_note?.severity !== filterSeverity) return false;
    if (filterStatus !== 'all' && incident.status !== filterStatus) return false;
    if (searchQuery && !incident.src_ip.includes(searchQuery) && !incident.reason.toLowerCase().includes(searchQuery.toLowerCase())) return false;
    return true;
  });

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

  // Show loading only if authenticated or still checking auth
  if (authLoading || (!authLoading && user && loading)) {
    return (
      <div className="min-h-screen bg-gray-950 text-white flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-400">Initializing SOC Command Center...</p>
        </div>
      </div>
    );
  }

  // If not authenticated and not loading, return null (will redirect to login)
  if (!authLoading && !user) {
    return null;
  }

  // Check if onboarding is incomplete - show banner instead of blocking
  const needsOnboarding = !authLoading && organization && (!organization.onboarding_status || organization.onboarding_status === 'not_started' || organization.onboarding_status === 'in_progress');

  return (
    <div className="min-h-screen bg-gray-950 text-white flex">
      {/* Onboarding Banner */}
      {needsOnboarding && (
        <div className="fixed top-0 left-0 right-0 z-50 bg-gradient-to-r from-amber-600 to-orange-600 px-6 py-3 flex items-center justify-between shadow-lg">
          <div className="flex items-center gap-3">
            <Database className="w-5 h-5 text-white" />
            <div>
              <p className="text-white font-semibold">Complete your setup to unlock full functionality</p>
              <p className="text-amber-100 text-sm">Network discovery and agent deployment pending</p>
            </div>
          </div>
          <Link href="/onboarding" className="inline-flex items-center gap-2 px-4 py-2 bg-white text-amber-700 rounded-lg font-medium hover:bg-amber-50 transition-colors">
            <CheckCircle className="w-4 h-4" />
            Complete Setup
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      )}
      
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
      {/* Sidebar */}
      <div className={`${sidebarCollapsed ? 'w-16' : 'w-80'} bg-gray-900 border-r border-gray-800 transition-all duration-300 flex flex-col`}>
        {/* Header */}
        <div className="p-4 border-b border-gray-800">
          <div className="flex items-center justify-between">
            {!sidebarCollapsed && (
              <div>
                <h1 className="text-xl font-bold text-white">SOC Command</h1>
                <p className="text-xs text-gray-400">Enterprise Security Center</p>
              </div>
            )}
            <button
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
            >
              {sidebarCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>
          </div>
        </div>

        {!sidebarCollapsed && (
          <>
            {/* Navigation */}
            <div className="p-4">
              <nav className="space-y-2">
                {[
                  { id: 'overview', label: 'Threat Overview', icon: BarChart3, isTab: true },
                  { id: 'incidents', label: 'Active Incidents', icon: AlertTriangle, isTab: true },
                  { id: 'intelligence', label: 'Threat Intel', icon: Globe, isTab: true, href: '/intelligence' },
                  { id: 'hunting', label: 'Threat Hunting', icon: Target, href: '/hunt' },
                  { id: 'forensics', label: 'Forensics', icon: Search, href: '/investigations' },
                  { id: 'response', label: 'Response Actions', icon: Shield, isTab: true },
                  { id: 'workflows', label: 'Workflow Automation', icon: Workflow, href: '/workflows' },
                  { id: 'visualizations', label: '3D Visualization', icon: Activity, href: '/visualizations' }
                ].map(({ id, label, icon: Icon, isTab, href }) => {
                  if (href) {
                    return (
                      <Link
                        key={id}
                        href={href}
                        className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left transition-colors hover:bg-gray-700/50 text-gray-300 hover:text-white"
                      >
                        <Icon className="w-4 h-4" />
                        <span className="text-sm font-medium">{label}</span>
                      </Link>
                    );
                  }
                  
                  return (
                    <button
                      key={id}
                      onClick={() => setActiveTab(id)}
                      className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left transition-colors ${
                        activeTab === id ? 'bg-blue-600/20 text-blue-300 border border-blue-500/30' : 'hover:bg-gray-700/50 text-gray-300'
                      }`}
                    >
                      <Icon className="w-4 h-4" />
                      <span className="text-sm font-medium">{label}</span>
                    </button>
                  );
                })}
              </nav>
            </div>

            {/* Quick Stats */}
            <div className="p-4 border-t border-gray-800">
              <h3 className="text-sm font-semibold text-gray-300 mb-3">System Status</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-400">Active Threats</span>
                  <span className="text-sm font-bold text-red-400">{metrics.high_priority}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-400">Contained</span>
                  <span className="text-sm font-bold text-green-400">{metrics.contained}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-400">AI Detected</span>
                  <span className="text-sm font-bold text-blue-400">{metrics.ml_detected}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-400">Avg Response</span>
                  <span className="text-sm font-bold text-purple-400">{metrics.avg_response_time}m</span>
                </div>
              </div>
            </div>

            {/* AI Agents & MCP Status */}
            <div className="p-4 border-t border-gray-800">
              <div className="flex items-center gap-2 mb-3">
                <Bot className="w-4 h-4 text-purple-400" />
                <h3 className="text-sm font-semibold text-gray-300">AI Agent Orchestra</h3>
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between p-2 bg-gray-700/30 rounded-lg">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                    <span className="text-xs text-gray-300">Attribution</span>
                  </div>
                  <span className="text-xs text-green-400 font-medium">Active</span>
                </div>
                <div className="flex items-center justify-between p-2 bg-gray-700/30 rounded-lg">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                    <span className="text-xs text-gray-300">Containment</span>
                  </div>
                  <span className="text-xs text-green-400 font-medium">Active</span>
                </div>
                <div className="flex items-center justify-between p-2 bg-gray-700/30 rounded-lg">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                    <span className="text-xs text-gray-300">Forensics</span>
                  </div>
                  <span className="text-xs text-green-400 font-medium">Active</span>
                </div>
                <div className="flex items-center justify-between p-2 bg-gray-700/30 rounded-lg">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                    <span className="text-xs text-gray-300">Deception</span>
                  </div>
                  <span className="text-xs text-green-400 font-medium">Active</span>
                </div>

                <div className="mt-3 pt-3 border-t border-gray-800">
                  <div className="flex items-center justify-between p-2 bg-blue-600/20 border border-blue-500/30 rounded-lg">
                    <div className="flex items-center gap-2">
                      <Zap className="w-3 h-3 text-blue-400" />
                      <span className="text-xs text-blue-300 font-medium">MCP Server</span>
                    </div>
                    <span className="text-xs text-green-400 font-medium">Online</span>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col" style={{ marginTop: needsOnboarding ? '52px' : '0' }}>
        {/* Top Bar */}
        <div className="bg-gray-900/50 border-b border-gray-800 p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <h2 className="text-2xl font-bold text-white">
                {activeTab === 'overview' && '📊 Threat Overview'}
                {activeTab === 'incidents' && '🚨 Active Incidents'}
                {activeTab === 'intelligence' && '🌐 Threat Intelligence'}
                {activeTab === 'hunting' && '🎯 Threat Hunting'}
                {activeTab === 'forensics' && '🔍 Digital Forensics'}
                {activeTab === 'response' && '🛡️ Response Actions'}
                {activeTab === 'visualizations' && '🌍 3D Threat Visualization'}
              </h2>
              {autoRefreshing && (
                <div className="flex items-center gap-2 px-3 py-1 bg-blue-500/20 border border-blue-500/50 rounded-full">
                  <RefreshCw className="w-3 h-3 text-blue-400 animate-spin" />
                  <span className="text-xs text-blue-300 font-medium">Live</span>
                </div>
              )}
            </div>

            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <Search className="w-4 h-4 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search incidents..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="bg-gray-700/50 border border-gray-600/50 rounded-lg px-3 py-1 text-sm text-white placeholder-gray-400 focus:outline-none focus:ring-1 focus:ring-blue-500/50 w-64"
                />
              </div>
              <button className="p-2 hover:bg-gray-700 rounded-lg transition-colors">
                <Bell className="w-4 h-4 text-gray-400" />
              </button>
              <button className="p-2 hover:bg-gray-700 rounded-lg transition-colors">
                <Settings className="w-4 h-4 text-gray-400" />
              </button>
              <button className="p-2 hover:bg-gray-700 rounded-lg transition-colors">
                <User className="w-4 h-4 text-gray-400" />
              </button>
            </div>
          </div>
        </div>

        {/* Content Area */}
        <div className="flex-1 flex">
          {/* Main Panel */}
          <div className="flex-1 p-6 overflow-y-auto">
            {activeTab === 'overview' && (
              <div className="space-y-6">
                {/* Zero-state when telemetry not flowing yet */}
                {telemetry && telemetry.hasLogs === false && (
                  <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-6">
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <h3 className="text-lg font-semibold text-amber-300">Awaiting first data</h3>
                        <p className="text-sm text-amber-200/80">
                          Network discovered. Agents deployed. Waiting for initial events to arrive. This can take a few minutes.
                        </p>
                      </div>
                      <div className="text-right text-sm text-amber-200/80">
                        <div>Assets: {telemetry.assetsDiscovered ?? 0}</div>
                        <div>Agents: {telemetry.agentsEnrolled ?? 0}</div>
                        <div>Incidents: {telemetry.incidents ?? 0}</div>
                      </div>
                    </div>
                  </div>
                )}
                {/* Metrics Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  <div className="bg-gray-900 border border-red-500/30 rounded-xl p-6">
                    <div className="flex items-center justify-between mb-4">
                      <div className="p-3 bg-red-500/20 rounded-lg">
                        <AlertTriangle className="w-6 h-6 text-red-400" />
                      </div>
                      <div className="text-right">
                        <div className="text-2xl font-bold text-red-300">{metrics.total_incidents}</div>
                        <div className="text-xs text-red-400/70">Total Incidents</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 text-xs">
                      <ArrowUpRight className="w-3 h-3 text-red-400" />
                      <span className="text-red-400">+12% from yesterday</span>
                    </div>
                  </div>

                  <div className="bg-gray-900 border border-orange-500/30 rounded-xl p-6">
                    <div className="flex items-center justify-between mb-4">
                      <div className="p-3 bg-orange-500/20 rounded-lg">
                        <Zap className="w-6 h-6 text-orange-400" />
                      </div>
                      <div className="text-right">
                        <div className="text-2xl font-bold text-orange-300">{metrics.high_priority}</div>
                        <div className="text-xs text-orange-400/70">High Priority</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 text-xs">
                      <ArrowDownRight className="w-3 h-3 text-green-400" />
                      <span className="text-green-400">-8% from yesterday</span>
                    </div>
                  </div>

                  <div className="bg-gray-900 border border-green-500/30 rounded-xl p-6">
                    <div className="flex items-center justify-between mb-4">
                      <div className="p-3 bg-green-500/20 rounded-lg">
                        <Shield className="w-6 h-6 text-green-400" />
                      </div>
                      <div className="text-right">
                        <div className="text-2xl font-bold text-green-300">{metrics.contained}</div>
                        <div className="text-xs text-green-400/70">Contained</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 text-xs">
                      <ArrowUpRight className="w-3 h-3 text-green-400" />
                      <span className="text-green-400">+23% effectiveness</span>
                    </div>
                  </div>

                  <div className="bg-gray-900 border border-blue-500/30 rounded-xl p-6">
                    <div className="flex items-center justify-between mb-4">
                      <div className="p-3 bg-blue-500/20 rounded-lg">
                        <Bot className="w-6 h-6 text-blue-400" />
                      </div>
                      <div className="text-right">
                        <div className="text-2xl font-bold text-blue-300">{metrics.ml_detected}</div>
                        <div className="text-xs text-blue-400/70">AI Detected</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 text-xs">
                      <Minus className="w-3 h-3 text-gray-400" />
                      <span className="text-gray-400">Stable detection rate</span>
                    </div>
                  </div>
                </div>

                {/* Recent Activity */}
                <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
                  <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <Activity className="w-5 h-5 text-green-400" />
                    Recent Activity
                  </h3>
                  <div className="space-y-3">
                    {incidents.slice(0, 5).map((incident) => (
                      <Link key={incident.id} href={`/incidents/incident/${incident.id}`}>
                        <div className="flex items-center gap-4 p-3 bg-gray-700/30 hover:bg-gray-700/50 rounded-lg cursor-pointer transition-all duration-200 border border-transparent hover:border-gray-600/50">
                          <div className={`w-3 h-3 rounded-full ${
                            incident.triage_note?.severity === 'high' ? 'bg-red-500' :
                            incident.triage_note?.severity === 'medium' ? 'bg-orange-500' : 'bg-green-500'
                          }`}></div>
                          <div className="flex-1">
                            <div className="flex items-center gap-2">
                              <span className="text-sm font-medium text-white">Incident #{incident.id}</span>
                              <span className="text-xs text-gray-400">from {incident.src_ip}</span>
                            </div>
                            <p className="text-xs text-gray-400 truncate">{incident.reason}</p>
                          </div>
                          <div className="flex items-center gap-2">
                            <div className="text-xs text-gray-500">{formatTimeAgo(incident.created_at)}</div>
                            <Eye className="w-4 h-4 text-gray-400 opacity-0 group-hover:opacity-100 transition-opacity" />
                          </div>
                        </div>
                      </Link>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'incidents' && (
              <div className="space-y-6">
                {/* Filters */}
                <div className="flex items-center gap-4 p-4 bg-gray-800/30 border border-gray-700/50 rounded-xl">
                  <div className="flex items-center gap-2">
                    <Filter className="w-4 h-4 text-gray-400" />
                    <span className="text-sm text-gray-300">Filters:</span>
                  </div>
                  <select
                    value={filterSeverity}
                    onChange={(e) => setFilterSeverity(e.target.value)}
                    className="bg-gray-700 border border-gray-600 rounded-lg px-3 py-1 text-sm text-white"
                  >
                    <option value="all">All Severities</option>
                    <option value="high">High</option>
                    <option value="medium">Medium</option>
                    <option value="low">Low</option>
                  </select>
                  <select
                    value={filterStatus}
                    onChange={(e) => setFilterStatus(e.target.value)}
                    className="bg-gray-700 border border-gray-600 rounded-lg px-3 py-1 text-sm text-white"
                  >
                    <option value="all">All Statuses</option>
                    <option value="open">Open</option>
                    <option value="contained">Contained</option>
                    <option value="dismissed">Dismissed</option>
                  </select>
                  <div className="ml-auto text-sm text-gray-400">
                    {filteredIncidents.length} of {incidents.length} incidents
                  </div>
                </div>

                {/* Incidents List */}
                <div className="space-y-6">
                  {filteredIncidents.map((incident) => (
                    <div key={incident.id} className="bg-gray-800/50 border border-gray-700/50 hover:border-gray-600/50 rounded-xl overflow-hidden transition-all duration-200 hover:bg-gray-800/70">
                      {/* Quick Actions Bar - Most Prominent */}
                      <div className="bg-gradient-to-r from-red-600/20 via-orange-600/20 to-blue-600/20 p-4 border-b border-gray-700/50">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <AlertTriangle className="w-5 h-5 text-red-400" />
                            <span className="text-sm font-semibold text-white">IMMEDIATE ACTIONS</span>
                            <span className={`px-2 py-1 rounded-full text-xs font-semibold ${getSeverityColor(incident.triage_note?.severity)}`}>
                              {incident.triage_note?.severity?.toUpperCase() || 'UNKNOWN'}
                            </span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className={`px-2 py-1 rounded-full text-xs font-semibold ${getStatusColor(incident.status)}`}>
                              {incident.status.toUpperCase()}
                            </span>
                            {incident.auto_contained && (
                              <span className="px-2 py-1 rounded-full text-xs font-semibold bg-purple-500/20 text-purple-300">
                                AUTO
                              </span>
                            )}
                          </div>
                        </div>
                        
                        <div className="flex items-center gap-3 mt-3">
                          {[
                            { action: 'block_ip', label: 'Block IP', icon: Ban, color: 'red' },
                            { action: 'isolate_host', label: 'Isolate Host', icon: Shield, color: 'orange' },
                            { action: 'reset_passwords', label: 'Reset Passwords', icon: Key, color: 'yellow' },
                            { action: 'threat_intel_lookup', label: 'Threat Intel', icon: Globe, color: 'blue' },
                            { action: 'hunt_similar_attacks', label: 'Hunt Similar', icon: Target, color: 'purple' }
                          ].map(({ action, label, icon: Icon, color }) => (
                            <button
                              key={action}
                              onClick={(e) => {
                                e.stopPropagation();
                                executeSOCAction(action, label, incident.id);
                              }}
                              disabled={actionLoading === `${action}-${incident.id}`}
                              className={`flex items-center gap-2 px-4 py-2 bg-${color}-600/20 hover:bg-${color}-600/30 border border-${color}-500/30 rounded-lg text-sm font-medium transition-all ${
                                actionLoading === `${action}-${incident.id}` ? 'opacity-50 cursor-not-allowed' : 'hover:scale-105'
                              }`}
                            >
                              {actionLoading === `${action}-${incident.id}` ? (
                                <Loader2 className="w-4 h-4 animate-spin" />
                              ) : (
                                <Icon className="w-4 h-4" />
                              )}
                              <span className="text-white">{label}</span>
                            </button>
                          ))}
                        </div>
                      </div>

                      {/* Incident Details */}
                      <div className="p-6">
                        <div className="flex items-start justify-between mb-4">
                          <div className="flex items-center gap-4">
                            <div className="p-3 bg-red-500/20 rounded-lg">
                              <AlertTriangle className="w-5 h-5 text-red-400" />
                            </div>
                            <div>
                              <h3 className="text-lg font-semibold text-white">Incident #{incident.id}</h3>
                              <div className="flex items-center gap-3 text-sm text-gray-400">
                                <span>Source: <code className="text-orange-400 font-mono">{incident.src_ip}</code></span>
                                <span>•</span>
                                <span>{formatTimeAgo(incident.created_at)}</span>
                              </div>
                            </div>
                          </div>
                        </div>

                        <div className="grid grid-cols-4 gap-4 mb-4">
                          <div className="text-center p-3 bg-gray-700/30 rounded-lg">
                            <div className="text-lg font-bold text-red-400">
                              {incident.risk_score ? `${Math.round(incident.risk_score * 100)}%` : 'N/A'}
                            </div>
                            <div className="text-xs text-gray-400">Risk Score</div>
                          </div>
                          <div className="text-center p-3 bg-gray-700/30 rounded-lg">
                            <div className="text-lg font-bold text-blue-400">
                              {incident.agent_confidence ? `${Math.round(incident.agent_confidence * 100)}%` : 'N/A'}
                            </div>
                            <div className="text-xs text-gray-400">ML Confidence</div>
                          </div>
                          <div className="text-center p-3 bg-gray-700/30 rounded-lg">
                            <div className="text-lg font-bold text-purple-400 capitalize">
                              {incident.escalation_level || 'Medium'}
                            </div>
                            <div className="text-xs text-gray-400">Escalation</div>
                          </div>
                          <div className="text-center p-3 bg-gray-700/30 rounded-lg">
                            <div className="text-lg font-bold text-green-400 capitalize">
                              {incident.containment_method || 'ML-driven'}
                            </div>
                            <div className="text-xs text-gray-400">Detection</div>
                          </div>
                        </div>

                        <p className="text-sm text-gray-300 mb-4 p-3 bg-gray-700/20 rounded-lg">{incident.reason}</p>

                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <Link href={`/incidents/incident/${incident.id}`}>
                              <button className="flex items-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors">
                                <Eye className="w-4 h-4" />
                                Full Investigation
                              </button>
                            </Link>
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                setSelectedIncident(incident);
                              }}
                              className="flex items-center gap-2 px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg text-sm font-medium transition-colors"
                            >
                              <MessageSquare className="w-4 h-4" />
                              AI Analysis
                            </button>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {activeTab === 'visualizations' && (
              <div className="h-full">
                <div className="mb-6">
                  <p className="text-gray-300">
                    Experience immersive 3D threat visualization with real-time data integration from our distributed intelligence network.
                  </p>
                </div>
                <div className="bg-gray-800/50 border border-gray-700/50 rounded-xl p-4">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                      <Activity className="w-5 h-5 text-blue-400" />
                      3D Threat Globe & Timeline
                    </h3>
                    <Link href="/visualizations" className="text-blue-400 hover:text-blue-300 transition-colors">
                      <button className="flex items-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-all hover:scale-105">
                        <Globe className="w-4 h-4" />
                        Launch 3D Visualization
                      </button>
                    </Link>
                  </div>
                  
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                    <div className="bg-gray-700/30 rounded-lg p-4">
                      <h4 className="text-white font-semibold mb-2 flex items-center gap-2">
                        <Globe className="w-4 h-4 text-blue-400" />
                        Interactive 3D Globe
                      </h4>
                      <ul className="text-sm text-gray-400 space-y-1">
                        <li>• Real-time threat origin mapping</li>
                        <li>• Country-based intelligence clustering</li>
                        <li>• Attack path visualization</li>
                        <li>• Interactive threat point details</li>
                        <li>• Performance optimized WebGL rendering</li>
                      </ul>
                    </div>
                    
                    <div className="bg-gray-700/30 rounded-lg p-4">
                      <h4 className="text-white font-semibold mb-2 flex items-center gap-2">
                        <Activity className="w-4 h-4 text-purple-400" />
                        3D Attack Timeline
                      </h4>
                      <ul className="text-sm text-gray-400 space-y-1">
                        <li>• Chronological attack progression</li>
                        <li>• Severity-based 3D positioning</li>
                        <li>• Playback controls with speed adjustment</li>
                        <li>• Attack chain connection visualization</li>
                        <li>• Real-time incident correlation</li>
                      </ul>
                    </div>
                  </div>
                  
                  <div className="bg-gradient-to-r from-blue-600/10 to-purple-600/10 border border-blue-500/20 rounded-lg p-4">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-blue-600/20 rounded-lg">
                        <Zap className="w-5 h-5 text-blue-400" />
                      </div>
                      <div>
                        <h4 className="text-white font-semibold">Powered by Phase 3 Distributed Architecture</h4>
                        <p className="text-sm text-gray-400">
                          Integrates real-time data from our distributed MCP network, federated learning insights, and threat intelligence APIs
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'intelligence' && (
              <div className="space-y-6">
                <div className="mb-6">
                  <p className="text-gray-300">
                    Advanced threat intelligence analysis with AI-powered insights and natural language processing.
                  </p>
                </div>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div className="bg-gray-800/50 border border-gray-700/50 rounded-xl p-6">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                      <Bot className="w-5 h-5 text-purple-400" />
                      AI Agent Interface
                    </h3>
                    <p className="text-sm text-gray-400 mb-4">
                      Natural language threat analysis with multi-agent coordination and predictive intelligence.
                    </p>
                    <Link href="/agents">
                      <button className="flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg text-sm font-medium transition-colors">
                        <Bot className="w-4 h-4" />
                        Launch AI Agents
                      </button>
                    </Link>
                  </div>
                  
                  <div className="bg-gray-800/50 border border-gray-700/50 rounded-xl p-6">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                      <BarChart3 className="w-5 h-5 text-blue-400" />
                      Advanced Analytics
                    </h3>
                    <p className="text-sm text-gray-400 mb-4">
                      ML monitoring, explainable AI, and federated learning insights dashboard.
                    </p>
                    <Link href="/analytics">
                      <button className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors">
                        <BarChart3 className="w-4 h-4" />
                        View Analytics
                      </button>
                    </Link>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'hunting' && (
              <div className="space-y-6">
                <div className="mb-6">
                  <p className="text-gray-300">
                    Proactive threat hunting using advanced search capabilities and behavioral analysis.
                  </p>
                </div>
                <div className="bg-gray-800/50 border border-gray-700/50 rounded-xl p-6">
                  <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <Target className="w-5 h-5 text-green-400" />
                    Threat Hunting Dashboard
                  </h3>
                  <p className="text-sm text-gray-400 mb-4">
                    Advanced hunting queries and suspicious activity detection.
                  </p>
                  <Link href="/hunt">
                    <button className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg text-sm font-medium transition-colors">
                      <Target className="w-4 h-4" />
                      Start Hunting
                    </button>
                  </Link>
                </div>
              </div>
            )}

            {activeTab === 'forensics' && (
              <div className="space-y-6">
                <div className="mb-6">
                  <p className="text-gray-300">
                    Digital forensics and incident investigation tools for comprehensive analysis.
                  </p>
                </div>
                <div className="bg-gray-800/50 border border-gray-700/50 rounded-xl p-6">
                  <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <Search className="w-5 h-5 text-yellow-400" />
                    Investigation Tools
                  </h3>
                  <p className="text-sm text-gray-400 mb-4">
                    Detailed forensic analysis and evidence collection capabilities.
                  </p>
                  <Link href="/investigations">
                    <button className="flex items-center gap-2 px-4 py-2 bg-yellow-600 hover:bg-yellow-700 text-white rounded-lg text-sm font-medium transition-colors">
                      <Search className="w-4 h-4" />
                      Open Investigations
                    </button>
                  </Link>
                </div>
              </div>
            )}

            {activeTab === 'response' && (
              <div className="space-y-6">
                <div className="mb-6">
                  <p className="text-gray-300">
                    Automated response actions and containment strategies for active threats.
                  </p>
                </div>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div className="bg-gray-800/50 border border-gray-700/50 rounded-xl p-6">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                      <Shield className="w-5 h-5 text-blue-400" />
                      Automated Response
                    </h3>
                    <p className="text-sm text-gray-400 mb-4">
                      Quick response actions: IP blocking, host isolation, password resets.
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {[
                        { label: 'Block IPs', color: 'red' },
                        { label: 'Isolate Hosts', color: 'orange' },
                        { label: 'Reset Passwords', color: 'yellow' }
                      ].map(({ label, color }) => (
                        <button
                          key={label}
                          className={`px-3 py-1 bg-${color}-600/20 border border-${color}-500/30 rounded text-sm text-${color}-300 hover:bg-${color}-600/30 transition-colors`}
                        >
                          {label}
                        </button>
                      ))}
                    </div>
                  </div>
                  
                  <div className="bg-gray-800/50 border border-gray-700/50 rounded-xl p-6">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                      <Settings className="w-5 h-5 text-purple-400" />
                      System Configuration
                    </h3>
                    <p className="text-sm text-gray-400 mb-4">
                      Response policies and system-wide security configurations.
                    </p>
                    <Link href="/settings">
                      <button className="flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg text-sm font-medium transition-colors">
                        <Settings className="w-4 h-4" />
                        Manage Settings
                      </button>
                    </Link>
                  </div>
                </div>
              </div>
            )}

            {/* Additional tabs can be added here */}
          </div>

          {/* AI Chat Panel */}
          <div className="w-96 bg-gray-800/50 border-l border-gray-700/50 flex flex-col">
            <div className="p-4 border-b border-gray-700/50">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-purple-600/20 rounded-lg">
                  <Bot className="w-5 h-5 text-purple-400" />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-white">AI Security Analyst</h3>
                  <p className="text-xs text-gray-400">
                    {selectedIncident ? `Analyzing Incident #${selectedIncident.id}` : 'Ready to assist'}
                  </p>
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
                        <span>Analyzing...</span>
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
                  placeholder="Ask about threats, incidents, or get analysis..."
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
                    'Send'
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
