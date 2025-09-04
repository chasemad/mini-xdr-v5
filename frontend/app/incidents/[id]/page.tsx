"use client";

import { useEffect, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { 
  getIncident, unblockIncident, containIncident, scheduleUnblock,
  socBlockIP, socIsolateHost, socResetPasswords, socCheckDBIntegrity,
  socThreatIntelLookup, socDeployWAFRules, socCaptureTraffic,
  socHuntSimilarAttacks, socAlertAnalysts, socCreateCase
} from "../../lib/api";
import { 
  Ban, Shield, UserX, Key, Database, History, UserMinus, Lock, 
  Search, Globe, Network, Clock, Target, Download, Archive, FileText, 
  Briefcase, TrendingUp, Code, Users, Crosshair, Flag, MapPin, 
  AlertTriangle, Ticket, Bell, Phone, Server, RefreshCw, X, Send,
  ChevronDown, ChevronRight, Eye, EyeOff, MessageSquare, Bot,
  CheckCircle, XCircle, AlertCircle, Info, Loader2, Copy, ExternalLink,
  Maximize2, Minimize2, Filter, MoreHorizontal
} from "lucide-react";

// Types for enhanced functionality
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
  duration?: number;
}

// Helper functions for compromise assessment
const getCompromiseStatus = (incident: IncidentDetail): 'CONFIRMED' | 'SUSPECTED' | 'UNLIKELY' => {
  if (incident.iocs?.successful_auth_indicators?.length > 0) {
    return 'CONFIRMED';
  }
  if (incident.iocs?.database_access_patterns?.length > 0 || 
      incident.iocs?.privilege_escalation_indicators?.length > 0) {
    return 'SUSPECTED';
  }
  return 'UNLIKELY';
};

const getCompromiseColor = (status: string) => {
  switch (status) {
    case 'CONFIRMED':
      return 'bg-red-500/20 border-red-500 text-red-100';
    case 'SUSPECTED':
      return 'bg-orange-500/20 border-orange-500 text-orange-100';
    case 'UNLIKELY':
      return 'bg-green-500/20 border-green-500 text-green-100';
    default:
      return 'bg-gray-700 border-gray-600 text-gray-300';
  }
};

const getRiskScoreColor = (score?: number) => {
  if (!score) return 'text-gray-400';
  if (score >= 0.8) return 'text-red-400';
  if (score >= 0.6) return 'text-orange-400';
  if (score >= 0.4) return 'text-yellow-400';
  return 'text-green-400';
};

const formatRiskScore = (score?: number) => {
  if (!score) return 'N/A';
  return `${(score * 100).toFixed(0)}%`;
};

const formatConfidence = (confidence?: number) => {
  if (!confidence) return 'N/A';
  const percent = (confidence * 100).toFixed(0);
  if (confidence >= 0.9) return `${percent}% (Very High)`;
  if (confidence >= 0.7) return `${percent}% (High)`;
  if (confidence >= 0.5) return `${percent}% (Medium)`;
  return `${percent}% (Low)`;
};

interface IncidentDetail {
  id: number;
  created_at: string;
  src_ip: string;
  reason: string;
  status: string;
  auto_contained: boolean;
  
  // Enhanced SOC fields
  escalation_level?: string;
  risk_score?: number;
  threat_category?: string;
  containment_confidence?: number;
  containment_method?: string;
  agent_id?: string;
  agent_actions?: any[];
  agent_confidence?: number;
  ml_features?: any;
  ensemble_scores?: any;
  
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
    params: any;
    due_at?: string;
  }>;
  
  // Detailed forensic data
  detailed_events: Array<{
    id: number;
    ts: string;
    src_ip: string;
    dst_ip?: string;
    dst_port?: number;
    eventid: string;
    message: string;
    raw: any;
    source_type: string;
    hostname?: string;
  }>;
  
  // Attack analysis
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
    raw_data: any;
    severity: string;
    attack_category: string;
  }>;
  
  event_summary: {
    total_events: number;
    event_types: string[];
    time_span_hours: number;
  };
}

// Toast Component
const Toast = ({ toast, onDismiss }: { toast: ToastMessage; onDismiss: (id: string) => void }) => {
  const icons = {
    success: CheckCircle,
    error: XCircle,
    warning: AlertCircle,
    info: Info
  };
  
  const colors = {
    success: 'bg-green-500/20 border-green-500 text-green-100',
    error: 'bg-red-500/20 border-red-500 text-red-100',
    warning: 'bg-orange-500/20 border-orange-500 text-orange-100',
    info: 'bg-blue-500/20 border-blue-500 text-blue-100'
  };
  
  const Icon = icons[toast.type];
  
  useEffect(() => {
    const timer = setTimeout(() => {
      onDismiss(toast.id);
    }, toast.duration || 5000);
    
    return () => clearTimeout(timer);
  }, [toast.id, toast.duration, onDismiss]);
  
  return (
    <div className={`p-4 rounded-xl border shadow-lg ${colors[toast.type]} animate-in slide-in-from-right duration-300`}>
      <div className="flex items-start gap-3">
        <Icon className="w-5 h-5 mt-0.5 flex-shrink-0" />
        <div className="flex-1">
          <div className="font-semibold text-sm">{toast.title}</div>
          <div className="text-sm opacity-90 mt-1">{toast.message}</div>
        </div>
        <button
          onClick={() => onDismiss(toast.id)}
          className="p-1 hover:bg-white/10 rounded-lg transition-colors"
          aria-label="Dismiss notification"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};

// Modal Component
const ConfirmModal = ({ 
  isOpen, 
  onClose, 
  onConfirm, 
  title, 
  message, 
  confirmText = "Confirm",
  confirmVariant = "danger"
}: {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title: string;
  message: string;
  confirmText?: string;
  confirmVariant?: "danger" | "primary";
}) => {
  if (!isOpen) return null;
  
  const confirmColors = {
    danger: 'bg-red-600 hover:bg-red-700 focus:ring-red-500',
    primary: 'bg-blue-600 hover:bg-blue-700 focus:ring-blue-500'
  };
  
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-gray-800 border border-gray-700 rounded-xl p-6 max-w-md w-full mx-4 shadow-2xl">
        <div className="flex items-center gap-3 mb-4">
          <AlertTriangle className="w-6 h-6 text-orange-400" />
          <h3 className="text-lg font-semibold text-white">{title}</h3>
        </div>
        <p className="text-gray-300 mb-6">{message}</p>
        <div className="flex gap-3 justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-300 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className={`px-4 py-2 text-white rounded-lg transition-colors focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 ${confirmColors[confirmVariant]}`}
          >
            {confirmText}
          </button>
        </div>
      </div>
    </div>
  );
};

export default function IncidentDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const router = useRouter();
  const [id, setId] = useState<number | null>(null);
  const [incident, setIncident] = useState<IncidentDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [actionResults, setActionResults] = useState<Record<string, any>>({});
  const [autoRefreshing, setAutoRefreshing] = useState(false);
  const [toasts, setToasts] = useState<ToastMessage[]>([]);
  const [confirmModal, setConfirmModal] = useState<{
    isOpen: boolean;
    title: string;
    message: string;
    action: () => void;
  }>({ isOpen: false, title: '', message: '', action: () => {} });
  
  // Chat state
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      type: 'ai',
      content: "Hi, I'm your AI SOC Assistant. Ask me about this incident, and I'll help analyze threats, explain IOCs, or suggest response actions!",
      timestamp: new Date()
    }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  
  // Expandable sections state
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    timeline: true,
    history: true,
    iocs: true
  });

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
  }, []);

  const dismissToast = useCallback((id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  const fetchIncident = useCallback(async () => {
    if (!id) return;
    
    try {
      const data = await getIncident(id);
      setIncident(data);
    } catch (error: any) {
      console.error("Failed to fetch incident:", error);
      showToast('error', 'Failed to Load', 'Could not fetch incident details');
    } finally {
      setLoading(false);
    }
  }, [id, showToast]);

  useEffect(() => {
    fetchIncident();

    // Set up automatic refresh every 5 seconds
    const interval = setInterval(async () => {
      if (id && !loading) {
        setAutoRefreshing(true);
        await fetchIncident();
        setAutoRefreshing(false);
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [fetchIncident, id, loading]);

  const handleSOCAction = async (actionType: string, actionLabel: string) => {
    if (!incident?.id) return;
    
    setConfirmModal({
      isOpen: true,
      title: `Confirm ${actionLabel}`,
      message: `Are you sure you want to ${actionLabel.toLowerCase()} for incident #${incident.id}?`,
      action: () => executeSOCAction(actionType, actionLabel)
    });
  };

  const executeSOCAction = async (actionType: string, actionLabel: string) => {
    if (!incident?.id) return;
    
    setActionLoading(actionType);
    try {
      let result;
      
      switch (actionType) {
        case 'block_ip':
          result = await socBlockIP(incident.id);
          break;
        case 'isolate_host':
          result = await socIsolateHost(incident.id);
          break;
        case 'reset_passwords':
          result = await socResetPasswords(incident.id);
          break;
        case 'check_db_integrity':
          result = await socCheckDBIntegrity(incident.id);
          break;
        case 'threat_intel_lookup':
          result = await socThreatIntelLookup(incident.id);
          break;
        case 'deploy_waf_rules':
          result = await socDeployWAFRules(incident.id);
          break;
        case 'capture_traffic':
          result = await socCaptureTraffic(incident.id);
          break;
        case 'hunt_similar_attacks':
          result = await socHuntSimilarAttacks(incident.id);
          break;
        case 'alert_analysts':
          result = await socAlertAnalysts(incident.id);
          break;
        case 'create_case':
          result = await socCreateCase(incident.id);
          break;
        default:
          throw new Error(`Unknown action type: ${actionType}`);
      }
      
      setActionResults(prev => ({ ...prev, [actionType]: result }));
      showToast('success', 'Action Completed', result.message || `${actionLabel} completed successfully`);
      
      // Refresh incident data
      await fetchIncident();
      
    } catch (error: any) {
      console.error(`SOC action ${actionType} failed:`, error);
      showToast('error', 'Action Failed', error.message || `${actionLabel} failed`);
    } finally {
      setActionLoading(null);
      setConfirmModal({ isOpen: false, title: '', message: '', action: () => {} });
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
      // Call the actual AI agent API with full incident context
      const response = await fetch('/api/agents/orchestrate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: userMessage.content,
          incident_id: incident?.id,
          context: {
            incident_data: {
              id: incident?.id,
              src_ip: incident?.src_ip,
              threat_category: incident?.threat_category,
              risk_score: incident?.risk_score,
              escalation_level: incident?.escalation_level,
              ml_confidence: incident?.agent_confidence,
              auto_contained: incident?.auto_contained,
              reason: incident?.reason
            },
            iocs: incident?.iocs,
            attack_timeline: incident?.attack_timeline?.slice(0, 10), // Last 10 events for context
            event_summary: incident?.event_summary,
            triage_note: incident?.triage_note,
            chat_history: chatMessages.slice(-5).map(msg => ({ // Last 5 messages for context
              role: msg.type === 'user' ? 'user' : 'assistant',
              content: msg.content
            }))
          }
        }),
      });

      let aiResponse = "I apologize, I'm having trouble analyzing this incident right now.";
      
      if (response.ok) {
        const data = await response.json();
        aiResponse = data.message || data.analysis || "I've analyzed your query but don't have a specific response.";
      } else {
        // Fallback to contextual analysis if API fails
        console.log('API call failed, using fallback analysis');
        aiResponse = generateContextualResponse(userMessage.content, incident!);
      }
      
      const aiMessage: ChatMessage = {
        id: (Date.now() + 2).toString(),
        type: 'ai',
        content: aiResponse,
        timestamp: new Date()
      };
      
      setChatMessages(prev => prev.slice(0, -1).concat(aiMessage));
      
    } catch (error) {
      setChatMessages(prev => prev.slice(0, -1));
      showToast('error', 'AI Error', 'Failed to get AI response');
    } finally {
      setChatLoading(false);
    }
  };

  const generateContextualResponse = (userInput: string, incident: IncidentDetail): string => {
    const input = userInput.toLowerCase();
    const iocCount = Object.values(incident?.iocs || {}).flat().length;
    const timelineCount = incident?.attack_timeline?.length || 0;
    
    // Analyze user intent and provide contextual response
    if (input.includes('ioc') || input.includes('indicator') || input.includes('compromise')) {
      return `I found ${iocCount} indicators of compromise in this incident. Key findings:
• ${incident?.iocs?.sql_injection_patterns?.length || 0} SQL injection patterns
• ${incident?.iocs?.reconnaissance_patterns?.length || 0} reconnaissance patterns  
• ${incident?.iocs?.database_access_patterns?.length || 0} database access patterns
• ${incident?.iocs?.successful_auth_indicators?.length || 0} successful authentication indicators

The most critical indicators suggest ${incident?.threat_category?.replace('_', ' ') || 'coordinated'} attack activity.`;
    }
    
    if (input.includes('timeline') || input.includes('attack') || input.includes('sequence')) {
      return `The attack timeline shows ${timelineCount} events over ${incident?.event_summary?.time_span_hours || 'several'} hours. Pattern analysis:
• Attack type: ${incident?.threat_category?.replace('_', ' ') || 'Multi-vector'}
• Escalation: ${incident?.escalation_level || 'Medium'} severity
• Source: ${incident?.src_ip}
• Status: ${incident?.auto_contained ? 'Automatically contained' : 'Monitoring'}

This appears to be a sophisticated ${incident?.threat_category?.includes('multi') ? 'coordinated' : 'targeted'} attack campaign.`;
    }
    
    if (input.includes('recommend') || input.includes('next') || input.includes('should') || input.includes('action')) {
      const riskLevel = incident?.risk_score || 0;
      if (riskLevel > 0.7) {
        return `⚠️ HIGH RISK INCIDENT - Immediate actions recommended:
1. **Block Source IP**: ${incident?.src_ip} (${Math.round((riskLevel) * 100)}% risk score)
2. **Isolate Affected Systems**: Prevent lateral movement
3. **Reset Credentials**: Change admin passwords immediately
4. **Database Security**: Verify integrity after SQL injection attempts
5. **Hunt Similar Patterns**: Search for related IOCs network-wide

The ML confidence is ${formatConfidence(incident?.agent_confidence)} - this requires urgent attention.`;
      } else {
        return `Based on the ${incident?.escalation_level || 'medium'} escalation level, I recommend:
• Continue monitoring ${incident?.src_ip}
• Deploy additional detection rules
• Review security controls for ${incident?.threat_category?.replace('_', ' ') || 'similar'} attacks
• Consider threat hunting for related activity

Current risk assessment: ${Math.round((riskLevel) * 100)}% with ${formatConfidence(incident?.agent_confidence)} ML confidence.`;
      }
    }
    
    if (input.includes('explain') || input.includes('what') || input.includes('how') || input.includes('why')) {
      return `This incident involves a ${incident?.threat_category?.replace('_', ' ') || 'security'} threat from ${incident?.src_ip}:

**Threat Analysis:**
• Risk Score: ${formatRiskScore(incident?.risk_score)} 
• ML Confidence: ${formatConfidence(incident?.agent_confidence)}
• Detection Method: ${incident?.containment_method?.replace('_', ' ') || 'Rule-based'}
• Status: ${incident?.auto_contained ? 'Auto-contained' : 'Under investigation'}

**Key Details:**
${incident?.reason || 'Multiple security violations detected'}

The system has classified this as ${incident?.escalation_level || 'medium'} priority based on behavioral analysis and threat intelligence.`;
    }
    
    // Handle conversational responses
    if (input.includes('yes') || input.includes('all') || input.includes('sure') || input.includes('ok')) {
      return `Here's a comprehensive analysis of incident #${incident?.id}:

📊 **IOC Analysis**: ${iocCount} total indicators detected
• SQL injection attempts: ${incident?.iocs?.sql_injection_patterns?.length || 0}
• Reconnaissance patterns: ${incident?.iocs?.reconnaissance_patterns?.length || 0}
• Database access attempts: ${incident?.iocs?.database_access_patterns?.length || 0}

⏰ **Attack Timeline**: ${timelineCount} events showing ${incident?.threat_category?.replace('_', ' ') || 'coordinated'} attack patterns

🎯 **Risk Assessment**: ${formatRiskScore(incident?.risk_score)} risk with ${formatConfidence(incident?.agent_confidence)} ML confidence

💡 **Bottom Line**: This appears to be a ${incident?.escalation_level || 'medium'} severity ${incident?.threat_category?.replace('_', ' ') || 'multi-vector'} attack from ${incident?.src_ip}.`;
    }
    
    if (input.includes('no') || input.includes('different') || input.includes('else')) {
      return `I understand you're looking for different information. I can help you with:

🔍 **Incident Analysis**: IOCs, attack patterns, timeline review
🚨 **Risk Assessment**: Threat scoring, ML confidence, severity analysis  
💡 **Recommendations**: Next steps, containment actions, investigation guidance
🛡️ **Context**: How this fits into your broader security posture

What specific aspect of incident #${incident?.id} would you like to explore?`;
    }
    
    // Default intelligent response based on incident context
    const severity = incident?.escalation_level || 'medium';
    const hasMultipleVectors = (incident?.iocs?.sql_injection_patterns?.length || 0) > 0 && 
                              (incident?.iocs?.reconnaissance_patterns?.length || 0) > 0;
    
    return `I'm analyzing incident #${incident?.id} from ${incident?.src_ip}. This ${severity} severity ${hasMultipleVectors ? 'multi-vector' : 'targeted'} attack shows:

• **${iocCount} IOCs detected** across multiple categories
• **${timelineCount} attack events** in the timeline
• **${formatRiskScore(incident?.risk_score)} risk score** with ${formatConfidence(incident?.agent_confidence)} ML confidence

I can help you understand the IOCs, analyze the attack timeline, assess the risk, or recommend next steps. What would you like to dive deeper into?`;
  };

  const toggleSection = (section: string) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }));
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
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

  if (loading || !id) {
    return (
      <div className="min-h-screen bg-gray-900 text-white">
        <div className="px-4 py-6">
          <div className="animate-pulse space-y-6">
            <div className="h-8 bg-gray-800 rounded w-1/3"></div>
            <div className="h-64 bg-gray-800 rounded-2xl"></div>
            <div className="grid md:grid-cols-2 gap-6">
              <div className="h-80 bg-gray-800 rounded-2xl"></div>
              <div className="h-80 bg-gray-800 rounded-2xl"></div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!incident) {
    return (
      <div className="min-h-screen bg-gray-900 text-white">
        <div className="px-4 py-6">
          <div className="text-center py-12">
            <AlertTriangle className="w-16 h-16 text-red-400 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-white mb-2">Incident Not Found</h3>
            <p className="text-gray-400 mb-6">The requested incident could not be found.</p>
            <button
              onClick={() => router.push("/incidents")}
              className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-medium transition-colors"
            >
              Back to Incidents
            </button>
          </div>
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
          <Toast key={toast.id} toast={toast} onDismiss={dismissToast} />
        ))}
      </div>

      {/* Confirmation Modal */}
      <ConfirmModal
        isOpen={confirmModal.isOpen}
        onClose={() => setConfirmModal({ isOpen: false, title: '', message: '', action: () => {} })}
        onConfirm={confirmModal.action}
        title={confirmModal.title}
        message={confirmModal.message}
        confirmText="Execute"
        confirmVariant="danger"
      />

      <div className="px-4 py-6 space-y-8 max-w-7xl mx-auto">
        {/* Enhanced Header with Live Status */}
        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-2xl p-6 shadow-2xl">
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 mb-6">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <h1 className="text-4xl font-bold text-white">
                  🚨 Incident #{incident.id}
                </h1>
                {autoRefreshing && (
                  <div className="flex items-center gap-2 px-3 py-1 bg-blue-500/20 border border-blue-500/50 rounded-full">
                    <RefreshCw className="w-4 h-4 text-blue-400 animate-spin" />
                    <span className="text-sm text-blue-300 font-medium">Live</span>
                  </div>
                )}
              </div>
              <div className="flex items-center gap-4 text-gray-400">
                <span>{formatDate(incident.created_at)}</span>
                <span>•</span>
                <span>Source: <code className="text-orange-400 font-mono">{incident.src_ip}</code></span>
                <span>•</span>
                <span className="text-gray-500">{formatTimeAgo(incident.created_at)}</span>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={fetchIncident}
                className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
                title="Refresh incident data"
              >
                <RefreshCw className="w-5 h-5 text-gray-400" />
              </button>
              <span className={`px-4 py-2 rounded-full text-sm font-semibold border ${
                incident.status === 'open' ? 'bg-yellow-500/20 border-yellow-500 text-yellow-300' :
                incident.status === 'contained' ? 'bg-red-500/20 border-red-500 text-red-300' :
                'bg-gray-700 border-gray-600 text-gray-300'
              }`}>
                {incident.status.toUpperCase()}
              </span>
              {incident.auto_contained && (
                <span className="px-4 py-2 rounded-full text-sm font-semibold bg-purple-500/20 border border-purple-500 text-purple-300">
                  AUTO-CONTAINED
                </span>
              )}
            </div>
          </div>

          {/* Critical Metrics Grid - Enhanced */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-red-500/10 border border-red-500/30 p-4 rounded-xl">
              <div className="flex items-center gap-2 mb-2">
                <AlertTriangle className="w-5 h-5 text-red-400" />
                <span className="text-sm text-red-400 font-medium">Risk Score</span>
              </div>
              <div className={`text-3xl font-bold ${getRiskScoreColor(incident.risk_score)}`}>
                {formatRiskScore(incident.risk_score)}
              </div>
              <div className="mt-2 bg-gray-700 rounded-full h-2">
                <div 
                  className="bg-red-500 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${(incident.risk_score || 0) * 100}%` }}
                />
              </div>
            </div>
            
            <div className="bg-blue-500/10 border border-blue-500/30 p-4 rounded-xl">
              <div className="flex items-center gap-2 mb-2">
                <TrendingUp className="w-5 h-5 text-blue-400" />
                <span className="text-sm text-blue-400 font-medium">ML Confidence</span>
              </div>
              <div className="text-2xl font-bold text-blue-300">
                {formatConfidence(incident.agent_confidence)}
              </div>
              <div className="mt-2 bg-gray-700 rounded-full h-2">
                <div 
                  className="bg-blue-500 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${(incident.agent_confidence || 0) * 100}%` }}
                />
              </div>
            </div>
            
            <div className="bg-purple-500/10 border border-purple-500/30 p-4 rounded-xl">
              <div className="flex items-center gap-2 mb-2">
                <Flag className="w-5 h-5 text-purple-400" />
                <span className="text-sm text-purple-400 font-medium">Escalation</span>
              </div>
              <div className="text-xl font-bold text-purple-300 capitalize">
                {incident.escalation_level || 'Medium'}
              </div>
            </div>
            
            <div className="bg-green-500/10 border border-green-500/30 p-4 rounded-xl">
              <div className="flex items-center gap-2 mb-2">
                <Bot className="w-5 h-5 text-green-400" />
                <span className="text-sm text-green-400 font-medium">Detection</span>
              </div>
              <div className="text-sm font-bold text-green-300 capitalize">
                {incident.containment_method || 'Rule-based'}
              </div>
            </div>
          </div>
        </div>

        {/* Compromise Assessment Alert - Enhanced */}
        <div className={`p-6 rounded-2xl border-2 shadow-xl ${getCompromiseColor(compromiseStatus)}`}>
          <div className="flex items-start gap-4 mb-6">
            <div className="p-2 rounded-full bg-white/10">
              <AlertTriangle className="w-8 h-8" />
            </div>
            <div className="flex-1">
              <h2 className="text-2xl font-bold mb-2">
                Compromise Assessment: {compromiseStatus}
              </h2>
              <p className="text-sm opacity-90 mb-4">
                {compromiseStatus === 'CONFIRMED' && '⚠️ Active compromise detected - immediate response required'}
                {compromiseStatus === 'SUSPECTED' && '🔍 Potential compromise indicators - investigation needed'}
                {compromiseStatus === 'UNLIKELY' && '✅ No clear signs of successful compromise'}
              </p>
              
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                {[
                  { key: 'successful_auth_indicators', label: 'Auth Success', icon: '🔑' },
                  { key: 'database_access_patterns', label: 'DB Access', icon: '🗄️' },
                  { key: 'privilege_escalation_indicators', label: 'Privilege Esc.', icon: '⬆️' },
                  { key: 'data_exfiltration_indicators', label: 'Data Exfil.', icon: '📤' }
                ].map(({ key, label, icon }) => (
                  <div key={key} className="flex items-center gap-2 p-2 bg-white/5 rounded-lg">
                    <div className={`w-3 h-3 rounded-full ${
                      (incident.iocs as any)?.[key]?.length > 0 ? 'bg-current animate-pulse' : 'bg-gray-600'
                    }`}></div>
                    <span className="text-sm font-medium">{icon} {label}</span>
                    <span className="text-xs ml-auto">
                      {(incident.iocs as any)?.[key]?.length || 0}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* SOC Action Panels Grid - Enhanced */}
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
          {/* Immediate Response Panel */}
          <div className="bg-red-500/10 border border-red-500/30 p-6 rounded-xl shadow-lg hover:shadow-xl transition-all">
            <h3 className="text-lg font-semibold text-red-300 mb-4 flex items-center gap-2">
              <Ban className="w-5 h-5" />
              🚨 Immediate Response
            </h3>
            <div className="space-y-3">
              {[
                { action: 'block_ip', label: `Block IP: ${incident.src_ip}`, icon: Ban },
                { action: 'isolate_host', label: 'Isolate Host', icon: Shield },
                { action: 'reset_passwords', label: 'Reset Admin Passwords', icon: Key }
              ].map(({ action, label, icon: Icon }) => (
                <button
                  key={action}
                  onClick={() => handleSOCAction(action, label)}
                  disabled={actionLoading === action}
                  className="w-full flex items-center gap-3 px-4 py-3 bg-red-600/20 hover:bg-red-600/30 disabled:bg-red-600/10 text-red-100 rounded-lg font-medium transition-all border border-red-500/30 hover:border-red-500/50 group"
                  title={`Execute ${label}`}
                >
                  {actionLoading === action ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Icon className="w-4 h-4 group-hover:scale-110 transition-transform" />
                  )}
                  <span className="text-sm">{label}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Database Security Panel */}
          <div className="bg-purple-500/10 border border-purple-500/30 p-6 rounded-xl shadow-lg hover:shadow-xl transition-all">
            <h3 className="text-lg font-semibold text-purple-300 mb-4 flex items-center gap-2">
              <Database className="w-5 h-5" />
              🗄️ Database Security
            </h3>
            <div className="space-y-3">
              {[
                { action: 'check_db_integrity', label: 'Check DB Integrity', icon: Database },
                { action: 'deploy_waf_rules', label: 'Deploy WAF Rules', icon: Shield }
              ].map(({ action, label, icon: Icon }) => (
                <button
                  key={action}
                  onClick={() => handleSOCAction(action, label)}
                  disabled={actionLoading === action}
                  className="w-full flex items-center gap-3 px-4 py-3 bg-purple-600/20 hover:bg-purple-600/30 disabled:bg-purple-600/10 text-purple-100 rounded-lg font-medium transition-all border border-purple-500/30 hover:border-purple-500/50 group"
                  title={`Execute ${label}`}
                >
                  {actionLoading === action ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Icon className="w-4 h-4 group-hover:scale-110 transition-transform" />
                  )}
                  <span className="text-sm">{label}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Threat Intelligence Panel */}
          <div className="bg-blue-500/10 border border-blue-500/30 p-6 rounded-xl shadow-lg hover:shadow-xl transition-all">
            <h3 className="text-lg font-semibold text-blue-300 mb-4 flex items-center gap-2">
              <Search className="w-5 h-5" />
              🔍 Threat Intelligence
            </h3>
            <div className="space-y-3">
              <button
                onClick={() => handleSOCAction('threat_intel_lookup', `Intel Lookup: ${incident.src_ip}`)}
                disabled={actionLoading === 'threat_intel_lookup'}
                className="w-full flex items-center gap-3 px-4 py-3 bg-blue-600/20 hover:bg-blue-600/30 disabled:bg-blue-600/10 text-blue-100 rounded-lg font-medium transition-all border border-blue-500/30 hover:border-blue-500/50 group"
                title="Lookup threat intelligence for source IP"
              >
                {actionLoading === 'threat_intel_lookup' ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Globe className="w-4 h-4 group-hover:scale-110 transition-transform" />
                )}
                <span className="text-sm">Intel Lookup: {incident.src_ip}</span>
              </button>
            </div>
          </div>

          {/* Forensics Panel */}
          <div className="bg-cyan-500/10 border border-cyan-500/30 p-6 rounded-xl shadow-lg hover:shadow-xl transition-all">
            <h3 className="text-lg font-semibold text-cyan-300 mb-4 flex items-center gap-2">
              <Archive className="w-5 h-5" />
              🔬 Forensics
            </h3>
            <div className="space-y-3">
              {[
                { action: 'capture_traffic', label: 'Capture Traffic', icon: Network },
                { action: 'create_case', label: 'Create SOAR Case', icon: FileText }
              ].map(({ action, label, icon: Icon }) => (
                <button
                  key={action}
                  onClick={() => handleSOCAction(action, label)}
                  disabled={actionLoading === action}
                  className="w-full flex items-center gap-3 px-4 py-3 bg-cyan-600/20 hover:bg-cyan-600/30 disabled:bg-cyan-600/10 text-cyan-100 rounded-lg font-medium transition-all border border-cyan-500/30 hover:border-cyan-500/50 group"
                  title={`Execute ${label}`}
                >
                  {actionLoading === action ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Icon className="w-4 h-4 group-hover:scale-110 transition-transform" />
                  )}
                  <span className="text-sm">{label}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Threat Hunting Panel */}
          <div className="bg-yellow-500/10 border border-yellow-500/30 p-6 rounded-xl shadow-lg hover:shadow-xl transition-all">
            <h3 className="text-lg font-semibold text-yellow-300 mb-4 flex items-center gap-2">
              <Target className="w-5 h-5" />
              🎯 Threat Hunting
            </h3>
            <div className="space-y-3">
              <button
                onClick={() => handleSOCAction('hunt_similar_attacks', 'Hunt Similar Attacks')}
                disabled={actionLoading === 'hunt_similar_attacks'}
                className="w-full flex items-center gap-3 px-4 py-3 bg-yellow-600/20 hover:bg-yellow-600/30 disabled:bg-yellow-600/10 text-yellow-100 rounded-lg font-medium transition-all border border-yellow-500/30 hover:border-yellow-500/50 group"
                title="Hunt for similar attack patterns"
              >
                {actionLoading === 'hunt_similar_attacks' ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Crosshair className="w-4 h-4 group-hover:scale-110 transition-transform" />
                )}
                <span className="text-sm">Hunt Similar Attacks</span>
              </button>
            </div>
          </div>

          {/* Escalation Panel */}
          <div className="bg-orange-500/10 border border-orange-500/30 p-6 rounded-xl shadow-lg hover:shadow-xl transition-all">
            <h3 className="text-lg font-semibold text-orange-300 mb-4 flex items-center gap-2">
              <Bell className="w-5 h-5" />
              📢 Escalation
            </h3>
            <div className="space-y-3">
              <button
                onClick={() => handleSOCAction('alert_analysts', 'Alert Senior Analysts')}
                disabled={actionLoading === 'alert_analysts'}
                className="w-full flex items-center gap-3 px-4 py-3 bg-orange-600/20 hover:bg-orange-600/30 disabled:bg-orange-600/10 text-orange-100 rounded-lg font-medium transition-all border border-orange-500/30 hover:border-orange-500/50 group"
                title="Alert senior security analysts"
              >
                {actionLoading === 'alert_analysts' ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Users className="w-4 h-4 group-hover:scale-110 transition-transform" />
                )}
                <span className="text-sm">Alert Senior Analysts</span>
              </button>
            </div>
          </div>
        </div>

        {/* IOC Analysis Grid - Enhanced */}
        {incident.iocs && Object.values(incident.iocs).some(arr => arr.length > 0) && (
          <div className="bg-gray-800/50 border border-gray-700/50 p-6 rounded-xl">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold text-white flex items-center gap-2">
                <Flag className="w-6 h-6 text-orange-400" />
                🔍 Indicators of Compromise
              </h2>
              <button
                onClick={() => toggleSection('iocs')}
                className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
              >
                {expandedSections.iocs ? <ChevronDown className="w-5 h-5" /> : <ChevronRight className="w-5 h-5" />}
              </button>
            </div>
            
            {expandedSections.iocs && (
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                {incident.iocs.successful_auth_indicators?.length > 0 && (
                  <div className="p-4 bg-red-500/20 border border-red-500/50 rounded-xl">
                    <h3 className="font-semibold text-red-300 mb-2 flex items-center gap-2">
                      <CheckCircle className="w-4 h-4" />
                      🚨 COMPROMISE CONFIRMED
                    </h3>
                    <div className="text-sm text-red-400 mb-3">
                      {incident.iocs.successful_auth_indicators.length} successful auth indicators
                    </div>
                    <div className="space-y-2">
                      {incident.iocs.successful_auth_indicators.slice(0, 2).map((indicator, idx) => (
                        <div key={idx} className="text-xs font-mono bg-red-800/30 p-2 rounded border border-red-700/50">
                          <div className="text-red-200 break-all">{indicator}</div>
                        </div>
                      ))}
                      {incident.iocs.successful_auth_indicators.length > 2 && (
                        <div className="text-xs text-red-400 text-center py-1">
                          +{incident.iocs.successful_auth_indicators.length - 2} more
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {incident.iocs.sql_injection_patterns?.length > 0 && (
                  <div className="p-4 bg-red-500/20 border border-red-500/50 rounded-xl">
                    <h3 className="font-semibold text-red-300 mb-2 flex items-center gap-2">
                      <Code className="w-4 h-4" />
                      🚨 SQL INJECTION
                    </h3>
                    <div className="text-sm text-red-400 mb-3">
                      {incident.iocs.sql_injection_patterns.length} injection patterns detected
                    </div>
                    <div className="space-y-2">
                      {incident.iocs.sql_injection_patterns.slice(0, 2).map((pattern, idx) => (
                        <div key={idx} className="text-xs font-mono bg-red-800/30 p-2 rounded border border-red-700/50">
                          <div className="text-red-200 break-all">{pattern}</div>
                        </div>
                      ))}
                      {incident.iocs.sql_injection_patterns.length > 2 && (
                        <div className="text-xs text-red-400 text-center py-1">
                          +{incident.iocs.sql_injection_patterns.length - 2} more
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {incident.iocs.database_access_patterns?.length > 0 && (
                  <div className="p-4 bg-orange-500/20 border border-orange-500/50 rounded-xl">
                    <h3 className="font-semibold text-orange-300 mb-2 flex items-center gap-2">
                      <Database className="w-4 h-4" />
                      🗄️ DATABASE ACCESS
                    </h3>
                    <div className="text-sm text-orange-400 mb-3">
                      {incident.iocs.database_access_patterns.length} access patterns detected
                    </div>
                    <div className="space-y-2">
                      {incident.iocs.database_access_patterns.slice(0, 2).map((pattern, idx) => (
                        <div key={idx} className="text-xs font-mono bg-orange-800/30 p-2 rounded border border-orange-700/50">
                          <div className="text-orange-200 break-all">{pattern}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {incident.iocs.command_patterns?.length > 0 && (
                  <div className="p-4 bg-orange-500/20 border border-orange-500/50 rounded-xl">
                    <h3 className="font-semibold text-orange-300 mb-2 flex items-center gap-2">
                      <Server className="w-4 h-4" />
                      ⚡ COMMAND EXECUTION
                    </h3>
                    <div className="text-sm text-orange-400 mb-3">
                      {incident.iocs.command_patterns.length} commands detected
                    </div>
                    <div className="space-y-2">
                      {incident.iocs.command_patterns.slice(0, 2).map((cmd, idx) => (
                        <div key={idx} className="text-xs font-mono bg-orange-800/30 p-2 rounded border border-orange-700/50">
                          <div className="text-orange-200 break-all">{cmd}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Attack Timeline - Enhanced */}
        <div className="bg-gray-800/50 border border-gray-700/50 p-6 rounded-xl">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold text-white flex items-center gap-2">
              <Clock className="w-6 h-6 text-blue-400" />
              ⏱️ Attack Timeline
            </h2>
            <button
              onClick={() => toggleSection('timeline')}
              className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
            >
              {expandedSections.timeline ? <ChevronDown className="w-5 h-5" /> : <ChevronRight className="w-5 h-5" />}
            </button>
          </div>
          
          {expandedSections.timeline && (
            <div className="max-h-96 overflow-y-auto space-y-4">
              {incident.attack_timeline?.map((event, idx) => (
                <div key={idx} className="flex gap-4 border-l-4 border-gray-600 pl-6 pb-4 relative group hover:border-blue-500 transition-colors">
                  <div className={`absolute -left-2 w-4 h-4 rounded-full border-2 border-gray-800 ${
                    event.severity === 'critical' ? 'bg-red-500' :
                    event.severity === 'high' ? 'bg-red-400' :
                    event.severity === 'medium' ? 'bg-yellow-400' :
                    'bg-gray-400'
                  }`}></div>
                  
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <span className={`px-3 py-1 text-xs rounded-full font-medium ${
                        event.attack_category === 'web_attack' ? 'bg-red-500/20 text-red-300 border border-red-500/30' :
                        event.attack_category === 'authentication' ? 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/30' :
                        event.attack_category === 'command_execution' ? 'bg-orange-500/20 text-orange-300 border border-orange-500/30' :
                        'bg-gray-700 text-gray-300 border border-gray-600'
                      }`}>
                        {event.attack_category.replace('_', ' ').toUpperCase()}
                      </span>
                      <span className="text-xs text-gray-400 font-mono">
                        {formatTimeAgo(event.timestamp)}
                      </span>
                      <span className={`px-2 py-1 text-xs rounded font-medium ${
                        event.severity === 'critical' ? 'bg-red-500/20 text-red-300' :
                        event.severity === 'high' ? 'bg-red-400/20 text-red-300' :
                        event.severity === 'medium' ? 'bg-yellow-400/20 text-yellow-300' :
                        'bg-gray-600/20 text-gray-300'
                      }`}>
                        {event.severity.toUpperCase()}
                      </span>
                    </div>
                    
                    <div className="text-sm text-gray-300 mb-2 group-hover:text-white transition-colors">
                      {event.description || 'Attack event detected'}
                    </div>
                    
                    <div className="text-xs text-gray-500 font-mono">
                      Event ID: {event.event_id}
                    </div>
                  </div>
                </div>
              ))}
              
              {(!incident.attack_timeline || incident.attack_timeline.length === 0) && (
                <div className="text-center py-8 text-gray-500">
                  <Clock className="w-12 h-12 mx-auto mb-2 opacity-50" />
                  <p>No timeline events available</p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Action History - Enhanced */}
        <div className="bg-gray-800/50 border border-gray-700/50 p-6 rounded-xl">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold text-white flex items-center gap-2">
              <History className="w-6 h-6 text-green-400" />
              🔧 Action History
            </h2>
            <button
              onClick={() => toggleSection('history')}
              className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
            >
              {expandedSections.history ? <ChevronDown className="w-5 h-5" /> : <ChevronRight className="w-5 h-5" />}
            </button>
          </div>
          
          {expandedSections.history && (
            <div className="space-y-3 max-h-80 overflow-y-auto">
              {incident.actions.map((action) => (
                <div key={action.id} className="p-4 bg-gray-700/30 border border-gray-600/50 rounded-lg hover:bg-gray-700/50 transition-colors">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-3">
                      <span className="font-mono text-sm bg-gray-700 text-gray-300 px-3 py-1 rounded-full">
                        {action.action}
                      </span>
                      <span className={`text-xs px-2 py-1 rounded-full font-medium ${
                        action.result === "success" ? "bg-green-500/20 text-green-300 border border-green-500/30" :
                        action.result === "failed" ? "bg-red-500/20 text-red-300 border border-red-500/30" :
                        action.result === "pending" ? "bg-yellow-500/20 text-yellow-300 border border-yellow-500/30" :
                        "bg-gray-600/20 text-gray-300 border border-gray-600/30"
                      }`}>
                        {action.result.toUpperCase()}
                      </span>
                    </div>
                    <span className="text-xs text-gray-400">
                      {formatTimeAgo(action.created_at)}
                    </span>
                  </div>
                  
                  <div className="text-xs text-gray-400">
                    {formatDate(action.created_at)}
                    {action.due_at && (
                      <span className="ml-2">• Due: {formatDate(action.due_at)}</span>
                    )}
                  </div>
                  
                  {action.detail && (
                    <div className="text-xs text-gray-300 mt-2 p-2 bg-gray-800/50 rounded border border-gray-600/30">
                      {action.detail}
                    </div>
                  )}
                </div>
              ))}
              
              {incident.actions.length === 0 && (
                <div className="text-center py-8 text-gray-500">
                  <History className="w-12 h-12 mx-auto mb-2 opacity-50" />
                  <p>No actions taken yet</p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* AI Assistant Chat Interface - Enhanced */}
        <div className="bg-gray-800/50 border border-gray-700/50 p-6 rounded-xl">
          <h2 className="text-xl font-semibold text-white mb-6 flex items-center gap-2">
            <Bot className="w-6 h-6 text-purple-400" />
            🤖 AI Security Analyst
          </h2>
          
          {/* Chat Messages */}
          <div className="bg-gray-900/50 border border-gray-700/50 rounded-xl p-4 h-80 overflow-y-auto mb-4 space-y-4">
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
                
                <div className={`max-w-[80%] ${
                  message.type === 'user'
                    ? 'bg-blue-600/20 border border-blue-500/30 text-blue-100'
                    : 'bg-purple-600/20 border border-purple-500/30 text-purple-100'
                } p-3 rounded-xl`}>
                  {message.loading ? (
                    <div className="flex items-center gap-2">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span className="text-sm">AI is analyzing...</span>
                    </div>
                  ) : (
                    <>
                      <div className="text-sm whitespace-pre-wrap">{message.content}</div>
                      <div className="text-xs opacity-70 mt-2">
                        {formatTimeAgo(message.timestamp.toISOString())}
                      </div>
                    </>
                  )}
                </div>
                
                {message.type === 'user' && (
                  <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center flex-shrink-0">
                    <Users className="w-4 h-4 text-white" />
                  </div>
                )}
              </div>
            ))}
          </div>
          
          {/* Chat Input */}
          <div className="flex gap-3">
            <input
              type="text"
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && sendChatMessage()}
              placeholder="Ask me about this incident... (e.g., 'What IOCs should I investigate?')"
              className="flex-1 bg-gray-700/50 border border-gray-600/50 rounded-lg px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500/50"
              disabled={chatLoading}
            />
            <button
              onClick={sendChatMessage}
              disabled={!chatInput.trim() || chatLoading}
              className="px-4 py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-purple-800 disabled:opacity-50 text-white rounded-lg transition-colors"
              title="Send message"
            >
              {chatLoading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </button>
          </div>
          
          {/* Quick Action Buttons */}
          <div className="flex flex-wrap gap-2 mt-4">
            {[
              "Analyze this timeline",
              "Explain these IOCs", 
              "What should I do next?",
              "Is this a false positive?"
            ].map((prompt) => (
              <button
                key={prompt}
                onClick={() => {
                  setChatInput(prompt);
                  setTimeout(sendChatMessage, 100);
                }}
                disabled={chatLoading}
                className="px-3 py-1 text-xs bg-gray-700/50 hover:bg-gray-600/50 text-gray-300 rounded-full transition-colors border border-gray-600/50"
              >
                {prompt}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
