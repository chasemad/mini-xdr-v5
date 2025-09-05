import Link from "next/link";
import { 
  AlertTriangle, Shield, Clock, MapPin, TrendingUp, 
  Bot, Zap, CheckCircle, XCircle, AlertCircle 
} from "lucide-react";

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
  triage_note?: {
    summary: string;
    severity: string;
    recommendation: string;
    rationale: string[];
  };
}

interface IncidentCardProps {
  incident: Incident;
}

function StatusBadge({ status, autoContained }: { status: string; autoContained: boolean }) {
  const getStatusConfig = () => {
    switch (status) {
      case "open":
        return {
          bg: "bg-yellow-500/20 border-yellow-500/50",
          text: "text-yellow-200",
          icon: AlertCircle
        };
      case "contained":
        return {
          bg: "bg-red-500/20 border-red-500/50",
          text: "text-red-200",
          icon: XCircle
        };
      case "dismissed":
        return {
          bg: "bg-gray-500/20 border-gray-500/50",
          text: "text-gray-200",
          icon: CheckCircle
        };
      default:
        return {
          bg: "bg-gray-500/20 border-gray-500/50",
          text: "text-gray-200",
          icon: AlertCircle
        };
    }
  };

  const config = getStatusConfig();
  const StatusIcon = config.icon;

  return (
    <div className="flex items-center gap-3">
      <div className={`flex items-center gap-2 px-4 py-2 rounded-2xl border backdrop-blur-sm ${config.bg}`}>
        <StatusIcon className="w-4 h-4" />
        <span className={`text-sm font-bold uppercase ${config.text}`}>
          {status}
        </span>
      </div>
      {autoContained && (
        <div className="flex items-center gap-2 px-3 py-2 rounded-2xl border bg-purple-500/20 border-purple-500/50 backdrop-blur-sm">
          <Shield className="w-4 h-4 text-purple-400" />
          <span className="text-sm font-bold text-purple-200">
            AUTO
          </span>
        </div>
      )}
    </div>
  );
}

function SeverityBadge({ severity }: { severity: string }) {
  const getSeverityConfig = () => {
    switch (severity) {
      case "high":
        return {
          bg: "bg-red-500/20 border-red-500/50",
          text: "text-red-200",
          icon: "🔴"
        };
      case "medium":
        return {
          bg: "bg-orange-500/20 border-orange-500/50",
          text: "text-orange-200",
          icon: "🟡"
        };
      case "low":
        return {
          bg: "bg-green-500/20 border-green-500/50",
          text: "text-green-200",
          icon: "🟢"
        };
      default:
        return {
          bg: "bg-gray-500/20 border-gray-500/50",
          text: "text-gray-200",
          icon: "⚪"
        };
    }
  };

  const config = getSeverityConfig();

  return (
    <div className={`flex items-center gap-2 px-3 py-2 rounded-2xl border backdrop-blur-sm ${config.bg}`}>
      <span className="text-sm">{config.icon}</span>
      <span className={`text-sm font-bold uppercase ${config.text}`}>
        {severity}
      </span>
    </div>
  );
}

export default function IncidentCard({ incident }: IncidentCardProps) {
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

  const getRiskScoreColor = (score?: number) => {
    if (!score) return 'text-gray-400';
    if (score >= 0.75) return 'text-red-400';
    if (score >= 0.5) return 'text-orange-400';
    return 'text-yellow-400';
  };

  const getRiskScoreBg = (score?: number) => {
    if (!score) return 'bg-gray-500/10 border-gray-500/20';
    if (score >= 0.75) return 'bg-red-500/10 border-red-500/20';
    if (score >= 0.5) return 'bg-orange-500/10 border-orange-500/20';
    return 'bg-yellow-500/10 border-yellow-500/20';
  };

  return (
    <Link href={`/incidents/${incident.id}`}>
      <div className="group relative overflow-hidden bg-gradient-to-br from-gray-800/50 to-gray-900/50 backdrop-blur-sm border border-gray-700/50 hover:border-gray-600/50 rounded-3xl shadow-xl hover:shadow-2xl transition-all duration-500 cursor-pointer hover:scale-[1.02]">
        {/* Gradient overlay on hover */}
        <div className="absolute inset-0 bg-gradient-to-r from-blue-500/5 to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
        
        <div className="relative z-10 p-8">
          {/* Header */}
          <div className="flex items-start justify-between mb-6">
            <div className="flex items-center gap-4">
              <div className="p-3 bg-red-500/20 rounded-2xl">
                <AlertTriangle className="w-6 h-6 text-red-400" />
              </div>
              <div>
                <h3 className="text-2xl font-bold text-white group-hover:text-blue-200 transition-colors">
                  🚨 Incident #{incident.id}
                </h3>
                <div className="flex items-center gap-3 text-gray-400 mt-1">
                  <Clock className="w-4 h-4" />
                  <span className="text-sm">{formatTimeAgo(incident.created_at)}</span>
                  <span>•</span>
                  <span className="text-xs font-mono">{formatDate(incident.created_at)}</span>
                </div>
              </div>
            </div>
            <StatusBadge status={incident.status} autoContained={incident.auto_contained} />
          </div>

          {/* Metrics Row */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            {/* Risk Score */}
            <div className={`p-4 rounded-2xl border backdrop-blur-sm ${getRiskScoreBg(incident.risk_score)}`}>
              <div className="flex items-center gap-2 mb-2">
                <AlertTriangle className="w-4 h-4 text-red-400" />
                <span className="text-xs text-gray-400 font-medium">RISK</span>
              </div>
              <div className={`text-xl font-bold ${getRiskScoreColor(incident.risk_score)}`}>
                {incident.risk_score ? `${Math.round(incident.risk_score * 100)}%` : 'N/A'}
              </div>
            </div>

            {/* ML Confidence */}
            <div className="p-4 rounded-2xl border bg-blue-500/10 border-blue-500/20 backdrop-blur-sm">
              <div className="flex items-center gap-2 mb-2">
                <TrendingUp className="w-4 h-4 text-blue-400" />
                <span className="text-xs text-gray-400 font-medium">ML</span>
              </div>
              <div className="text-xl font-bold text-blue-300">
                {incident.agent_confidence ? `${Math.round(incident.agent_confidence * 100)}%` : 'N/A'}
              </div>
            </div>

            {/* Escalation */}
            <div className="p-4 rounded-2xl border bg-purple-500/10 border-purple-500/20 backdrop-blur-sm">
              <div className="flex items-center gap-2 mb-2">
                <Zap className="w-4 h-4 text-purple-400" />
                <span className="text-xs text-gray-400 font-medium">LEVEL</span>
              </div>
              <div className="text-sm font-bold text-purple-300 capitalize">
                {incident.escalation_level || 'High'}
              </div>
            </div>

            {/* Detection */}
            <div className="p-4 rounded-2xl border bg-green-500/10 border-green-500/20 backdrop-blur-sm">
              <div className="flex items-center gap-2 mb-2">
                <Bot className="w-4 h-4 text-green-400" />
                <span className="text-xs text-gray-400 font-medium">TYPE</span>
              </div>
              <div className="text-xs font-bold text-green-300 capitalize">
                {incident.containment_method || 'ML-driven'}
              </div>
            </div>
          </div>

          {/* Source IP and Reason */}
          <div className="space-y-4 mb-6">
            <div className="flex items-center gap-3">
              <MapPin className="w-4 h-4 text-orange-400" />
              <span className="text-sm font-medium text-gray-300">Source IP:</span>
              <span className="text-sm font-mono bg-orange-500/20 border border-orange-500/30 text-orange-200 px-3 py-1 rounded-lg">
                {incident.src_ip}
              </span>
            </div>

            <div className="bg-gray-800/50 border border-gray-700/50 rounded-2xl p-4">
              <div className="flex items-center gap-2 mb-2">
                <AlertCircle className="w-4 h-4 text-yellow-400" />
                <span className="text-sm font-medium text-gray-300">Attack Details:</span>
              </div>
              <p className="text-sm text-gray-400 leading-relaxed">{incident.reason}</p>
            </div>
          </div>

          {/* Triage Note */}
          {incident.triage_note && (
            <div className="bg-gray-700/30 border border-gray-600/50 rounded-2xl p-6 backdrop-blur-sm">
              <div className="flex items-center gap-3 mb-4">
                <SeverityBadge severity={incident.triage_note.severity} />
                <div className="flex items-center gap-2 text-xs">
                  <span className="text-gray-400">Recommendation:</span>
                  <span className={`px-2 py-1 rounded-full font-medium ${
                    incident.triage_note.recommendation === 'contain_now' ? 'bg-red-500/20 text-red-300' :
                    incident.triage_note.recommendation === 'watch' ? 'bg-yellow-500/20 text-yellow-300' :
                    'bg-gray-500/20 text-gray-300'
                  }`}>
                    {incident.triage_note.recommendation.replace('_', ' ').toUpperCase()}
                  </span>
                </div>
              </div>
              <p className="text-sm text-gray-300 leading-relaxed mb-3">{incident.triage_note.summary}</p>
              
              {incident.triage_note.rationale && incident.triage_note.rationale.length > 0 && (
                <div className="space-y-2">
                  <span className="text-xs text-gray-500 font-medium">Key Points:</span>
                  <div className="grid gap-2">
                    {incident.triage_note.rationale.slice(0, 2).map((point, idx) => (
                      <div key={idx} className="flex items-start gap-2">
                        <div className="w-1.5 h-1.5 bg-blue-400 rounded-full mt-2 flex-shrink-0"></div>
                        <span className="text-xs text-gray-400">{point}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Bottom gradient indicator */}
        <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-red-500 via-orange-500 to-yellow-500 opacity-50 group-hover:opacity-100 transition-opacity duration-300"></div>
      </div>
    </Link>
  );
}
