"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { getIncident, unblockIncident, containIncident, scheduleUnblock } from "../../lib/api";
import { 
  Ban, Shield, UserX, Key, Database, History, UserMinus, Lock, 
  Search, Globe, Network, Clock, Target, Download, Archive, FileText, 
  Briefcase, TrendingUp, Code, Users, Crosshair, Flag, MapPin, 
  AlertTriangle, Ticket, Bell, Phone, Server 
} from "lucide-react";

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

export default function IncidentDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const router = useRouter();
  const [id, setId] = useState<number | null>(null);

  useEffect(() => {
    params.then((resolvedParams) => {
      setId(Number(resolvedParams.id));
    });
  }, [params]);
  const [incident, setIncident] = useState<IncidentDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [actionResults, setActionResults] = useState<any>({});

  // Handle SOC action clicks
  const handleAction = async (actionType: string, parameter: string) => {
    setActionLoading(actionType);
    try {
      // Simulate API call for now - replace with actual API calls
      console.log(`Executing ${actionType} with parameter: ${parameter}`);
      
      // Mock response for demonstration
      const mockResponses: { [key: string]: string } = {
        'block_ip': `✅ IP ${parameter} blocked successfully`,
        'isolate_host': `✅ Host ${parameter} isolated from network`,
        'revoke_sessions': `✅ All ${parameter} sessions revoked`,
        'reset_passwords': `✅ ${parameter} passwords reset`,
        'check_db_integrity': `✅ Database integrity check initiated`,
        'virustotal_lookup': `🔍 VirusTotal: ${parameter} - Clean (0/70 engines)`,
        'abuseipdb_query': `🔍 AbuseIPDB: ${parameter} - Confidence: 85% malicious`,
        'dns_sinkhole': `✅ DNS sinkhole deployed for ${parameter}`,
        'deploy_waf_rules': `✅ WAF rules deployed for ${parameter} attacks`,
        'capture_traffic': `✅ Traffic capture started for ${parameter}`,
        'hunt_similar_attacks': `🎯 Found 3 similar ${parameter} attacks in last 30 days`,
        'alert_senior_analysts': `📧 Senior analysts notified about incident ${parameter}`
      };
      
      const result = mockResponses[actionType] || `✅ ${actionType} executed for ${parameter}`;
      
      // Update action results
      setActionResults(prev => ({
        ...prev,
        [actionType]: result
      }));
      
      // Show toast notification (you can add a toast library)
      alert(result);
      
    } catch (error) {
      console.error('Action failed:', error);
      alert('❌ Action failed');
    } finally {
      setActionLoading(null);
    }
  };
  const [scheduleMinutes, setScheduleMinutes] = useState(15);
  const [result, setResult] = useState<string>("");
  const [autoRefreshing, setAutoRefreshing] = useState(false);

  useEffect(() => {
    const fetchIncident = async () => {
      if (!id) return;
      
      try {
        const data = await getIncident(id);
        setIncident(data);
      } catch (error) {
        console.error("Failed to fetch incident:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchIncident();

    // Set up automatic refresh every 5 seconds to catch scheduled actions
    const interval = setInterval(async () => {
      if (id && !loading) {
        setAutoRefreshing(true);
        await fetchIncident();
        setAutoRefreshing(false);
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [id, loading]);

  const handleUnblock = async () => {
    if (!incident || !id) return;
    
    setActionLoading("unblock");
    try {
      const response = await unblockIncident(incident.id);
      setResult(`Unblock: ${response.status}`);
      
      // Refresh incident data
      const updatedIncident = await getIncident(id);
      setIncident(updatedIncident);
    } catch (error) {
      setResult(`Unblock failed: ${error}`);
    } finally {
      setActionLoading(null);
    }
  };

  const handleContain = async (durationSeconds?: number) => {
    if (!incident || !id) return;
    
    setActionLoading("contain");
    try {
      const response = await containIncident(incident.id, durationSeconds);
      const duration = durationSeconds ? ` for ${durationSeconds}s` : "";
      setResult(`Contain${duration}: ${response.status}`);
      
      // Refresh incident data
      const updatedIncident = await getIncident(id);
      setIncident(updatedIncident);
    } catch (error) {
      setResult(`Contain failed: ${error}`);
    } finally {
      setActionLoading(null);
    }
  };

  const handleScheduleUnblock = async () => {
    if (!incident || !id) return;
    
    setActionLoading("schedule");
    try {
      const response = await scheduleUnblock(incident.id, scheduleMinutes);
      setResult(`Scheduled unblock: ${response.status} at ${response.due_at}`);
      
      // Refresh incident data
      const updatedIncident = await getIncident(id);
      setIncident(updatedIncident);
    } catch (error) {
      setResult(`Schedule failed: ${error}`);
    } finally {
      setActionLoading(null);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "high":
        return "bg-red-100 text-red-800";
      case "medium":
        return "bg-orange-100 text-orange-800";
      case "low":
        return "bg-green-100 text-green-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "open":
        return "bg-yellow-100 text-yellow-800";
      case "contained":
        return "bg-red-100 text-red-800";
      case "dismissed":
        return "bg-gray-100 text-gray-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

  if (loading || !id) {
    return (
      <div className="px-4 py-6">
        <div className="animate-pulse space-y-6">
          <div className="h-8 bg-gray-200 rounded w-1/3"></div>
          <div className="h-64 bg-gray-200 rounded-2xl"></div>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="h-80 bg-gray-200 rounded-2xl"></div>
            <div className="h-80 bg-gray-200 rounded-2xl"></div>
          </div>
        </div>
      </div>
    );
  }

  if (!incident) {
    return (
      <div className="px-4 py-6">
        <div className="text-center py-12">
          <h3 className="text-lg font-medium text-gray-900">Incident not found</h3>
          <p className="mt-2 text-gray-600">The requested incident could not be found.</p>
          <button
            onClick={() => router.push("/incidents")}
            className="mt-4 px-4 py-2 bg-indigo-600 text-white rounded-xl hover:bg-indigo-700"
          >
            Back to Incidents
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="px-4 py-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-3xl font-bold text-gray-900">
              Incident #{incident.id}
            </h1>
            {autoRefreshing && (
              <div className="flex items-center gap-2 text-sm text-blue-600">
                <div className="animate-spin h-4 w-4 border-2 border-blue-600 border-t-transparent rounded-full"></div>
                <span>Refreshing...</span>
              </div>
            )}
          </div>
          <p className="mt-2 text-gray-600">
            {formatDate(incident.created_at)}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(incident.status)}`}>
            {incident.status}
          </span>
          {incident.auto_contained && (
            <span className="px-3 py-1 rounded-full text-sm font-medium bg-purple-100 text-purple-800">
              AUTO-CONTAINED
            </span>
          )}
        </div>
      </div>

      {/* Triage Note */}
      {incident.triage_note && (
        <div className="p-6 rounded-2xl shadow-sm border border-gray-200 bg-white">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Triage Analysis</h2>
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <span className={`px-3 py-1 rounded-full text-sm font-medium ${getSeverityColor(incident.triage_note.severity)}`}>
                Severity: {incident.triage_note.severity}
              </span>
              <span className="px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-800">
                Recommend: {incident.triage_note.recommendation}
              </span>
            </div>
            <p className="text-gray-700">{incident.triage_note.summary}</p>
            <div>
              <h4 className="font-medium text-gray-900 mb-2">Analysis:</h4>
              <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
                {incident.triage_note.rationale?.map((point, index) => (
                  <li key={index}>{point}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="p-6 rounded-2xl shadow-sm border border-gray-200 bg-white">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Actions</h2>
        <div className="space-y-4">
          {/* Immediate Actions */}
          <div className="flex flex-wrap gap-4 items-center">
            <button
              onClick={() => handleContain()}
              disabled={actionLoading === "contain"}
              className="px-4 py-2 bg-red-600 text-white rounded-xl hover:bg-red-700 disabled:opacity-50"
            >
              {actionLoading === "contain" ? "Containing..." : "Block Permanently"}
            </button>
            
            <button
              onClick={() => handleContain(30)}
              disabled={actionLoading === "contain"}
              className="px-4 py-2 bg-orange-600 text-white rounded-xl hover:bg-orange-700 disabled:opacity-50"
            >
              {actionLoading === "contain" ? "Containing..." : "Block 30s (Test)"}
            </button>
            
            <button
              onClick={handleUnblock}
              disabled={actionLoading === "unblock"}
              className="px-4 py-2 bg-green-600 text-white rounded-xl hover:bg-green-700 disabled:opacity-50"
            >
              {actionLoading === "unblock" ? "Unblocking..." : "Unblock Now"}
            </button>
          </div>
          
          {/* Scheduled Actions */}
          <div className="p-4 bg-gray-50 rounded-lg">
            <h3 className="text-sm font-medium text-gray-700 mb-3">Scheduled Unblock</h3>
            <div className="flex items-center gap-2">
              <input
                type="number"
                min={1}
                max={1440}
                value={scheduleMinutes}
                onChange={(e) => setScheduleMinutes(parseInt(e.target.value) || 15)}
                className="border border-gray-300 rounded-lg px-3 py-2 w-20 text-sm"
              />
              <span className="text-sm text-gray-600">minutes</span>
              <button
                onClick={handleScheduleUnblock}
                disabled={actionLoading === "schedule"}
                className="px-4 py-2 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:opacity-50"
              >
                {actionLoading === "schedule" ? "Scheduling..." : "Schedule Unblock"}
              </button>
            </div>
            <p className="text-xs text-gray-500 mt-2">
              Automatically unblock this IP after the specified time period
            </p>
          </div>
        </div>
        
        {result && (
          <div className="mt-4 p-3 bg-gray-50 rounded-xl">
            <pre className="text-sm text-gray-700 whitespace-pre-wrap">{result}</pre>
          </div>
        )}
      </div>

      {/* Enhanced SOC Analysis Section */}
      <div className="p-6 rounded-2xl shadow-sm border border-gray-200 bg-white">
        <h2 className="text-lg font-semibold text-gray-900 mb-6">SOC Analysis Dashboard</h2>
        
        {/* Risk & Confidence Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-gradient-to-r from-red-50 to-red-100 p-4 rounded-xl">
            <div className="text-sm text-red-600 font-medium">Risk Score</div>
            <div className="text-2xl font-bold text-red-900">
              {incident.risk_score ? (incident.risk_score * 100).toFixed(0) + '%' : 'N/A'}
            </div>
          </div>
          <div className="bg-gradient-to-r from-blue-50 to-blue-100 p-4 rounded-xl">
            <div className="text-sm text-blue-600 font-medium">ML Confidence</div>
            <div className="text-2xl font-bold text-blue-900">
              {incident.agent_confidence ? (incident.agent_confidence * 100).toFixed(0) + '%' : 'N/A'}
            </div>
          </div>
          <div className="bg-gradient-to-r from-purple-50 to-purple-100 p-4 rounded-xl">
            <div className="text-sm text-purple-600 font-medium">Escalation</div>
            <div className="text-xl font-bold text-purple-900 capitalize">
              {incident.escalation_level || 'Medium'}
            </div>
          </div>
          <div className="bg-gradient-to-r from-green-50 to-green-100 p-4 rounded-xl">
            <div className="text-sm text-green-600 font-medium">Detection</div>
            <div className="text-sm font-bold text-green-900 capitalize">
              {incident.containment_method || 'Rule-based'}
            </div>
          </div>
        </div>

        {/* Threat Intelligence */}
        {incident.threat_category && (
          <div className="mb-6 p-4 bg-yellow-50 rounded-xl border border-yellow-200">
            <h3 className="font-semibold text-yellow-800 mb-2">🎯 Threat Classification</h3>
            <span className="px-3 py-1 bg-yellow-200 text-yellow-800 rounded-full text-sm font-medium">
              {incident.threat_category.replace(/_/g, ' ').toUpperCase()}
            </span>
            {incident.agent_id && (
              <div className="mt-2 text-sm text-yellow-700">
                Analyzed by: <span className="font-mono">{incident.agent_id}</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Critical SOC Analysis - Compromise Assessment */}
      <div className="p-6 rounded-2xl shadow-sm border border-gray-200 bg-white">
        <h2 className="text-lg font-semibold text-gray-900 mb-6">🚨 Compromise Assessment & Impact Analysis</h2>
        
        {/* Success Indicators */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
          {/* Successful Authentication */}
          {incident.iocs?.successful_auth_indicators?.length > 0 && (
            <div className="p-4 bg-red-50 rounded-xl border border-red-300">
              <h3 className="font-semibold text-red-800 mb-2">🚨 COMPROMISE CONFIRMED</h3>
              <div className="text-sm text-red-700 mb-2">Successful authentication detected!</div>
              <div className="space-y-1">
                {incident.iocs.successful_auth_indicators.slice(0, 3).map((indicator, idx) => (
                  <div key={idx} className="text-xs font-mono bg-red-100 p-2 rounded text-red-800 break-all">
                    {indicator}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Database Access */}
          {incident.iocs?.database_access_patterns?.length > 0 && (
            <div className="p-4 bg-orange-50 rounded-xl border border-orange-300">
              <h3 className="font-semibold text-orange-800 mb-2">🗄️ DATABASE ACCESS</h3>
              <div className="text-sm text-orange-700 mb-2">Data access patterns detected</div>
              <div className="space-y-1">
                {incident.iocs.database_access_patterns.slice(0, 2).map((pattern, idx) => (
                  <div key={idx} className="text-xs font-mono bg-orange-100 p-2 rounded text-orange-800 break-all">
                    {pattern}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Privilege Escalation */}
          {incident.iocs?.privilege_escalation_indicators?.length > 0 && (
            <div className="p-4 bg-purple-50 rounded-xl border border-purple-300">
              <h3 className="font-semibold text-purple-800 mb-2">⬆️ PRIVILEGE ESCALATION</h3>
              <div className="text-sm text-purple-700 mb-2">Escalation attempts detected</div>
              <div className="space-y-1">
                {incident.iocs.privilege_escalation_indicators.slice(0, 2).map((indicator, idx) => (
                  <div key={idx} className="text-xs font-mono bg-purple-100 p-2 rounded text-purple-800 break-all">
                    {indicator}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Post-Exploitation Activity */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          {/* Data Exfiltration */}
          {incident.iocs?.data_exfiltration_indicators?.length > 0 && (
            <div className="p-4 bg-red-50 rounded-xl border border-red-200">
              <h3 className="font-semibold text-red-800 mb-2">📤 DATA EXFILTRATION</h3>
              <div className="space-y-1">
                {incident.iocs.data_exfiltration_indicators.slice(0, 2).map((indicator, idx) => (
                  <div key={idx} className="text-xs font-mono bg-red-100 p-2 rounded text-red-700 break-all">
                    {indicator}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Persistence */}
          {incident.iocs?.persistence_mechanisms?.length > 0 && (
            <div className="p-4 bg-yellow-50 rounded-xl border border-yellow-200">
              <h3 className="font-semibold text-yellow-800 mb-2">🔒 PERSISTENCE</h3>
              <div className="space-y-1">
                {incident.iocs.persistence_mechanisms.slice(0, 2).map((mechanism, idx) => (
                  <div key={idx} className="text-xs font-mono bg-yellow-100 p-2 rounded text-yellow-700 break-all">
                    {mechanism}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Lateral Movement */}
          {incident.iocs?.lateral_movement_indicators?.length > 0 && (
            <div className="p-4 bg-pink-50 rounded-xl border border-pink-200">
              <h3 className="font-semibold text-pink-800 mb-2">↔️ LATERAL MOVEMENT</h3>
              <div className="space-y-1">
                {incident.iocs.lateral_movement_indicators.slice(0, 2).map((indicator, idx) => (
                  <div key={idx} className="text-xs font-mono bg-pink-100 p-2 rounded text-pink-700 break-all">
                    {indicator}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Reconnaissance */}
          {incident.iocs?.reconnaissance_patterns?.length > 0 && (
            <div className="p-4 bg-blue-50 rounded-xl border border-blue-200">
              <h3 className="font-semibold text-blue-800 mb-2">🔍 RECONNAISSANCE</h3>
              <div className="space-y-1">
                {incident.iocs.reconnaissance_patterns.slice(0, 2).map((pattern, idx) => (
                  <div key={idx} className="text-xs font-mono bg-blue-100 p-2 rounded text-blue-700 break-all">
                    {pattern}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Traditional IOCs */}
      <div className="p-6 rounded-2xl shadow-sm border border-gray-200 bg-white">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">🔍 Technical Indicators of Compromise</h2>
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* SQL Injection Patterns */}
          {incident.iocs?.sql_injection_patterns?.length > 0 && (
            <div className="p-4 bg-red-50 rounded-xl border border-red-200">
              <h3 className="font-semibold text-red-800 mb-2">🚨 SQL Injection</h3>
              <div className="space-y-1">
                {incident.iocs.sql_injection_patterns.slice(0, 3).map((pattern, idx) => (
                  <div key={idx} className="text-xs font-mono bg-red-100 p-2 rounded text-red-700 break-all">
                    {pattern}
                  </div>
                ))}
                {incident.iocs.sql_injection_patterns.length > 3 && (
                  <div className="text-xs text-red-600">
                    +{incident.iocs.sql_injection_patterns.length - 3} more patterns
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Command Patterns */}
          {incident.iocs?.command_patterns?.length > 0 && (
            <div className="p-4 bg-orange-50 rounded-xl border border-orange-200">
              <h3 className="font-semibold text-orange-800 mb-2">⚡ Commands</h3>
              <div className="space-y-1">
                {incident.iocs.command_patterns.slice(0, 3).map((cmd, idx) => (
                  <div key={idx} className="text-xs font-mono bg-orange-100 p-2 rounded text-orange-700 break-all">
                    {cmd}
                  </div>
                ))}
                {incident.iocs.command_patterns.length > 3 && (
                  <div className="text-xs text-orange-600">
                    +{incident.iocs.command_patterns.length - 3} more commands
                  </div>
                )}
              </div>
            </div>
          )}

          {/* URLs/Paths */}
          {incident.iocs?.urls?.length > 0 && (
            <div className="p-4 bg-blue-50 rounded-xl border border-blue-200">
              <h3 className="font-semibold text-blue-800 mb-2">🌐 URLs/Paths</h3>
              <div className="space-y-1">
                {incident.iocs.urls.slice(0, 3).map((url, idx) => (
                  <div key={idx} className="text-xs font-mono bg-blue-100 p-2 rounded text-blue-700 break-all">
                    {url}
                  </div>
                ))}
                {incident.iocs.urls.length > 3 && (
                  <div className="text-xs text-blue-600">
                    +{incident.iocs.urls.length - 3} more URLs
                  </div>
                )}
              </div>
            </div>
          )}

          {/* IP Addresses */}
          {incident.iocs?.ip_addresses?.length > 0 && (
            <div className="p-4 bg-purple-50 rounded-xl border border-purple-200">
              <h3 className="font-semibold text-purple-800 mb-2">🌍 IP Addresses</h3>
              <div className="space-y-1">
                {incident.iocs.ip_addresses.slice(0, 3).map((ip, idx) => (
                  <div key={idx} className="text-xs font-mono bg-purple-100 p-2 rounded text-purple-700">
                    {ip}
                  </div>
                ))}
                {incident.iocs.ip_addresses.length > 3 && (
                  <div className="text-xs text-purple-600">
                    +{incident.iocs.ip_addresses.length - 3} more IPs
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Attack Timeline */}
      <div className="p-6 rounded-2xl shadow-sm border border-gray-200 bg-white">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">⏱️ Attack Timeline</h2>
        <div className="max-h-96 overflow-y-auto">
          <div className="space-y-3">
            {incident.attack_timeline?.map((event, idx) => (
              <div key={idx} className="flex gap-4 border-l-4 border-gray-200 pl-4 pb-4 relative">
                {/* Severity indicator */}
                <div className={`absolute -left-2 w-4 h-4 rounded-full ${
                  event.severity === 'critical' ? 'bg-red-600' :
                  event.severity === 'high' ? 'bg-red-400' :
                  event.severity === 'medium' ? 'bg-yellow-400' :
                  'bg-gray-400'
                }`}></div>
                
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`px-2 py-1 text-xs rounded-full ${
                      event.attack_category === 'web_attack' ? 'bg-red-100 text-red-700' :
                      event.attack_category === 'authentication' ? 'bg-yellow-100 text-yellow-700' :
                      event.attack_category === 'command_execution' ? 'bg-orange-100 text-orange-700' :
                      'bg-gray-100 text-gray-700'
                    }`}>
                      {event.attack_category.replace('_', ' ')}
                    </span>
                    <span className="text-xs text-gray-500 font-mono">
                      {new Date(event.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  
                  <div className="text-sm text-gray-900 mb-1">
                    {event.description}
                  </div>
                  
                  <div className="text-xs text-gray-500 font-mono">
                    {event.event_id}
                  </div>
                  
                  {/* Raw data preview */}
                  {event.raw_data && Object.keys(event.raw_data).length > 0 && (
                    <details className="mt-2">
                      <summary className="text-xs text-blue-600 cursor-pointer hover:text-blue-800">
                        View raw data
                      </summary>
                      <pre className="text-xs bg-gray-50 p-2 rounded mt-1 overflow-x-auto">
                        {JSON.stringify(event.raw_data, null, 2)}
                      </pre>
                    </details>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Details Grid */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Detailed Events */}
        <div className="p-6 rounded-2xl shadow-sm border border-gray-200 bg-white">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            📊 Detailed Events ({incident.event_summary?.total_events || 0})
          </h2>
          <div className="space-y-3 max-h-80 overflow-y-auto">
            {incident.detailed_events?.map((event, index) => (
              <div key={event.id} className="border-b border-gray-100 pb-3 last:border-b-0">
                <div className="flex items-center gap-2 mb-1">
                  <div className="text-xs text-gray-500">
                    {formatDate(event.ts)}
                  </div>
                  <span className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded">
                    {event.source_type}
                  </span>
                </div>
                <div className="font-mono text-xs text-gray-700 mb-1">
                  {event.eventid}
                </div>
                {event.message && (
                  <div className="text-sm text-gray-600 mb-1">
                    {event.message}
                  </div>
                )}
                {event.dst_port && (
                  <div className="text-xs text-gray-500">
                    Port: {event.dst_port}
                  </div>
                )}
                {/* Raw event data */}
                {event.raw && Object.keys(event.raw).length > 0 && (
                  <details className="mt-2">
                    <summary className="text-xs text-blue-600 cursor-pointer hover:text-blue-800">
                      View raw event
                    </summary>
                    <pre className="text-xs bg-gray-50 p-2 rounded mt-1 overflow-x-auto max-h-32">
                      {JSON.stringify(event.raw, null, 2)}
                    </pre>
                  </details>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Action History */}
        <div className="p-6 rounded-2xl shadow-sm border border-gray-200 bg-white">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">🔧 Action History</h2>
          <div className="space-y-3 max-h-80 overflow-y-auto">
            {incident.actions.map((action) => (
              <div key={action.id} className="border-b border-gray-100 pb-3 last:border-b-0">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-xs bg-gray-100 px-2 py-1 rounded">
                    {action.action}
                  </span>
                  <span className={`text-xs px-2 py-1 rounded ${
                    action.result === "success" ? "bg-green-100 text-green-800" :
                    action.result === "failed" ? "bg-red-100 text-red-800" :
                    action.result === "pending" ? "bg-yellow-100 text-yellow-800" :
                    "bg-gray-100 text-gray-800"
                  }`}>
                    {action.result}
                  </span>
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  {formatDate(action.created_at)}
                  {action.due_at && (
                    <span className="ml-2">• Due: {formatDate(action.due_at)}</span>
                  )}
                </div>
                {action.detail && (
                  <div className="text-xs text-gray-600 mt-2 bg-gray-50 p-2 rounded">
                    {action.detail}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Advanced SOC Capabilities */}
      <div className="p-6 rounded-2xl shadow-sm border border-gray-200 bg-white">
        <h2 className="text-lg font-semibold text-gray-900 mb-6">🎯 Advanced Response & Investigation</h2>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {/* Threat Intelligence */}
          <div className="p-4 border border-gray-200 rounded-xl">
            <h3 className="font-semibold text-gray-800 mb-3">🌐 Threat Intelligence</h3>
            <div className="space-y-2">
              <button className="w-full text-left px-3 py-2 text-sm bg-blue-50 hover:bg-blue-100 rounded-lg border border-blue-200">
                Query VirusTotal for IP reputation
              </button>
              <button className="w-full text-left px-3 py-2 text-sm bg-blue-50 hover:bg-blue-100 rounded-lg border border-blue-200">
                Check AlienVault OTX feeds
              </button>
              <button className="w-full text-left px-3 py-2 text-sm bg-blue-50 hover:bg-blue-100 rounded-lg border border-blue-200">
                Lookup in AbuseIPDB
              </button>
              <button className="w-full text-left px-3 py-2 text-sm bg-blue-50 hover:bg-blue-100 rounded-lg border border-blue-200">
                Search MISP threat feeds
              </button>
            </div>
          </div>

          {/* Network Actions */}
          <div className="p-4 border border-gray-200 rounded-xl">
            <h3 className="font-semibold text-gray-800 mb-3">🔒 Network Response</h3>
            <div className="space-y-2">
              <button className="w-full text-left px-3 py-2 text-sm bg-red-50 hover:bg-red-100 rounded-lg border border-red-200">
                Block IP at firewall
              </button>
              <button className="w-full text-left px-3 py-2 text-sm bg-red-50 hover:bg-red-100 rounded-lg border border-red-200">
                Add to DNS sinkhole
              </button>
              <button className="w-full text-left px-3 py-2 text-sm bg-orange-50 hover:bg-orange-100 rounded-lg border border-orange-200">
                Rate limit IP traffic
              </button>
              <button className="w-full text-left px-3 py-2 text-sm bg-orange-50 hover:bg-orange-100 rounded-lg border border-orange-200">
                Deploy honeypot redirect
              </button>
            </div>
          </div>

          {/* Forensic Actions */}
          <div className="p-4 border border-gray-200 rounded-xl">
            <h3 className="font-semibold text-gray-800 mb-3">🔬 Forensics</h3>
            <div className="space-y-2">
              <button className="w-full text-left px-3 py-2 text-sm bg-purple-50 hover:bg-purple-100 rounded-lg border border-purple-200">
                Capture network traffic
              </button>
              <button className="w-full text-left px-3 py-2 text-sm bg-purple-50 hover:bg-purple-100 rounded-lg border border-purple-200">
                Generate evidence package
              </button>
              <button className="w-full text-left px-3 py-2 text-sm bg-purple-50 hover:bg-purple-100 rounded-lg border border-purple-200">
                Export IOCs to STIX/TAXII
              </button>
              <button className="w-full text-left px-3 py-2 text-sm bg-purple-50 hover:bg-purple-100 rounded-lg border border-purple-200">
                Create case in SOAR
              </button>
            </div>
          </div>

          {/* Hunting & Pivoting */}
          <div className="p-4 border border-gray-200 rounded-xl">
            <h3 className="font-semibold text-gray-800 mb-3">🎯 Threat Hunting</h3>
            <div className="space-y-2">
              <button className="w-full text-left px-3 py-2 text-sm bg-green-50 hover:bg-green-100 rounded-lg border border-green-200">
                Hunt for similar attacks
              </button>
              <button className="w-full text-left px-3 py-2 text-sm bg-green-50 hover:bg-green-100 rounded-lg border border-green-200">
                Check other honeypots
              </button>
              <button className="w-full text-left px-3 py-2 text-sm bg-green-50 hover:bg-green-100 rounded-lg border border-green-200">
                Analyze attack patterns
              </button>
              <button className="w-full text-left px-3 py-2 text-sm bg-green-50 hover:bg-green-100 rounded-lg border border-green-200">
                Generate hunt queries
              </button>
            </div>
          </div>

          {/* Attribution */}
          <div className="p-4 border border-gray-200 rounded-xl">
            <h3 className="font-semibold text-gray-800 mb-3">🕵️ Attribution</h3>
            <div className="space-y-2">
              <button className="w-full text-left px-3 py-2 text-sm bg-yellow-50 hover:bg-yellow-100 rounded-lg border border-yellow-200">
                Link to threat actors
              </button>
              <button className="w-full text-left px-3 py-2 text-sm bg-yellow-50 hover:bg-yellow-100 rounded-lg border border-yellow-200">
                Analyze TTPs (MITRE ATT&CK)
              </button>
              <button className="w-full text-left px-3 py-2 text-sm bg-yellow-50 hover:bg-yellow-100 rounded-lg border border-yellow-200">
                Check campaign signatures
              </button>
              <button className="w-full text-left px-3 py-2 text-sm bg-yellow-50 hover:bg-yellow-100 rounded-lg border border-yellow-200">
                Geolocation analysis
              </button>
            </div>
          </div>

          {/* Escalation */}
          <div className="p-4 border border-gray-200 rounded-xl">
            <h3 className="font-semibold text-gray-800 mb-3">📢 Escalation</h3>
            <div className="space-y-2">
              <button className="w-full text-left px-3 py-2 text-sm bg-indigo-50 hover:bg-indigo-100 rounded-lg border border-indigo-200">
                Alert senior analysts
              </button>
              <button className="w-full text-left px-3 py-2 text-sm bg-indigo-50 hover:bg-indigo-100 rounded-lg border border-indigo-200">
                Create JIRA ticket
              </button>
              <button className="w-full text-left px-3 py-2 text-sm bg-indigo-50 hover:bg-indigo-100 rounded-lg border border-indigo-200">
                Notify threat intel team
              </button>
              <button className="w-full text-left px-3 py-2 text-sm bg-indigo-50 hover:bg-indigo-100 rounded-lg border border-indigo-200">
                Schedule war room
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* AI Assistant Chat Interface */}
      <div className="p-6 rounded-2xl shadow-sm border border-gray-200 bg-white">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">🤖 AI Security Analyst</h2>
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 p-4 rounded-xl border border-blue-200 mb-4">
          <div className="flex items-start gap-3">
            <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center text-white text-sm font-bold">
              AI
            </div>
            <div className="flex-1">
              <div className="text-sm text-gray-700 mb-2">
                <strong>AI Security Analyst:</strong> I've analyzed incident #{incident.id} involving {incident.src_ip}. 
                {incident.iocs?.sql_injection_patterns?.length > 0 && 
                  ` I detected ${incident.iocs.sql_injection_patterns.length} SQL injection patterns indicating a web application attack.`
                }
                {incident.risk_score && incident.risk_score > 0.7 && 
                  ` The risk score of ${(incident.risk_score * 100).toFixed(0)}% suggests this is a high-priority threat.`
                }
              </div>
              <div className="text-sm text-blue-700">
                <strong>Recommendations:</strong>
                <ul className="list-disc list-inside mt-1 space-y-1">
                  <li>Immediate IP blocking implemented - monitoring for evasion attempts</li>
                  <li>Check web application logs for successful exploitation</li>
                  <li>Validate database integrity and check for data exfiltration</li>
                  <li>Deploy additional monitoring for similar attack patterns</li>
                  {incident.escalation_level === 'high' && 
                    <li>Consider escalating to incident response team due to high severity</li>
                  }
                </ul>
              </div>
            </div>
          </div>
        </div>

        {/* Chat Interface */}
        <div className="border border-gray-200 rounded-xl">
          <div className="p-4 bg-gray-50 border-b border-gray-200 rounded-t-xl">
            <h3 className="font-medium text-gray-800">Chat with AI about this incident</h3>
            <p className="text-sm text-gray-600">Ask questions about attack patterns, recommendations, or next steps</p>
          </div>
          
          <div className="p-4 max-h-60 overflow-y-auto space-y-3">
            {/* Example AI responses */}
            <div className="flex gap-3">
              <div className="w-6 h-6 bg-blue-600 rounded-full flex items-center justify-center text-white text-xs">
                AI
              </div>
              <div className="flex-1 bg-blue-50 p-3 rounded-lg">
                <p className="text-sm text-gray-700">
                  Based on the attack patterns, this appears to be automated SQL injection reconnaissance. 
                  The attacker used basic injection patterns but hasn't shown signs of advanced persistence techniques.
                </p>
              </div>
            </div>
          </div>
          
          <div className="p-4 border-t border-gray-200">
            <div className="flex gap-2">
              <input
                type="text"
                placeholder="Ask me about this incident... (e.g., 'What are the next steps?' or 'Should we escalate this?')"
                className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm font-medium">
                Ask AI
              </button>
            </div>
            <div className="mt-2 flex flex-wrap gap-2">
              <button className="px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded-full text-xs text-gray-700">
                What TTPs were used?
              </button>
              <button className="px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded-full text-xs text-gray-700">
                Should we escalate?
              </button>
              <button className="px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded-full text-xs text-gray-700">
                Generate hunt queries
              </button>
              <button className="px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded-full text-xs text-gray-700">
                Create MITRE mapping
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Playbook Recommendations */}
      <div className="p-6 rounded-2xl shadow-sm border border-gray-200 bg-white">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">📋 Response Playbooks</h2>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="border border-orange-200 bg-orange-50 p-4 rounded-xl">
            <h3 className="font-semibold text-orange-800 mb-2">🌐 Web Application Attack Response</h3>
            <div className="text-sm text-orange-700 space-y-2">
              <div className="flex items-center gap-2">
                <span className="w-4 h-4 bg-green-500 rounded-full text-xs flex items-center justify-center text-white">✓</span>
                <span>IP blocking activated</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-4 h-4 bg-yellow-500 rounded-full text-xs flex items-center justify-center text-white">2</span>
                <span>Check application logs for successful exploitation</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-4 h-4 bg-gray-400 rounded-full text-xs flex items-center justify-center text-white">3</span>
                <span>Validate database integrity</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-4 h-4 bg-gray-400 rounded-full text-xs flex items-center justify-center text-white">4</span>
                <span>Update WAF rules</span>
              </div>
            </div>
          </div>

          <div className="border border-blue-200 bg-blue-50 p-4 rounded-xl">
            <h3 className="font-semibold text-blue-800 mb-2">🕵️ Advanced Threat Hunting</h3>
            <div className="text-sm text-blue-700 space-y-2">
              <div className="flex items-center gap-2">
                <span className="w-4 h-4 bg-gray-400 rounded-full text-xs flex items-center justify-center text-white">1</span>
                <span>Search for similar IPs in last 30 days</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-4 h-4 bg-gray-400 rounded-full text-xs flex items-center justify-center text-white">2</span>
                <span>Analyze SQL injection payload sophistication</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-4 h-4 bg-gray-400 rounded-full text-xs flex items-center justify-center text-white">3</span>
                <span>Check for privilege escalation attempts</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-4 h-4 bg-gray-400 rounded-full text-xs flex items-center justify-center text-white">4</span>
                <span>Generate threat intelligence report</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
