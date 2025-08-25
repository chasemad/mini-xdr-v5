"use client";

import { useEffect, useState } from "react";
import { getIncidents, containIncident, unblockIncident } from "../lib/api";
import Link from "next/link";

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

interface RealTimeIncidentsProps {
  autoRefresh?: boolean;
  refreshInterval?: number;
}

export default function RealTimeIncidents({ 
  autoRefresh = true, 
  refreshInterval = 5000 
}: RealTimeIncidentsProps) {
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState<Record<number, string>>({});
  const [actionResults, setActionResults] = useState<Record<number, {type: 'success' | 'error', message: string}>>({});
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const fetchIncidents = async () => {
    try {
      const data = await getIncidents();
      setIncidents(data);
      setLastUpdated(new Date());
    } catch (error) {
      console.error("Failed to fetch incidents:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchIncidents();
  }, []);

  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(fetchIncidents, refreshInterval);
    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval]);

  const handleQuickAction = async (incidentId: number, action: "contain" | "unblock") => {
    setActionLoading(prev => ({ ...prev, [incidentId]: action }));
    
    try {
      let response;
      if (action === "contain") {
        response = await containIncident(incidentId);
      } else {
        response = await unblockIncident(incidentId);
      }
      
      // Show success message
      setActionResults(prev => ({ 
        ...prev, 
        [incidentId]: { 
          type: 'success', 
          message: `${action === 'contain' ? 'Block' : 'Unblock'} successful: ${response.status}` 
        }
      }));
      
      // Clear success message after 3 seconds
      setTimeout(() => {
        setActionResults(prev => {
          const { [incidentId]: _, ...rest } = prev;
          return rest;
        });
      }, 3000);
      
      // Refresh incidents after action
      await fetchIncidents();
    } catch (error) {
      console.error(`Failed to ${action} incident:`, error);
      
      // Show error message
      setActionResults(prev => ({ 
        ...prev, 
        [incidentId]: { 
          type: 'error', 
          message: `${action === 'contain' ? 'Block' : 'Unblock'} failed: ${error}` 
        }
      }));
      
      // Clear error message after 5 seconds
      setTimeout(() => {
        setActionResults(prev => {
          const { [incidentId]: _, ...rest } = prev;
          return rest;
        });
      }, 5000);
    } finally {
      setActionLoading(prev => {
        const { [incidentId]: _, ...rest } = prev;
        return rest;
      });
    }
  };

  const getSeverityBadge = (severity: string) => {
    const colors = {
      high: "bg-red-100 text-red-800 border-red-200",
      medium: "bg-orange-100 text-orange-800 border-orange-200",
      low: "bg-green-100 text-green-800 border-green-200"
    };
    return colors[severity as keyof typeof colors] || "bg-gray-100 text-gray-800 border-gray-200";
  };

  const getStatusBadge = (status: string) => {
    const colors = {
      open: "bg-yellow-100 text-yellow-800 border-yellow-200",
      contained: "bg-red-100 text-red-800 border-red-200",
      dismissed: "bg-gray-100 text-gray-800 border-gray-200"
    };
    return colors[status as keyof typeof colors] || "bg-gray-100 text-gray-800 border-gray-200";
  };

  if (loading) {
    return (
      <div className="space-y-4">
        {[1, 2, 3].map((i) => (
          <div key={i} className="h-24 bg-gray-200 rounded-xl animate-pulse"></div>
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header with refresh status */}
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-semibold text-gray-900">
          Active Incidents ({incidents.length})
        </h2>
        <div className="flex items-center gap-3">
          {lastUpdated && (
            <span className="text-xs text-gray-500">
              Last updated: {lastUpdated.toLocaleTimeString()}
            </span>
          )}
          <div className={`h-2 w-2 rounded-full ${autoRefresh ? 'bg-green-500' : 'bg-gray-400'}`}>
          </div>
          <span className="text-xs text-gray-500">
            {autoRefresh ? 'Live' : 'Manual'}
          </span>
        </div>
      </div>

      {/* Incidents list */}
      {incidents.length === 0 ? (
        <div className="text-center py-8 bg-green-50 rounded-xl border border-green-200">
          <div className="h-12 w-12 mx-auto bg-green-100 rounded-full flex items-center justify-center mb-3">
            <svg className="h-6 w-6 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <h3 className="text-sm font-medium text-green-900">All Clear</h3>
          <p className="text-sm text-green-700">No active security incidents detected</p>
        </div>
      ) : (
        <div className="space-y-3">
          {incidents.map((incident) => (
            <div
              key={incident.id}
              className="bg-white border border-gray-200 rounded-xl p-6 hover:shadow-lg transition-all duration-200"
            >
              <div className="space-y-4">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <Link href={`/incidents/${incident.id}`}>
                      <h3 className="font-semibold text-gray-900 hover:text-indigo-600 cursor-pointer">
                        Incident #{incident.id}
                      </h3>
                    </Link>
                    
                    <div className="flex items-center gap-2">
                      <span className={`px-2 py-1 text-xs font-medium rounded-full border ${getStatusBadge(incident.status)}`}>
                        {incident.status.toUpperCase()}
                      </span>
                      
                      {incident.auto_contained && (
                        <span className="px-2 py-1 text-xs font-medium rounded-full bg-purple-100 text-purple-800 border border-purple-200">
                          AUTO
                        </span>
                      )}
                      
                      {incident.triage_note && (
                        <span className={`px-2 py-1 text-xs font-medium rounded-full border ${getSeverityBadge(incident.triage_note.severity)}`}>
                          {incident.triage_note.severity.toUpperCase()}
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="space-y-3 text-sm">
                    <div className="flex flex-col sm:flex-row sm:items-center gap-2">
                      <span className="text-gray-600 font-medium min-w-[80px]">Source IP:</span>
                      <span className="font-mono bg-gray-100 px-3 py-2 rounded-lg text-gray-900 text-base font-medium">
                        {incident.src_ip}
                      </span>
                    </div>
                    <div className="flex flex-col sm:flex-row sm:items-center gap-2">
                      <span className="text-gray-600 font-medium min-w-[80px]">Time:</span>
                      <span className="text-gray-900">
                        {new Date(incident.created_at).toLocaleString()}
                      </span>
                    </div>
                    <div className="flex flex-col gap-2">
                      <span className="text-gray-600 font-medium">Reason:</span>
                      <p className="text-gray-700 bg-gray-50 px-3 py-2 rounded-lg">{incident.reason}</p>
                    </div>
                  </div>

                  {incident.triage_note && (
                    <div className="mt-4 p-3 bg-indigo-50 border border-indigo-200 rounded-lg">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-xs font-medium text-indigo-700">AI ANALYSIS:</span>
                        <span className={`px-2 py-1 text-xs font-medium rounded-full border ${getSeverityBadge(incident.triage_note.severity)}`}>
                          {incident.triage_note.severity.toUpperCase()}
                        </span>
                      </div>
                      <p className="text-sm text-indigo-800 font-medium">
                        "{incident.triage_note.summary}"
                      </p>
                      <p className="text-xs text-indigo-600 mt-1">
                        Recommendation: {incident.triage_note.recommendation.replace('_', ' ').toUpperCase()}
                      </p>
                    </div>
                  )}
                </div>

                {/* Action buttons at bottom */}
                <div className="flex items-center justify-between pt-4 border-t border-gray-100">
                  <div className="flex items-center gap-3">
                    {incident.status === "open" && (
                      <button
                        onClick={() => handleQuickAction(incident.id, "contain")}
                        disabled={actionLoading[incident.id] === "contain"}
                        className="px-4 py-2 bg-red-600 text-white text-sm font-medium rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                      >
                        {actionLoading[incident.id] === "contain" ? "Blocking..." : "🚫 Block Now"}
                      </button>
                    )}
                    
                    {incident.status === "contained" && (
                      <button
                        onClick={() => handleQuickAction(incident.id, "unblock")}
                        disabled={actionLoading[incident.id] === "unblock"}
                        className="px-4 py-2 bg-green-600 text-white text-sm font-medium rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                      >
                        {actionLoading[incident.id] === "unblock" ? "Unblocking..." : "✅ Unblock"}
                      </button>
                    )}
                  </div>
                  
                  <Link href={`/incidents/${incident.id}`}>
                    <button className="px-4 py-2 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 transition-colors">
                      📋 Details
                    </button>
                  </Link>
                </div>
              </div>
              
              {/* Action Result Feedback */}
              {actionResults[incident.id] && (
                <div className={`mt-3 p-3 rounded-lg text-sm font-medium ${
                  actionResults[incident.id].type === 'success' 
                    ? 'bg-green-50 text-green-800 border border-green-200' 
                    : 'bg-red-50 text-red-800 border border-red-200'
                }`}>
                  <div className="flex items-center gap-2">
                    <span>
                      {actionResults[incident.id].type === 'success' ? '✅' : '❌'}
                    </span>
                    <span>{actionResults[incident.id].message}</span>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
