"use client";

import { useEffect, useState } from "react";
import { getIncidents, unblockIncident } from "../lib/api";

interface Incident {
  id: number;
  src_ip: string;
  status: string;
  created_at: string;
  reason: string;
  auto_contained: boolean;
}

interface BlockedIP {
  ip: string;
  incidentId: number;
  blockedAt: string;
  reason: string;
  autoBlocked: boolean;
}

export default function BlockedIPs() {
  const [blockedIPs, setBlockedIPs] = useState<BlockedIP[]>([]);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState<Record<string, boolean>>({});

  const fetchBlockedIPs = async () => {
    try {
      const incidents: Incident[] = await getIncidents();
      
      // Filter for contained incidents to show currently blocked IPs
      const blocked = incidents
        .filter((incident: Incident) => incident.status === "contained")
        .map((incident: Incident) => ({
          ip: incident.src_ip,
          incidentId: incident.id,
          blockedAt: incident.created_at,
          reason: incident.reason,
          autoBlocked: incident.auto_contained
        }));
      
      setBlockedIPs(blocked);
    } catch (error) {
      console.error("Failed to fetch blocked IPs:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchBlockedIPs();
    
    // Refresh every 10 seconds
    const interval = setInterval(fetchBlockedIPs, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleUnblock = async (ip: string, incidentId: number) => {
    setActionLoading(prev => ({ ...prev, [ip]: true }));
    
    try {
      await unblockIncident(incidentId);
      await fetchBlockedIPs(); // Refresh the list
    } catch (error) {
      console.error(`Failed to unblock ${ip}:`, error);
    } finally {
      setActionLoading(prev => ({ ...prev, [ip]: false }));
    }
  };

  if (loading) {
    return (
      <div className="bg-white border border-gray-200 rounded-xl p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Blocked IPs</h2>
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-16 bg-gray-200 rounded-lg animate-pulse"></div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-6">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-semibold text-gray-900">
          Blocked IPs ({blockedIPs.length})
        </h2>
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 bg-red-500 rounded-full"></div>
          <span className="text-xs text-gray-500">Active Blocks</span>
        </div>
      </div>

      {blockedIPs.length === 0 ? (
        <div className="text-center py-6 bg-green-50 rounded-lg border border-green-200">
          <div className="h-10 w-10 mx-auto bg-green-100 rounded-full flex items-center justify-center mb-2">
            <svg className="h-5 w-5 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <p className="text-sm text-green-700 font-medium">No IPs Currently Blocked</p>
          <p className="text-xs text-green-600">All threats have been resolved</p>
        </div>
      ) : (
        <div className="space-y-3 max-h-80 overflow-y-auto">
          {blockedIPs.map((blocked) => (
            <div
              key={blocked.ip}
              className="flex items-center justify-between p-3 bg-red-50 border border-red-200 rounded-lg"
            >
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                  <span className="font-mono text-sm font-semibold text-red-900">
                    {blocked.ip}
                  </span>
                  {blocked.autoBlocked && (
                    <span className="px-2 py-1 text-xs font-medium bg-purple-100 text-purple-800 rounded-full">
                      AUTO
                    </span>
                  )}
                </div>
                <p className="text-xs text-red-700 mb-1">{blocked.reason}</p>
                <p className="text-xs text-red-600">
                  Blocked: {new Date(blocked.blockedAt).toLocaleString()}
                </p>
              </div>
              
              <button
                onClick={() => handleUnblock(blocked.ip, blocked.incidentId)}
                disabled={actionLoading[blocked.ip]}
                className="px-3 py-1 bg-green-600 text-white text-xs rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {actionLoading[blocked.ip] ? "Unblocking..." : "Unblock"}
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
