"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { getIncident, unblockIncident, containIncident, scheduleUnblock } from "../../lib/api";

interface IncidentDetail {
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
  actions: Array<{
    id: number;
    created_at: string;
    action: string;
    result: string;
    detail: string;
    params: any;
    due_at?: string;
  }>;
  recent_events: Array<{
    ts: string;
    eventid: string;
    message: string;
  }>;
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
  const [scheduleMinutes, setScheduleMinutes] = useState(15);
  const [result, setResult] = useState<string>("");

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
  }, [id]);

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
          <h1 className="text-3xl font-bold text-gray-900">
            Incident #{incident.id}
          </h1>
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

      {/* Details Grid */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Recent Events */}
        <div className="p-6 rounded-2xl shadow-sm border border-gray-200 bg-white">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Recent Events</h2>
          <div className="space-y-3 max-h-80 overflow-y-auto">
            {incident.recent_events.map((event, index) => (
              <div key={index} className="border-b border-gray-100 pb-3 last:border-b-0">
                <div className="text-xs text-gray-500">
                  {formatDate(event.ts)}
                </div>
                <div className="font-mono text-xs text-gray-700 mt-1">
                  {event.eventid}
                </div>
                {event.message && (
                  <div className="text-sm text-gray-600 mt-1">
                    {event.message}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Action History */}
        <div className="p-6 rounded-2xl shadow-sm border border-gray-200 bg-white">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Action History</h2>
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
    </div>
  );
}
