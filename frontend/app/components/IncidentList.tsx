"use client";

import { useState } from "react";
import Link from "next/link";
import { Filter, Eye } from "lucide-react";

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

interface IncidentListProps {
  incidents: Incident[];
  onIncidentSelect?: (incident: Incident) => void;
}

export default function IncidentList({ incidents, onIncidentSelect, onQuickView }: IncidentListProps) {
  const [filterSeverity, setFilterSeverity] = useState('all');
  const [filterStatus, setFilterStatus] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');

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

  return (
    <div className="space-y-6">
      {/* Filters */}
      <div className="flex items-center gap-4 p-4 bg-surface-2/50 border border-border/50 rounded-xl">
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-text-muted" />
          <span className="text-sm text-text">Filters:</span>
        </div>
        <select
          value={filterSeverity}
          onChange={(e) => setFilterSeverity(e.target.value)}
          className="bg-surface-1 border border-border rounded-lg px-3 py-1 text-sm text-text"
        >
          <option value="all">All Severities</option>
          <option value="high">High</option>
          <option value="medium">Medium</option>
          <option value="low">Low</option>
        </select>
        <select
          value={filterStatus}
          onChange={(e) => setFilterStatus(e.target.value)}
          className="bg-surface-1 border border-border rounded-lg px-3 py-1 text-sm text-text"
        >
          <option value="all">All Statuses</option>
          <option value="open">Open</option>
          <option value="contained">Contained</option>
          <option value="dismissed">Dismissed</option>
        </select>
        <div className="ml-auto text-sm text-text-muted">
          {filteredIncidents.length} of {incidents.length} incidents
        </div>
      </div>

      {/* Incidents List */}
      <div className="space-y-4">
        {filteredIncidents.length === 0 ? (
          <div className="text-center py-12 text-text-muted">
            <p>No incidents match your filters</p>
          </div>
        ) : (
          filteredIncidents.map((incident) => (
            <div key={incident.id} className="bg-surface-0 border border-border/50 hover:border-border rounded-xl overflow-hidden transition-all duration-200 hover:bg-surface-1/50 group">
              <div className="p-4">
                <div className="flex items-center gap-4">
                  <div className={`w-3 h-3 rounded-full ${
                    incident.triage_note?.severity === 'high' ? 'bg-severity-critical' :
                    incident.triage_note?.severity === 'medium' ? 'bg-severity-high' : 'bg-severity-low'
                  }`}></div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-text">Incident #{incident.id}</span>
                      <span className="text-xs text-text-muted">from {incident.src_ip}</span>
                    </div>
                    <p className="text-xs text-text-muted truncate mt-1">{incident.reason}</p>
                  </div>
                  <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <Button
                      size="sm"
                      variant="default"
                      onClick={() => onQuickView?.(incident)}
                      className="gap-1.5"
                    >
                      <MessageSquare className="w-3.5 h-3.5" />
                      <span className="text-xs">Quick View</span>
                    </Button>
                    <Button size="sm" variant="ghost" asChild>
                      <Link href={`/incidents/incident/${incident.id}`}>
                        <ExternalLink className="w-3.5 h-3.5" />
                      </Link>
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
