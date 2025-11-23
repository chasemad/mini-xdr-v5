"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import {
  AlertTriangle, Shield, Eye, Filter,
  Globe, MoreHorizontal, Search, RefreshCw,
  Inbox, Bot, MessageSquare, ExternalLink
} from "lucide-react";
import { DashboardLayout } from "@/components/DashboardLayout";
import { getIncidents } from "../lib/api";
import { useDashboard } from "../contexts/DashboardContext";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { IncidentQuickView } from "@/components/IncidentQuickView";

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

export default function IncidentsPage() {
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [loading, setLoading] = useState(true);
  const [autoRefreshing, setAutoRefreshing] = useState(false);
  const [filterSeverity, setFilterSeverity] = useState('all');
  const [filterStatus, setFilterStatus] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [quickViewOpen, setQuickViewOpen] = useState(false);
  const [selectedIncident, setSelectedIncident] = useState<Incident | null>(null);

  const { toggleCopilot, setCopilotContext } = useDashboard();

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
    const interval = setInterval(async () => {
      if (!loading) {
        setAutoRefreshing(true);
        await fetchIncidents();
        setAutoRefreshing(false);
      }
    }, 10000);
    return () => clearInterval(interval);
  }, [fetchIncidents, loading]);

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
      case 'high': return 'text-red-500 border-red-500/20 bg-red-500/10';
      case 'medium': return 'text-orange-500 border-orange-500/20 bg-orange-500/10';
      case 'low': return 'text-green-500 border-green-500/20 bg-green-500/10';
      default: return 'text-muted-foreground border-border bg-muted';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'open': return 'text-yellow-500 border-yellow-500/20 bg-yellow-500/10';
      case 'contained': return 'text-red-500 border-red-500/20 bg-red-500/10';
      case 'dismissed': return 'text-muted-foreground border-border bg-muted';
      default: return 'text-muted-foreground border-border bg-muted';
    }
  };

  const handleChatWithAI = (e: React.MouseEvent, incident: Incident) => {
    e.stopPropagation();
    e.preventDefault();
    setCopilotContext({ incidentId: incident.id, incidentData: incident });
    toggleCopilot();
  };

  const handleQuickView = (e: React.MouseEvent, incident: Incident) => {
    e.stopPropagation();
    e.preventDefault();
    setSelectedIncident(incident);
    setQuickViewOpen(true);
  };

  return (
    <DashboardLayout breadcrumbs={[{ label: "Incidents" }]}>
      {/* Header Controls */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
        <div className="flex items-center gap-2">
          <div className="relative">
            <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search incidents..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9 h-9 w-64 rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
            />
          </div>

          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-muted-foreground ml-2" />
            <select
              value={filterSeverity}
              onChange={(e) => setFilterSeverity(e.target.value)}
              className="h-9 rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            >
              <option value="all">All Severities</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
            </select>
            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              className="h-9 rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            >
              <option value="all">All Statuses</option>
              <option value="open">Open</option>
              <option value="contained">Contained</option>
              <option value="dismissed">Dismissed</option>
            </select>
          </div>
        </div>

        {autoRefreshing && (
          <div className="flex items-center gap-2 px-3 py-1 bg-primary/10 rounded-full self-start md:self-center">
            <RefreshCw className="w-3 h-3 text-primary animate-spin" />
            <span className="text-xs font-medium text-primary">Live Updates</span>
          </div>
        )}
      </div>

      {/* Empty State */}
      {!loading && filteredIncidents.length === 0 && (
        <div className="flex flex-col items-center justify-center py-20 text-center border border-dashed border-border rounded-xl bg-muted/5">
          <div className="p-4 bg-muted rounded-full mb-4">
            <Inbox className="w-8 h-8 text-muted-foreground" />
          </div>
          <h3 className="text-lg font-semibold mb-1">No incidents found</h3>
          <p className="text-sm text-muted-foreground max-w-sm mb-6">
            {searchQuery || filterSeverity !== 'all' || filterStatus !== 'all'
              ? "Try adjusting your filters or search query to find what you're looking for."
              : "Great job! There are no active incidents requiring your attention right now."}
          </p>
          {(searchQuery || filterSeverity !== 'all' || filterStatus !== 'all') && (
            <Button
              variant="outline"
              onClick={() => {
                setSearchQuery('');
                setFilterSeverity('all');
                setFilterStatus('all');
              }}
            >
              Clear Filters
            </Button>
          )}
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div className="space-y-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-48 rounded-xl bg-muted/50 animate-pulse" />
          ))}
        </div>
      )}

      {/* Incidents List */}
      <div className="grid gap-4">
        {filteredIncidents.map((incident) => (
          <Card
            key={incident.id}
            className="group transition-all hover:shadow-md hover:border-primary/20 overflow-hidden"
          >
            <div className="p-6">
              <div className="flex flex-col md:flex-row md:items-start justify-between gap-4 mb-6">
                <div className="flex items-start gap-4">
                  <div className={cn("p-3 rounded-xl",
                    incident.triage_note?.severity === 'high' ? 'bg-red-500/10' :
                    incident.triage_note?.severity === 'medium' ? 'bg-orange-500/10' : 'bg-blue-500/10'
                  )}>
                    <AlertTriangle className={cn("w-5 h-5",
                      incident.triage_note?.severity === 'high' ? 'text-red-500' :
                      incident.triage_note?.severity === 'medium' ? 'text-orange-500' : 'text-blue-500'
                    )} />
                  </div>

                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <h3 className="text-lg font-semibold">Incident #{incident.id}</h3>
                      {incident.auto_contained && (
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-purple-500/10 text-purple-500 border border-purple-500/20 uppercase tracking-wide">
                          Auto-Contained
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-3 text-sm text-muted-foreground">
                      <span className="flex items-center gap-1.5">
                        <Globe className="w-3.5 h-3.5" />
                        {incident.src_ip}
                      </span>
                      <span>•</span>
                      <span>{formatTimeAgo(incident.created_at)}</span>
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <span className={cn("px-2.5 py-1 rounded-full text-xs font-medium border capitalize", getSeverityColor(incident.triage_note?.severity))}>
                    {incident.triage_note?.severity || 'Unknown'} Severity
                  </span>
                  <span className={cn("px-2.5 py-1 rounded-full text-xs font-medium border capitalize", getStatusColor(incident.status))}>
                    {incident.status}
                  </span>
                </div>
              </div>

              {/* Stats Grid */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                {[
                  { label: 'Risk Score', value: incident.risk_score ? `${Math.round(incident.risk_score * 100)}%` : 'N/A', color: 'text-red-500' },
                  { label: 'Confidence', value: incident.agent_confidence ? `${Math.round(incident.agent_confidence * 100)}%` : 'N/A', color: 'text-blue-500' },
                  { label: 'Escalation', value: incident.escalation_level || 'Medium', color: 'text-purple-500', capitalize: true },
                  { label: 'Detection', value: incident.containment_method || 'ML-driven', color: 'text-green-500', capitalize: true },
                ].map((stat, i) => (
                  <div key={i} className="bg-muted/30 border border-border rounded-lg p-3 text-center">
                    <div className={cn("text-xl font-bold mb-0.5", stat.color, stat.capitalize && "capitalize")}>
                      {stat.value}
                    </div>
                    <div className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider">{stat.label}</div>
                  </div>
                ))}
              </div>

              <div className="bg-muted/30 rounded-lg p-4 mb-6 border border-border/50">
                <p className="text-sm text-muted-foreground leading-relaxed">{incident.reason}</p>
              </div>

              <div className="flex items-center justify-between pt-4 border-t border-border">
                <div className="flex items-center gap-3">
                  <Button
                    variant="default"
                    size="sm"
                    onClick={(e) => handleQuickView(e, incident)}
                  >
                    <MessageSquare className="w-4 h-4 mr-2" />
                    Quick View
                  </Button>
                  <Button asChild variant="outline" size="sm">
                    <Link href={`/incidents/incident/${incident.id}`}>
                      <ExternalLink className="w-4 h-4 mr-2" />
                      Full Analysis
                    </Link>
                  </Button>
                </div>
                <Button variant="ghost" size="icon">
                  <MoreHorizontal className="w-4 h-4 text-muted-foreground" />
                </Button>
              </div>
            </div>
          </Card>
        ))}
      </div>

      {/* Quick View Drawer */}
      <IncidentQuickView
        open={quickViewOpen}
        onOpenChange={setQuickViewOpen}
        incident={selectedIncident}
        onExecuteAction={async (actionType) => {
          // Quick action handler - you can expand this
          console.log(`Execute ${actionType} for incident ${selectedIncident?.id}`);
        }}
      />
    </DashboardLayout>
  );
}
