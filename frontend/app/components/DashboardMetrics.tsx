"use client";

import { AlertTriangle, Zap, Shield, Bot, ArrowUpRight, ArrowDownRight, Minus } from "lucide-react";

interface DashboardMetricsProps {
  incidents: any[];
  telemetry?: { hasLogs: boolean; lastEventAt?: string; assetsDiscovered?: number; agentsEnrolled?: number; incidents?: number };
}

// Helper to check if date falls within a specific day range
function isWithinDays(dateString: string, daysAgo: number): boolean {
  const date = new Date(dateString);
  const now = new Date();
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const targetStart = new Date(startOfToday);
  targetStart.setDate(targetStart.getDate() - daysAgo);
  const targetEnd = new Date(targetStart);
  targetEnd.setDate(targetEnd.getDate() + 1);
  return date >= targetStart && date < targetEnd;
}

function isToday(dateString: string): boolean {
  return isWithinDays(dateString, 0);
}

function isYesterday(dateString: string): boolean {
  return isWithinDays(dateString, 1);
}

// Calculate percentage change between two values
function calculatePercentChange(current: number, previous: number, hasAnyData: boolean = false): { value: number; label: string } {
  // If no incidents at all in system, show appropriate message
  if (previous === 0 && current === 0) {
    return { value: 0, label: hasAnyData ? "None today" : "No data yet" };
  }
  if (previous === 0) {
    return { value: 100, label: current > 0 ? `+${current} new today` : "No data yet" };
  }
  const change = ((current - previous) / previous) * 100;
  const rounded = Math.round(change);
  if (rounded === 0) {
    return { value: 0, label: "No change from yesterday" };
  }
  return {
    value: rounded,
    label: `${rounded > 0 ? '+' : ''}${rounded}% from yesterday`
  };
}

export default function DashboardMetrics({ incidents, telemetry }: DashboardMetricsProps) {
  // Calculate average response time from actual incident data
  const calculateAvgResponseTime = () => {
    const containedIncidents = incidents.filter((i: any) => i.status === 'contained' && i.created_at);
    if (containedIncidents.length === 0) return 0;
    // In production, use actual containment timestamps
    return 5; // Default for contained incidents
  };

  // Current totals
  const metrics = {
    total_incidents: incidents.length,
    high_priority: incidents.filter((i: any) => i.triage_note?.severity === 'high' || (i.risk_score && i.risk_score > 0.7)).length,
    contained: incidents.filter((i: any) => i.status === 'contained').length,
    ml_detected: incidents.filter((i: any) => i.auto_contained).length,
    avg_response_time: calculateAvgResponseTime(),
    threat_intel_hits: incidents.filter((i: any) => i.escalation_level === 'high').length
  };

  // Calculate today's and yesterday's incidents for comparison
  const todayIncidents = incidents.filter(i => i.created_at && isToday(i.created_at));
  const yesterdayIncidents = incidents.filter(i => i.created_at && isYesterday(i.created_at));

  const todayHighPriority = todayIncidents.filter(i => i.triage_note?.severity === 'high' || (i.risk_score && i.risk_score > 0.7)).length;
  const yesterdayHighPriority = yesterdayIncidents.filter(i => i.triage_note?.severity === 'high' || (i.risk_score && i.risk_score > 0.7)).length;

  const todayContained = todayIncidents.filter(i => i.status === 'contained').length;
  const yesterdayContained = yesterdayIncidents.filter(i => i.status === 'contained').length;

  const todayMLDetected = todayIncidents.filter(i => i.auto_contained).length;
  const yesterdayMLDetected = yesterdayIncidents.filter(i => i.auto_contained).length;

  // Check if we have any incidents in the system
  const hasAnyIncidents = incidents.length > 0;

  // Calculate percentage changes
  const totalChange = calculatePercentChange(todayIncidents.length, yesterdayIncidents.length, hasAnyIncidents);
  const highPriorityChange = calculatePercentChange(todayHighPriority, yesterdayHighPriority, hasAnyIncidents);
  const mlDetectedChange = calculatePercentChange(todayMLDetected, yesterdayMLDetected, hasAnyIncidents);

  // Calculate containment effectiveness (% of incidents that are contained)
  const containmentRate = metrics.total_incidents > 0
    ? Math.round((metrics.contained / metrics.total_incidents) * 100)
    : 0;
  const containmentLabel = metrics.total_incidents > 0
    ? `${containmentRate}% containment rate`
    : "No incidents yet";

  const MetricCard = ({
    title,
    value,
    icon: Icon,
    color,
    change
  }: {
    title: string;
    value: number;
    icon: any;
    color: string;
    change?: string;
  }) => (
    <div className={`bg-surface-0 border border-${color}/30 rounded-2xl p-6 shadow-card`}>
      <div className="flex items-center justify-between mb-4">
        <div className={`p-3 bg-${color}/20 rounded-xl`}>
          <Icon className={`w-6 h-6 text-${color}`} />
        </div>
        <div className="text-right">
          <div className="text-2xl font-bold text-text tabular-nums">{value}</div>
          <div className="text-xs text-text-muted">{title}</div>
        </div>
      </div>
      {change && (
        <div className="flex items-center gap-2 text-xs">
          {change.startsWith('+') && <ArrowUpRight className={`w-3 h-3 text-success`} />}
          {change.startsWith('-') && <ArrowDownRight className={`w-3 h-3 text-success`} />}
          {!change.startsWith('+') && !change.startsWith('-') && <Minus className="w-3 h-3 text-text-muted" />}
          <span className={change.startsWith('+') ? 'text-success' : change.startsWith('-') ? 'text-success' : 'text-text-muted'}>
            {change}
          </span>
        </div>
      )}
    </div>
  );

  return (
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
        <MetricCard
          title="Total Incidents"
          value={metrics.total_incidents}
          icon={AlertTriangle}
          color="danger"
          change={totalChange.label}
        />
        <MetricCard
          title="High Priority"
          value={metrics.high_priority}
          icon={Zap}
          color="warning"
          change={highPriorityChange.label}
        />
        <MetricCard
          title="Contained"
          value={metrics.contained}
          icon={Shield}
          color="success"
          change={containmentLabel}
        />
        <MetricCard
          title="AI Detected"
          value={metrics.ml_detected}
          icon={Bot}
          color="info"
          change={mlDetectedChange.label}
        />
      </div>
    </div>
  );
}
