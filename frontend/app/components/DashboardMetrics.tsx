"use client";

import { AlertTriangle, Zap, Shield, Bot, ArrowUpRight, ArrowDownRight, Minus } from "lucide-react";

interface DashboardMetricsProps {
  incidents: any[];
  telemetry?: { hasLogs: boolean; lastEventAt?: string; assetsDiscovered?: number; agentsEnrolled?: number; incidents?: number };
}

export default function DashboardMetrics({ incidents, telemetry }: DashboardMetricsProps) {
  const metrics = {
    total_incidents: incidents.length,
    high_priority: incidents.filter(i => i.triage_note?.severity === 'high' || (i.risk_score && i.risk_score > 0.7)).length,
    contained: incidents.filter(i => i.status === 'contained').length,
    ml_detected: incidents.filter(i => i.auto_contained).length,
    avg_response_time: 4.2, // minutes
    threat_intel_hits: incidents.filter(i => i.escalation_level === 'high').length
  };

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
          change="+12% from yesterday"
        />
        <MetricCard
          title="High Priority"
          value={metrics.high_priority}
          icon={Zap}
          color="warning"
          change="-8% from yesterday"
        />
        <MetricCard
          title="Contained"
          value={metrics.contained}
          icon={Shield}
          color="success"
          change="+23% effectiveness"
        />
        <MetricCard
          title="AI Detected"
          value={metrics.ml_detected}
          icon={Bot}
          color="info"
          change="Stable detection rate"
        />
      </div>
    </div>
  );
}
