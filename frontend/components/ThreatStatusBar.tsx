"use client";

import React, { useState, useEffect } from 'react';
import {
  AlertTriangle, CheckCircle, Shield, Clock,
  TrendingUp, Bot
} from 'lucide-react';

interface ThreatStatus {
  attackActive: boolean;
  containmentStatus: 'none' | 'partial' | 'complete';
  agentCount: number;
  workflowCount: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  confidence: number;
  threatCategory: string;
  sourceIp: string;
  duration: string;
  status: string;
}

interface ThreatStatusBarProps {
  incident: any;
  onExpand?: () => void;
}

export default function ThreatStatusBar({ incident, onExpand }: ThreatStatusBarProps) {
  const [threatStatus, setThreatStatus] = useState<ThreatStatus>({
    attackActive: true,
    containmentStatus: 'partial',
    agentCount: 0,
    workflowCount: 0,
    severity: 'high',
    confidence: 0,
    threatCategory: 'Unknown',
    sourceIp: '',
    duration: '',
    status: 'open'
  });

  useEffect(() => {
    if (incident) {
      // Calculate threat status from incident data
      const agentActions = incident.agent_actions || [];
      const advancedActions = incident.advanced_actions || [];
      const workflowActions = advancedActions.filter((a: any) => a.workflow_id);

      // Determine containment status
      let containmentStatus: 'none' | 'partial' | 'complete' = 'none';
      if (incident.auto_contained) {
        containmentStatus = 'complete';
      } else if (agentActions.length > 0 || workflowActions.length > 0) {
        containmentStatus = 'partial';
      }

      // Determine if attack is still active
      const attackActive = incident.status === 'open' || incident.status === 'investigating';

      // Calculate duration
      const createdAt = new Date(incident.created_at);
      const now = new Date();
      const diffMs = now.getTime() - createdAt.getTime();
      const diffMins = Math.floor(diffMs / 60000);
      const diffHours = Math.floor(diffMins / 60);
      const diffDays = Math.floor(diffHours / 24);

      let duration = '';
      if (diffDays > 0) duration = `${diffDays}d ago`;
      else if (diffHours > 0) duration = `${diffHours}h ago`;
      else if (diffMins > 0) duration = `${diffMins}m ago`;
      else duration = 'just now';

      // Map severity
      const severityMap: Record<string, 'low' | 'medium' | 'high' | 'critical'> = {
        'low': 'low',
        'medium': 'medium',
        'high': 'high',
        'critical': 'critical'
      };

      const severity = incident.triage_note?.severity
        ? severityMap[incident.triage_note.severity.toLowerCase()] || 'medium'
        : 'medium';

      setThreatStatus({
        attackActive,
        containmentStatus,
        agentCount: agentActions.length,
        workflowCount: workflowActions.length,
        severity,
        confidence: incident.containment_confidence || incident.agent_confidence || 0,
        threatCategory: incident.threat_category || incident.triage_note?.severity || 'Unknown',
        sourceIp: incident.src_ip || '',
        duration,
        status: incident.status || 'open'
      });
    }
  }, [incident]);

  const getSeverityColor = () => {
    switch (threatStatus.severity) {
      case 'critical': return 'red';
      case 'high': return 'orange';
      case 'medium': return 'yellow';
      case 'low': return 'blue';
      default: return 'gray';
    }
  };

  const getContainmentColor = () => {
    switch (threatStatus.containmentStatus) {
      case 'complete': return 'green';
      case 'partial': return 'yellow';
      case 'none': return 'red';
      default: return 'gray';
    }
  };

  const severityColor = getSeverityColor();
  const containmentColor = getContainmentColor();

  return (
    <div className={`glass-card border-l-4 border-l-${severityColor}-500 mb-6 rounded-lg overflow-hidden`}>
      {/* Main Status Bar */}
      <div className="p-6">
        {/* Top Row: Primary Status */}
        <div className="flex flex-wrap items-center justify-between mb-6 gap-4">
          <div className="flex items-center gap-4 flex-wrap">
            {threatStatus.attackActive ? (
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-red-500/10 border border-red-500/20">
                <div className="h-2 w-2 bg-red-500 rounded-full animate-pulse"></div>
                <span className="text-red-400 font-bold font-heading uppercase tracking-wide text-sm">
                  {threatStatus.status === 'open' ? 'Active Threat' : 'Investigating'}
                </span>
              </div>
            ) : (
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-green-500/10 border border-green-500/20">
                <CheckCircle className="h-4 w-4 text-green-400" />
                <span className="text-green-400 font-bold font-heading uppercase tracking-wide text-sm">
                  Threat Resolved
                </span>
              </div>
            )}

            <div className="h-4 w-px bg-white/10 hidden sm:block"></div>

            <div className="flex items-center gap-2">
              <Shield className={`h-4 w-4 text-${severityColor}-400`} />
              <span className={`text-${severityColor}-300 font-bold font-mono uppercase text-sm`}>
                {threatStatus.threatCategory}
              </span>
            </div>

            <div className="flex items-center gap-2">
              <span className={`px-2 py-0.5 rounded text-[10px] font-bold font-mono bg-${severityColor}-500/10 text-${severityColor}-300 border border-${severityColor}-500/20 uppercase tracking-wider`}>
                {threatStatus.severity} SEVERITY
              </span>
            </div>
          </div>

          <div className="flex items-center gap-3 text-sm text-gray-500 font-mono">
            <div className="flex items-center gap-1.5">
              <Clock className="h-3 w-3" />
              <span>{threatStatus.duration}</span>
            </div>
            {onExpand && (
              <button
                onClick={onExpand}
                className="text-primary hover:text-primary/80 transition-colors font-bold uppercase tracking-wider text-xs ml-4"
              >
                View Timeline →
              </button>
            )}
          </div>
        </div>

        {/* Bottom Row: Status Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {/* Attack Status */}
          <div className={`bg-${severityColor}-500/5 border border-${severityColor}-500/20 rounded-lg p-3 group hover:bg-${severityColor}-500/10 transition-colors`}>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <AlertTriangle className={`h-3 w-3 text-${severityColor}-400`} />
                <span className="text-[10px] font-bold text-gray-400 uppercase tracking-wider font-mono">Attack</span>
              </div>
            </div>
            <div className={`text-lg font-bold text-${severityColor}-300 mb-1 font-heading`}>
              {threatStatus.attackActive ? 'ACTIVE' : 'INACTIVE'}
            </div>
            <div className="text-[10px] text-gray-500 font-mono truncate">
              SRC: {threatStatus.sourceIp}
            </div>
          </div>

          {/* Containment Status */}
          <div className={`bg-${containmentColor}-500/5 border border-${containmentColor}-500/20 rounded-lg p-3 group hover:bg-${containmentColor}-500/10 transition-colors`}>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Shield className={`h-3 w-3 text-${containmentColor}-400`} />
                <span className="text-[10px] font-bold text-gray-400 uppercase tracking-wider font-mono">Containment</span>
              </div>
            </div>
            <div className={`text-lg font-bold text-${containmentColor}-300 mb-1 font-heading uppercase`}>
              {threatStatus.containmentStatus}
            </div>
            <div className="text-[10px] text-gray-500 font-mono">
              {threatStatus.containmentStatus === 'complete' ? 'SECURE' :
               threatStatus.containmentStatus === 'partial' ? 'IN PROGRESS' :
               'VULNERABLE'}
            </div>
          </div>

          {/* Agent Activity */}
          <div className="bg-blue-500/5 border border-blue-500/20 rounded-lg p-3 group hover:bg-blue-500/10 transition-colors">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Bot className="h-3 w-3 text-blue-400" />
                <span className="text-[10px] font-bold text-gray-400 uppercase tracking-wider font-mono">Agents</span>
              </div>
            </div>
            <div className="text-lg font-bold text-blue-300 mb-1 font-heading">
              {threatStatus.agentCount}
            </div>
            <div className="text-[10px] text-gray-500 font-mono">
              ACTIVE DEFENSE UNITS
            </div>
          </div>

          {/* Confidence Score */}
          <div className="bg-purple-500/5 border border-purple-500/20 rounded-lg p-3 group hover:bg-purple-500/10 transition-colors">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <TrendingUp className="h-3 w-3 text-purple-400" />
                <span className="text-[10px] font-bold text-gray-400 uppercase tracking-wider font-mono">Confidence</span>
              </div>
            </div>
            <div className="text-lg font-bold text-purple-300 mb-1 font-heading">
              {Math.round(threatStatus.confidence * 100)}%
            </div>
            <div className="text-[10px] text-gray-500 font-mono">
              AI CERTAINTY SCORE
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
