"use client";

import React, { useState, useEffect } from 'react';
import { 
  AlertTriangle, CheckCircle, Shield, Activity, Clock,
  TrendingUp, Zap, Bot, Eye
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
    <div className={`border-l-4 border-${severityColor}-500 bg-gradient-to-r from-${severityColor}-500/10 to-transparent mb-6 rounded-lg overflow-hidden`}>
      {/* Main Status Bar */}
      <div className="bg-gray-800/90 backdrop-blur-sm p-6">
        {/* Top Row: Primary Status */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-4">
            {threatStatus.attackActive ? (
              <div className="flex items-center gap-2">
                <div className={`h-3 w-3 bg-${severityColor}-500 rounded-full animate-pulse`}></div>
                <span className={`text-${severityColor}-400 font-bold text-lg uppercase tracking-wide`}>
                  {threatStatus.status === 'open' ? 'Active Threat' : 'Investigating'}
                </span>
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <CheckCircle className="h-5 w-5 text-green-400" />
                <span className="text-green-400 font-bold text-lg uppercase tracking-wide">
                  Threat Resolved
                </span>
              </div>
            )}
            
            <div className="h-6 w-px bg-gray-600"></div>
            
            <div className="flex items-center gap-2">
              <Shield className={`h-5 w-5 text-${severityColor}-400`} />
              <span className={`text-${severityColor}-300 font-semibold uppercase text-sm`}>
                {threatStatus.threatCategory}
              </span>
            </div>

            <div className="h-6 w-px bg-gray-600"></div>

            <div className="flex items-center gap-2">
              <span className={`px-3 py-1 rounded-full text-xs font-bold bg-${severityColor}-500/20 text-${severityColor}-300 border border-${severityColor}-500/30`}>
                {threatStatus.severity.toUpperCase()} SEVERITY
              </span>
            </div>
          </div>

          <div className="flex items-center gap-3 text-sm text-gray-400">
            <div className="flex items-center gap-1">
              <Clock className="h-4 w-4" />
              <span>{threatStatus.duration}</span>
            </div>
            {onExpand && (
              <button
                onClick={onExpand}
                className="text-blue-400 hover:text-blue-300 transition-colors font-medium"
              >
                View Timeline →
              </button>
            )}
          </div>
        </div>

        {/* Bottom Row: Status Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {/* Attack Status */}
          <div className={`bg-${severityColor}-500/10 border border-${severityColor}-500/30 rounded-lg p-4`}>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <AlertTriangle className={`h-4 w-4 text-${severityColor}-400`} />
                <span className="text-xs font-semibold text-gray-300 uppercase">Attack</span>
              </div>
            </div>
            <div className={`text-2xl font-bold text-${severityColor}-300 mb-1`}>
              {threatStatus.attackActive ? 'ACTIVE' : 'INACTIVE'}
            </div>
            <div className="text-xs text-gray-400">
              Source: <span className="font-mono text-gray-300">{threatStatus.sourceIp}</span>
            </div>
          </div>

          {/* Containment Status */}
          <div className={`bg-${containmentColor}-500/10 border border-${containmentColor}-500/30 rounded-lg p-4`}>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Shield className={`h-4 w-4 text-${containmentColor}-400`} />
                <span className="text-xs font-semibold text-gray-300 uppercase">Containment</span>
              </div>
            </div>
            <div className={`text-2xl font-bold text-${containmentColor}-300 mb-1 uppercase`}>
              {threatStatus.containmentStatus}
            </div>
            <div className="text-xs text-gray-400">
              {threatStatus.containmentStatus === 'complete' ? 'Fully Contained' :
               threatStatus.containmentStatus === 'partial' ? 'In Progress' :
               'Not Contained'}
            </div>
          </div>

          {/* Agent Activity */}
          <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Bot className="h-4 w-4 text-blue-400" />
                <span className="text-xs font-semibold text-gray-300 uppercase">AI Agents</span>
              </div>
            </div>
            <div className="text-2xl font-bold text-blue-300 mb-1">
              {threatStatus.agentCount}
            </div>
            <div className="text-xs text-gray-400">
              {threatStatus.agentCount === 0 ? 'No actions yet' :
               threatStatus.agentCount === 1 ? '1 agent acting' :
               `${threatStatus.agentCount} agents acting`}
            </div>
          </div>

          {/* Confidence Score */}
          <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <TrendingUp className="h-4 w-4 text-purple-400" />
                <span className="text-xs font-semibold text-gray-300 uppercase">Confidence</span>
              </div>
            </div>
            <div className="text-2xl font-bold text-purple-300 mb-1">
              {Math.round(threatStatus.confidence * 100)}%
            </div>
            <div className="text-xs text-gray-400">
              {threatStatus.confidence >= 0.8 ? 'Very High' :
               threatStatus.confidence >= 0.6 ? 'High' :
               threatStatus.confidence >= 0.4 ? 'Medium' :
               'Low'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

