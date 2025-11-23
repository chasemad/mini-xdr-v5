"use client";

import React, { useState } from 'react';
import {
  Shield, Search, AlertOctagon, FileText, Bot,
  MessageSquare, Workflow, TrendingUp, Command
} from 'lucide-react';

interface TacticalDecisionCenterProps {
  incidentId: number;
  onContainNow?: () => void;
  onHuntThreats?: () => void;
  onEscalate?: () => void;
  onCreatePlaybook?: () => void;
  onGenerateReport?: () => void;
  onAskAI?: () => void;
}

export default function TacticalDecisionCenter({
  incidentId,
  onContainNow,
  onHuntThreats,
  onEscalate,
  onCreatePlaybook,
  onGenerateReport,
  onAskAI
}: TacticalDecisionCenterProps) {
  const [processing, setProcessing] = useState<string | null>(null);

  const handleAction = async (actionName: string, callback?: () => void) => {
    if (!callback) return;

    try {
      setProcessing(actionName);
      await callback();
    } catch (err) {
      console.error(`Action ${actionName} failed:`, err);
    } finally {
      setProcessing(null);
    }
  };

  const actionButtons = [
    {
      id: 'contain',
      label: 'Contain Now',
      icon: Shield,
      onClick: onContainNow,
      color: 'red',
      description: 'Emergency containment'
    },
    {
      id: 'hunt',
      label: 'Hunt Threats',
      icon: Search,
      onClick: onHuntThreats,
      color: 'purple',
      description: 'Search for IOCs'
    },
    {
      id: 'escalate',
      label: 'Escalate',
      icon: AlertOctagon,
      onClick: onEscalate,
      color: 'amber',
      description: 'Alert SOC team'
    },
    {
      id: 'playbook',
      label: 'Playbook',
      icon: Workflow,
      onClick: onCreatePlaybook,
      color: 'cyan',
      description: 'Automated response'
    },
    {
      id: 'report',
      label: 'Report',
      icon: FileText,
      onClick: onGenerateReport,
      color: 'emerald',
      description: 'Incident summary'
    },
    {
      id: 'ai',
      label: 'Ask AI',
      icon: MessageSquare,
      onClick: onAskAI,
      color: 'blue',
      description: 'AI assistance'
    }
  ];

  return (
    <div className="bg-gradient-to-br from-background to-background/95 rounded-xl p-6 border border-primary/20 relative overflow-hidden shadow-2xl">
      <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-secondary/5 opacity-30 pointer-events-none" />
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-primary/50 to-transparent" />

      {/* Enhanced Header */}
      <div className="flex items-center justify-between mb-6 relative z-10">
        <div className="flex items-center gap-4">
          <div className="p-3 rounded-lg bg-gradient-to-br from-primary/20 to-primary/10 border border-primary/30 shadow-lg">
            <Command className="w-6 h-6 text-primary" />
          </div>
          <div>
            <h3 className="text-xl font-bold font-heading text-foreground tracking-wide">Tactical Operations</h3>
            <p className="text-xs text-muted-foreground font-mono mt-1 uppercase tracking-wider">COMMAND AUTHORIZATION REQUIRED</p>
          </div>
        </div>

        {/* Status indicator */}
        <div className="flex items-center gap-2 px-3 py-1 bg-green-500/10 border border-green-500/20 rounded-full">
          <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
          <span className="text-xs font-mono text-green-400 uppercase tracking-wide">SYSTEM ACTIVE</span>
        </div>
      </div>

      {/* Action Buttons Grid - Enhanced Control Deck */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 relative z-10">
        {actionButtons.map((action) => {
          const Icon = action.icon;
          const isProcessing = processing === action.id;
          const isDisabled = !action.onClick;

          const colorClasses = {
            red: {
              bg: 'bg-red-500/10 hover:bg-red-500/20',
              border: 'border-red-500/30 hover:border-red-500/60',
              icon: 'text-red-400 group-hover:text-red-300',
              text: 'text-red-300 group-hover:text-red-200'
            },
            purple: {
              bg: 'bg-purple-500/10 hover:bg-purple-500/20',
              border: 'border-purple-500/30 hover:border-purple-500/60',
              icon: 'text-purple-400 group-hover:text-purple-300',
              text: 'text-purple-300 group-hover:text-purple-200'
            },
            amber: {
              bg: 'bg-amber-500/10 hover:bg-amber-500/20',
              border: 'border-amber-500/30 hover:border-amber-500/60',
              icon: 'text-amber-400 group-hover:text-amber-300',
              text: 'text-amber-300 group-hover:text-amber-200'
            },
            cyan: {
              bg: 'bg-cyan-500/10 hover:bg-cyan-500/20',
              border: 'border-cyan-500/30 hover:border-cyan-500/60',
              icon: 'text-cyan-400 group-hover:text-cyan-300',
              text: 'text-cyan-300 group-hover:text-cyan-200'
            },
            emerald: {
              bg: 'bg-emerald-500/10 hover:bg-emerald-500/20',
              border: 'border-emerald-500/30 hover:border-emerald-500/60',
              icon: 'text-emerald-400 group-hover:text-emerald-300',
              text: 'text-emerald-300 group-hover:text-emerald-200'
            },
            blue: {
              bg: 'bg-blue-500/10 hover:bg-blue-500/20',
              border: 'border-blue-500/30 hover:border-blue-500/60',
              icon: 'text-blue-400 group-hover:text-blue-300',
              text: 'text-blue-300 group-hover:text-blue-200'
            }
          };

          const colors = colorClasses[action.color as keyof typeof colorClasses] || colorClasses.blue;

          return (
            <button
              key={action.id}
              onClick={() => handleAction(action.id, action.onClick)}
              disabled={isDisabled || isProcessing}
              className={`
                group relative aspect-square w-full
                ${colors.bg} ${colors.border}
                rounded-lg transition-all duration-200
                flex flex-col items-center justify-center gap-2
                disabled:opacity-50 disabled:cursor-not-allowed
                hover:shadow-lg hover:shadow-black/20
              `}
            >
              {/* Icon */}
              <div className="p-1.5">
                <Icon className={`w-4 h-4 ${colors.icon} transition-colors`} />
              </div>

              {/* Label */}
              <div className="text-center px-1">
                <span className={`block text-[10px] font-medium font-mono uppercase tracking-wide ${colors.text} leading-tight transition-colors`}>
                  {action.label}
                </span>
              </div>

              {/* Processing Indicator */}
              {isProcessing && (
                <div className={`absolute inset-0 flex items-center justify-center ${colors.bg} rounded-lg border ${colors.border}`}>
                  <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                </div>
              )}

              {/* Status accent */}
              <div className={`absolute top-0 right-0 w-1 h-1 rounded-full ${action.color}-500/60`} />
            </button>
          );
        })}
      </div>
    </div>
  );
}
