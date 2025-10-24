"use client";

import React, { useState } from 'react';
import { 
  Shield, Search, AlertOctagon, FileText, Bot, 
  MessageSquare, Workflow, TrendingUp
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
      gradient: 'from-red-600 to-orange-600',
      hoverGradient: 'from-red-700 to-orange-700',
      description: 'Emergency containment'
    },
    {
      id: 'hunt',
      label: 'Hunt Threats',
      icon: Search,
      onClick: onHuntThreats,
      gradient: 'from-purple-600 to-pink-600',
      hoverGradient: 'from-purple-700 to-pink-700',
      description: 'Search for IOCs'
    },
    {
      id: 'escalate',
      label: 'Escalate',
      icon: AlertOctagon,
      onClick: onEscalate,
      gradient: 'from-yellow-600 to-orange-600',
      hoverGradient: 'from-yellow-700 to-orange-700',
      description: 'Alert SOC team'
    },
    {
      id: 'playbook',
      label: 'Create Playbook',
      icon: Workflow,
      onClick: onCreatePlaybook,
      gradient: 'from-blue-600 to-cyan-600',
      hoverGradient: 'from-blue-700 to-cyan-700',
      description: 'Automated response'
    },
    {
      id: 'report',
      label: 'Generate Report',
      icon: FileText,
      onClick: onGenerateReport,
      gradient: 'from-gray-600 to-gray-700',
      hoverGradient: 'from-gray-700 to-gray-800',
      description: 'Incident summary'
    },
    {
      id: 'ai',
      label: 'Ask AI',
      icon: MessageSquare,
      onClick: onAskAI,
      gradient: 'from-purple-600 to-blue-600',
      hoverGradient: 'from-purple-700 to-blue-700',
      description: 'AI assistance'
    }
  ];

  return (
    <div className="bg-gradient-to-br from-gray-800/90 to-gray-900/90 backdrop-blur-sm border border-gray-700 rounded-xl p-6 shadow-2xl">
      {/* Header */}
      <div className="flex items-center gap-2 mb-4">
        <TrendingUp className="w-5 h-5 text-blue-400" />
        <h3 className="text-lg font-bold text-white">Tactical Decision Center</h3>
        <span className="text-xs text-gray-400 ml-auto">Quick Actions</span>
      </div>

      {/* Action Buttons Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        {actionButtons.map((action) => {
          const Icon = action.icon;
          const isProcessing = processing === action.id;
          const isDisabled = !action.onClick;

          return (
            <button
              key={action.id}
              onClick={() => handleAction(action.id, action.onClick)}
              disabled={isDisabled || isProcessing}
              className={`
                group relative bg-gradient-to-br ${action.gradient}
                hover:${action.hoverGradient}
                disabled:opacity-50 disabled:cursor-not-allowed
                text-white rounded-lg p-4 
                transition-all duration-200 transform hover:scale-105 hover:shadow-lg
                flex flex-col items-center justify-center gap-2
                ${isProcessing ? 'animate-pulse' : ''}
              `}
              title={action.description}
            >
              {/* Icon */}
              <div className="relative">
                <Icon className={`w-6 h-6 ${isProcessing ? 'animate-spin' : ''}`} />
                {isProcessing && (
                  <div className="absolute inset-0 bg-white/20 rounded-full animate-ping"></div>
                )}
              </div>

              {/* Label */}
              <span className="text-xs font-bold text-center leading-tight">
                {action.label}
              </span>

              {/* Tooltip on hover */}
              <div className="absolute -bottom-10 left-1/2 transform -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-10">
                <div className="bg-gray-900 border border-gray-700 px-3 py-1 rounded text-xs text-gray-300 whitespace-nowrap shadow-lg">
                  {action.description}
                </div>
              </div>

              {/* Glow effect on hover */}
              <div className="absolute inset-0 rounded-lg bg-white/0 group-hover:bg-white/10 transition-all duration-200"></div>
            </button>
          );
        })}
      </div>

      {/* Info Text */}
      <div className="mt-4 text-xs text-gray-400 text-center">
        Click any action for immediate response • AI-powered recommendations available
      </div>
    </div>
  );
}

