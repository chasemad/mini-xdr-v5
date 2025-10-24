"use client";

import React, { useState, useEffect } from 'react';
import { 
  Brain, Lightbulb, AlertTriangle, Target, Shield, 
  Zap, RefreshCw, ChevronDown, ChevronUp, CheckCircle,
  TrendingUp, Globe, Info, Loader2
} from 'lucide-react';

interface AIRecommendation {
  action: string;
  displayName: string;
  reason: string;
  impact: string;
  priority: 'high' | 'medium' | 'low';
  estimatedDuration?: string;
  parameters?: Record<string, any>;
}

interface AIAnalysis {
  summary: string;
  severity: string;
  recommendation: string;
  rationale: string[];
  confidence_score: number;
  threat_attribution?: string;
  estimated_impact?: string;
  next_steps?: string[];
  recommendations?: AIRecommendation[];
}

interface EnhancedAIAnalysisProps {
  incident: any;
  onExecuteRecommendation?: (action: string, params?: Record<string, any>) => Promise<void>;
  onExecuteAllRecommendations?: () => Promise<void>;
}

export default function EnhancedAIAnalysis({ 
  incident, 
  onExecuteRecommendation,
  onExecuteAllRecommendations 
}: EnhancedAIAnalysisProps) {
  const [analysis, setAnalysis] = useState<AIAnalysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedRationale, setExpandedRationale] = useState(false);
  const [executingAction, setExecutingAction] = useState<string | null>(null);
  const [executedActions, setExecutedActions] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (incident?.id) {
      generateAIAnalysis();
    }
  }, [incident?.id]);

  const generateAIAnalysis = async () => {
    if (!incident?.id) return;

    try {
      setLoading(true);
      setError(null);

      const response = await fetch(`http://localhost:8000/api/incidents/${incident.id}/ai-analysis`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': process.env.NEXT_PUBLIC_API_KEY || 'demo-minixdr-api-key'
        },
        body: JSON.stringify({
          provider: 'openai',
          analysis_type: 'comprehensive',
          include_recommendations: true
        })
      });

      if (!response.ok) {
        throw new Error(`AI analysis failed: ${response.statusText}`);
      }

      const data = await response.json();
      if (data.success) {
        // Generate actionable recommendations from analysis
        const recommendations = generateRecommendations(data.analysis, incident);
        setAnalysis({ ...data.analysis, recommendations });
      } else {
        throw new Error(data.error || 'AI analysis failed');
      }

    } catch (err) {
      console.error('AI analysis failed:', err);
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const generateRecommendations = (analysis: AIAnalysis, incident: any): AIRecommendation[] => {
    const recommendations: AIRecommendation[] = [];

    // Based on analysis, generate actionable recommendations
    const severity = analysis.severity?.toLowerCase();
    const category = incident.threat_category?.toLowerCase();

    // High priority: Block source IP
    if (incident.src_ip && !incident.auto_contained) {
      recommendations.push({
        action: 'block_ip',
        displayName: `Block IP: ${incident.src_ip}`,
        reason: 'Prevent further attacks from this source',
        impact: 'Source IP will be blocked for 30 minutes',
        priority: 'high',
        estimatedDuration: '< 1 minute',
        parameters: { ip: incident.src_ip, duration: 30 }
      });
    }

    // High priority: Isolate affected host if ransomware or malware
    if (category && ['ransomware', 'malware'].includes(category)) {
      recommendations.push({
        action: 'isolate_host',
        displayName: 'Isolate Affected Host',
        reason: 'Stop lateral movement and prevent spread',
        impact: 'Host will be isolated from network until cleared',
        priority: 'high',
        estimatedDuration: '< 2 minutes'
      });
    }

    // Medium priority: Force password reset if credential compromise
    if (category && ['brute_force', 'credential_access'].includes(category)) {
      recommendations.push({
        action: 'reset_passwords',
        displayName: 'Force Password Reset',
        reason: 'Compromised credentials detected',
        impact: 'Users will be required to reset passwords',
        priority: 'high',
        estimatedDuration: '< 5 minutes'
      });
    }

    // Medium priority: Threat intel lookup
    recommendations.push({
      action: 'threat_intel_lookup',
      displayName: 'Threat Intelligence Lookup',
      reason: 'Check external threat feeds for IOCs',
      impact: 'Read-only analysis, no system changes',
      priority: 'medium',
      estimatedDuration: '< 1 minute',
      parameters: { ip: incident.src_ip }
    });

    // Medium priority: Hunt similar attacks
    if (severity === 'high' || severity === 'critical') {
      recommendations.push({
        action: 'hunt_similar_attacks',
        displayName: 'Hunt for Similar Attacks',
        reason: 'Identify if this is part of broader campaign',
        impact: 'System-wide search for similar indicators',
        priority: 'medium',
        estimatedDuration: '2-5 minutes'
      });
    }

    // Low priority: Deploy WAF rules for web attacks
    if (category && ['web_attack', 'sql_injection', 'xss'].includes(category)) {
      recommendations.push({
        action: 'deploy_waf_rules',
        displayName: 'Deploy WAF Protection Rules',
        reason: 'Block similar attack patterns',
        impact: 'WAF rules updated to block attack signatures',
        priority: 'low',
        estimatedDuration: '< 2 minutes'
      });
    }

    return recommendations;
  };

  const executeRecommendation = async (recommendation: AIRecommendation) => {
    if (!onExecuteRecommendation) return;

    try {
      setExecutingAction(recommendation.action);
      await onExecuteRecommendation(recommendation.action, recommendation.parameters);
      setExecutedActions(prev => new Set(prev).add(recommendation.action));
    } catch (err) {
      console.error('Failed to execute recommendation:', err);
    } finally {
      setExecutingAction(null);
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return 'red';
      case 'medium': return 'yellow';
      case 'low': return 'blue';
      default: return 'gray';
    }
  };

  const getSeverityColor = (severity: string) => {
    const s = severity?.toLowerCase();
    if (s === 'critical') return 'red';
    if (s === 'high') return 'orange';
    if (s === 'medium') return 'yellow';
    if (s === 'low') return 'blue';
    return 'gray';
  };

  if (loading) {
    return (
      <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-8">
        <div className="flex items-center justify-center">
          <Brain className="w-8 h-8 text-purple-400 animate-pulse mr-3" />
          <div>
            <p className="text-gray-300 font-medium">AI analyzing incident...</p>
            <p className="text-sm text-gray-500">Using GPT-4 for comprehensive analysis</p>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-6">
        <div className="flex items-center gap-3 mb-3">
          <AlertTriangle className="w-6 h-6 text-red-400" />
          <h3 className="text-lg font-semibold text-red-300">AI Analysis Failed</h3>
        </div>
        <p className="text-red-200 mb-4">{error}</p>
        <button
          onClick={generateAIAnalysis}
          className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg flex items-center gap-2 transition-colors"
        >
          <RefreshCw className="w-4 h-4" />
          Retry Analysis
        </button>
      </div>
    );
  }

  if (!analysis) return null;

  const severityColor = getSeverityColor(analysis.severity);

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <Brain className="w-6 h-6 text-purple-400" />
          AI Security Analysis
        </h2>
        <button
          onClick={generateAIAnalysis}
          disabled={loading}
          className="px-3 py-1.5 bg-purple-600 hover:bg-purple-700 disabled:opacity-50 text-white rounded-lg text-sm flex items-center gap-2 transition-colors"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* AI Summary Card */}
      <div className="bg-gradient-to-r from-purple-500/10 via-blue-500/10 to-cyan-500/10 border border-purple-500/30 rounded-lg p-6">
        <div className="flex items-start gap-3 mb-4">
          <Lightbulb className="w-6 h-6 text-yellow-400 flex-shrink-0 mt-1" />
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-white mb-2">AI Security Summary</h3>
            <p className="text-gray-300 leading-relaxed">{analysis.summary}</p>
          </div>
        </div>

        {/* Severity and Confidence */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
          <div className={`bg-${severityColor}-500/10 border border-${severityColor}-500/30 rounded-lg p-4`}>
            <div className="flex items-center gap-2 mb-2">
              <Target className={`w-5 h-5 text-${severityColor}-400`} />
              <span className="text-sm font-semibold text-gray-300 uppercase">Severity</span>
            </div>
            <div className={`text-3xl font-bold text-${severityColor}-300 mb-1 uppercase`}>
              {analysis.severity}
            </div>
            <div className="text-xs text-gray-400">
              {analysis.confidence_score && `${Math.round(analysis.confidence_score * 100)}% confidence`}
            </div>
          </div>

          <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <Shield className="w-5 h-5 text-blue-400" />
              <span className="text-sm font-semibold text-gray-300 uppercase">Recommendation</span>
            </div>
            <div className="text-lg font-semibold text-blue-300">
              {analysis.recommendation || 'Investigate Further'}
            </div>
          </div>
        </div>
      </div>

      {/* AI Rationale (Expandable) */}
      <div className="bg-gray-800/50 border border-gray-700 rounded-lg overflow-hidden">
        <button
          onClick={() => setExpandedRationale(!expandedRationale)}
          className="w-full flex items-center justify-between p-4 hover:bg-gray-800/70 transition-colors"
        >
          <div className="flex items-center gap-2">
            <Info className="w-5 h-5 text-gray-400" />
            <span className="font-semibold text-white">Why AI Recommends This</span>
          </div>
          {expandedRationale ? (
            <ChevronUp className="w-5 h-5 text-gray-400" />
          ) : (
            <ChevronDown className="w-5 h-5 text-gray-400" />
          )}
        </button>

        {expandedRationale && (
          <div className="p-4 pt-0 space-y-2">
            {analysis.rationale?.map((reason, idx) => (
              <div key={idx} className="flex items-start gap-2 text-sm text-gray-300">
                <span className="text-purple-400 font-bold">{idx + 1}.</span>
                <span>{reason}</span>
              </div>
            ))}
            
            {analysis.threat_attribution && (
              <div className="mt-4 pt-4 border-t border-gray-700">
                <div className="flex items-center gap-2 mb-2">
                  <Globe className="w-4 h-4 text-cyan-400" />
                  <span className="text-sm font-semibold text-gray-400 uppercase">Attribution</span>
                </div>
                <p className="text-sm text-gray-300">{analysis.threat_attribution}</p>
              </div>
            )}
          </div>
        )}
      </div>

      {/* AI-Recommended Actions */}
      {analysis.recommendations && analysis.recommendations.length > 0 && (
        <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white flex items-center gap-2">
              <Zap className="w-5 h-5 text-yellow-400" />
              AI-Recommended Actions
            </h3>
            {onExecuteAllRecommendations && (
              <button
                onClick={onExecuteAllRecommendations}
                className="px-4 py-2 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white rounded-lg text-sm font-medium transition-all flex items-center gap-2"
              >
                <Zap className="w-4 h-4" />
                Execute All Priority Actions
              </button>
            )}
          </div>

          <div className="space-y-3">
            {analysis.recommendations.map((rec, idx) => {
              const priorityColor = getPriorityColor(rec.priority);
              const isExecuting = executingAction === rec.action;
              const isExecuted = executedActions.has(rec.action);

              return (
                <div
                  key={idx}
                  className={`bg-gray-900/50 border rounded-lg p-4 transition-all ${
                    isExecuted
                      ? 'border-green-500/50 bg-green-500/5'
                      : `border-${priorityColor}-500/30`
                  }`}
                >
                  <div className="flex items-start gap-4">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <span className={`px-2 py-0.5 rounded text-xs font-bold bg-${priorityColor}-500/20 text-${priorityColor}-300 uppercase`}>
                          {rec.priority} Priority
                        </span>
                        {isExecuted && (
                          <span className="px-2 py-0.5 rounded text-xs font-bold bg-green-500/20 text-green-300 flex items-center gap-1">
                            <CheckCircle className="w-3 h-3" />
                            Executed
                          </span>
                        )}
                      </div>

                      <h4 className="font-semibold text-white mb-1">{rec.displayName}</h4>
                      <p className="text-sm text-gray-400 mb-2">{rec.reason}</p>

                      <div className="flex items-center gap-4 text-xs text-gray-500">
                        <span>Impact: <span className="text-gray-400">{rec.impact}</span></span>
                        {rec.estimatedDuration && (
                          <span>Duration: <span className="text-gray-400">{rec.estimatedDuration}</span></span>
                        )}
                      </div>
                    </div>

                    {onExecuteRecommendation && !isExecuted && (
                      <button
                        onClick={() => executeRecommendation(rec)}
                        disabled={isExecuting}
                        className={`px-4 py-2 bg-${priorityColor}-600 hover:bg-${priorityColor}-700 disabled:opacity-50 text-white rounded-lg text-sm font-medium transition-colors flex items-center gap-2 whitespace-nowrap`}
                      >
                        {isExecuting ? (
                          <>
                            <Loader2 className="w-4 h-4 animate-spin" />
                            Executing...
                          </>
                        ) : (
                          <>
                            <Zap className="w-4 h-4" />
                            Execute
                          </>
                        )}
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Threat Intelligence Context */}
      {(analysis.estimated_impact || analysis.threat_attribution) && (
        <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white flex items-center gap-2 mb-3">
            <Globe className="w-5 h-5 text-cyan-400" />
            Threat Intelligence Context
          </h3>
          {analysis.threat_attribution && (
            <div className="mb-3">
              <span className="text-sm font-semibold text-gray-400">Attribution: </span>
              <span className="text-gray-300">{analysis.threat_attribution}</span>
            </div>
          )}
          {analysis.estimated_impact && (
            <div>
              <span className="text-sm font-semibold text-gray-400">Estimated Impact: </span>
              <span className="text-gray-300">{analysis.estimated_impact}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

