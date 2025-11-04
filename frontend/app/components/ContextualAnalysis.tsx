'use client'

/**
 * Contextual Analysis Component
 *
 * Comprehensive multi-dimensional context visualization for incidents,
 * including threat intelligence, behavioral analysis, and predictive insights.
 */

import React, { useState, useEffect, useCallback } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Brain,
  Target,
  Clock,
  Eye,
  Shield,
  Activity,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  MapPin,
  Network,
  User,
  Database,
  Zap,
  Sparkles,
  RefreshCw,
  Globe,
  Search,
  BarChart3
} from 'lucide-react'
import { analyzeIncidentContext } from '@/app/lib/api'

interface ThreatContext {
  severity_score: number
  attack_vector: string
  threat_category: string
  confidence: number
  indicators: string[]
  attribution: {
    confidence: number
    probable_actor_type: string
    sophistication_level: string
    geographical_indicators: any
  }
}

interface TemporalAnalysis {
  pattern: string
  confidence: number
  total_duration_seconds: number
  event_rate_per_minute: number
  intervals_analysis: {
    min: number
    max: number
    variance: number
  }
}

interface BehavioralAnalysis {
  overall_score: number
  primary_behavior: string
  behavior_scores: Record<string, number>
  sophistication_indicators: string[]
  attacker_intent: string
  attack_progression: string[]
}

interface PredictiveAnalysis {
  escalation_probability: number
  lateral_movement_risk: number
  predicted_duration_hours: number
  next_likely_targets: Array<{
    target_type: string
    probability: number
  }>
  early_warning_indicators: string[]
}

interface ContextAnalysisData {
  threat_context: ThreatContext
  temporal_analysis: TemporalAnalysis
  behavioral_analysis: BehavioralAnalysis
  predictive_analysis: PredictiveAnalysis
  analysis_quality: {
    score: number
    completeness: string
  }
}

interface ContextualAnalysisProps {
  incidentId: number
  onInsightGenerated?: (insight: any) => void
  className?: string
}

const ContextualAnalysis: React.FC<ContextualAnalysisProps> = ({
  incidentId,
  onInsightGenerated,
  className = ""
}) => {
  const [contextData, setContextData] = useState<ContextAnalysisData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState('threat')
  const [autoRefresh, setAutoRefresh] = useState(false)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)

  // Load context analysis
  const loadContextAnalysis = useCallback(async () => {
    try {
      setError(null)

      const result = await analyzeIncidentContext(incidentId)

      if (result.success) {
        setContextData(result.context_analysis)
        setLastUpdated(new Date())

        // Generate insights for parent component
        onInsightGenerated?.({
          type: 'context_analysis',
          incidentId,
          insights: result.context_analysis
        })
      } else {
        setError(result.error || 'Failed to load context analysis')
      }

    } catch (err) {
      console.error('Failed to load context analysis:', err)
      setError(err instanceof Error ? err.message : 'Unknown error')
    }
  }, [incidentId, onInsightGenerated])

  // Initialize and setup auto-refresh
  useEffect(() => {
    const initialize = async () => {
      setLoading(true)
      await loadContextAnalysis()
      setLoading(false)
    }

    initialize()

    if (autoRefresh) {
      const interval = setInterval(loadContextAnalysis, 60000) // 1 minute
      return () => clearInterval(interval)
    }
  }, [loadContextAnalysis, autoRefresh])

  // Get severity color
  const getSeverityColor = (score: number): string => {
    if (score >= 0.8) return 'text-red-600 bg-red-100 border-red-200'
    if (score >= 0.6) return 'text-orange-600 bg-orange-100 border-orange-200'
    if (score >= 0.4) return 'text-yellow-600 bg-yellow-100 border-yellow-200'
    return 'text-green-600 bg-green-100 border-green-200'
  }

  // Get confidence level
  const getConfidenceLevel = (confidence: number): string => {
    if (confidence >= 0.9) return 'Very High'
    if (confidence >= 0.7) return 'High'
    if (confidence >= 0.5) return 'Medium'
    if (confidence >= 0.3) return 'Low'
    return 'Very Low'
  }

  // Format duration
  const formatDuration = (seconds: number): string => {
    if (seconds < 60) return `${seconds}s`
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m`
    return `${(seconds / 3600).toFixed(1)}h`
  }

  if (loading) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5 animate-pulse" />
            Contextual Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-32">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <span className="ml-2">Analyzing incident context...</span>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!contextData) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Contextual Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <AlertTriangle className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h4 className="font-medium text-gray-900 mb-2">Analysis Unavailable</h4>
            <p className="text-gray-600">Unable to load contextual analysis for incident #{incidentId}</p>
            <Button variant="outline" className="mt-4" onClick={loadContextAnalysis}>
              <RefreshCw className="h-4 w-4 mr-1" />
              Retry Analysis
            </Button>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Analysis Header */}
      <Card className="bg-gradient-to-r from-blue-50 to-purple-50 border-blue-200">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5 text-blue-600" />
              Contextual Analysis - Incident #{incidentId}
              <Sparkles className="h-4 w-4 text-purple-500" />
            </CardTitle>

            <div className="flex items-center gap-2">
              {lastUpdated && (
                <span className="text-xs text-gray-500">
                  Updated {lastUpdated.toLocaleTimeString()}
                </span>
              )}
              <Button
                size="sm"
                variant="outline"
                onClick={() => setAutoRefresh(!autoRefresh)}
              >
                <RefreshCw className={`h-3 w-3 ${autoRefresh ? 'animate-spin' : ''}`} />
                {autoRefresh ? 'Auto' : 'Manual'}
              </Button>
            </div>
          </div>

          <div className="flex items-center gap-4 mt-2">
            <Badge
              variant="outline"
              className={getSeverityColor(contextData.threat_context.severity_score)}
            >
              {Math.round(contextData.threat_context.severity_score * 100)}% Severity
            </Badge>

            <Badge variant="outline">
              {getConfidenceLevel(contextData.threat_context.confidence)} Confidence
            </Badge>

            <Badge variant="outline">
              {contextData.analysis_quality.completeness} Completeness
            </Badge>
          </div>
        </CardHeader>
      </Card>

      {/* Error Display */}
      {error && (
        <Card className="border-red-200 bg-red-50">
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-red-500" />
              <span className="text-red-700 text-sm">{error}</span>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Context Analysis Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="threat">Threat</TabsTrigger>
          <TabsTrigger value="temporal">Temporal</TabsTrigger>
          <TabsTrigger value="behavioral">Behavioral</TabsTrigger>
          <TabsTrigger value="predictive">Predictive</TabsTrigger>
        </TabsList>

        <TabsContent value="threat" className="space-y-4">
          {/* Threat Context Analysis */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <Shield className="h-5 w-5" />
                  Threat Assessment
                </CardTitle>
              </CardHeader>

              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center p-3 bg-gray-50 rounded-lg">
                    <div className={`text-2xl font-bold ${getSeverityColor(contextData.threat_context.severity_score).split(' ')[0]}`}>
                      {Math.round(contextData.threat_context.severity_score * 100)}%
                    </div>
                    <div className="text-sm text-gray-600">Severity Score</div>
                  </div>

                  <div className="text-center p-3 bg-gray-50 rounded-lg">
                    <div className="text-2xl font-bold text-blue-600">
                      {Math.round(contextData.threat_context.confidence * 100)}%
                    </div>
                    <div className="text-sm text-gray-600">Analysis Confidence</div>
                  </div>
                </div>

                <div>
                  <h5 className="font-semibold text-sm mb-2">Threat Classification</h5>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Category</span>
                      <Badge variant="outline">{contextData.threat_context.threat_category}</Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Attack Vector</span>
                      <Badge variant="outline">{contextData.threat_context.attack_vector}</Badge>
                    </div>
                  </div>
                </div>

                <div>
                  <h5 className="font-semibold text-sm mb-2">Threat Indicators</h5>
                  <div className="flex flex-wrap gap-1">
                    {contextData.threat_context.indicators.map((indicator, index) => (
                      <Badge key={index} variant="secondary" className="text-xs">
                        {indicator.replace('_', ' ')}
                      </Badge>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <Globe className="h-5 w-5" />
                  Attribution Analysis
                </CardTitle>
              </CardHeader>

              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center p-3 bg-gray-50 rounded-lg">
                    <div className="text-lg font-bold text-purple-600">
                      {Math.round(contextData.threat_context.attribution.confidence * 100)}%
                    </div>
                    <div className="text-sm text-gray-600">Attribution Confidence</div>
                  </div>

                  <div className="text-center p-3 bg-gray-50 rounded-lg">
                    <div className="text-lg font-bold text-orange-600">
                      {contextData.threat_context.attribution.sophistication_level}
                    </div>
                    <div className="text-sm text-gray-600">Sophistication</div>
                  </div>
                </div>

                <div>
                  <h5 className="font-semibold text-sm mb-2">Actor Assessment</h5>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Actor Type</span>
                      <Badge variant="outline">{contextData.threat_context.attribution.probable_actor_type}</Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Sophistication</span>
                      <Badge variant="outline">{contextData.threat_context.attribution.sophistication_level}</Badge>
                    </div>
                  </div>
                </div>

                <div>
                  <h5 className="font-semibold text-sm mb-2">Geographic Indicators</h5>
                  <div className="p-3 bg-blue-50 rounded-lg">
                    <div className="flex items-center gap-2 text-sm">
                      <MapPin className="h-4 w-4 text-blue-600" />
                      <span>
                        {contextData.threat_context.attribution.geographical_indicators?.country || 'Unknown'}
                        {contextData.threat_context.attribution.geographical_indicators?.city &&
                          `, ${contextData.threat_context.attribution.geographical_indicators.city}`
                        }
                      </span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="temporal" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Clock className="h-5 w-5" />
                Temporal Pattern Analysis
              </CardTitle>
            </CardHeader>

            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div className="text-center p-4 bg-blue-50 rounded-lg">
                  <div className="text-2xl font-bold text-blue-600">
                    {contextData.temporal_analysis.pattern.replace('_', ' ').toUpperCase()}
                  </div>
                  <div className="text-sm text-gray-600">Attack Pattern</div>
                  <Badge variant="outline" className="mt-1">
                    {Math.round(contextData.temporal_analysis.confidence * 100)}% confidence
                  </Badge>
                </div>

                <div className="text-center p-4 bg-green-50 rounded-lg">
                  <div className="text-2xl font-bold text-green-600">
                    {formatDuration(contextData.temporal_analysis.total_duration_seconds)}
                  </div>
                  <div className="text-sm text-gray-600">Attack Duration</div>
                </div>

                <div className="text-center p-4 bg-orange-50 rounded-lg">
                  <div className="text-2xl font-bold text-orange-600">
                    {contextData.temporal_analysis.event_rate_per_minute.toFixed(1)}/min
                  </div>
                  <div className="text-sm text-gray-600">Event Rate</div>
                </div>
              </div>

              <div className="space-y-4">
                <h4 className="font-semibold">Timing Analysis</h4>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <span>Minimum Interval</span>
                      <span className="font-medium">
                        {formatDuration(contextData.temporal_analysis.intervals_analysis.min)}
                      </span>
                    </div>
                    <Progress value={25} className="h-1" />
                  </div>

                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <span>Maximum Interval</span>
                      <span className="font-medium">
                        {formatDuration(contextData.temporal_analysis.intervals_analysis.max)}
                      </span>
                    </div>
                    <Progress value={75} className="h-1" />
                  </div>

                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <span>Variance</span>
                      <span className="font-medium">
                        {contextData.temporal_analysis.intervals_analysis.variance.toFixed(2)}
                      </span>
                    </div>
                    <Progress value={50} className="h-1" />
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="behavioral" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <User className="h-5 w-5" />
                Behavioral Analysis
              </CardTitle>
            </CardHeader>

            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Behavior Scores */}
                <div className="space-y-4">
                  <h4 className="font-semibold">Behavioral Patterns</h4>

                  <div className="space-y-3">
                    {Object.entries(contextData.behavioral_analysis.behavior_scores).map(([behavior, score]) => (
                      <div key={behavior}>
                        <div className="flex items-center justify-between text-sm mb-1">
                          <span className="capitalize">{behavior.replace('_', ' ')}</span>
                          <span className="font-medium">{Math.round(score * 100)}%</span>
                        </div>
                        <Progress value={score * 100} className="h-2" />
                      </div>
                    ))}
                  </div>
                </div>

                {/* Sophistication Analysis */}
                <div className="space-y-4">
                  <h4 className="font-semibold">Sophistication Assessment</h4>

                  <div className="p-4 bg-gray-50 rounded-lg">
                    <div className="text-center mb-3">
                      <div className="text-2xl font-bold text-purple-600">
                        {contextData.behavioral_analysis.primary_behavior.replace('_', ' ').toUpperCase()}
                      </div>
                      <div className="text-sm text-gray-600">Primary Behavior</div>
                    </div>

                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span>Overall Score</span>
                        <span className="font-medium">
                          {Math.round(contextData.behavioral_analysis.overall_score * 100)}%
                        </span>
                      </div>

                      <div className="flex items-center justify-between text-sm">
                        <span>Attacker Intent</span>
                        <Badge variant="outline">
                          {contextData.behavioral_analysis.attacker_intent.replace('_', ' ')}
                        </Badge>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h5 className="font-medium text-sm mb-2">Sophistication Indicators</h5>
                    <div className="flex flex-wrap gap-1">
                      {contextData.behavioral_analysis.sophistication_indicators.map((indicator, index) => (
                        <Badge key={index} variant="secondary" className="text-xs">
                          {indicator.replace('_', ' ')}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* Attack Progression */}
              <div className="mt-6 pt-6 border-t">
                <h4 className="font-semibold mb-3">Attack Progression</h4>
                <div className="flex items-center gap-2 overflow-x-auto pb-2">
                  {contextData.behavioral_analysis.attack_progression.map((phase, index) => (
                    <div key={index} className="flex items-center gap-2 min-w-fit">
                      <div className="flex items-center gap-2 px-3 py-2 bg-blue-100 rounded-lg">
                        <div className="w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs font-bold">
                          {index + 1}
                        </div>
                        <span className="text-sm font-medium">{phase.replace('_', ' ')}</span>
                      </div>
                      {index < contextData.behavioral_analysis.attack_progression.length - 1 && (
                        <ArrowRight className="h-4 w-4 text-gray-400" />
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="predictive" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Brain className="h-5 w-5" />
                Predictive Analysis
              </CardTitle>
              <CardDescription>
                AI-powered predictions for incident evolution and impact
              </CardDescription>
            </CardHeader>

            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                {/* Escalation Probability */}
                <div className="p-4 border rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <TrendingUp className="h-5 w-5 text-red-500" />
                    <Badge variant="outline">Escalation Risk</Badge>
                  </div>
                  <div className="text-2xl font-bold text-red-600 mb-1">
                    {Math.round(contextData.predictive_analysis.escalation_probability * 100)}%
                  </div>
                  <Progress
                    value={contextData.predictive_analysis.escalation_probability * 100}
                    className="h-2"
                  />
                </div>

                {/* Lateral Movement Risk */}
                <div className="p-4 border rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <Network className="h-5 w-5 text-orange-500" />
                    <Badge variant="outline">Lateral Movement</Badge>
                  </div>
                  <div className="text-2xl font-bold text-orange-600 mb-1">
                    {Math.round(contextData.predictive_analysis.lateral_movement_risk * 100)}%
                  </div>
                  <Progress
                    value={contextData.predictive_analysis.lateral_movement_risk * 100}
                    className="h-2"
                  />
                </div>

                {/* Predicted Duration */}
                <div className="p-4 border rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <Clock className="h-5 w-5 text-blue-500" />
                    <Badge variant="outline">Duration</Badge>
                  </div>
                  <div className="text-2xl font-bold text-blue-600 mb-1">
                    {contextData.predictive_analysis.predicted_duration_hours.toFixed(1)}h
                  </div>
                  <div className="text-xs text-gray-600">Predicted remaining</div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Next Likely Targets */}
                <div>
                  <h4 className="font-semibold mb-3 flex items-center gap-2">
                    <Target className="h-4 w-4" />
                    Next Likely Targets
                  </h4>

                  <div className="space-y-2">
                    {contextData.predictive_analysis.next_likely_targets.map((target, index) => (
                      <div key={index} className="flex items-center justify-between p-2 border rounded">
                        <span className="text-sm capitalize">{target.target_type.replace('_', ' ')}</span>
                        <div className="flex items-center gap-2">
                          <Progress value={target.probability * 100} className="w-16 h-1" />
                          <span className="text-xs font-medium">
                            {Math.round(target.probability * 100)}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Early Warning Indicators */}
                <div>
                  <h4 className="font-semibold mb-3 flex items-center gap-2">
                    <AlertTriangle className="h-4 w-4" />
                    Early Warning Indicators
                  </h4>

                  <div className="space-y-2">
                    {contextData.predictive_analysis.early_warning_indicators.map((indicator, index) => (
                      <div key={index} className="flex items-center gap-2 p-2 bg-yellow-50 border border-yellow-200 rounded">
                        <Eye className="h-3 w-3 text-yellow-600" />
                        <span className="text-sm">{indicator.replace('_', ' ')}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Analysis Quality */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Analysis Quality Assessment
          </CardTitle>
        </CardHeader>

        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">Analysis Completeness</span>
                <span className="text-sm font-bold">
                  {Math.round(contextData.analysis_quality.score * 100)}%
                </span>
              </div>
              <Progress value={contextData.analysis_quality.score * 100} className="h-2 mb-2" />
              <div className="text-xs text-gray-600">
                {contextData.analysis_quality.completeness} quality data available
              </div>
            </div>

            <div className="space-y-2">
              <h5 className="font-medium text-sm">Quality Factors</h5>
              <div className="space-y-1">
                <div className="flex items-center gap-2 text-xs">
                  <CheckCircle className="h-3 w-3 text-green-500" />
                  <span>Multi-dimensional analysis completed</span>
                </div>
                <div className="flex items-center gap-2 text-xs">
                  <CheckCircle className="h-3 w-3 text-green-500" />
                  <span>Threat intelligence integrated</span>
                </div>
                <div className="flex items-center gap-2 text-xs">
                  <CheckCircle className="h-3 w-3 text-green-500" />
                  <span>Behavioral patterns identified</span>
                </div>
                <div className="flex items-center gap-2 text-xs">
                  <CheckCircle className="h-3 w-3 text-green-500" />
                  <span>Predictive models applied</span>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default ContextualAnalysis
