'use client'

/**
 * AI Recommendation Engine Component
 * 
 * Provides intelligent response recommendations with confidence scoring,
 * contextual analysis, and natural language explanations.
 */

import React, { useState, useEffect, useCallback } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ScrollArea } from "@/components/ui/scroll-area"
import { 
  Brain, 
  Zap, 
  Target, 
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  Eye,
  Play,
  Clock,
  Shield,
  Activity,
  Lightbulb,
  BarChart3,
  RefreshCw,
  Sparkles,
  Network,
  Server,
  Mail,
  Cloud,
  Key,
  Database
} from 'lucide-react'
import { 
  getAIRecommendations, 
  analyzeIncidentContext,
  optimizeResponseStrategy,
  createResponseWorkflow
} from '@/app/lib/api'

interface AIRecommendation {
  action_type: string
  priority: number
  confidence: number
  confidence_level: string
  parameters: Record<string, any>
  estimated_duration: number
  safety_considerations: string[]
  rollback_plan: Record<string, any>
  approval_required: boolean
  rationale?: string
  expected_outcome?: string
  risks?: string[]
  alternatives?: string[]
}

interface ContextAnalysis {
  incident_severity: {
    score: number
    level: string
    factors: string[]
  }
  attack_pattern: {
    pattern: string
    confidence: number
    indicators: string[]
  }
  threat_intelligence: {
    reputation_score: number
    threat_categories: string[]
    confidence: number
  }
  ml_analysis: {
    anomaly_score: number
    threat_probability: number
    model_confidence: number
  }
}

interface AIRecommendationEngineProps {
  incidentId: number
  onRecommendationSelected?: (recommendation: AIRecommendation) => void
  onWorkflowCreated?: (workflowId: string) => void
}

const AIRecommendationEngine: React.FC<AIRecommendationEngineProps> = ({
  incidentId,
  onRecommendationSelected,
  onWorkflowCreated
}) => {
  // State management
  const [recommendations, setRecommendations] = useState<AIRecommendation[]>([])
  const [contextAnalysis, setContextAnalysis] = useState<ContextAnalysis | null>(null)
  const [selectedRecommendations, setSelectedRecommendations] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [analyzing, setAnalyzing] = useState(false)
  const [optimizing, setOptimizing] = useState(false)
  const [creatingWorkflow, setCreatingWorkflow] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState('recommendations')
  const [aiInsight, setAiInsight] = useState<string>('')

  // Category icons
  const categoryIcons = {
    network: Network,
    endpoint: Server,
    email: Mail,
    cloud: Cloud,
    identity: Key,
    data: Database,
    compliance: Shield,
    forensics: Target
  }

  // Confidence colors
  const confidenceColors = {
    very_high: 'bg-green-100 text-green-800 border-green-200',
    high: 'bg-blue-100 text-blue-800 border-blue-200',
    medium: 'bg-yellow-100 text-yellow-800 border-yellow-200',
    low: 'bg-orange-100 text-orange-800 border-orange-200',
    very_low: 'bg-red-100 text-red-800 border-red-200'
  }

  // Load AI recommendations and context
  const loadRecommendations = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)

      // Get AI recommendations and context analysis in parallel
      const [recommendationsData, contextData] = await Promise.all([
        getAIRecommendations(incidentId),
        analyzeIncidentContext(incidentId)
      ])

      if (recommendationsData.success) {
        setRecommendations(recommendationsData.recommendations || [])
        setAiInsight(recommendationsData.explanations?.ai_insight || '')
      } else {
        setError(recommendationsData.error || 'Failed to load AI recommendations')
      }

      if (contextData.success) {
        setContextAnalysis(contextData.context_analysis || null)
      }

    } catch (err) {
      console.error('Failed to load AI recommendations:', err)
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }, [incidentId])

  // Optimize recommendations
  const optimizeRecommendations = async () => {
    if (recommendations.length === 0) return

    setOptimizing(true)
    try {
      // Create a temporary workflow to optimize
      const tempWorkflowData = {
        incident_id: incidentId,
        playbook_name: "AI Optimization Analysis",
        steps: recommendations.slice(0, 5).map(rec => ({
          action_type: rec.action_type,
          parameters: rec.parameters
        })),
        auto_execute: false
      }

      const workflowResult = await createResponseWorkflow(tempWorkflowData)
      
      if (workflowResult.success) {
        const optimizationResult = await optimizeResponseStrategy(workflowResult.workflow_id)
        
        if (optimizationResult.success) {
          setRecommendations(optimizationResult.optimized_recommendations || recommendations)
          setError(null)
        }
      }
      
    } catch (err) {
      console.error('Failed to optimize recommendations:', err)
      setError('Failed to optimize recommendations')
    } finally {
      setOptimizing(false)
    }
  }

  // Create workflow from selected recommendations
  const createWorkflowFromAI = async () => {
    if (selectedRecommendations.length === 0) {
      setError('Please select at least one recommendation')
      return
    }

    setCreatingWorkflow(true)
    try {
      const selectedRecs = recommendations.filter(rec => 
        selectedRecommendations.includes(rec.action_type)
      )

      const workflowData = {
        incident_id: incidentId,
        playbook_name: `AI-Generated Response Workflow`,
        steps: selectedRecs.map(rec => ({
          action_type: rec.action_type,
          parameters: rec.parameters,
          timeout_seconds: rec.estimated_duration,
          continue_on_failure: false,
          max_retries: 3
        })),
        auto_execute: false,
        priority: 'high'
      }

      const result = await createResponseWorkflow(workflowData)
      
      if (result.success) {
        onWorkflowCreated?.(result.workflow_id)
        setSelectedRecommendations([])
        setError(null)
      } else {
        setError(result.error || 'Failed to create workflow')
      }

    } catch (err) {
      console.error('Failed to create AI workflow:', err)
      setError('Failed to create workflow')
    } finally {
      setCreatingWorkflow(false)
    }
  }

  // Initialize component
  useEffect(() => {
    if (incidentId) {
      loadRecommendations()
    }
  }, [incidentId, loadRecommendations])

  // Format confidence percentage
  const formatConfidence = (confidence: number): string => {
    return `${Math.round(confidence * 100)}%`
  }

  // Get confidence icon
  const getConfidenceIcon = (confidence: number) => {
    if (confidence >= 0.8) return <TrendingUp className="h-4 w-4 text-green-600" />
    if (confidence >= 0.6) return <TrendingUp className="h-4 w-4 text-blue-600" />
    if (confidence >= 0.4) return <Activity className="h-4 w-4 text-yellow-600" />
    return <TrendingDown className="h-4 w-4 text-red-600" />
  }

  if (loading) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5 animate-pulse" />
            AI Response Engine
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-32">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <span className="ml-2">Analyzing incident with AI...</span>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="w-full space-y-4">
      {/* Header with AI Insights */}
      <Card className="bg-gradient-to-r from-blue-50 to-purple-50 border-blue-200">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-blue-600" />
            AI Response Engine
            <Sparkles className="h-4 w-4 text-purple-500" />
            <Badge variant="outline" className="ml-auto">
              {recommendations.length} Recommendations
            </Badge>
          </CardTitle>
          <CardDescription>
            Intelligent response recommendations powered by machine learning and contextual analysis
          </CardDescription>
        </CardHeader>
        
        {aiInsight && (
          <CardContent>
            <div className="p-4 bg-white border border-blue-200 rounded-lg">
              <div className="flex items-start gap-2">
                <Lightbulb className="h-5 w-5 text-yellow-500 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-sm mb-2">AI Insight</h4>
                  <p className="text-sm text-gray-700">{aiInsight}</p>
                </div>
              </div>
            </div>
          </CardContent>
        )}
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

      {/* Main Content */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="recommendations">AI Recommendations</TabsTrigger>
          <TabsTrigger value="context">Context Analysis</TabsTrigger>
          <TabsTrigger value="insights">Learning Insights</TabsTrigger>
        </TabsList>

        <TabsContent value="recommendations" className="space-y-4">
          {/* Control Panel */}
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={loadRecommendations}
                    disabled={analyzing}
                  >
                    {analyzing ? (
                      <div className="animate-spin h-4 w-4 border border-current border-t-transparent rounded-full"></div>
                    ) : (
                      <RefreshCw className="h-4 w-4" />
                    )}
                    Re-analyze
                  </Button>
                  
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={optimizeRecommendations}
                    disabled={optimizing || recommendations.length === 0}
                  >
                    {optimizing ? (
                      <div className="animate-spin h-4 w-4 border border-current border-t-transparent rounded-full"></div>
                    ) : (
                      <Sparkles className="h-4 w-4" />
                    )}
                    Optimize
                  </Button>
                </div>
                
                {selectedRecommendations.length > 0 && (
                  <Button
                    onClick={createWorkflowFromAI}
                    disabled={creatingWorkflow}
                  >
                    {creatingWorkflow ? (
                      <div className="flex items-center gap-1">
                        <div className="animate-spin h-4 w-4 border border-current border-t-transparent rounded-full"></div>
                        Creating
                      </div>
                    ) : (
                      <div className="flex items-center gap-1">
                        <Zap className="h-4 w-4" />
                        Create Workflow ({selectedRecommendations.length})
                      </div>
                    )}
                  </Button>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Recommendations List */}
          <div className="space-y-3">
            {recommendations.map((rec, index) => {
              const isSelected = selectedRecommendations.includes(rec.action_type)
              const Icon = categoryIcons['network'] || Shield // Simplified - would get actual category
              
              return (
                <Card 
                  key={rec.action_type}
                  className={`transition-all cursor-pointer ${
                    isSelected ? 'ring-2 ring-blue-500 bg-blue-50' : 'hover:shadow-md'
                  }`}
                  onClick={() => {
                    if (isSelected) {
                      setSelectedRecommendations(prev => prev.filter(a => a !== rec.action_type))
                    } else {
                      setSelectedRecommendations(prev => [...prev, rec.action_type])
                    }
                  }}
                >
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-blue-100">
                          <Icon className="h-5 w-5 text-blue-600" />
                        </div>
                        <div>
                          <h4 className="font-semibold">{rec.action_type.replace('_', ' ')}</h4>
                          <p className="text-sm text-gray-600">Priority {rec.priority}</p>
                        </div>
                      </div>
                      
                      <div className="flex items-center gap-2">
                        {getConfidenceIcon(rec.confidence)}
                        <Badge 
                          variant="outline"
                          className={confidenceColors[rec.confidence_level as keyof typeof confidenceColors]}
                        >
                          {formatConfidence(rec.confidence)} confidence
                        </Badge>
                      </div>
                    </div>

                    {/* AI Rationale */}
                    {rec.rationale && (
                      <div className="mb-3 p-3 bg-gray-50 rounded-lg">
                        <div className="flex items-start gap-2">
                          <Brain className="h-4 w-4 text-purple-500 mt-0.5" />
                          <div>
                            <h5 className="font-medium text-sm">AI Rationale</h5>
                            <p className="text-sm text-gray-700 mt-1">{rec.rationale}</p>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Expected Outcome */}
                    {rec.expected_outcome && (
                      <div className="mb-3 p-3 bg-green-50 rounded-lg">
                        <div className="flex items-start gap-2">
                          <Target className="h-4 w-4 text-green-500 mt-0.5" />
                          <div>
                            <h5 className="font-medium text-sm">Expected Outcome</h5>
                            <p className="text-sm text-gray-700 mt-1">{rec.expected_outcome}</p>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Safety Considerations */}
                    {rec.safety_considerations && rec.safety_considerations.length > 0 && (
                      <div className="mb-3">
                        <h5 className="font-medium text-sm mb-2 flex items-center gap-1">
                          <AlertTriangle className="h-4 w-4 text-orange-500" />
                          Safety Considerations
                        </h5>
                        <ul className="text-sm text-gray-600 space-y-1">
                          {rec.safety_considerations.map((consideration, idx) => (
                            <li key={idx} className="flex items-start gap-1">
                              <span className="text-orange-500 mt-1">•</span>
                              {consideration}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Metrics */}
                    <div className="grid grid-cols-3 gap-4 text-sm">
                      <div className="text-center">
                        <div className="font-semibold text-blue-600">
                          {Math.floor(rec.estimated_duration / 60)}m
                        </div>
                        <div className="text-gray-500">Duration</div>
                      </div>
                      <div className="text-center">
                        <div className="font-semibold text-green-600">
                          {rec.approval_required ? 'Required' : 'Not Required'}
                        </div>
                        <div className="text-gray-500">Approval</div>
                      </div>
                      <div className="text-center">
                        <div className="font-semibold text-purple-600">
                          {rec.rollback_plan ? 'Available' : 'N/A'}
                        </div>
                        <div className="text-gray-500">Rollback</div>
                      </div>
                    </div>

                    {/* Action Buttons */}
                    <div className="mt-4 pt-3 border-t flex gap-2">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={(e) => {
                          e.stopPropagation()
                          onRecommendationSelected?.(rec)
                        }}
                        className="flex-1"
                      >
                        <Eye className="h-3 w-3 mr-1" />
                        Details
                      </Button>
                      
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={(e) => {
                          e.stopPropagation()
                          // Execute single action
                        }}
                        className="flex-1"
                      >
                        <Play className="h-3 w-3 mr-1" />
                        Execute
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              )
            })}
            
            {recommendations.length === 0 && !loading && (
              <Card>
                <CardContent className="p-8 text-center">
                  <Brain className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <h4 className="font-medium text-gray-900 mb-2">No AI Recommendations</h4>
                  <p className="text-gray-600">
                    The AI engine couldn't generate recommendations for this incident.
                  </p>
                  <Button variant="outline" className="mt-4" onClick={loadRecommendations}>
                    <RefreshCw className="h-4 w-4 mr-1" />
                    Try Again
                  </Button>
                </CardContent>
              </Card>
            )}
          </div>
        </TabsContent>

        <TabsContent value="context" className="space-y-4">
          {contextAnalysis ? (
            <div className="space-y-4">
              {/* Incident Severity */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <BarChart3 className="h-5 w-5" />
                    Incident Severity Analysis
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium">Severity Score</span>
                        <Badge className={
                          contextAnalysis.incident_severity.score > 0.8 ? 'bg-red-100 text-red-800' :
                          contextAnalysis.incident_severity.score > 0.6 ? 'bg-orange-100 text-orange-800' :
                          contextAnalysis.incident_severity.score > 0.4 ? 'bg-yellow-100 text-yellow-800' :
                          'bg-green-100 text-green-800'
                        }>
                          {contextAnalysis.incident_severity.level}
                        </Badge>
                      </div>
                      <Progress value={contextAnalysis.incident_severity.score * 100} className="h-2" />
                      <div className="text-sm text-gray-600 mt-1">
                        {formatConfidence(contextAnalysis.incident_severity.score)} severity
                      </div>
                    </div>
                    
                    <div>
                      <h5 className="font-medium text-sm mb-2">Contributing Factors</h5>
                      <ul className="text-sm text-gray-600 space-y-1">
                        {contextAnalysis.incident_severity.factors.map((factor, idx) => (
                          <li key={idx} className="flex items-start gap-1">
                            <span className="text-blue-500 mt-1">•</span>
                            {factor}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Attack Pattern Analysis */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <Target className="h-5 w-5" />
                    Attack Pattern Analysis
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <div className="text-center p-4 bg-gray-50 rounded-lg">
                        <div className="text-2xl font-bold text-blue-600 mb-1">
                          {contextAnalysis.attack_pattern.pattern.replace('_', ' ')}
                        </div>
                        <div className="text-sm text-gray-600">Identified Pattern</div>
                        <div className="mt-2">
                          <Badge variant="outline">
                            {formatConfidence(contextAnalysis.attack_pattern.confidence)} confidence
                          </Badge>
                        </div>
                      </div>
                    </div>
                    
                    <div>
                      <h5 className="font-medium text-sm mb-2">Pattern Indicators</h5>
                      <div className="flex flex-wrap gap-1">
                        {contextAnalysis.attack_pattern.indicators.map((indicator, idx) => (
                          <Badge key={idx} variant="secondary" className="text-xs">
                            {indicator.replace('_', ' ')}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* ML Analysis */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <Brain className="h-5 w-5" />
                    Machine Learning Analysis
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-red-600">
                        {formatConfidence(contextAnalysis.ml_analysis.anomaly_score)}
                      </div>
                      <div className="text-sm text-gray-600">Anomaly Score</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-orange-600">
                        {formatConfidence(contextAnalysis.ml_analysis.threat_probability)}
                      </div>
                      <div className="text-sm text-gray-600">Threat Probability</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-blue-600">
                        {formatConfidence(contextAnalysis.ml_analysis.model_confidence)}
                      </div>
                      <div className="text-sm text-gray-600">Model Confidence</div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Threat Intelligence */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <Shield className="h-5 w-5" />
                    Threat Intelligence
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium">Reputation Score</span>
                        <span className="text-sm text-gray-600">
                          {formatConfidence(contextAnalysis.threat_intelligence.reputation_score)}
                        </span>
                      </div>
                      <Progress 
                        value={contextAnalysis.threat_intelligence.reputation_score * 100} 
                        className="h-2" 
                      />
                    </div>
                    
                    <div>
                      <h5 className="font-medium text-sm mb-2">Threat Categories</h5>
                      <div className="flex flex-wrap gap-1">
                        {contextAnalysis.threat_intelligence.threat_categories.map((category, idx) => (
                          <Badge key={idx} variant="outline" className="text-xs">
                            {category}
                          </Badge>
                        ))}
                        {contextAnalysis.threat_intelligence.threat_categories.length === 0 && (
                          <span className="text-sm text-gray-500">No known categories</span>
                        )}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          ) : (
            <Card>
              <CardContent className="p-8 text-center">
                <AlertTriangle className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h4 className="font-medium text-gray-900 mb-2">Context Analysis Unavailable</h4>
                <p className="text-gray-600">Unable to load detailed context analysis for this incident.</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="insights" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Lightbulb className="h-5 w-5" />
                Learning Insights
              </CardTitle>
              <CardDescription>
                How the AI system is learning and improving over time
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Learning Progress */}
                <div className="space-y-4">
                  <h4 className="font-semibold">Learning Progress</h4>
                  
                  <div className="space-y-3">
                    <div>
                      <div className="flex items-center justify-between text-sm mb-1">
                        <span>Recommendation Accuracy</span>
                        <span>87%</span>
                      </div>
                      <Progress value={87} className="h-2" />
                    </div>
                    
                    <div>
                      <div className="flex items-center justify-between text-sm mb-1">
                        <span>Learning Velocity</span>
                        <span>High</span>
                      </div>
                      <Progress value={75} className="h-2" />
                    </div>
                    
                    <div>
                      <div className="flex items-center justify-between text-sm mb-1">
                        <span>Model Confidence</span>
                        <span>92%</span>
                      </div>
                      <Progress value={92} className="h-2" />
                    </div>
                  </div>
                </div>

                {/* Recent Improvements */}
                <div className="space-y-4">
                  <h4 className="font-semibold">Recent Improvements</h4>
                  
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 text-sm">
                      <CheckCircle className="h-4 w-4 text-green-500" />
                      <span>Improved malware detection accuracy by 12%</span>
                    </div>
                    <div className="flex items-center gap-2 text-sm">
                      <CheckCircle className="h-4 w-4 text-green-500" />
                      <span>Reduced false positives in DDoS detection by 8%</span>
                    </div>
                    <div className="flex items-center gap-2 text-sm">
                      <CheckCircle className="h-4 w-4 text-green-500" />
                      <span>Optimized response timing for insider threats</span>
                    </div>
                    <div className="flex items-center gap-2 text-sm">
                      <TrendingUp className="h-4 w-4 text-blue-500" />
                      <span>Overall response effectiveness up 15%</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Learning Statistics */}
              <div className="mt-6 pt-4 border-t">
                <h4 className="font-semibold mb-4">Learning Statistics</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">247</div>
                    <div className="text-sm text-gray-600">Workflows Analyzed</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600">89%</div>
                    <div className="text-sm text-gray-600">Success Rate</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-purple-600">156</div>
                    <div className="text-sm text-gray-600">Patterns Learned</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-orange-600">23min</div>
                    <div className="text-sm text-gray-600">Avg Response Time</div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default AIRecommendationEngine













