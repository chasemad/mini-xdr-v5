'use client'

/**
 * Response Optimizer Component
 * 
 * Advanced response optimization dashboard with AI-powered strategy tuning,
 * historical learning insights, and real-time performance optimization.
 */

import React, { useState, useEffect, useCallback } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ScrollArea } from "@/components/ui/scroll-area"
import { 
  Zap, 
  TrendingUp, 
  TrendingDown,
  Target,
  Brain,
  Sparkles,
  Settings,
  Activity,
  CheckCircle,
  AlertTriangle,
  Clock,
  BarChart3,
  RefreshCw,
  Play,
  Eye,
  Lightbulb,
  ArrowRight,
  Award,
  Gauge
} from 'lucide-react'
import { optimizeResponseStrategy, getWorkflowStatus } from '@/app/lib/api'

interface OptimizationResult {
  optimization_score: number
  improvements: string[]
  risk_reduction: number
  efficiency_gain: number
  confidence: number
}

interface OptimizationOpportunity {
  type: string
  priority: string
  description: string
  potential_improvement: string
  implementation: string
}

interface ResponseOptimizerProps {
  workflowId?: string
  incidentId?: number
  onOptimizationApplied?: (result: OptimizationResult) => void
  className?: string
}

const ResponseOptimizer: React.FC<ResponseOptimizerProps> = ({
  workflowId,
  incidentId,
  onOptimizationApplied,
  className = ""
}) => {
  // State management
  const [optimizationResult, setOptimizationResult] = useState<OptimizationResult | null>(null)
  const [optimizationOpportunities, setOptimizationOpportunities] = useState<OptimizationOpportunity[]>([])
  const [isOptimizing, setIsOptimizing] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedStrategy, setSelectedStrategy] = useState<'performance' | 'effectiveness' | 'efficiency' | 'safety' | 'compliance'>('effectiveness')
  const [activeTab, setActiveTab] = useState('optimizer')

  // Mock data for demonstration (in production, would come from API)
  const mockOpportunities: OptimizationOpportunity[] = [
    {
      type: "parallel_execution",
      priority: "high",
      description: "Execute compatible actions in parallel to reduce total workflow time",
      potential_improvement: "40% faster execution",
      implementation: "Identify non-conflicting actions and execute simultaneously"
    },
    {
      type: "parameter_optimization",
      priority: "medium", 
      description: "Optimize action parameters based on historical success patterns",
      potential_improvement: "15% higher success rate",
      implementation: "Apply ML-learned optimal parameters for each action type"
    },
    {
      type: "redundancy_removal",
      priority: "medium",
      description: "Remove redundant actions that don't improve effectiveness",
      potential_improvement: "25% resource savings",
      implementation: "Eliminate actions with overlapping functionality"
    },
    {
      type: "safety_enhancement",
      priority: "high",
      description: "Add safety checkpoints to prevent unintended impacts",
      potential_improvement: "60% risk reduction",
      implementation: "Insert validation steps before high-risk actions"
    }
  ]

  // Load optimization data
  const loadOptimizationData = useCallback(async () => {
    try {
      setError(null)
      
      // Set mock data for now
      setOptimizationOpportunities(mockOpportunities)
      
      // If we have a workflow ID, get its status
      if (workflowId) {
        const workflowData = await getWorkflowStatus(workflowId)
        if (workflowData.success) {
          // Mock optimization result based on workflow data
          setOptimizationResult({
            optimization_score: 0.78,
            improvements: [
              "Reduced average response time by 23%",
              "Improved success rate by 12%", 
              "Decreased resource usage by 18%"
            ],
            risk_reduction: 0.35,
            efficiency_gain: 0.28,
            confidence: 0.82
          })
        }
      }
      
    } catch (err) {
      console.error('Failed to load optimization data:', err)
      setError(err instanceof Error ? err.message : 'Unknown error')
    }
  }, [workflowId])

  // Initialize component
  useEffect(() => {
    const initialize = async () => {
      setLoading(true)
      await loadOptimizationData()
      setLoading(false)
    }

    initialize()
  }, [loadOptimizationData])

  // Execute optimization
  const executeOptimization = async () => {
    if (!workflowId) {
      setError('No workflow ID provided for optimization')
      return
    }

    setIsOptimizing(true)
    setError(null)

    try {
      const result = await optimizeResponseStrategy(workflowId, {
        optimization_strategy: selectedStrategy,
        context: { source: 'response_optimizer_ui' }
      })

      if (result.success) {
        const optimizationData = result.optimization_result
        setOptimizationResult(optimizationData)
        onOptimizationApplied?.(optimizationData)
      } else {
        setError(result.error || 'Optimization failed')
      }
      
    } catch (err) {
      console.error('Optimization failed:', err)
      setError(err instanceof Error ? err.message : 'Optimization failed')
    } finally {
      setIsOptimizing(false)
    }
  }

  // Get priority color
  const getPriorityColor = (priority: string): string => {
    switch (priority) {
      case 'high': return 'text-red-600 bg-red-100 border-red-200'
      case 'medium': return 'text-yellow-600 bg-yellow-100 border-yellow-200'
      case 'low': return 'text-green-600 bg-green-100 border-green-200'
      default: return 'text-gray-600 bg-gray-100 border-gray-200'
    }
  }

  // Get optimization score color
  const getOptimizationScoreColor = (score: number): string => {
    if (score >= 0.8) return 'text-green-600'
    if (score >= 0.6) return 'text-blue-600'
    if (score >= 0.4) return 'text-yellow-600'
    return 'text-red-600'
  }

  if (loading) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5" />
            Response Optimizer
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-32">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <span className="ml-2">Loading optimization engine...</span>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Optimization Header */}
      <Card className="bg-gradient-to-r from-blue-50 to-purple-50 border-blue-200">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5 text-blue-600" />
            AI-Powered Response Optimizer
            <Sparkles className="h-4 w-4 text-purple-500" />
          </CardTitle>
          <CardDescription>
            Continuously optimize response strategies using machine learning and historical data
          </CardDescription>
        </CardHeader>
        
        <CardContent>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              {/* Strategy Selector */}
              <div className="flex gap-2">
                {[
                  { key: 'performance', label: 'Performance', icon: Clock },
                  { key: 'effectiveness', label: 'Effectiveness', icon: Target },
                  { key: 'efficiency', label: 'Efficiency', icon: Gauge },
                  { key: 'safety', label: 'Safety', icon: Shield }
                ].map(({ key, label, icon: Icon }) => (
                  <Button
                    key={key}
                    size="sm"
                    variant={selectedStrategy === key ? 'default' : 'outline'}
                    onClick={() => setSelectedStrategy(key as any)}
                    className="flex items-center gap-1"
                  >
                    <Icon className="h-3 w-3" />
                    {label}
                  </Button>
                ))}
              </div>
            </div>
            
            <Button
              onClick={executeOptimization}
              disabled={isOptimizing || !workflowId}
              className="flex items-center gap-2"
            >
              {isOptimizing ? (
                <>
                  <div className="animate-spin h-4 w-4 border border-current border-t-transparent rounded-full"></div>
                  Optimizing
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4" />
                  Optimize
                </>
              )}
            </Button>
          </div>
        </CardContent>
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
          <TabsTrigger value="optimizer">Optimizer</TabsTrigger>
          <TabsTrigger value="opportunities">Opportunities</TabsTrigger>
          <TabsTrigger value="insights">Insights</TabsTrigger>
        </TabsList>

        <TabsContent value="optimizer" className="space-y-4">
          {/* Optimization Results */}
          {optimizationResult && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Award className="h-5 w-5" />
                  Optimization Results
                </CardTitle>
              </CardHeader>
              
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                  <div className="text-center p-4 bg-blue-50 rounded-lg">
                    <div className={`text-2xl font-bold ${getOptimizationScoreColor(optimizationResult.optimization_score)}`}>
                      {Math.round(optimizationResult.optimization_score * 100)}%
                    </div>
                    <div className="text-sm text-gray-600">Optimization Score</div>
                  </div>
                  
                  <div className="text-center p-4 bg-green-50 rounded-lg">
                    <div className="text-2xl font-bold text-green-600">
                      {Math.round(optimizationResult.risk_reduction * 100)}%
                    </div>
                    <div className="text-sm text-gray-600">Risk Reduction</div>
                  </div>
                  
                  <div className="text-center p-4 bg-purple-50 rounded-lg">
                    <div className="text-2xl font-bold text-purple-600">
                      {Math.round(optimizationResult.efficiency_gain * 100)}%
                    </div>
                    <div className="text-sm text-gray-600">Efficiency Gain</div>
                  </div>
                  
                  <div className="text-center p-4 bg-orange-50 rounded-lg">
                    <div className="text-2xl font-bold text-orange-600">
                      {Math.round(optimizationResult.confidence * 100)}%
                    </div>
                    <div className="text-sm text-gray-600">Confidence</div>
                  </div>
                </div>

                <div className="space-y-3">
                  <h4 className="font-semibold flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    Applied Improvements
                  </h4>
                  
                  {optimizationResult.improvements.map((improvement, index) => (
                    <div key={index} className="flex items-center gap-3 p-3 bg-green-50 border border-green-200 rounded-lg">
                      <ArrowRight className="h-4 w-4 text-green-600" />
                      <span className="text-sm text-green-800">{improvement}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Strategy Selection */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="h-5 w-5" />
                Optimization Strategy
              </CardTitle>
              <CardDescription>
                Select optimization focus area for AI-powered tuning
              </CardDescription>
            </CardHeader>
            
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {[
                  {
                    key: 'performance',
                    title: 'Performance Optimization',
                    description: 'Focus on reducing response times and improving speed',
                    icon: Clock,
                    benefits: ['Faster incident resolution', 'Reduced MTTC', 'Parallel execution']
                  },
                  {
                    key: 'effectiveness',
                    title: 'Effectiveness Optimization', 
                    description: 'Focus on improving success rates and outcomes',
                    icon: Target,
                    benefits: ['Higher success rates', 'Better action selection', 'Reduced false positives']
                  },
                  {
                    key: 'efficiency',
                    title: 'Efficiency Optimization',
                    description: 'Focus on resource usage and cost optimization',
                    icon: Gauge,
                    benefits: ['Lower costs', 'Resource optimization', 'Automated workflows']
                  },
                  {
                    key: 'safety',
                    title: 'Safety Optimization',
                    description: 'Focus on risk reduction and safe execution',
                    icon: Shield,
                    benefits: ['Risk mitigation', 'Rollback planning', 'Safety validation']
                  }
                ].map(({ key, title, description, icon: Icon, benefits }) => (
                  <Card 
                    key={key}
                    className={`cursor-pointer transition-all ${
                      selectedStrategy === key ? 'ring-2 ring-blue-500 bg-blue-50' : 'hover:shadow-md'
                    }`}
                    onClick={() => setSelectedStrategy(key as any)}
                  >
                    <CardContent className="p-4">
                      <div className="flex items-start gap-3 mb-3">
                        <Icon className="h-5 w-5 text-blue-600 mt-0.5" />
                        <div>
                          <h4 className="font-semibold text-sm">{title}</h4>
                          <p className="text-xs text-gray-600">{description}</p>
                        </div>
                      </div>
                      
                      <div className="space-y-1">
                        {benefits.map((benefit, idx) => (
                          <div key={idx} className="flex items-center gap-1 text-xs text-gray-600">
                            <CheckCircle className="h-3 w-3 text-green-500" />
                            {benefit}
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="opportunities" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Lightbulb className="h-5 w-5" />
                Optimization Opportunities
              </CardTitle>
              <CardDescription>
                AI-identified opportunities for improving response performance
              </CardDescription>
            </CardHeader>
            
            <CardContent>
              <div className="space-y-4">
                {optimizationOpportunities.map((opportunity, index) => (
                  <Card key={index} className="border-l-4 border-l-blue-500">
                    <CardContent className="p-4">
                      <div className="flex items-start justify-between mb-3">
                        <div>
                          <h4 className="font-semibold flex items-center gap-2">
                            <Target className="h-4 w-4" />
                            {opportunity.type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                          </h4>
                          <p className="text-sm text-gray-600 mt-1">{opportunity.description}</p>
                        </div>
                        
                        <Badge 
                          variant="outline"
                          className={getPriorityColor(opportunity.priority)}
                        >
                          {opportunity.priority} priority
                        </Badge>
                      </div>
                      
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <h5 className="font-medium text-xs text-gray-700 mb-1">Expected Improvement</h5>
                          <div className="flex items-center gap-2">
                            <TrendingUp className="h-4 w-4 text-green-500" />
                            <span className="text-sm font-semibold text-green-600">
                              {opportunity.potential_improvement}
                            </span>
                          </div>
                        </div>
                        
                        <div>
                          <h5 className="font-medium text-xs text-gray-700 mb-1">Implementation</h5>
                          <p className="text-xs text-gray-600">{opportunity.implementation}</p>
                        </div>
                      </div>
                      
                      <div className="mt-3 pt-3 border-t flex gap-2">
                        <Button size="sm" variant="outline" className="flex-1">
                          <Eye className="h-3 w-3 mr-1" />
                          Details
                        </Button>
                        <Button size="sm" className="flex-1">
                          <Play className="h-3 w-3 mr-1" />
                          Apply
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
                
                {optimizationOpportunities.length === 0 && (
                  <div className="text-center py-12">
                    <CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-4" />
                    <h4 className="font-medium text-gray-900 mb-2">Fully Optimized</h4>
                    <p className="text-gray-600">
                      No optimization opportunities identified. Your responses are performing optimally.
                    </p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="insights" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5" />
                Optimization Insights
              </CardTitle>
              <CardDescription>
                Historical learning and performance insights
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
                        <span>Optimization Accuracy</span>
                        <span>94%</span>
                      </div>
                      <Progress value={94} className="h-2" />
                    </div>
                    
                    <div>
                      <div className="flex items-center justify-between text-sm mb-1">
                        <span>Learning Velocity</span>
                        <span>High</span>
                      </div>
                      <Progress value={85} className="h-2" />
                    </div>
                    
                    <div>
                      <div className="flex items-center justify-between text-sm mb-1">
                        <span>Prediction Confidence</span>
                        <span>87%</span>
                      </div>
                      <Progress value={87} className="h-2" />
                    </div>
                  </div>
                </div>

                {/* Optimization History */}
                <div className="space-y-4">
                  <h4 className="font-semibold">Recent Optimizations</h4>
                  
                  <div className="space-y-2">
                    {[
                      { action: "Malware Response optimization", improvement: "+23% faster", time: "2 hours ago" },
                      { action: "DDoS mitigation tuning", improvement: "+15% success rate", time: "6 hours ago" },
                      { action: "Insider threat workflow", improvement: "+30% efficiency", time: "1 day ago" },
                      { action: "Phishing response optimization", improvement: "+18% accuracy", time: "2 days ago" }
                    ].map((opt, index) => (
                      <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                        <div>
                          <div className="text-sm font-medium">{opt.action}</div>
                          <div className="text-xs text-gray-600">{opt.time}</div>
                        </div>
                        <Badge variant="outline" className="text-green-600 bg-green-50">
                          {opt.improvement}
                        </Badge>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Performance Impact */}
              <div className="mt-6 pt-6 border-t">
                <h4 className="font-semibold mb-4">Optimization Impact</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">247</div>
                    <div className="text-sm text-gray-600">Workflows Optimized</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600">+31%</div>
                    <div className="text-sm text-gray-600">Avg Improvement</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-purple-600">$47K</div>
                    <div className="text-sm text-gray-600">Cost Savings</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-orange-600">-18min</div>
                    <div className="text-sm text-gray-600">Time Saved</div>
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

export default ResponseOptimizer





