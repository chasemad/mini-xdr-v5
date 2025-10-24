'use client'

/**
 * Response Analytics Dashboard
 * 
 * Comprehensive analytics dashboard for response effectiveness, performance monitoring,
 * and optimization insights with real-time metrics and trend analysis.
 */

import React, { useState, useEffect, useCallback } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { 
  BarChart3, 
  TrendingUp, 
  TrendingDown,
  Target,
  Clock,
  Activity,
  AlertTriangle,
  CheckCircle,
  Zap,
  RefreshCw,
  Download,
  Filter,
  Calendar,
  Users,
  DollarSign,
  Shield,
  Award,
  Eye,
  Settings
} from 'lucide-react'
import ResponseMetrics from '../../components/ResponseMetrics'
import EffectivenessAnalysis from '../../components/EffectivenessAnalysis'
import TrendAnalysis from '../../components/TrendAnalysis'
import { getResponseImpactMetrics } from '@/app/lib/api'

interface AnalyticsData {
  response_metrics: {
    total_workflows: number
    successful_workflows: number
    average_response_time: number
    success_rate: number
    false_positive_rate: number
    mean_time_to_containment: number
    cost_effectiveness_score: number
  }
  effectiveness_analysis: {
    action_effectiveness: Record<string, number>
    workflow_effectiveness: Record<string, number>
    improvement_recommendations: string[]
  }
  trend_analysis: {
    trend_data: Array<{
      timestamp: string
      success_rate: number
      response_time: number
      incident_volume: number
      effectiveness_score: number
    }>
    success_rate_trend: {
      slope: number
      direction: string
      confidence: string
    }
  }
  business_impact: {
    total_cost_impact_usd: number
    total_downtime_minutes: number
    estimated_roi: number
    cost_effectiveness_grade: string
  }
  executive_summary: {
    performance_grade: string
    overall_score: number
    key_insights: string[]
    recommendation: string
  }
}

const ResponseAnalyticsPage: React.FC = () => {
  const [analyticsData, setAnalyticsData] = useState<AnalyticsData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedTimeframe, setSelectedTimeframe] = useState('7d')
  const [activeTab, setActiveTab] = useState('overview')
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)

  // Load analytics data
  const loadAnalytics = useCallback(async () => {
    try {
      setError(null)
      
      // Get comprehensive analytics data
      const result = await getResponseImpactMetrics({
        days_back: parseInt(selectedTimeframe.replace('d', ''))
      })

      if (result.success) {
        // Transform the data to match our interface
        const transformedData: AnalyticsData = {
          response_metrics: {
            total_workflows: result.summary?.total_workflows || 0,
            successful_workflows: result.summary?.successful_workflows || 0,
            average_response_time: result.summary?.average_response_time_ms || 0,
            success_rate: result.summary?.average_success_rate || 0,
            false_positive_rate: result.summary?.false_positive_rate || 0,
            mean_time_to_containment: result.summary?.mean_time_to_containment || 0,
            cost_effectiveness_score: result.summary?.cost_effectiveness_score || 0
          },
          effectiveness_analysis: {
            action_effectiveness: result.effectiveness_analysis?.action_effectiveness || {},
            workflow_effectiveness: result.effectiveness_analysis?.workflow_effectiveness || {},
            improvement_recommendations: result.effectiveness_analysis?.improvement_recommendations || []
          },
          trend_analysis: {
            trend_data: result.trend_analysis?.trend_data || [],
            success_rate_trend: result.trend_analysis?.success_rate_trend || {
              slope: 0,
              direction: 'stable',
              confidence: 'low'
            }
          },
          business_impact: {
            total_cost_impact_usd: result.business_impact?.total_cost_impact_usd || 0,
            total_downtime_minutes: result.business_impact?.total_downtime_minutes || 0,
            estimated_roi: result.business_impact?.estimated_roi || 0,
            cost_effectiveness_grade: result.business_impact?.cost_effectiveness_grade || 'Unknown'
          },
          executive_summary: {
            performance_grade: result.executive_summary?.performance_grade || 'C',
            overall_score: result.executive_summary?.overall_score || 0.7,
            key_insights: result.executive_summary?.key_insights || [],
            recommendation: result.executive_summary?.recommendation || 'Continue monitoring'
          }
        }
        
        setAnalyticsData(transformedData)
        setLastUpdated(new Date())
      } else {
        setError(result.error || 'Failed to load analytics')
      }
      
    } catch (err) {
      console.error('Failed to load analytics:', err)
      setError(err instanceof Error ? err.message : 'Unknown error')
    }
  }, [selectedTimeframe])

  // Auto-refresh effect
  useEffect(() => {
    const initialize = async () => {
      setLoading(true)
      await loadAnalytics()
      setLoading(false)
    }

    initialize()

    if (autoRefresh) {
      const interval = setInterval(loadAnalytics, 30000) // 30 seconds
      return () => clearInterval(interval)
    }
  }, [loadAnalytics, autoRefresh])

  // Format currency
  const formatCurrency = (amount: number): string => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(amount)
  }

  // Format duration
  const formatDuration = (ms: number): string => {
    if (ms < 1000) return `${ms}ms`
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
    return `${(ms / 60000).toFixed(1)}m`
  }

  // Get trend icon
  const getTrendIcon = (direction: string) => {
    switch (direction) {
      case 'improving': return <TrendingUp className="h-4 w-4 text-green-500" />
      case 'declining': return <TrendingDown className="h-4 w-4 text-red-500" />
      default: return <Activity className="h-4 w-4 text-gray-500" />
    }
  }

  // Get performance color
  const getPerformanceColor = (grade: string): string => {
    switch (grade) {
      case 'A': return 'text-green-600 bg-green-100'
      case 'B': return 'text-blue-600 bg-blue-100'
      case 'C': return 'text-yellow-600 bg-yellow-100'
      case 'D': return 'text-orange-600 bg-orange-100'
      case 'F': return 'text-red-600 bg-red-100'
      default: return 'text-gray-600 bg-gray-100'
    }
  }

  if (loading) {
    return (
      <div className="container mx-auto p-6">
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
          <span className="ml-4 text-lg">Loading response analytics...</span>
        </div>
      </div>
    )
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <BarChart3 className="h-8 w-8 text-blue-600" />
            Response Analytics Dashboard
          </h1>
          <p className="text-gray-600 mt-1">
            Comprehensive response effectiveness monitoring and optimization insights
          </p>
        </div>
        
        <div className="flex items-center gap-4">
          {/* Timeframe Selector */}
          <div className="flex gap-2">
            {['1d', '7d', '30d', '90d'].map(timeframe => (
              <Button
                key={timeframe}
                variant={selectedTimeframe === timeframe ? 'default' : 'outline'}
                size="sm"
                onClick={() => setSelectedTimeframe(timeframe)}
              >
                {timeframe}
              </Button>
            ))}
          </div>
          
          {/* Controls */}
          <Button
            variant="outline"
            size="sm"
            onClick={() => setAutoRefresh(!autoRefresh)}
          >
            <RefreshCw className={`h-4 w-4 ${autoRefresh ? 'animate-spin' : ''}`} />
            {autoRefresh ? 'Auto' : 'Manual'}
          </Button>
          
          <Button variant="outline" size="sm" onClick={loadAnalytics}>
            <Eye className="h-4 w-4 mr-1" />
            Refresh
          </Button>
          
          <Button variant="outline" size="sm">
            <Download className="h-4 w-4 mr-1" />
            Export
          </Button>
        </div>
      </div>

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

      {/* Executive Summary */}
      {analyticsData && (
        <Card className="bg-gradient-to-r from-blue-50 to-purple-50 border-blue-200">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Award className="h-5 w-5 text-gold-500" />
              Executive Summary
              {lastUpdated && (
                <span className="text-xs text-gray-500 ml-auto">
                  Updated {lastUpdated.toLocaleTimeString()}
                </span>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className={`text-3xl font-bold px-4 py-2 rounded-full ${getPerformanceColor(analyticsData.executive_summary.performance_grade)}`}>
                  {analyticsData.executive_summary.performance_grade}
                </div>
                <div className="text-sm text-gray-600 mt-2">Performance Grade</div>
              </div>
              
              <div className="text-center">
                <div className="text-3xl font-bold text-blue-600">
                  {Math.round(analyticsData.executive_summary.overall_score * 100)}%
                </div>
                <div className="text-sm text-gray-600 mt-2">Overall Score</div>
              </div>
              
              <div className="text-center">
                <div className="text-3xl font-bold text-green-600">
                  {analyticsData.response_metrics.successful_workflows}
                </div>
                <div className="text-sm text-gray-600 mt-2">Successful Workflows</div>
              </div>
              
              <div className="text-center">
                <div className="text-3xl font-bold text-purple-600">
                  {formatCurrency(analyticsData.business_impact.estimated_roi)}
                </div>
                <div className="text-sm text-gray-600 mt-2">Estimated ROI</div>
              </div>
            </div>
            
            <div className="mt-6 p-4 bg-white rounded-lg border">
              <h4 className="font-semibold mb-2">Key Insights</h4>
              <ul className="space-y-1">
                {analyticsData.executive_summary.key_insights.map((insight, index) => (
                  <li key={index} className="flex items-start gap-2 text-sm">
                    <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                    {insight}
                  </li>
                ))}
              </ul>
              
              <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                <div className="flex items-start gap-2">
                  <Target className="h-4 w-4 text-blue-500 mt-0.5" />
                  <div>
                    <h5 className="font-medium text-sm">Executive Recommendation</h5>
                    <p className="text-sm text-gray-700 mt-1">
                      {analyticsData.executive_summary.recommendation}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Main Analytics Content */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="effectiveness">Effectiveness</TabsTrigger>
          <TabsTrigger value="trends">Trends</TabsTrigger>
          <TabsTrigger value="optimization">Optimization</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          {analyticsData && (
            <>
              {/* Core Metrics */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <Card>
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-gray-600">Success Rate</p>
                        <p className="text-2xl font-bold text-green-600">
                          {Math.round(analyticsData.response_metrics.success_rate * 100)}%
                        </p>
                      </div>
                      <CheckCircle className="h-8 w-8 text-green-600" />
                    </div>
                    <div className="flex items-center mt-2">
                      {getTrendIcon(analyticsData.trend_analysis.success_rate_trend.direction)}
                      <span className="text-xs text-gray-500 ml-1">
                        {analyticsData.trend_analysis.success_rate_trend.direction}
                      </span>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-gray-600">Avg Response Time</p>
                        <p className="text-2xl font-bold text-blue-600">
                          {formatDuration(analyticsData.response_metrics.average_response_time)}
                        </p>
                      </div>
                      <Clock className="h-8 w-8 text-blue-600" />
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-gray-600">False Positives</p>
                        <p className="text-2xl font-bold text-orange-600">
                          {Math.round(analyticsData.response_metrics.false_positive_rate * 100)}%
                        </p>
                      </div>
                      <AlertTriangle className="h-8 w-8 text-orange-600" />
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-gray-600">Cost Effectiveness</p>
                        <p className="text-2xl font-bold text-purple-600">
                          {analyticsData.business_impact.cost_effectiveness_grade}
                        </p>
                      </div>
                      <DollarSign className="h-8 w-8 text-purple-600" />
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Response Metrics Component */}
              <ResponseMetrics 
                data={analyticsData.response_metrics}
                timeframe={selectedTimeframe}
              />
            </>
          )}
        </TabsContent>

        <TabsContent value="effectiveness" className="space-y-6">
          {analyticsData && (
            <EffectivenessAnalysis 
              data={analyticsData.effectiveness_analysis}
              timeframe={selectedTimeframe}
            />
          )}
        </TabsContent>

        <TabsContent value="trends" className="space-y-6">
          {analyticsData && (
            <TrendAnalysis 
              data={analyticsData.trend_analysis}
              timeframe={selectedTimeframe}
            />
          )}
        </TabsContent>

        <TabsContent value="optimization" className="space-y-6">
          {analyticsData && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="h-5 w-5" />
                  Optimization Opportunities
                </CardTitle>
                <CardDescription>
                  AI-powered recommendations for improving response effectiveness
                </CardDescription>
              </CardHeader>
              
              <CardContent>
                <div className="space-y-4">
                  {analyticsData.effectiveness_analysis.improvement_recommendations.map((recommendation, index) => (
                    <div key={index} className="p-4 border rounded-lg">
                      <div className="flex items-start justify-between">
                        <div className="flex items-start gap-3">
                          <div className="p-2 rounded-lg bg-blue-100">
                            <Target className="h-4 w-4 text-blue-600" />
                          </div>
                          <div>
                            <h4 className="font-semibold">Optimization #{index + 1}</h4>
                            <p className="text-sm text-gray-600 mt-1">{recommendation}</p>
                          </div>
                        </div>
                        <Badge variant="outline">Priority</Badge>
                      </div>
                    </div>
                  ))}
                  
                  {analyticsData.effectiveness_analysis.improvement_recommendations.length === 0 && (
                    <div className="text-center py-8">
                      <CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-4" />
                      <h4 className="font-medium text-gray-900 mb-2">System Optimized</h4>
                      <p className="text-gray-600">
                        No immediate optimization opportunities identified. 
                        Continue monitoring for emerging patterns.
                      </p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default ResponseAnalyticsPage








