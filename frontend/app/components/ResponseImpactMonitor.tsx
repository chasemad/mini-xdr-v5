'use client'

/**
 * Response Impact Monitor Component
 * 
 * Real-time monitoring and analytics for response action effectiveness,
 * performance metrics, and business impact assessment.
 */

import React, { useState, useEffect, useCallback } from 'react'
import { getResponseImpactMetrics } from '@/app/lib/api'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { 
  BarChart3, 
  TrendingUp, 
  TrendingDown,
  Shield, 
  Clock, 
  Target,
  AlertTriangle,
  CheckCircle,
  Activity,
  Zap,
  Users,
  DollarSign,
  Server,
  RefreshCw
} from 'lucide-react'

interface ImpactMetrics {
  attacks_blocked: number
  false_positives: number
  systems_affected: number
  users_affected: number
  response_time_ms: number
  success_rate: number
  confidence_score: number
  downtime_minutes: number
  cost_impact_usd: number
  compliance_impact: string
}

interface ResponseAnalytics {
  summary: {
    total_attacks_blocked: number
    total_false_positives: number
    average_response_time_ms: number
    average_success_rate: number
    metrics_count: number
  }
  detailed_metrics: ImpactMetrics[]
}

interface PerformanceTrend {
  timestamp: string
  success_rate: number
  response_time: number
  attacks_blocked: number
}

interface ResponseImpactMonitorProps {
  workflowId?: string
  refreshInterval?: number
  className?: string
}

const ResponseImpactMonitor: React.FC<ResponseImpactMonitorProps> = ({
  workflowId,
  refreshInterval = 30000, // 30 seconds
  className = ""
}) => {
  // State management
  const [analytics, setAnalytics] = useState<ResponseAnalytics | null>(null)
  const [trends, setTrends] = useState<PerformanceTrend[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const [activeTab, setActiveTab] = useState('overview')

  // Load analytics data
  const loadAnalytics = useCallback(async () => {
    try {
      // Get real analytics data from API
      const data = await getResponseImpactMetrics({
        workflow_id: workflowId,
        days_back: 7
      })

      if (data.success) {
        setAnalytics(data)
        setError(null)
        setLastUpdated(new Date())
        
        // Generate trend data from real metrics (simplified for now)
        const trendData: PerformanceTrend[] = data.detailed_metrics?.slice(0, 24).map((metric, i) => ({
          timestamp: new Date(Date.now() - (23 - i) * 60 * 60 * 1000).toISOString(),
          success_rate: metric.success_rate || 0,
          response_time: metric.response_time_ms || 0,
          attacks_blocked: metric.attacks_blocked || 0
        })) || []
        
        setTrends(trendData)
      } else {
        throw new Error(data.error || 'Failed to load analytics')
      }
      
    } catch (err) {
      console.error('Failed to load analytics:', err)
      setError(err instanceof Error ? err.message : 'Failed to load analytics')
      
      // Set empty data on error
      setAnalytics(null)
      setTrends([])
    }
  }, [workflowId])

  // Initialize and set up refresh interval
  useEffect(() => {
    const initialize = async () => {
      setLoading(true)
      await loadAnalytics()
      setLoading(false)
    }

    initialize()

    // Set up refresh interval
    const interval = setInterval(loadAnalytics, refreshInterval)
    return () => clearInterval(interval)
  }, [loadAnalytics, refreshInterval])

  // Helper functions
  const formatDuration = (ms: number): string => {
    if (ms < 1000) return `${ms}ms`
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
    return `${(ms / 60000).toFixed(1)}m`
  }

  const formatCurrency = (amount: number): string => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(amount)
  }

  const getComplianceColor = (impact: string): string => {
    switch (impact) {
      case 'none': return 'text-green-600 bg-green-100'
      case 'low': return 'text-yellow-600 bg-yellow-100'
      case 'medium': return 'text-orange-600 bg-orange-100'
      case 'high': return 'text-red-600 bg-red-100'
      case 'critical': return 'text-red-800 bg-red-200'
      default: return 'text-gray-600 bg-gray-100'
    }
  }

  const getTrendDirection = (current: number, previous: number): 'up' | 'down' | 'stable' => {
    const change = ((current - previous) / previous) * 100
    if (Math.abs(change) < 5) return 'stable'
    return change > 0 ? 'up' : 'down'
  }

  if (loading) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Response Impact Monitor
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-32">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <span className="ml-2">Loading analytics...</span>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!analytics) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Response Impact Monitor
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <AlertTriangle className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-600">No analytics data available</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Response Impact Monitor
            </CardTitle>
            <CardDescription>
              Real-time effectiveness and performance analytics
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            {lastUpdated && (
              <span className="text-xs text-gray-500">
                Updated {lastUpdated.toLocaleTimeString()}
              </span>
            )}
            <button
              onClick={loadAnalytics}
              className="p-1 rounded-full hover:bg-gray-100"
              title="Refresh data"
            >
              <RefreshCw className="h-4 w-4" />
            </button>
          </div>
        </div>
      </CardHeader>

      <CardContent>
        {error && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-md">
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-red-500" />
              <span className="text-red-700 text-sm">{error}</span>
            </div>
          </div>
        )}

        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="performance">Performance</TabsTrigger>
            <TabsTrigger value="impact">Business Impact</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-4">
            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-600">Attacks Blocked</p>
                      <p className="text-2xl font-bold text-green-600">
                        {analytics.summary.total_attacks_blocked}
                      </p>
                    </div>
                    <Shield className="h-8 w-8 text-green-600" />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-600">Success Rate</p>
                      <p className="text-2xl font-bold text-blue-600">
                        {Math.round(analytics.summary.average_success_rate * 100)}%
                      </p>
                    </div>
                    <Target className="h-8 w-8 text-blue-600" />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-600">Avg Response Time</p>
                      <p className="text-2xl font-bold text-orange-600">
                        {formatDuration(analytics.summary.average_response_time_ms)}
                      </p>
                    </div>
                    <Clock className="h-8 w-8 text-orange-600" />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-600">False Positives</p>
                      <p className="text-2xl font-bold text-red-600">
                        {analytics.summary.total_false_positives}
                      </p>
                    </div>
                    <AlertTriangle className="h-8 w-8 text-red-600" />
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Recent Metrics */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Recent Response Metrics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {analytics.detailed_metrics.slice(0, 3).map((metric, index) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div className="flex items-center gap-3">
                        <div className="flex flex-col">
                          <span className="font-medium">Response #{index + 1}</span>
                          <span className="text-sm text-gray-600">
                            {metric.attacks_blocked} attacks blocked
                          </span>
                        </div>
                      </div>
                      
                      <div className="flex items-center gap-4">
                        <div className="text-right">
                          <div className="text-sm font-medium">
                            {Math.round(metric.success_rate * 100)}% success
                          </div>
                          <div className="text-xs text-gray-500">
                            {formatDuration(metric.response_time_ms)}
                          </div>
                        </div>
                        
                        <Badge 
                          variant="outline"
                          className={getComplianceColor(metric.compliance_impact)}
                        >
                          {metric.compliance_impact} impact
                        </Badge>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="performance" className="space-y-4">
            {/* Performance Trends */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Performance Trends (24h)</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">
                      {Math.round(trends[trends.length - 1]?.success_rate * 100 || 0)}%
                    </div>
                    <div className="text-sm text-gray-600">Current Success Rate</div>
                    <div className="flex items-center justify-center mt-1">
                      {getTrendDirection(
                        trends[trends.length - 1]?.success_rate || 0,
                        trends[trends.length - 2]?.success_rate || 0
                      ) === 'up' ? (
                        <TrendingUp className="h-4 w-4 text-green-500" />
                      ) : (
                        <TrendingDown className="h-4 w-4 text-red-500" />
                      )}
                    </div>
                  </div>
                  
                  <div className="text-center">
                    <div className="text-2xl font-bold text-orange-600">
                      {formatDuration(trends[trends.length - 1]?.response_time || 0)}
                    </div>
                    <div className="text-sm text-gray-600">Current Response Time</div>
                    <div className="flex items-center justify-center mt-1">
                      {getTrendDirection(
                        trends[trends.length - 2]?.response_time || 0,
                        trends[trends.length - 1]?.response_time || 0
                      ) === 'up' ? (
                        <TrendingUp className="h-4 w-4 text-green-500" />
                      ) : (
                        <TrendingDown className="h-4 w-4 text-red-500" />
                      )}
                    </div>
                  </div>
                  
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600">
                      {trends.reduce((sum, t) => sum + t.attacks_blocked, 0)}
                    </div>
                    <div className="text-sm text-gray-600">Total Attacks Blocked</div>
                    <div className="flex items-center justify-center mt-1">
                      <Activity className="h-4 w-4 text-blue-500" />
                    </div>
                  </div>
                </div>

                {/* Simple trend visualization */}
                <div className="space-y-4">
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium">Success Rate Trend</span>
                      <span className="text-sm text-gray-500">Last 24 hours</span>
                    </div>
                    <Progress 
                      value={Math.round((trends[trends.length - 1]?.success_rate || 0) * 100)} 
                      className="h-2" 
                    />
                  </div>
                  
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium">Response Time Performance</span>
                      <span className="text-sm text-gray-500">
                        {formatDuration(trends[trends.length - 1]?.response_time || 0)} current
                      </span>
                    </div>
                    <Progress 
                      value={Math.max(0, 100 - ((trends[trends.length - 1]?.response_time || 0) / 5000) * 100)} 
                      className="h-2" 
                    />
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="impact" className="space-y-4">
            {/* Business Impact Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <Users className="h-5 w-5" />
                    User Impact
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {analytics.detailed_metrics.slice(0, 3).map((metric, index) => (
                      <div key={index} className="flex items-center justify-between">
                        <span className="text-sm">Response #{index + 1}</span>
                        <div className="text-right">
                          <div className="font-medium">{metric.users_affected} users</div>
                          <div className="text-xs text-gray-500">
                            {metric.systems_affected} systems
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <DollarSign className="h-5 w-5" />
                    Cost Impact
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {analytics.detailed_metrics.slice(0, 3).map((metric, index) => (
                      <div key={index} className="flex items-center justify-between">
                        <span className="text-sm">Response #{index + 1}</span>
                        <div className="text-right">
                          <div className="font-medium">{formatCurrency(metric.cost_impact_usd)}</div>
                          <div className="text-xs text-gray-500">
                            {metric.downtime_minutes}m downtime
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Compliance Impact Summary */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Compliance Impact Summary</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {['none', 'low', 'medium', 'high'].map(level => {
                    const count = analytics.detailed_metrics.filter(m => m.compliance_impact === level).length
                    return (
                      <div key={level} className="text-center">
                        <div className={`text-lg font-bold px-3 py-1 rounded-full ${getComplianceColor(level)}`}>
                          {count}
                        </div>
                        <div className="text-sm text-gray-600 mt-1 capitalize">{level} Impact</div>
                      </div>
                    )
                  })}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}

export default ResponseImpactMonitor
