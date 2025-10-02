'use client'

/**
 * Trend Analysis Component
 * 
 * Advanced trend visualization and analysis for response metrics,
 * predictive insights, and performance forecasting.
 */

import React, { useState, useEffect, useMemo } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { 
  TrendingUp, 
  TrendingDown,
  Activity,
  BarChart3,
  Target,
  Clock,
  AlertTriangle,
  CheckCircle,
  Eye,
  Zap,
  Calendar,
  ArrowUp,
  ArrowDown,
  Minus,
  Brain,
  Sparkles
} from 'lucide-react'

interface TrendDataPoint {
  timestamp: string
  success_rate: number
  response_time: number
  incident_volume: number
  effectiveness_score: number
}

interface TrendInfo {
  slope: number
  direction: string
  confidence: string
}

interface TrendAnalysisData {
  trend_data: TrendDataPoint[]
  success_rate_trend: TrendInfo
  response_time_trend?: TrendInfo
  incident_volume_trend?: TrendInfo
}

interface TrendAnalysisProps {
  data: TrendAnalysisData
  timeframe: string
  className?: string
}

const TrendAnalysis: React.FC<TrendAnalysisProps> = ({
  data,
  timeframe,
  className = ""
}) => {
  const [selectedMetric, setSelectedMetric] = useState<'success_rate' | 'response_time' | 'incident_volume' | 'effectiveness'>('success_rate')
  const [showPredictions, setShowPredictions] = useState(true)
  const [viewMode, setViewMode] = useState<'chart' | 'table'>('chart')

  // Calculate trend statistics
  const trendStats = useMemo(() => {
    if (!data.trend_data || data.trend_data.length < 2) {
      return { min: 0, max: 0, average: 0, change: 0, changePercent: 0 }
    }

    const values = data.trend_data.map(d => {
      switch (selectedMetric) {
        case 'success_rate': return d.success_rate
        case 'response_time': return d.response_time
        case 'incident_volume': return d.incident_volume
        case 'effectiveness': return d.effectiveness_score
        default: return 0
      }
    })

    const min = Math.min(...values)
    const max = Math.max(...values)
    const average = values.reduce((sum, val) => sum + val, 0) / values.length
    const firstValue = values[0]
    const lastValue = values[values.length - 1]
    const change = lastValue - firstValue
    const changePercent = firstValue !== 0 ? (change / firstValue) * 100 : 0

    return { min, max, average, change, changePercent }
  }, [data.trend_data, selectedMetric])

  // Get trend direction icon
  const getTrendIcon = (direction: string, size: string = "h-4 w-4") => {
    switch (direction) {
      case 'improving':
        return <TrendingUp className={`${size} text-green-500`} />
      case 'declining':
        return <TrendingDown className={`${size} text-red-500`} />
      default:
        return <Minus className={`${size} text-gray-500`} />
    }
  }

  // Get confidence color
  const getConfidenceColor = (confidence: string): string => {
    switch (confidence) {
      case 'high': return 'text-green-600 bg-green-100'
      case 'medium': return 'text-yellow-600 bg-yellow-100'
      case 'low': return 'text-red-600 bg-red-100'
      default: return 'text-gray-600 bg-gray-100'
    }
  }

  // Format metric value
  const formatMetricValue = (value: number, metric: string): string => {
    switch (metric) {
      case 'success_rate':
      case 'effectiveness':
        return `${Math.round(value * 100)}%`
      case 'response_time':
        return value < 1000 ? `${value}ms` : 
               value < 60000 ? `${(value / 1000).toFixed(1)}s` : 
               `${(value / 60000).toFixed(1)}m`
      case 'incident_volume':
        return value.toString()
      default:
        return value.toFixed(2)
    }
  }

  // Generate simple chart visualization
  const generateSimpleChart = (dataPoints: TrendDataPoint[]) => {
    if (!dataPoints || dataPoints.length === 0) return null

    const values = dataPoints.map(d => {
      switch (selectedMetric) {
        case 'success_rate': return d.success_rate * 100
        case 'response_time': return d.response_time / 1000 // Convert to seconds
        case 'incident_volume': return d.incident_volume
        case 'effectiveness': return d.effectiveness_score * 100
        default: return 0
      }
    })

    const max = Math.max(...values)
    const min = Math.min(...values)
    const range = max - min || 1

    return (
      <div className="flex items-end justify-between h-32 p-4 bg-gray-50 rounded-lg">
        {values.map((value, index) => {
          const height = ((value - min) / range) * 100
          const isLast = index === values.length - 1
          
          return (
            <div key={index} className="flex flex-col items-center gap-1">
              <div 
                className={`w-2 rounded-t transition-all ${
                  isLast ? 'bg-blue-500' : 'bg-gray-300'
                }`}
                style={{ height: `${Math.max(height, 5)}%` }}
              />
              <div className="text-xs text-gray-500 transform -rotate-45 origin-bottom-left">
                {formatMetricValue(value, selectedMetric)}
              </div>
            </div>
          )
        })}
      </div>
    )
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Trend Overview */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Trend Analysis Overview
            <Badge variant="outline" className="ml-auto">
              {data.trend_data.length} data points
            </Badge>
          </CardTitle>
          <CardDescription>
            Historical performance trends and predictive insights
          </CardDescription>
        </CardHeader>
        
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Success Rate Trend */}
            <div className="p-4 border rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  {getTrendIcon(data.success_rate_trend.direction)}
                  <span className="font-medium text-sm">Success Rate</span>
                </div>
                <Badge 
                  variant="outline"
                  className={getConfidenceColor(data.success_rate_trend.confidence)}
                >
                  {data.success_rate_trend.confidence} confidence
                </Badge>
              </div>
              
              <div className="text-lg font-bold text-blue-600 mb-1">
                {data.success_rate_trend.direction.charAt(0).toUpperCase() + data.success_rate_trend.direction.slice(1)}
              </div>
              
              <div className="text-xs text-gray-600">
                Slope: {data.success_rate_trend.slope.toFixed(4)}
              </div>
            </div>

            {/* Response Time Trend */}
            {data.response_time_trend && (
              <div className="p-4 border rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    {getTrendIcon(data.response_time_trend.direction)}
                    <span className="font-medium text-sm">Response Time</span>
                  </div>
                  <Badge 
                    variant="outline"
                    className={getConfidenceColor(data.response_time_trend.confidence)}
                  >
                    {data.response_time_trend.confidence} confidence
                  </Badge>
                </div>
                
                <div className="text-lg font-bold text-orange-600 mb-1">
                  {data.response_time_trend.direction.charAt(0).toUpperCase() + data.response_time_trend.direction.slice(1)}
                </div>
                
                <div className="text-xs text-gray-600">
                  Slope: {data.response_time_trend.slope.toFixed(4)}
                </div>
              </div>
            )}

            {/* Incident Volume Trend */}
            {data.incident_volume_trend && (
              <div className="p-4 border rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    {getTrendIcon(data.incident_volume_trend.direction)}
                    <span className="font-medium text-sm">Incident Volume</span>
                  </div>
                  <Badge 
                    variant="outline"
                    className={getConfidenceColor(data.incident_volume_trend.confidence)}
                  >
                    {data.incident_volume_trend.confidence} confidence
                  </Badge>
                </div>
                
                <div className="text-lg font-bold text-purple-600 mb-1">
                  {data.incident_volume_trend.direction.charAt(0).toUpperCase() + data.incident_volume_trend.direction.slice(1)}
                </div>
                
                <div className="text-xs text-gray-600">
                  Slope: {data.incident_volume_trend.slope.toFixed(4)}
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Detailed Trend Visualization */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Trend Visualization
            </CardTitle>
            
            <div className="flex items-center gap-2">
              {/* Metric Selector */}
              <div className="flex gap-1">
                {[
                  { key: 'success_rate', label: 'Success', icon: CheckCircle },
                  { key: 'response_time', label: 'Time', icon: Clock },
                  { key: 'incident_volume', label: 'Volume', icon: BarChart3 },
                  { key: 'effectiveness', label: 'Effectiveness', icon: Target }
                ].map(({ key, label, icon: Icon }) => (
                  <Button
                    key={key}
                    size="sm"
                    variant={selectedMetric === key ? 'default' : 'outline'}
                    onClick={() => setSelectedMetric(key as any)}
                    className="flex items-center gap-1"
                  >
                    <Icon className="h-3 w-3" />
                    {label}
                  </Button>
                ))}
              </div>
              
              <div className="flex gap-1">
                <Button
                  size="sm"
                  variant={viewMode === 'chart' ? 'default' : 'outline'}
                  onClick={() => setViewMode('chart')}
                >
                  Chart
                </Button>
                <Button
                  size="sm"
                  variant={viewMode === 'table' ? 'default' : 'outline'}
                  onClick={() => setViewMode('table')}
                >
                  Table
                </Button>
              </div>
            </div>
          </div>
        </CardHeader>
        
        <CardContent>
          <Tabs value={viewMode} onValueChange={(value) => setViewMode(value as 'chart' | 'table')}>
            <TabsContent value="chart">
              {/* Simple Chart Visualization */}
              <div className="space-y-4">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-medium">
                    {selectedMetric.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())} Trend
                  </span>
                  <div className="flex items-center gap-4">
                    <span>Min: {formatMetricValue(trendStats.min, selectedMetric)}</span>
                    <span>Avg: {formatMetricValue(trendStats.average, selectedMetric)}</span>
                    <span>Max: {formatMetricValue(trendStats.max, selectedMetric)}</span>
                  </div>
                </div>
                
                {generateSimpleChart(data.trend_data)}
                
                <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center gap-2">
                    {trendStats.changePercent > 5 ? (
                      <ArrowUp className="h-4 w-4 text-green-500" />
                    ) : trendStats.changePercent < -5 ? (
                      <ArrowDown className="h-4 w-4 text-red-500" />
                    ) : (
                      <Minus className="h-4 w-4 text-gray-500" />
                    )}
                    <span className="text-sm font-medium">
                      {Math.abs(trendStats.changePercent).toFixed(1)}% 
                      {trendStats.changePercent > 0 ? ' increase' : trendStats.changePercent < 0 ? ' decrease' : ' no change'}
                    </span>
                  </div>
                  
                  <span className="text-sm text-gray-600">
                    Period-over-period change
                  </span>
                </div>
              </div>
            </TabsContent>
            
            <TabsContent value="table">
              {/* Data Table */}
              <div className="space-y-4">
                <div className="max-h-64 overflow-y-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-gray-50 sticky top-0">
                      <tr>
                        <th className="text-left p-2">Time</th>
                        <th className="text-right p-2">Success Rate</th>
                        <th className="text-right p-2">Response Time</th>
                        <th className="text-right p-2">Volume</th>
                        <th className="text-right p-2">Effectiveness</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.trend_data.map((point, index) => (
                        <tr key={index} className="border-t hover:bg-gray-50">
                          <td className="p-2">
                            {new Date(point.timestamp).toLocaleDateString()}
                          </td>
                          <td className="text-right p-2">
                            {Math.round(point.success_rate * 100)}%
                          </td>
                          <td className="text-right p-2">
                            {formatMetricValue(point.response_time, 'response_time')}
                          </td>
                          <td className="text-right p-2">
                            {point.incident_volume}
                          </td>
                          <td className="text-right p-2">
                            {Math.round(point.effectiveness_score * 100)}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* Predictive Analysis */}
      {showPredictions && (
        <Card className="border-purple-200 bg-gradient-to-r from-purple-50 to-pink-50">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5 text-purple-600" />
                Predictive Analysis
                <Sparkles className="h-4 w-4 text-pink-500" />
              </CardTitle>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowPredictions(false)}
              >
                ×
              </Button>
            </div>
            <CardDescription>
              AI-powered predictions based on current trends
            </CardDescription>
          </CardHeader>
          
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Next Period Predictions */}
              <div className="space-y-4">
                <h4 className="font-semibold flex items-center gap-1">
                  <Target className="h-4 w-4" />
                  Next 24 Hour Predictions
                </h4>
                
                <div className="space-y-3">
                  <div className="flex items-center justify-between p-3 bg-white border rounded-lg">
                    <span className="text-sm">Predicted Success Rate</span>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="text-green-600">
                        {Math.round((trendStats.average + (trendStats.change * 0.1)) * 100)}%
                      </Badge>
                      {getTrendIcon(data.success_rate_trend.direction, "h-3 w-3")}
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between p-3 bg-white border rounded-lg">
                    <span className="text-sm">Predicted Response Time</span>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="text-blue-600">
                        {formatMetricValue(trendStats.average * 0.95, 'response_time')}
                      </Badge>
                      <TrendingUp className="h-3 w-3 text-green-500" />
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between p-3 bg-white border rounded-lg">
                    <span className="text-sm">Predicted Volume</span>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="text-purple-600">
                        {Math.round(trendStats.average * 1.1)}
                      </Badge>
                      <Activity className="h-3 w-3 text-blue-500" />
                    </div>
                  </div>
                </div>
              </div>

              {/* Trend Insights */}
              <div className="space-y-4">
                <h4 className="font-semibold flex items-center gap-1">
                  <Eye className="h-4 w-4" />
                  Trend Insights
                </h4>
                
                <div className="space-y-2">
                  {data.success_rate_trend.direction === 'improving' && (
                    <div className="flex items-start gap-2 text-sm">
                      <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                      <div>
                        <div className="font-medium">Positive Success Trend</div>
                        <div className="text-gray-600">Response effectiveness is improving over time</div>
                      </div>
                    </div>
                  )}
                  
                  {data.success_rate_trend.direction === 'declining' && (
                    <div className="flex items-start gap-2 text-sm">
                      <AlertTriangle className="h-4 w-4 text-orange-500 mt-0.5" />
                      <div>
                        <div className="font-medium">Declining Success Trend</div>
                        <div className="text-gray-600">Response effectiveness needs attention</div>
                      </div>
                    </div>
                  )}
                  
                  {trendStats.changePercent > 10 && (
                    <div className="flex items-start gap-2 text-sm">
                      <Zap className="h-4 w-4 text-blue-500 mt-0.5" />
                      <div>
                        <div className="font-medium">Significant Change Detected</div>
                        <div className="text-gray-600">
                          {Math.abs(trendStats.changePercent).toFixed(1)}% change in {selectedMetric.replace('_', ' ')}
                        </div>
                      </div>
                    </div>
                  )}
                  
                  <div className="flex items-start gap-2 text-sm">
                    <Brain className="h-4 w-4 text-purple-500 mt-0.5" />
                    <div>
                      <div className="font-medium">AI Recommendation</div>
                      <div className="text-gray-600">
                        {data.success_rate_trend.direction === 'improving' 
                          ? "Continue current strategies - performance is improving"
                          : data.success_rate_trend.direction === 'declining'
                          ? "Review and optimize response strategies immediately"
                          : "Monitor closely for emerging patterns"
                        }
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Detailed Trend Chart */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Detailed Trend Chart
            </CardTitle>
            
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-600">Metric:</span>
              <Badge variant="outline">
                {selectedMetric.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
              </Badge>
            </div>
          </div>
        </CardHeader>
        
        <CardContent>
          {data.trend_data && data.trend_data.length > 0 ? (
            <div className="space-y-4">
              {generateSimpleChart(data.trend_data)}
              
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t">
                <div className="text-center">
                  <div className="text-lg font-bold text-blue-600">
                    {formatMetricValue(trendStats.min, selectedMetric)}
                  </div>
                  <div className="text-xs text-gray-600">Minimum</div>
                </div>
                
                <div className="text-center">
                  <div className="text-lg font-bold text-green-600">
                    {formatMetricValue(trendStats.average, selectedMetric)}
                  </div>
                  <div className="text-xs text-gray-600">Average</div>
                </div>
                
                <div className="text-center">
                  <div className="text-lg font-bold text-purple-600">
                    {formatMetricValue(trendStats.max, selectedMetric)}
                  </div>
                  <div className="text-xs text-gray-600">Maximum</div>
                </div>
                
                <div className="text-center">
                  <div className={`text-lg font-bold ${
                    trendStats.changePercent > 0 ? 'text-green-600' : 
                    trendStats.changePercent < 0 ? 'text-red-600' : 'text-gray-600'
                  }`}>
                    {trendStats.changePercent > 0 ? '+' : ''}{trendStats.changePercent.toFixed(1)}%
                  </div>
                  <div className="text-xs text-gray-600">Change</div>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-12">
              <BarChart3 className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h4 className="font-medium text-gray-900 mb-2">No Trend Data</h4>
              <p className="text-gray-600">
                Insufficient data points for trend analysis. 
                At least 2 data points required.
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

export default TrendAnalysis





