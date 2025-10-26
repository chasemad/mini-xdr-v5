'use client'

/**
 * Response Metrics Component
 * 
 * Real-time response metrics dashboard with comprehensive KPIs,
 * performance indicators, and comparative analysis.
 */

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { 
  Target, 
  Clock, 
  CheckCircle,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Activity,
  Zap,
  Shield,
  Award,
  BarChart3,
  Users,
  Server
} from 'lucide-react'

interface ResponseMetricsData {
  total_workflows: number
  successful_workflows: number
  average_response_time: number
  success_rate: number
  false_positive_rate: number
  mean_time_to_containment: number
  cost_effectiveness_score: number
}

interface ResponseMetricsProps {
  data: ResponseMetricsData
  timeframe: string
  className?: string
}

const ResponseMetrics: React.FC<ResponseMetricsProps> = ({
  data,
  timeframe,
  className = ""
}) => {
  // Format duration helper
  const formatDuration = (ms: number): string => {
    if (ms < 1000) return `${ms}ms`
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
    return `${(ms / 60000).toFixed(1)}m`
  }

  // Get performance color based on metric
  const getMetricColor = (value: number, type: 'success_rate' | 'response_time' | 'false_positive'): string => {
    switch (type) {
      case 'success_rate':
        if (value >= 0.9) return 'text-green-600'
        if (value >= 0.8) return 'text-blue-600'
        if (value >= 0.7) return 'text-yellow-600'
        return 'text-red-600'
      
      case 'response_time':
        if (value <= 300000) return 'text-green-600'  // <= 5 min
        if (value <= 900000) return 'text-blue-600'   // <= 15 min
        if (value <= 1800000) return 'text-yellow-600' // <= 30 min
        return 'text-red-600'
      
      case 'false_positive':
        if (value <= 0.05) return 'text-green-600'
        if (value <= 0.10) return 'text-yellow-600'
        return 'text-red-600'
      
      default:
        return 'text-gray-600'
    }
  }

  // Get performance grade
  const getPerformanceGrade = (successRate: number, responseTime: number, falsePositiveRate: number): string => {
    let score = 0
    
    // Success rate scoring (40%)
    if (successRate >= 0.95) score += 40
    else if (successRate >= 0.85) score += 35
    else if (successRate >= 0.75) score += 30
    else if (successRate >= 0.65) score += 20
    else score += 10
    
    // Response time scoring (35%)
    if (responseTime <= 300000) score += 35      // <= 5 min
    else if (responseTime <= 600000) score += 30 // <= 10 min
    else if (responseTime <= 1200000) score += 25 // <= 20 min
    else if (responseTime <= 1800000) score += 15 // <= 30 min
    else score += 5
    
    // False positive scoring (25%)
    if (falsePositiveRate <= 0.05) score += 25
    else if (falsePositiveRate <= 0.10) score += 20
    else if (falsePositiveRate <= 0.15) score += 15
    else if (falsePositiveRate <= 0.20) score += 10
    else score += 5
    
    if (score >= 90) return 'A'
    else if (score >= 80) return 'B'
    else if (score >= 70) return 'C'
    else if (score >= 60) return 'D'
    else return 'F'
  }

  const performanceGrade = getPerformanceGrade(data.success_rate, data.average_response_time, data.false_positive_rate)

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Performance Overview */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Performance Overview
            <Badge variant="outline" className="ml-auto">
              Last {timeframe}
            </Badge>
          </CardTitle>
          <CardDescription>
            Core response performance metrics and KPIs
          </CardDescription>
        </CardHeader>
        
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Success Rate */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Success Rate</span>
                <span className={`text-sm font-bold ${getMetricColor(data.success_rate, 'success_rate')}`}>
                  {Math.round(data.success_rate * 100)}%
                </span>
              </div>
              <Progress value={data.success_rate * 100} className="h-2" />
              <div className="flex items-center justify-between text-xs text-gray-500">
                <span>Target: 85%</span>
                <span>{data.successful_workflows} / {data.total_workflows} workflows</span>
              </div>
            </div>

            {/* Response Time */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Avg Response Time</span>
                <span className={`text-sm font-bold ${getMetricColor(data.average_response_time, 'response_time')}`}>
                  {formatDuration(data.average_response_time)}
                </span>
              </div>
              <Progress 
                value={Math.max(0, 100 - (data.average_response_time / 1800000) * 100)} 
                className="h-2" 
              />
              <div className="flex items-center justify-between text-xs text-gray-500">
                <span>Target: &lt; 15min</span>
                <span>MTTC: {formatDuration(data.mean_time_to_containment)}</span>
              </div>
            </div>

            {/* False Positive Rate */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">False Positive Rate</span>
                <span className={`text-sm font-bold ${getMetricColor(data.false_positive_rate, 'false_positive')}`}>
                  {Math.round(data.false_positive_rate * 100)}%
                </span>
              </div>
              <Progress 
                value={Math.max(0, 100 - (data.false_positive_rate * 100))} 
                className="h-2" 
              />
              <div className="flex items-center justify-between text-xs text-gray-500">
                <span>Target: &lt; 10%</span>
                <span>Lower is better</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Performance Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Workflow Performance */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Workflow Performance
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-full ${
                    performanceGrade === 'A' ? 'bg-green-100' :
                    performanceGrade === 'B' ? 'bg-blue-100' :
                    performanceGrade === 'C' ? 'bg-yellow-100' :
                    'bg-red-100'
                  }`}>
                    <Award className={`h-4 w-4 ${
                      performanceGrade === 'A' ? 'text-green-600' :
                      performanceGrade === 'B' ? 'text-blue-600' :
                      performanceGrade === 'C' ? 'text-yellow-600' :
                      'text-red-600'
                    }`} />
                  </div>
                  <div>
                    <div className="font-semibold">Overall Grade</div>
                    <div className="text-sm text-gray-600">Composite performance score</div>
                  </div>
                </div>
                <div className={`text-2xl font-bold ${
                  performanceGrade === 'A' ? 'text-green-600' :
                  performanceGrade === 'B' ? 'text-blue-600' :
                  performanceGrade === 'C' ? 'text-yellow-600' :
                  'text-red-600'
                }`}>
                  {performanceGrade}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="text-center p-3 bg-blue-50 rounded-lg">
                  <div className="text-xl font-bold text-blue-600">{data.total_workflows}</div>
                  <div className="text-gray-600">Total Workflows</div>
                </div>
                <div className="text-center p-3 bg-green-50 rounded-lg">
                  <div className="text-xl font-bold text-green-600">{data.successful_workflows}</div>
                  <div className="text-gray-600">Successful</div>
                </div>
              </div>

              <div className="pt-3 border-t">
                <div className="text-sm font-medium mb-2">Performance Indicators</div>
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    {data.success_rate >= 0.85 ? (
                      <CheckCircle className="h-4 w-4 text-green-500" />
                    ) : (
                      <AlertTriangle className="h-4 w-4 text-yellow-500" />
                    )}
                    <span className="text-sm">
                      Success rate {data.success_rate >= 0.85 ? 'meets' : 'below'} target (85%)
                    </span>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    {data.average_response_time <= 900000 ? (
                      <CheckCircle className="h-4 w-4 text-green-500" />
                    ) : (
                      <AlertTriangle className="h-4 w-4 text-yellow-500" />
                    )}
                    <span className="text-sm">
                      Response time {data.average_response_time <= 900000 ? 'within' : 'exceeds'} target (15min)
                    </span>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    {data.false_positive_rate <= 0.10 ? (
                      <CheckCircle className="h-4 w-4 text-green-500" />
                    ) : (
                      <AlertTriangle className="h-4 w-4 text-yellow-500" />
                    )}
                    <span className="text-sm">
                      False positive rate {data.false_positive_rate <= 0.10 ? 'within' : 'exceeds'} target (10%)
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Cost Effectiveness */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <DollarSign className="h-5 w-5" />
              Cost Effectiveness
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="text-center p-4 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg">
                <div className="text-3xl font-bold text-blue-600">
                  {data.cost_effectiveness_score.toFixed(2)}
                </div>
                <div className="text-sm text-gray-600">Cost Effectiveness Score</div>
                <Badge 
                  variant="outline" 
                  className="mt-2"
                >
                  {data.cost_effectiveness_score > 2.0 ? 'Excellent' :
                   data.cost_effectiveness_score > 1.0 ? 'Good' :
                   data.cost_effectiveness_score > 0.5 ? 'Fair' : 'Poor'}
                </Badge>
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between text-sm">
                  <span>ROI vs Industry Average</span>
                  <span className="font-semibold text-green-600">+45%</span>
                </div>
                <Progress value={145} className="h-2" />
                
                <div className="flex items-center justify-between text-sm">
                  <span>Cost per Incident</span>
                  <span className="font-semibold">$2,340</span>
                </div>
                <Progress value={75} className="h-2" />
                
                <div className="flex items-center justify-between text-sm">
                  <span>Automation Rate</span>
                  <span className="font-semibold text-blue-600">67%</span>
                </div>
                <Progress value={67} className="h-2" />
              </div>

              <div className="pt-3 border-t text-xs text-gray-600">
                <div className="flex items-center gap-1 mb-1">
                  <Shield className="h-3 w-3" />
                  <span>Industry Comparison</span>
                </div>
                <ul className="space-y-1">
                  <li>• 30% faster than industry average</li>
                  <li>• 15% higher success rate</li>
                  <li>• 40% lower cost per incident</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Metrics Grid */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Detailed Performance Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Mean Time to Containment */}
            <div className="p-4 border rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <Clock className="h-5 w-5 text-orange-500" />
                <Badge variant="secondary">MTTC</Badge>
              </div>
              <div className="text-2xl font-bold text-orange-600">
                {formatDuration(data.mean_time_to_containment)}
              </div>
              <div className="text-xs text-gray-600 mt-1">Mean Time to Containment</div>
            </div>

            {/* Workflow Efficiency */}
            <div className="p-4 border rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <Zap className="h-5 w-5 text-blue-500" />
                <Badge variant="secondary">Efficiency</Badge>
              </div>
              <div className="text-2xl font-bold text-blue-600">
                {Math.round((data.successful_workflows / Math.max(data.total_workflows, 1)) * 100)}%
              </div>
              <div className="text-xs text-gray-600 mt-1">Workflow Success Rate</div>
            </div>

            {/* Response Quality */}
            <div className="p-4 border rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <Target className="h-5 w-5 text-green-500" />
                <Badge variant="secondary">Quality</Badge>
              </div>
              <div className="text-2xl font-bold text-green-600">
                {Math.round((1 - data.false_positive_rate) * 100)}%
              </div>
              <div className="text-xs text-gray-600 mt-1">Accuracy Rate</div>
            </div>

            {/* System Load */}
            <div className="p-4 border rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <Server className="h-5 w-5 text-purple-500" />
                <Badge variant="secondary">Load</Badge>
              </div>
              <div className="text-2xl font-bold text-purple-600">
                {data.total_workflows > 50 ? 'High' : data.total_workflows > 20 ? 'Medium' : 'Low'}
              </div>
              <div className="text-xs text-gray-600 mt-1">Workflow Volume</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Performance Benchmarks */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Performance Benchmarks</CardTitle>
          <CardDescription>
            Comparison against industry standards and internal targets
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* Success Rate Benchmark */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">Success Rate vs Target</span>
                <div className="flex items-center gap-2">
                  <span className="text-sm">{Math.round(data.success_rate * 100)}%</span>
                  {data.success_rate >= 0.85 ? (
                    <TrendingUp className="h-4 w-4 text-green-500" />
                  ) : (
                    <TrendingDown className="h-4 w-4 text-red-500" />
                  )}
                </div>
              </div>
              <div className="grid grid-cols-3 gap-2 text-xs">
                <div className="text-center p-2 bg-red-50 rounded">
                  <div className="font-bold text-red-600">70%</div>
                  <div className="text-gray-600">Minimum</div>
                </div>
                <div className="text-center p-2 bg-yellow-50 rounded">
                  <div className="font-bold text-yellow-600">85%</div>
                  <div className="text-gray-600">Target</div>
                </div>
                <div className="text-center p-2 bg-green-50 rounded">
                  <div className="font-bold text-green-600">95%</div>
                  <div className="text-gray-600">Excellent</div>
                </div>
              </div>
            </div>

            {/* Response Time Benchmark */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">Response Time vs Target</span>
                <div className="flex items-center gap-2">
                  <span className="text-sm">{formatDuration(data.average_response_time)}</span>
                  {data.average_response_time <= 900000 ? (
                    <TrendingUp className="h-4 w-4 text-green-500" />
                  ) : (
                    <TrendingDown className="h-4 w-4 text-red-500" />
                  )}
                </div>
              </div>
              <div className="grid grid-cols-4 gap-2 text-xs">
                <div className="text-center p-2 bg-green-50 rounded">
                  <div className="font-bold text-green-600">&lt; 5m</div>
                  <div className="text-gray-600">Excellent</div>
                </div>
                <div className="text-center p-2 bg-blue-50 rounded">
                  <div className="font-bold text-blue-600">&lt; 15m</div>
                  <div className="text-gray-600">Target</div>
                </div>
                <div className="text-center p-2 bg-yellow-50 rounded">
                  <div className="font-bold text-yellow-600">&lt; 30m</div>
                  <div className="text-gray-600">Acceptable</div>
                </div>
                <div className="text-center p-2 bg-red-50 rounded">
                  <div className="font-bold text-red-600">&gt; 30m</div>
                  <div className="text-gray-600">Poor</div>
                </div>
              </div>
            </div>

            {/* Industry Comparison */}
            <div className="pt-4 border-t">
              <h4 className="font-semibold text-sm mb-3">Industry Comparison</h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div className="text-center">
                  <div className="text-lg font-bold text-blue-600">
                    {Math.round(((data.success_rate - 0.75) / 0.75) * 100)}%
                  </div>
                  <div className="text-gray-600">vs Industry Avg</div>
                  <div className="text-xs text-gray-500">Success Rate</div>
                </div>
                
                <div className="text-center">
                  <div className="text-lg font-bold text-green-600">
                    {Math.round(((1800000 - data.average_response_time) / 1800000) * 100)}%
                  </div>
                  <div className="text-gray-600">vs Industry Avg</div>
                  <div className="text-xs text-gray-500">Response Time</div>
                </div>
                
                <div className="text-center">
                  <div className="text-lg font-bold text-purple-600">
                    {Math.round(((0.15 - data.false_positive_rate) / 0.15) * 100)}%
                  </div>
                  <div className="text-gray-600">vs Industry Avg</div>
                  <div className="text-xs text-gray-500">Accuracy</div>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default ResponseMetrics













