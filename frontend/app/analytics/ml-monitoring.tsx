'use client'

import React, { useState, useEffect, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { ScrollArea } from '@/components/ui/scroll-area'
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
import { RefreshCw, Brain, AlertTriangle, TrendingUp, TrendingDown, Activity, Zap, Target, Settings, PlayCircle } from 'lucide-react'

interface OnlineLearningStatus {
  online_learning_status: {
    buffer_size: number
    last_drift_time: string | null
    detection_sensitivity: number
    window_size: number
  }
  phase_2b_features: {
    online_adaptation_enabled: boolean
    ensemble_optimization_enabled: boolean
    explainable_ai_enabled: boolean
    online_adaptations: number
    ensemble_optimizations: number
    drift_detections: number
    last_online_adaptation: string | null
    last_ensemble_optimization: string | null
    last_performance_check: string | null
  }
  drift_detections: number
}

interface ModelPerformance {
  model_performance: Array<{
    model_id: string
    version: string
    algorithm: string
    status: string
    accuracy: number
    error_rate: number
    latency_ms: number
    data_points: number
    last_evaluation: string | null
  }>
  recent_alerts: number
  alerts_detail: Array<{
    timestamp: string
    model_id: string
    version: string
    status: string
    accuracy: number
    error_rate: number
  }>
  total_production_models: number
}

interface DriftStatus {
  drift_detection: {
    buffer_size: number
    last_drift_time: string | null
    detection_sensitivity: number
    window_size: number
  }
  recent_adaptations: number
  adaptation_metrics: Array<{
    timestamp: string
    accuracy_before: number
    accuracy_after: number
    adaptation_time: number
    samples_processed: number
    drift_magnitude: number
    strategy_used: string
    model_version: string
    success: boolean
  }>
  last_drift_time: string | null
  buffer_size: number
  detection_sensitivity: number
}

interface EnsembleStatus {
  ensemble_models: Array<{
    model_id: string
    version: string
    algorithm: string
    status: string
    created_at: string
    performance_metrics: {
      accuracy?: number
      precision?: number
      recall?: number
      f1_score?: number
    }
  }>
  total_ensembles: number
  production_ensembles: number
}

const statusColors = {
  healthy: 'bg-green-500',
  degraded: 'bg-yellow-500',
  unhealthy: 'bg-red-500',
  production: 'bg-blue-500',
  training: 'bg-gray-500',
  testing: 'bg-purple-500'
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d']

export default function MLMonitoringDashboard() {
  const [onlineLearningStatus, setOnlineLearningStatus] = useState<OnlineLearningStatus | null>(null)
  const [modelPerformance, setModelPerformance] = useState<ModelPerformance | null>(null)
  const [driftStatus, setDriftStatus] = useState<DriftStatus | null>(null)
  const [ensembleStatus, setEnsembleStatus] = useState<EnsembleStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchData = useCallback(async () => {
    try {
      setRefreshing(true)
      setError(null)

      const [
        onlineLearningResponse,
        modelPerformanceResponse,
        driftResponse,
        ensembleResponse
      ] = await Promise.all([
        fetch('/api/ml/online-learning/status').then(r => r.json()),
        fetch('/api/ml/models/performance').then(r => r.json()),
        fetch('/api/ml/drift/status').then(r => r.json()),
        fetch('/api/ml/ensemble/status').then(r => r.json())
      ])

      if (onlineLearningResponse.success) {
        setOnlineLearningStatus(onlineLearningResponse)
      }

      if (modelPerformanceResponse.success) {
        setModelPerformance(modelPerformanceResponse)
      }

      if (driftResponse.success) {
        setDriftStatus(driftResponse)
      }

      if (ensembleResponse.success) {
        setEnsembleStatus(ensembleResponse)
      }

    } catch (err) {
      console.error('Failed to fetch ML monitoring data:', err)
      setError('Failed to load monitoring data')
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }, [])

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 30000) // Refresh every 30 seconds
    return () => clearInterval(interval)
  }, [fetchData])

  const triggerOnlineAdaptation = async () => {
    try {
      const response = await fetch('/api/ml/online-learning/adapt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })
      const result = await response.json()
      
      if (result.success) {
        alert(`Online adaptation completed: ${result.samples_processed} samples processed`)
        fetchData() // Refresh data
      } else {
        alert(`Adaptation failed: ${result.message}`)
      }
    } catch (err) {
      console.error('Failed to trigger adaptation:', err)
      alert('Failed to trigger adaptation')
    }
  }

  const triggerEnsembleOptimization = async () => {
    try {
      const response = await fetch('/api/ml/ensemble/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })
      const result = await response.json()
      
      if (result.success) {
        alert(`Ensemble optimization completed with ${result.training_events} training events`)
        fetchData() // Refresh data
      } else {
        alert('Ensemble optimization failed')
      }
    } catch (err) {
      console.error('Failed to trigger optimization:', err)
      alert('Failed to trigger optimization')
    }
  }

  if (loading) {
    return (
      <div className="container mx-auto p-6">
        <div className="flex items-center justify-center h-64">
          <RefreshCw className="h-8 w-8 animate-spin" />
          <span className="ml-2 text-lg">Loading ML monitoring data...</span>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="container mx-auto p-6">
        <Alert>
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      </div>
    )
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Advanced ML Monitoring</h1>
          <p className="text-muted-foreground">
            Real-time monitoring of Phase 2B machine learning capabilities
          </p>
        </div>
        <div className="flex gap-2">
          <Button 
            onClick={fetchData} 
            disabled={refreshing}
            variant="outline"
            size="sm"
          >
            <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button 
            onClick={triggerOnlineAdaptation}
            size="sm"
            className="bg-blue-600 hover:bg-blue-700"
          >
            <Zap className="h-4 w-4 mr-2" />
            Trigger Adaptation
          </Button>
          <Button 
            onClick={triggerEnsembleOptimization}
            size="sm"
            className="bg-green-600 hover:bg-green-700"
          >
            <Target className="h-4 w-4 mr-2" />
            Optimize Ensemble
          </Button>
        </div>
      </div>

      {/* System Status Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Online Learning</CardTitle>
            <Brain className="h-4 w-4 text-blue-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-blue-600">
              {onlineLearningStatus?.phase_2b_features.online_adaptation_enabled ? 'Active' : 'Disabled'}
            </div>
            <p className="text-xs text-muted-foreground">
              {onlineLearningStatus?.phase_2b_features.online_adaptations || 0} adaptations completed
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Concept Drift</CardTitle>
            <TrendingUp className="h-4 w-4 text-orange-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-orange-600">
              {driftStatus?.recent_adaptations || 0}
            </div>
            <p className="text-xs text-muted-foreground">
              Recent drift adaptations
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Model Performance</CardTitle>
            <Activity className="h-4 w-4 text-green-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">
              {modelPerformance?.total_production_models || 0}
            </div>
            <p className="text-xs text-muted-foreground">
              Production models active
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Ensemble Models</CardTitle>
            <Target className="h-4 w-4 text-purple-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-purple-600">
              {ensembleStatus?.total_ensembles || 0}
            </div>
            <p className="text-xs text-muted-foreground">
              {ensembleStatus?.production_ensembles || 0} in production
            </p>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="performance" className="space-y-4">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="performance">Model Performance</TabsTrigger>
          <TabsTrigger value="drift">Concept Drift</TabsTrigger>
          <TabsTrigger value="online">Online Learning</TabsTrigger>
          <TabsTrigger value="ensemble">Ensembles</TabsTrigger>
          <TabsTrigger value="alerts">Alerts</TabsTrigger>
        </TabsList>

        <TabsContent value="performance" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Model Performance Chart */}
            <Card>
              <CardHeader>
                <CardTitle>Model Accuracy Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={modelPerformance?.model_performance || []}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="model_id" />
                    <YAxis domain={[0, 1]} />
                    <Tooltip formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, 'Accuracy']} />
                    <Bar dataKey="accuracy" fill="#0088FE" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Model Status Overview */}
            <Card>
              <CardHeader>
                <CardTitle>Model Status Overview</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {modelPerformance?.model_performance.map((model, index) => (
                    <div key={`${model.model_id}-${model.version}`} className="flex items-center justify-between p-3 border rounded-lg">
                      <div>
                        <div className="font-medium">{model.model_id}</div>
                        <div className="text-sm text-muted-foreground">
                          v{model.version} • {model.algorithm}
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge className={statusColors[model.status as keyof typeof statusColors] || 'bg-gray-500'}>
                          {model.status}
                        </Badge>
                        <div className="text-right">
                          <div className="text-sm font-medium">
                            {(model.accuracy * 100).toFixed(1)}%
                          </div>
                          <div className="text-xs text-muted-foreground">
                            {model.latency_ms.toFixed(1)}ms
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="drift" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Drift Detection Status */}
            <Card>
              <CardHeader>
                <CardTitle>Concept Drift Detection</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-between items-center">
                  <span>Detection Sensitivity</span>
                  <span className="font-medium">
                    {((driftStatus?.detection_sensitivity || 0) * 100).toFixed(1)}%
                  </span>
                </div>
                <Progress value={(driftStatus?.detection_sensitivity || 0) * 100} />
                
                <div className="flex justify-between items-center">
                  <span>Buffer Size</span>
                  <span className="font-medium">{driftStatus?.buffer_size || 0} samples</span>
                </div>
                <Progress value={Math.min((driftStatus?.buffer_size || 0) / 1000 * 100, 100)} />

                <div className="pt-4 border-t">
                  <div className="text-sm font-medium mb-2">Last Drift Detection</div>
                  <div className="text-sm text-muted-foreground">
                    {driftStatus?.last_drift_time 
                      ? new Date(driftStatus.last_drift_time).toLocaleString()
                      : 'No drift detected recently'
                    }
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Adaptation Timeline */}
            <Card>
              <CardHeader>
                <CardTitle>Recent Adaptations</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={driftStatus?.adaptation_metrics.slice(-10) || []}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="timestamp" 
                      tickFormatter={(value) => new Date(value).toLocaleDateString()}
                    />
                    <YAxis domain={[0, 1]} />
                    <Tooltip 
                      labelFormatter={(value) => new Date(value as string).toLocaleString()}
                      formatter={(value: number, name: string) => [`${(value * 100).toFixed(1)}%`, name]}
                    />
                    <Line type="monotone" dataKey="accuracy_before" stroke="#ff7c7c" strokeDasharray="5 5" name="Before" />
                    <Line type="monotone" dataKey="accuracy_after" stroke="#8884d8" name="After" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="online" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Online Learning Status */}
            <Card>
              <CardHeader>
                <CardTitle>Online Learning Configuration</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center p-4 border rounded-lg">
                    <div className="text-2xl font-bold text-blue-600">
                      {onlineLearningStatus?.phase_2b_features.online_adaptations || 0}
                    </div>
                    <div className="text-sm text-muted-foreground">Total Adaptations</div>
                  </div>
                  <div className="text-center p-4 border rounded-lg">
                    <div className="text-2xl font-bold text-green-600">
                      {onlineLearningStatus?.drift_detections || 0}
                    </div>
                    <div className="text-sm text-muted-foreground">Drift Events</div>
                  </div>
                </div>

                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span>Online Adaptation</span>
                    <Badge variant={onlineLearningStatus?.phase_2b_features.online_adaptation_enabled ? "default" : "secondary"}>
                      {onlineLearningStatus?.phase_2b_features.online_adaptation_enabled ? "Enabled" : "Disabled"}
                    </Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Ensemble Optimization</span>
                    <Badge variant={onlineLearningStatus?.phase_2b_features.ensemble_optimization_enabled ? "default" : "secondary"}>
                      {onlineLearningStatus?.phase_2b_features.ensemble_optimization_enabled ? "Enabled" : "Disabled"}
                    </Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Explainable AI</span>
                    <Badge variant={onlineLearningStatus?.phase_2b_features.explainable_ai_enabled ? "default" : "secondary"}>
                      {onlineLearningStatus?.phase_2b_features.explainable_ai_enabled ? "Enabled" : "Disabled"}
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Recent Activity */}
            <Card>
              <CardHeader>
                <CardTitle>Recent Activity Timeline</CardTitle>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-64">
                  <div className="space-y-3">
                    {[
                      {
                        time: onlineLearningStatus?.phase_2b_features.last_online_adaptation,
                        event: 'Online Adaptation',
                        icon: <Brain className="h-4 w-4" />,
                        color: 'text-blue-600'
                      },
                      {
                        time: onlineLearningStatus?.phase_2b_features.last_ensemble_optimization,
                        event: 'Ensemble Optimization',
                        icon: <Target className="h-4 w-4" />,
                        color: 'text-green-600'
                      },
                      {
                        time: onlineLearningStatus?.phase_2b_features.last_performance_check,
                        event: 'Performance Check',
                        icon: <Activity className="h-4 w-4" />,
                        color: 'text-orange-600'
                      }
                    ].filter(item => item.time).map((item, index) => (
                      <div key={index} className="flex items-center gap-3 p-2 border-l-2 border-gray-200">
                        <div className={item.color}>{item.icon}</div>
                        <div className="flex-1">
                          <div className="text-sm font-medium">{item.event}</div>
                          <div className="text-xs text-muted-foreground">
                            {item.time ? new Date(item.time).toLocaleString() : 'Never'}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="ensemble" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Ensemble Distribution */}
            <Card>
              <CardHeader>
                <CardTitle>Ensemble Algorithm Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={ensembleStatus?.ensemble_models.reduce((acc, model) => {
                        const existing = acc.find(item => item.name === model.algorithm)
                        if (existing) {
                          existing.value += 1
                        } else {
                          acc.push({ name: model.algorithm, value: 1 })
                        }
                        return acc
                      }, [] as Array<{name: string, value: number}>) || []}
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      dataKey="value"
                    >
                      {ensembleStatus?.ensemble_models.map((_, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Ensemble Models List */}
            <Card>
              <CardHeader>
                <CardTitle>Ensemble Models</CardTitle>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-64">
                  <div className="space-y-3">
                    {ensembleStatus?.ensemble_models.map((model, index) => (
                      <div key={`${model.model_id}-${model.version}`} className="flex items-center justify-between p-3 border rounded-lg">
                        <div>
                          <div className="font-medium">{model.model_id}</div>
                          <div className="text-sm text-muted-foreground">
                            {model.algorithm} • v{model.version}
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge className={statusColors[model.status as keyof typeof statusColors] || 'bg-gray-500'}>
                            {model.status}
                          </Badge>
                          {model.performance_metrics.accuracy && (
                            <div className="text-sm font-medium">
                              {(model.performance_metrics.accuracy * 100).toFixed(1)}%
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="alerts" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Recent Performance Alerts</CardTitle>
              <div className="text-sm text-muted-foreground">
                {modelPerformance?.recent_alerts || 0} alerts in the last 24 hours
              </div>
            </CardHeader>
            <CardContent>
              {modelPerformance?.alerts_detail && modelPerformance.alerts_detail.length > 0 ? (
                <div className="space-y-3">
                  {modelPerformance.alerts_detail.map((alert, index) => (
                    <Alert key={index}>
                      <AlertTriangle className="h-4 w-4" />
                      <AlertDescription>
                        <div className="flex justify-between items-start">
                          <div>
                            <div className="font-medium">
                              {alert.model_id} v{alert.version} - {alert.status}
                            </div>
                            <div className="text-sm">
                              Accuracy: {(alert.accuracy * 100).toFixed(1)}% | 
                              Error Rate: {(alert.error_rate * 100).toFixed(1)}%
                            </div>
                          </div>
                          <div className="text-sm text-muted-foreground">
                            {new Date(alert.timestamp).toLocaleString()}
                          </div>
                        </div>
                      </AlertDescription>
                    </Alert>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  No recent alerts
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
