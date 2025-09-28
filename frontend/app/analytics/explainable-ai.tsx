'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Button } from '@/components/ui/button'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadialBarChart, RadialBar } from 'recharts'
import { Brain, Lightbulb, TrendingUp, TrendingDown, AlertTriangle, CheckCircle, XCircle, HelpCircle, Zap, Target, ArrowRight, ArrowLeft } from 'lucide-react'

interface FeatureAttribution {
  name: string
  value: number
  importance: number
  description: string
}

interface Counterfactual {
  changes: Record<string, [number, number]>
  summary: string
  feasibility: number
}

interface IncidentExplanation {
  incident_id: number
  explanation_id: string
  prediction: number
  confidence: number
  summary: string
  technical_details: string
  top_features: FeatureAttribution[]
  counterfactuals: Counterfactual[]
}

interface ABTestResult {
  test_id: string
  winner: 'a_wins' | 'b_wins' | 'tie' | 'insufficient_data'
  confidence: number
  p_value: number
  effect_size: number
  samples_a: number
  samples_b: number
  metric_a: number
  metric_b: number
  statistical_significance: boolean
  practical_significance: boolean
  recommendation: string
  detailed_metrics: {
    metric_name: string
    mean_a: number
    mean_b: number
    std_a: number
    std_b: number
    t_statistic: number
  }
}

const getImportanceColor = (importance: number): string => {
  if (importance > 0.7) return 'text-red-600 bg-red-100'
  if (importance > 0.4) return 'text-orange-600 bg-orange-100'
  if (importance > 0.2) return 'text-yellow-600 bg-yellow-100'
  return 'text-green-600 bg-green-100'
}

const getConfidenceColor = (confidence: number): string => {
  if (confidence > 0.8) return 'text-green-600'
  if (confidence > 0.6) return 'text-yellow-600'
  return 'text-red-600'
}

export default function ExplainableAIDashboard() {
  const [selectedIncidentId, setSelectedIncidentId] = useState<number | null>(null)
  const [explanation, setExplanation] = useState<IncidentExplanation | null>(null)
  const [abTestResult, setABTestResult] = useState<ABTestResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [abTestId, setABTestId] = useState('')
  const [newTestData, setNewTestData] = useState({
    model_a_id: '',
    model_a_version: '',
    model_b_id: '',
    model_b_version: '',
    test_name: '',
    description: ''
  })

  const explainIncident = async () => {
    if (!selectedIncidentId) return

    try {
      setLoading(true)
      setError(null)

      const response = await fetch(`/api/ml/explain/${selectedIncidentId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_context: {
            analyst_request: 'Dashboard explanation request'
          }
        })
      })

      const result = await response.json()

      if (result.success) {
        setExplanation(result.explanation)
      } else {
        setError('Failed to generate explanation')
      }
    } catch (err) {
      console.error('Failed to explain incident:', err)
      setError('Failed to explain incident')
    } finally {
      setLoading(false)
    }
  }

  const createABTest = async () => {
    try {
      setError(null)
      
      const response = await fetch('/api/ml/ab-test/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newTestData)
      })

      const result = await response.json()

      if (result.success) {
        alert(`A/B test created successfully: ${result.test_id}`)
        setNewTestData({
          model_a_id: '',
          model_a_version: '',
          model_b_id: '',
          model_b_version: '',
          test_name: '',
          description: ''
        })
      } else {
        setError('Failed to create A/B test')
      }
    } catch (err) {
      console.error('Failed to create A/B test:', err)
      setError('Failed to create A/B test')
    }
  }

  const getABTestResults = async () => {
    if (!abTestId) return

    try {
      setError(null)

      const response = await fetch(`/api/ml/ab-test/${abTestId}/results`)
      const result = await response.json()

      if (result.success) {
        setABTestResult(result)
      } else {
        setError(result.detail || 'Failed to get A/B test results')
      }
    } catch (err) {
      console.error('Failed to get A/B test results:', err)
      setError('Failed to get A/B test results')
    }
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Explainable AI Dashboard</h1>
          <p className="text-muted-foreground">
            Understanding AI decisions with SHAP, LIME, and OpenAI integration
          </p>
        </div>
      </div>

      {error && (
        <Alert>
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <Tabs defaultValue="explain" className="space-y-4">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="explain">Incident Explanation</TabsTrigger>
          <TabsTrigger value="abtests">A/B Testing</TabsTrigger>
          <TabsTrigger value="insights">Model Insights</TabsTrigger>
        </TabsList>

        <TabsContent value="explain" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5" />
                Explain Incident Prediction
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-4 items-end">
                <div className="flex-1">
                  <Label htmlFor="incidentId">Incident ID</Label>
                  <Input
                    id="incidentId"
                    type="number"
                    value={selectedIncidentId || ''}
                    onChange={(e) => setSelectedIncidentId(parseInt(e.target.value) || null)}
                    placeholder="Enter incident ID to explain"
                  />
                </div>
                <Button 
                  onClick={explainIncident} 
                  disabled={!selectedIncidentId || loading}
                  className="bg-blue-600 hover:bg-blue-700"
                >
                  <Lightbulb className="h-4 w-4 mr-2" />
                  {loading ? 'Generating...' : 'Explain'}
                </Button>
              </div>

              {explanation && (
                <div className="mt-6 space-y-6">
                  {/* Prediction Summary */}
                  <Card className="bg-slate-50">
                    <CardContent className="p-6">
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className="text-center">
                          <div className="text-3xl font-bold text-blue-600">
                            {explanation.prediction === 1 ? 'THREAT' : 'NORMAL'}
                          </div>
                          <div className="text-sm text-muted-foreground">Prediction</div>
                        </div>
                        <div className="text-center">
                          <div className={`text-3xl font-bold ${getConfidenceColor(explanation.confidence)}`}>
                            {(explanation.confidence * 100).toFixed(1)}%
                          </div>
                          <div className="text-sm text-muted-foreground">Confidence</div>
                        </div>
                        <div className="text-center">
                          <div className="text-lg font-medium">#{explanation.incident_id}</div>
                          <div className="text-sm text-muted-foreground">Incident ID</div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* AI-Generated Summary */}
                    <Card>
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                          <Brain className="h-5 w-5" />
                          AI Analysis Summary
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-4">
                          <div className="p-4 bg-blue-50 rounded-lg">
                            <div className="font-medium text-blue-900 mb-2">Executive Summary</div>
                            <div className="text-sm text-blue-800">{explanation.summary}</div>
                          </div>
                          <div className="p-4 bg-slate-50 rounded-lg">
                            <div className="font-medium text-slate-900 mb-2">Technical Details</div>
                            <div className="text-sm text-slate-700 whitespace-pre-wrap">
                              {explanation.technical_details}
                            </div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    {/* Feature Importance */}
                    <Card>
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                          <BarChart className="h-5 w-5" />
                          Feature Importance
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <ResponsiveContainer width="100%" height={300}>
                          <BarChart data={explanation.top_features} layout="horizontal">
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis type="number" />
                            <YAxis dataKey="name" type="category" width={100} />
                            <Tooltip 
                              formatter={(value: number) => [value.toFixed(3), 'Importance']}
                              labelFormatter={(label) => `Feature: ${label}`}
                            />
                            <Bar 
                              dataKey="importance" 
                              fill="#0088FE"
                              radius={[0, 4, 4, 0]}
                            />
                          </BarChart>
                        </ResponsiveContainer>
                      </CardContent>
                    </Card>
                  </div>

                  {/* Detailed Feature Analysis */}
                  <Card>
                    <CardHeader>
                      <CardTitle>Feature Impact Analysis</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3">
                        {explanation.top_features.map((feature, index) => (
                          <div key={index} className="flex items-center justify-between p-4 border rounded-lg">
                            <div className="flex-1">
                              <div className="flex items-center gap-3">
                                <div className="font-medium">{feature.name}</div>
                                <Badge className={getImportanceColor(Math.abs(feature.importance))}>
                                  {Math.abs(feature.importance) > 0.7 ? 'High' : 
                                   Math.abs(feature.importance) > 0.4 ? 'Medium' : 
                                   Math.abs(feature.importance) > 0.2 ? 'Low' : 'Minimal'}
                                </Badge>
                              </div>
                              <div className="text-sm text-muted-foreground mt-1">
                                {feature.description}
                              </div>
                              <div className="text-xs text-slate-500 mt-1">
                                Value: {typeof feature.value === 'number' ? feature.value.toFixed(3) : feature.value}
                              </div>
                            </div>
                            <div className="text-right">
                              <div className={`text-lg font-bold ${feature.importance >= 0 ? 'text-red-600' : 'text-green-600'}`}>
                                {feature.importance >= 0 ? <TrendingUp className="h-5 w-5" /> : <TrendingDown className="h-5 w-5" />}
                              </div>
                              <div className="text-sm font-medium">
                                {Math.abs(feature.importance).toFixed(3)}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>

                  {/* Counterfactual Explanations */}
                  {explanation.counterfactuals.length > 0 && (
                    <Card>
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                          <Target className="h-5 w-5" />
                          &quot;What If&quot; Scenarios
                        </CardTitle>
                        <div className="text-sm text-muted-foreground">
                          How to change the prediction outcome
                        </div>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-4">
                          {explanation.counterfactuals.map((cf, index) => (
                            <div key={index} className="p-4 border rounded-lg bg-gradient-to-r from-blue-50 to-purple-50">
                              <div className="flex items-center justify-between mb-3">
                                <div className="font-medium">Scenario {index + 1}</div>
                                <Badge variant="outline">
                                  {(cf.feasibility * 100).toFixed(0)}% feasible
                                </Badge>
                              </div>
                              <div className="text-sm text-slate-700 mb-3">
                                {cf.summary}
                              </div>
                              <div className="space-y-2">
                                {Object.entries(cf.changes).slice(0, 3).map(([feature, [original, modified]]) => (
                                  <div key={feature} className="flex items-center gap-3 text-sm">
                                    <div className="font-medium w-24 truncate">{feature}</div>
                                    <div className="flex items-center gap-2">
                                      <span className="px-2 py-1 bg-red-100 text-red-800 rounded text-xs">
                                        {original.toFixed(3)}
                                      </span>
                                      <ArrowRight className="h-3 w-3" />
                                      <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">
                                        {modified.toFixed(3)}
                                      </span>
                                    </div>
                                  </div>
                                ))}
                              </div>
                              <Progress value={cf.feasibility * 100} className="mt-3" />
                            </div>
                          ))}
                        </div>
                      </CardContent>
                    </Card>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="abtests" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Create A/B Test */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="h-5 w-5" />
                  Create A/B Test
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="modelAId">Model A ID</Label>
                    <Input
                      id="modelAId"
                      value={newTestData.model_a_id}
                      onChange={(e) => setNewTestData({...newTestData, model_a_id: e.target.value})}
                      placeholder="model_a"
                    />
                  </div>
                  <div>
                    <Label htmlFor="modelAVersion">Model A Version</Label>
                    <Input
                      id="modelAVersion"
                      value={newTestData.model_a_version}
                      onChange={(e) => setNewTestData({...newTestData, model_a_version: e.target.value})}
                      placeholder="v1.0"
                    />
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="modelBId">Model B ID</Label>
                    <Input
                      id="modelBId"
                      value={newTestData.model_b_id}
                      onChange={(e) => setNewTestData({...newTestData, model_b_id: e.target.value})}
                      placeholder="model_b"
                    />
                  </div>
                  <div>
                    <Label htmlFor="modelBVersion">Model B Version</Label>
                    <Input
                      id="modelBVersion"
                      value={newTestData.model_b_version}
                      onChange={(e) => setNewTestData({...newTestData, model_b_version: e.target.value})}
                      placeholder="v2.0"
                    />
                  </div>
                </div>
                <div>
                  <Label htmlFor="testName">Test Name</Label>
                  <Input
                    id="testName"
                    value={newTestData.test_name}
                    onChange={(e) => setNewTestData({...newTestData, test_name: e.target.value})}
                    placeholder="Model comparison test"
                  />
                </div>
                <div>
                  <Label htmlFor="description">Description</Label>
                  <Textarea
                    id="description"
                    value={newTestData.description}
                    onChange={(e) => setNewTestData({...newTestData, description: e.target.value})}
                    placeholder="Optional test description"
                    rows={3}
                  />
                </div>
                <Button 
                  onClick={createABTest}
                  className="w-full bg-green-600 hover:bg-green-700"
                  disabled={!newTestData.model_a_id || !newTestData.model_b_id || !newTestData.test_name}
                >
                  <Target className="h-4 w-4 mr-2" />
                  Create A/B Test
                </Button>
              </CardContent>
            </Card>

            {/* Get A/B Test Results */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart className="h-5 w-5" />
                  A/B Test Results
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex gap-2">
                  <Input
                    value={abTestId}
                    onChange={(e) => setABTestId(e.target.value)}
                    placeholder="Enter test ID"
                    className="flex-1"
                  />
                  <Button 
                    onClick={getABTestResults}
                    disabled={!abTestId}
                  >
                    Get Results
                  </Button>
                </div>

                {abTestResult && (
                  <div className="space-y-4">
                    {/* Winner Declaration */}
                    <div className="p-4 border rounded-lg bg-gradient-to-r from-blue-50 to-green-50">
                      <div className="text-center">
                        <div className="text-2xl font-bold mb-2">
                          {abTestResult.winner === 'a_wins' ? '🏆 Model A Wins!' :
                           abTestResult.winner === 'b_wins' ? '🏆 Model B Wins!' :
                           abTestResult.winner === 'tie' ? '🤝 Tie Result' :
                           '⏳ Insufficient Data'}
                        </div>
                        <div className="text-sm text-muted-foreground mb-2">
                          Confidence: {(abTestResult.confidence * 100).toFixed(1)}%
                        </div>
                        <Badge 
                          variant={abTestResult.statistical_significance ? "default" : "secondary"}
                          className="mr-2"
                        >
                          {abTestResult.statistical_significance ? "Statistically Significant" : "Not Significant"}
                        </Badge>
                        <Badge 
                          variant={abTestResult.practical_significance ? "default" : "secondary"}
                        >
                          {abTestResult.practical_significance ? "Practically Significant" : "Not Significant"}
                        </Badge>
                      </div>
                    </div>

                    {/* Detailed Metrics */}
                    <div className="grid grid-cols-2 gap-4">
                      <div className="p-4 border rounded-lg">
                        <div className="text-center">
                          <div className="text-2xl font-bold text-blue-600">
                            {(abTestResult.metric_a * 100).toFixed(1)}%
                          </div>
                          <div className="text-sm text-muted-foreground">Model A Performance</div>
                          <div className="text-xs text-slate-500">
                            {abTestResult.samples_a} samples
                          </div>
                        </div>
                      </div>
                      <div className="p-4 border rounded-lg">
                        <div className="text-center">
                          <div className="text-2xl font-bold text-green-600">
                            {(abTestResult.metric_b * 100).toFixed(1)}%
                          </div>
                          <div className="text-sm text-muted-foreground">Model B Performance</div>
                          <div className="text-xs text-slate-500">
                            {abTestResult.samples_b} samples
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Statistical Details */}
                    <div className="p-4 border rounded-lg bg-slate-50">
                      <div className="grid grid-cols-3 gap-4 text-sm">
                        <div>
                          <div className="font-medium">P-Value</div>
                          <div>{abTestResult.p_value.toFixed(4)}</div>
                        </div>
                        <div>
                          <div className="font-medium">Effect Size</div>
                          <div>{abTestResult.effect_size.toFixed(3)}</div>
                        </div>
                        <div>
                          <div className="font-medium">T-Statistic</div>
                          <div>{abTestResult.detailed_metrics.t_statistic.toFixed(3)}</div>
                        </div>
                      </div>
                    </div>

                    {/* Recommendation */}
                    <Alert>
                      <Lightbulb className="h-4 w-4" />
                      <AlertDescription>
                        <div className="font-medium mb-1">Recommendation</div>
                        {abTestResult.recommendation}
                      </AlertDescription>
                    </Alert>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="insights" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Model Interpretability Insights</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="text-center p-6 border rounded-lg">
                  <Brain className="h-12 w-12 mx-auto mb-4 text-blue-600" />
                  <h3 className="text-lg font-semibold mb-2">SHAP Explanations</h3>
                  <p className="text-sm text-muted-foreground">
                    Feature importance analysis using Shapley values for precise attribution
                  </p>
                </div>
                <div className="text-center p-6 border rounded-lg">
                  <Target className="h-12 w-12 mx-auto mb-4 text-green-600" />
                  <h3 className="text-lg font-semibold mb-2">LIME Analysis</h3>
                  <p className="text-sm text-muted-foreground">
                    Local explanations with perturbation-based feature analysis
                  </p>
                </div>
                <div className="text-center p-6 border rounded-lg">
                  <Zap className="h-12 w-12 mx-auto mb-4 text-purple-600" />
                  <h3 className="text-lg font-semibold mb-2">AI-Enhanced Insights</h3>
                  <p className="text-sm text-muted-foreground">
                    OpenAI-powered natural language explanations and recommendations
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
