/**
 * AI Incident Analysis Component
 * Provides OpenAI/XAI powered analysis and explanations for incidents
 */

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { ScrollArea } from "@/components/ui/scroll-area"
import { 
  Brain, 
  Lightbulb, 
  AlertTriangle, 
  CheckCircle, 
  Target,
  Shield,
  Zap,
  RefreshCw,
  MessageSquare
} from 'lucide-react'

interface AIIncidentAnalysisProps {
  incident: any
  onRecommendationAction?: (action: string) => void
}

interface AIAnalysis {
  summary: string
  severity: string
  recommendation: string
  rationale: string[]
  confidence_score?: number
  threat_attribution?: string
  response_priority?: string
  estimated_impact?: string
  next_steps?: string[]
}

export default function AIIncidentAnalysis({ incident, onRecommendationAction }: AIIncidentAnalysisProps) {
  const [analysis, setAnalysis] = useState<AIAnalysis | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [aiProvider, setAiProvider] = useState<'openai' | 'xai'>('openai')

  useEffect(() => {
    if (incident?.id) {
      generateAIAnalysis()
    }
  }, [incident?.id])

  const generateAIAnalysis = async () => {
    if (!incident?.id) return

    try {
      setLoading(true)
      setError(null)

      const response = await fetch(`http://localhost:8000/api/incidents/${incident.id}/ai-analysis`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': process.env.NEXT_PUBLIC_API_KEY || 'demo-minixdr-api-key'
        },
        body: JSON.stringify({
          provider: aiProvider,
          analysis_type: 'comprehensive',
          include_recommendations: true
        })
      })

      if (!response.ok) {
        throw new Error(`AI analysis failed: ${response.statusText}`)
      }

      const data = await response.json()
      if (data.success) {
        setAnalysis(data.analysis)
      } else {
        throw new Error(data.error || 'AI analysis failed')
      }

    } catch (err) {
      console.error('AI analysis failed:', err)
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  const getSeverityColor = (severity: string) => {
    switch (severity?.toLowerCase()) {
      case 'critical': return 'bg-red-100 text-red-800'
      case 'high': return 'bg-orange-100 text-orange-800'
      case 'medium': return 'bg-yellow-100 text-yellow-800'
      case 'low': return 'bg-green-100 text-green-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  const getRecommendationIcon = (recommendation: string) => {
    switch (recommendation?.toLowerCase()) {
      case 'contain_now': return <Shield className="w-4 h-4 text-red-500" />
      case 'monitor': return <Target className="w-4 h-4 text-yellow-500" />
      case 'investigate': return <MessageSquare className="w-4 h-4 text-blue-500" />
      default: return <CheckCircle className="w-4 h-4 text-gray-500" />
    }
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="w-5 h-5 text-purple-400" />
          AI Security Analysis
          <div className="ml-auto flex items-center gap-2">
            <select 
              value={aiProvider} 
              onChange={(e) => setAiProvider(e.target.value as 'openai' | 'xai')}
              className="text-sm bg-gray-800 border border-gray-600 rounded px-2 py-1"
            >
              <option value="openai">GPT-4</option>
              <option value="xai">Grok</option>
            </select>
            <Button
              variant="outline"
              size="sm"
              onClick={generateAIAnalysis}
              disabled={loading}
            >
              <RefreshCw className={`w-4 h-4 mr-1 ${loading ? 'animate-spin' : ''}`} />
              {loading ? 'Analyzing...' : 'Refresh'}
            </Button>
          </div>
        </CardTitle>
        <CardDescription>
          AI-powered incident analysis and response recommendations
        </CardDescription>
      </CardHeader>
      <CardContent>
        {error && (
          <Alert className="mb-4">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {loading ? (
          <div className="flex items-center justify-center py-8">
            <Brain className="w-8 h-8 text-purple-400 animate-pulse mr-3" />
            <div className="text-center">
              <p className="text-gray-400">AI analyzing incident...</p>
              <p className="text-sm text-gray-500">Using {aiProvider === 'openai' ? 'GPT-4' : 'Grok'} for comprehensive analysis</p>
            </div>
          </div>
        ) : analysis ? (
          <div className="space-y-6">
            {/* AI Summary */}
            <div className="bg-gradient-to-r from-purple-500/10 to-blue-500/10 border border-purple-500/30 rounded-lg p-4">
              <h4 className="font-semibold text-white mb-2 flex items-center gap-2">
                <Lightbulb className="w-4 h-4 text-yellow-400" />
                AI Security Summary
              </h4>
              <p className="text-gray-300 leading-relaxed">{analysis.summary}</p>
            </div>

            {/* Severity and Recommendation */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
                <h5 className="font-medium text-gray-300 mb-2">AI Severity Assessment</h5>
                <Badge className={getSeverityColor(analysis.severity)} size="lg">
                  {analysis.severity?.toUpperCase()}
                </Badge>
                {analysis.confidence_score && (
                  <p className="text-sm text-gray-400 mt-2">
                    Confidence: {Math.round(analysis.confidence_score * 100)}%
                  </p>
                )}
              </div>

              <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
                <h5 className="font-medium text-gray-300 mb-2">AI Recommendation</h5>
                <div className="flex items-center gap-2 mb-2">
                  {getRecommendationIcon(analysis.recommendation)}
                  <span className="font-medium text-white">
                    {analysis.recommendation?.replace('_', ' ').toUpperCase()}
                  </span>
                </div>
                {onRecommendationAction && analysis.recommendation === 'contain_now' && (
                  <Button 
                    size="sm" 
                    onClick={() => onRecommendationAction('auto_contain')}
                    className="bg-red-600 hover:bg-red-700"
                  >
                    <Zap className="w-3 h-3 mr-1" />
                    Execute AI Recommendation
                  </Button>
                )}
              </div>
            </div>

            {/* AI Rationale */}
            <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
              <h5 className="font-medium text-gray-300 mb-3">AI Analysis Rationale</h5>
              <div className="space-y-2">
                {analysis.rationale?.map((reason, idx) => (
                  <div key={idx} className="flex items-start gap-2">
                    <div className="w-5 h-5 rounded-full bg-purple-600 text-white flex items-center justify-center text-xs mt-0.5">
                      {idx + 1}
                    </div>
                    <p className="text-gray-300 text-sm leading-relaxed">{reason}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Additional AI Insights */}
            {(analysis.threat_attribution || analysis.estimated_impact || analysis.next_steps) && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                {analysis.threat_attribution && (
                  <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
                    <h5 className="font-medium text-gray-300 mb-2">Threat Attribution</h5>
                    <p className="text-sm text-gray-400">{analysis.threat_attribution}</p>
                  </div>
                )}

                {analysis.estimated_impact && (
                  <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
                    <h5 className="font-medium text-gray-300 mb-2">Estimated Impact</h5>
                    <p className="text-sm text-gray-400">{analysis.estimated_impact}</p>
                  </div>
                )}

                {analysis.next_steps && (
                  <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
                    <h5 className="font-medium text-gray-300 mb-2">Next Steps</h5>
                    <div className="space-y-1">
                      {analysis.next_steps.map((step, idx) => (
                        <p key={idx} className="text-sm text-gray-400">• {step}</p>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* AI Model Info */}
            <div className="flex items-center justify-between text-xs text-gray-500 pt-2 border-t border-gray-700">
              <span>Analysis powered by {aiProvider === 'openai' ? 'OpenAI GPT-4' : 'xAI Grok'}</span>
              <span>Generated {new Date().toLocaleTimeString()}</span>
            </div>
          </div>
        ) : (
          <div className="text-center py-8 text-gray-400">
            <Brain className="w-12 h-12 mx-auto mb-4 text-purple-400" />
            <p>Click "Refresh" to generate AI analysis</p>
            <p className="text-sm">AI will analyze the incident and provide recommendations</p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
