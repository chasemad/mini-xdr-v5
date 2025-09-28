'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Textarea } from '@/components/ui/textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { ScrollArea } from '@/components/ui/scroll-area'
import { 
  Search, 
  Bot, 
  Brain, 
  Target, 
  AlertTriangle,
  CheckCircle,
  Loader2,
  TrendingUp,
  FileSearch,
  MessageSquare,
  Lightbulb,
  HelpCircle,
  Copy,
  ExternalLink
} from 'lucide-react'

interface NLPFinding {
  type: string
  description?: string
  incident_id?: number
  src_ip?: string
  incident_count?: number
  confidence_score?: number
  relevance_score?: number
  threat_type?: string
  unique_ips?: number
  time_span_hours?: number
  iocs?: {
    ip_addresses?: string[]
    domains?: string[]
    urls?: string[]
  }
  recommendations?: string[]
  [key: string]: unknown
}

interface NLPResponse {
  success: boolean
  query: string
  query_understanding: string
  structured_query: Record<string, unknown>
  findings: NLPFinding[]
  recommendations: string[]
  confidence_score: number
  reasoning: string
  follow_up_questions: string[]
  processing_stats?: Record<string, unknown>
  analysis_metadata?: Record<string, unknown>
}

interface SemanticSearchIncident {
  incident: {
    id: number
    src_ip: string
    reason: string
    status: string
    created_at: string
    risk_score?: number
    escalation_level?: string
  }
  similarity_score: number
}

interface SemanticSearchResponse {
  success?: boolean
  total_found?: number
  avg_similarity?: number
  similar_incidents?: SemanticSearchIncident[]
  [key: string]: unknown
}

const EXAMPLE_QUERIES = [
  "Show me all brute force attacks from the last 24 hours",
  "Find incidents similar to IP 192.168.1.100", 
  "What patterns do you see in recent malware incidents?",
  "Timeline of high-severity incidents this week",
  "Extract IOCs from all contained incidents",
  "Who might be behind the recent attack campaign?",
  "Recommend actions for open incidents"
]

const ANALYSIS_TYPES = [
  { value: "pattern_recognition", label: "Pattern Recognition" },
  { value: "timeline_analysis", label: "Timeline Analysis" },
  { value: "attribution", label: "Attribution Analysis" },
  { value: "ioc_extraction", label: "IOC Extraction" },
  { value: "recommendation", label: "Recommendations" }
]

export default function NLPInterface() {
  const [query, setQuery] = useState('')
  const [analysisType, setAnalysisType] = useState('pattern_recognition')
  const [timeRange, setTimeRange] = useState(24)
  const [isLoading, setIsLoading] = useState(false)
  const [results, setResults] = useState<NLPResponse | null>(null)
  const [semanticResults, setSemanticResults] = useState<SemanticSearchResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState('query')
  const [nlpStatus, setNlpStatus] = useState<{
    nlp_system?: {
      langchain_available?: boolean
      [key: string]: unknown
    }
    capabilities?: {
      semantic_search?: boolean
      ai_powered_insights?: boolean
      [key: string]: unknown
    }
    [key: string]: unknown
  } | null>(null)

  // Load NLP status on mount
  useEffect(() => {
    loadNLPStatus()
  }, [])

  const loadNLPStatus = async () => {
    try {
      const response = await fetch('/api/nlp/status')
      const data = await response.json()
      if (data.success) {
        setNlpStatus(data)
      }
    } catch (err) {
      console.error('Failed to load NLP status:', err)
    }
  }

  const handleNaturalLanguageQuery = async () => {
    if (!query.trim()) return

    setIsLoading(true)
    setError(null)
    setResults(null)

    try {
      const response = await fetch('/api/nlp/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query.trim(),
          include_context: true,
          max_results: 10,
          semantic_search: true
        }),
      })

      const data = await response.json()
      
      if (data.success) {
        setResults(data)
        setActiveTab('results')
      } else {
        setError(data.detail || 'Query failed')
      }
    } catch (err) {
      setError(`Failed to process query: ${err}`)
    } finally {
      setIsLoading(false)
    }
  }

  const handleThreatAnalysis = async () => {
    if (!query.trim()) return

    setIsLoading(true)
    setError(null)
    setResults(null)

    try {
      const response = await fetch('/api/nlp/threat-analysis', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query.trim(),
          analysis_type: analysisType,
          time_range_hours: timeRange
        }),
      })

      const data = await response.json()
      
      if (data.success) {
        setResults(data)
        setActiveTab('results')
      } else {
        setError(data.detail || 'Analysis failed')
      }
    } catch (err) {
      setError(`Failed to process analysis: ${err}`)
    } finally {
      setIsLoading(false)
    }
  }

  const handleSemanticSearch = async () => {
    if (!query.trim()) return

    setIsLoading(true)
    setError(null)
    setSemanticResults(null)

    try {
      const response = await fetch('/api/nlp/semantic-search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query.trim(),
          similarity_threshold: 0.7,
          max_results: 10
        }),
      })

      const data = await response.json()
      
      if (data.success) {
        setSemanticResults(data)
        setActiveTab('semantic')
      } else {
        setError(data.detail || 'Search failed')
      }
    } catch (err) {
      setError(`Failed to perform semantic search: ${err}`)
    } finally {
      setIsLoading(false)
    }
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
  }

  const formatConfidence = (score?: number) => {
    if (typeof score !== 'number') return 'N/A'
    const percentage = Math.round(score * 100)
    const color = percentage >= 80 ? 'bg-green-500' : percentage >= 60 ? 'bg-yellow-500' : 'bg-red-500'
    return <Badge className={`${color} text-white`}>{percentage}%</Badge>
  }

  const formatFindingType = (type: string) => {
    return type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
  }

  const renderFindings = (findings: NLPFinding[]) => {
    if (!findings || findings.length === 0) {
      return <div className="text-gray-500 text-center py-8">No findings available</div>
    }

    return (
      <div className="space-y-4">
        {findings.map((finding, index) => (
          <Card key={index} className="border-l-4 border-l-blue-500">
            <CardHeader className="pb-2">
              <div className="flex justify-between items-start">
                <CardTitle className="text-sm font-medium">
                  {formatFindingType(finding.type)}
                </CardTitle>
                <div className="flex gap-2">
                  {finding.confidence_score && formatConfidence(finding.confidence_score)}
                  {finding.relevance_score && (
                    <Badge variant="outline">
                      {Math.round(finding.relevance_score * 100)}% relevant
                    </Badge>
                  )}
                </div>
              </div>
            </CardHeader>
            <CardContent className="pt-0">
              {finding.description && (
                <p className="text-sm text-gray-600 mb-2">{finding.description}</p>
              )}
              
              <div className="grid grid-cols-2 gap-4 text-xs">
                {finding.incident_id && (
                  <div>
                    <span className="font-medium">Incident ID:</span> 
                    <Button 
                      variant="link" 
                      className="p-0 h-auto font-normal text-blue-600"
                      onClick={() => window.open(`/incidents/${finding.incident_id}`, '_blank')}
                    >
                      #{finding.incident_id} <ExternalLink className="ml-1 w-3 h-3" />
                    </Button>
                  </div>
                )}
                {finding.src_ip && (
                  <div><span className="font-medium">Source IP:</span> {finding.src_ip}</div>
                )}
                {finding.incident_count && (
                  <div><span className="font-medium">Incidents:</span> {finding.incident_count}</div>
                )}
                {finding.unique_ips && (
                  <div><span className="font-medium">Unique IPs:</span> {finding.unique_ips}</div>
                )}
                {finding.threat_type && (
                  <div><span className="font-medium">Threat Type:</span> {finding.threat_type}</div>
                )}
                {finding.time_span_hours && (
                  <div><span className="font-medium">Time Span:</span> {finding.time_span_hours}h</div>
                )}
              </div>


              {finding.recommendations && (
                <div className="mt-3">
                  <span className="font-medium text-xs">Recommendations:</span>
                  <ul className="text-xs text-gray-600 mt-1 space-y-1">
                    {finding.recommendations.slice(0, 3).map((rec: string, i: number) => (
                      <li key={i} className="flex items-start gap-2">
                        <Lightbulb className="w-3 h-3 mt-0.5 text-yellow-500" />
                        {rec}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </CardContent>
          </Card>
        ))}
      </div>
    )
  }

  const renderSemanticResults = (results: SemanticSearchIncident[]) => {
    if (!results || results.length === 0) {
      return <div className="text-gray-500 text-center py-8">No similar incidents found</div>
    }

    return (
      <div className="space-y-4">
        {results.map((result, index) => (
          <Card key={index} className="border-l-4 border-l-green-500">
            <CardHeader className="pb-2">
              <div className="flex justify-between items-start">
                <CardTitle className="text-sm font-medium">
                  Incident #{result.incident.id}
                </CardTitle>
                <Badge className="bg-green-500 text-white">
                  {Math.round(result.similarity_score * 100)}% similar
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="pt-0">
              <div className="grid grid-cols-2 gap-4 text-xs">
                <div><span className="font-medium">Source IP:</span> {result.incident.src_ip}</div>
                <div><span className="font-medium">Status:</span> {result.incident.status}</div>
                <div><span className="font-medium">Reason:</span> {result.incident.reason}</div>
                <div><span className="font-medium">Created:</span> {new Date(result.incident.created_at).toLocaleString()}</div>
                {result.incident.risk_score && (
                  <div><span className="font-medium">Risk Score:</span> {Math.round(result.incident.risk_score * 100)}%</div>
                )}
                <div><span className="font-medium">Severity:</span> {result.incident.escalation_level}</div>
              </div>
              
              <div className="mt-3 flex gap-2">
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => window.open(`/incidents/${result.incident.id}`, '_blank')}
                >
                  View Details <ExternalLink className="ml-1 w-3 h-3" />
                </Button>
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => copyToClipboard(result.incident.src_ip)}
                >
                  Copy IP <Copy className="ml-1 w-3 h-3" />
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    )
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Brain className="w-8 h-8 text-blue-600" />
        <div>
          <h1 className="text-3xl font-bold">Natural Language Threat Analysis</h1>
          <p className="text-gray-600">Query your security data using natural language</p>
        </div>
      </div>

      {/* Status Card */}
      {nlpStatus && (
        <Card className="border-blue-200 bg-blue-50">
          <CardContent className="p-4">
            <div className="flex justify-between items-center">
              <div className="flex items-center gap-3">
                <Bot className="w-5 h-5 text-blue-600" />
                <div>
                  <div className="font-medium">NLP System Status</div>
                  <div className="text-sm text-gray-600">
                    {nlpStatus.nlp_system?.langchain_available ? 'Advanced AI' : 'Pattern-based'} analysis available
                  </div>
                </div>
              </div>
              <div className="flex gap-2">
                {nlpStatus.capabilities?.semantic_search && (
                  <Badge className="bg-green-500 text-white">Semantic Search</Badge>
                )}
                {nlpStatus.capabilities?.ai_powered_insights && (
                  <Badge className="bg-purple-500 text-white">AI Insights</Badge>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Query Interface */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <MessageSquare className="w-5 h-5" />
            Query Interface
          </CardTitle>
          <CardDescription>
            Ask questions about your security incidents in natural language
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Query Input */}
          <div className="flex gap-2">
            <div className="flex-1">
              <Textarea
                placeholder="Enter your security question... (e.g., 'Show me all brute force attacks from China in the last 24 hours')"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="min-h-[100px]"
              />
            </div>
          </div>

          {/* Example queries */}
          <div className="space-y-2">
            <span className="text-sm font-medium text-gray-600">Example Queries:</span>
            <div className="flex flex-wrap gap-2">
              {EXAMPLE_QUERIES.map((example, index) => (
                <Button
                  key={index}
                  variant="outline"
                  size="sm"
                  onClick={() => setQuery(example)}
                  className="text-xs"
                >
                  {example}
                </Button>
              ))}
            </div>
          </div>

          {/* Advanced options for threat analysis */}
          <Tabs defaultValue="basic" className="w-full">
            <TabsList>
              <TabsTrigger value="basic">Basic Query</TabsTrigger>
              <TabsTrigger value="advanced">Advanced Analysis</TabsTrigger>
            </TabsList>
            
            <TabsContent value="advanced" className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium">Analysis Type</label>
                  <Select value={analysisType} onValueChange={setAnalysisType}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {ANALYSIS_TYPES.map((type) => (
                        <SelectItem key={type.value} value={type.value}>
                          {type.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                
                <div>
                  <label className="text-sm font-medium">Time Range (hours)</label>
                  <Input
                    type="number"
                    value={timeRange}
                    onChange={(e) => setTimeRange(parseInt(e.target.value) || 24)}
                    min={1}
                    max={720}
                  />
                </div>
              </div>
            </TabsContent>
          </Tabs>

          {/* Action buttons */}
          <div className="flex gap-2">
            <Button 
              onClick={handleNaturalLanguageQuery} 
              disabled={isLoading || !query.trim()}
              className="flex-1"
            >
              {isLoading ? <Loader2 className="mr-2 w-4 h-4 animate-spin" /> : <Search className="mr-2 w-4 h-4" />}
              Natural Language Query
            </Button>
            
            <Button 
              onClick={handleThreatAnalysis} 
              disabled={isLoading || !query.trim()}
              variant="outline"
              className="flex-1"
            >
              {isLoading ? <Loader2 className="mr-2 w-4 h-4 animate-spin" /> : <Target className="mr-2 w-4 h-4" />}
              Threat Analysis
            </Button>
            
            <Button 
              onClick={handleSemanticSearch} 
              disabled={isLoading || !query.trim()}
              variant="outline"
              className="flex-1"
            >
              {isLoading ? <Loader2 className="mr-2 w-4 h-4 animate-spin" /> : <FileSearch className="mr-2 w-4 h-4" />}
              Semantic Search
            </Button>
          </div>

          {/* Error display */}
          {error && (
            <Alert className="border-red-200 bg-red-50">
              <AlertTriangle className="w-4 h-4 text-red-600" />
              <AlertDescription className="text-red-800">{error}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Results Display */}
      {(results || semanticResults) && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5" />
              Analysis Results
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Tabs value={activeTab} onValueChange={setActiveTab}>
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="results">Query Results</TabsTrigger>
                <TabsTrigger value="semantic">Semantic Search</TabsTrigger>
                <TabsTrigger value="insights">AI Insights</TabsTrigger>
              </TabsList>
              
              <TabsContent value="results" className="space-y-4">
                {results && (
                  <>
                    {/* Query understanding */}
                    <Alert className="border-blue-200 bg-blue-50">
                      <Bot className="w-4 h-4 text-blue-600" />
                      <AlertDescription>
                        <div className="font-medium">Query Understanding:</div>
                        {results.query_understanding}
                        <div className="mt-2 flex items-center gap-2">
                          <span className="text-sm">Confidence:</span>
                          {formatConfidence(results.confidence_score)}
                        </div>
                      </AlertDescription>
                    </Alert>

                    {/* Findings */}
                    <div>
                      <h3 className="font-medium mb-3">Findings ({results.findings?.length || 0})</h3>
                      <ScrollArea className="h-[400px]">
                        {renderFindings(results.findings)}
                      </ScrollArea>
                    </div>

                    {/* Recommendations */}
                    {results.recommendations && results.recommendations.length > 0 && (
                      <div>
                        <h3 className="font-medium mb-3 flex items-center gap-2">
                          <Lightbulb className="w-4 h-4 text-yellow-500" />
                          Recommendations
                        </h3>
                        <div className="space-y-2">
                          {results.recommendations.map((rec, index) => (
                            <div key={index} className="flex items-start gap-2 text-sm bg-gray-50 p-3 rounded">
                              <CheckCircle className="w-4 h-4 text-green-500 mt-0.5" />
                              {rec}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Follow-up questions */}
                    {results.follow_up_questions && results.follow_up_questions.length > 0 && (
                      <div>
                        <h3 className="font-medium mb-3 flex items-center gap-2">
                          <HelpCircle className="w-4 h-4 text-blue-500" />
                          Suggested Follow-up Questions
                        </h3>
                        <div className="flex flex-wrap gap-2">
                          {results.follow_up_questions.map((question, index) => (
                            <Button
                              key={index}
                              variant="outline"
                              size="sm"
                              onClick={() => setQuery(question)}
                              className="text-xs"
                            >
                              {question}
                            </Button>
                          ))}
                        </div>
                      </div>
                    )}
                  </>
                )}
              </TabsContent>
              
              <TabsContent value="semantic" className="space-y-4">
                {semanticResults && (
                  <>
                    {/* Search info */}
                    <Alert className="border-green-200 bg-green-50">
                      <FileSearch className="w-4 h-4 text-green-600" />
                      <AlertDescription>
                        <div className="font-medium">Semantic Search Results</div>
                        Found {semanticResults.total_found} similar incidents with {Math.round((semanticResults.avg_similarity || 0) * 100)}% average similarity
                      </AlertDescription>
                    </Alert>

                    {/* Results */}
                    <ScrollArea className="h-[500px]">
                      {renderSemanticResults(semanticResults.similar_incidents || [])}
                    </ScrollArea>
                  </>
                )}
              </TabsContent>
              
              <TabsContent value="insights" className="space-y-4">
                {results && (
                  <div className="space-y-4">
                    {/* Reasoning */}
                    <div>
                      <h3 className="font-medium mb-2">AI Reasoning</h3>
                      <div className="bg-gray-50 p-4 rounded text-sm">
                        {results.reasoning}
                      </div>
                    </div>

                    {/* Processing stats */}
                    {results.processing_stats && (
                      <div>
                        <h3 className="font-medium mb-2">Processing Statistics</h3>
                        <div className="grid grid-cols-3 gap-4 text-sm">
                          <div className="text-center p-3 bg-blue-50 rounded">
                            <div className="font-medium text-lg">{(results.processing_stats as { incidents_analyzed?: number })?.incidents_analyzed || 0}</div>
                            <div className="text-gray-600">Incidents Analyzed</div>
                          </div>
                          <div className="text-center p-3 bg-green-50 rounded">
                            <div className="font-medium text-lg">{(results.processing_stats as { events_analyzed?: number })?.events_analyzed || 0}</div>
                            <div className="text-gray-600">Events Processed</div>
                          </div>
                          <div className="text-center p-3 bg-purple-50 rounded">
                            <div className="font-medium text-lg">{Math.round(results.confidence_score * 100)}%</div>
                            <div className="text-gray-600">Analysis Confidence</div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
