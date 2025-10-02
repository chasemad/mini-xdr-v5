'use client'

/**
 * Natural Language Workflow Creator
 * Allows users to create workflows using natural language descriptions
 */

import React, { useState, useRef, useEffect, memo } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"
import {
  Brain,
  Send,
  Zap,
  Target,
  Shield,
  Network,
  Server,
  Mail,
  Cloud,
  Key,
  Database,
  AlertTriangle,
  CheckCircle,
  Loader2,
  Sparkles,
  MessageSquare
} from 'lucide-react'
import { useAppContext, appActions } from '../contexts/AppContext'
import { getAvailableResponseActions, parseNlpWorkflow, createNlpWorkflow, getIncidentContextForNLP } from '@/app/lib/api'

interface NaturalLanguageInputProps {
  selectedIncidentId?: number | null
  selectedIncident?: any | null  // Pass the full incident object to avoid state lookup
  onWorkflowCreated?: (workflowId: string) => void
  onSwitchToDesigner?: () => void
}

interface IncidentContext {
  incident_id: number
  src_ip: string
  threat_summary: string
  status: string
  risk_score: number
  escalation_level: string
  threat_category?: string
  attack_patterns: string[]
  total_events: number
  triage_note?: any
  actions_taken?: Array<{
    action: string
    result: string
    created_at: string
  }>
  suggested_actions?: string[]
  context_summary: string
}

interface ParsedWorkflow {
  suggested_playbook: string
  confidence: number
  actions: Array<{
    action_type: string
    category: string
    description: string
    parameters: Record<string, any>
    estimated_duration: number
  }>
  explanation: string
  risk_assessment: {
    level: string
    concerns: string[]
    mitigations: string[]
  }
  priority: string
  approval_required: boolean
  target_ip?: string | null
  conditions: Record<string, any>
}

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

const RISK_PROFILES: Record<string, { level: string; concerns: string[]; mitigations: string[] }> = {
  critical: {
    level: 'critical',
    concerns: [
      'Automated containment may disrupt production systems',
      'High-impact changes require analyst supervision'
    ],
    mitigations: [
      'Review each step with on-call lead',
      'Ensure rollback plans are staged before execution'
    ]
  },
  high: {
    level: 'high',
    concerns: [
      'Actions can affect multiple services',
      'Additional validation needed before execution'
    ],
    mitigations: [
      'Validate scope and targets with responders',
      'Monitor workflow progress in real time'
    ]
  },
  medium: {
    level: 'medium',
    concerns: [
      'Moderate automation risk if context is incomplete',
      'Manual approval may still be required'
    ],
    mitigations: [
      'Confirm incident details before running',
      'Schedule follow-up validation after completion'
    ]
  },
  low: {
    level: 'low',
    concerns: [
      'Minimal service impact expected',
      'Suitable for semi-automated execution'
    ],
    mitigations: [
      'Document results for audit trail',
      'Escalate only if anomalies are detected'
    ]
  }
}

const FALLBACK_RISK = RISK_PROFILES.medium
const FALLBACK_ESTIMATED_DURATION = 300

const toTitleCase = (value: string) =>
  value
    .replace(/_/g, ' ')
    .replace(/\b\w/g, char => char.toUpperCase())

const buildSuggestedPlaybook = (priority: string, threatType?: string | null) => {
  const priorityLabel = priority ? toTitleCase(priority) : 'Adaptive'
  if (threatType) {
    return `${toTitleCase(threatType)} Response (${priorityLabel} Priority)`
  }
  return `NLP ${priorityLabel} Workflow`
}

const getRiskProfile = (priority: string) => {
  const key = priority?.toLowerCase()
  return RISK_PROFILES[key as keyof typeof RISK_PROFILES] || FALLBACK_RISK
}

const getPriorityBadgeClass = (priority: string) => {
  switch (priority) {
    case 'critical':
      return 'bg-red-600 text-white hover:bg-red-600'
    case 'high':
      return 'bg-orange-500 text-white hover:bg-orange-500'
    case 'low':
      return 'bg-emerald-600 text-white hover:bg-emerald-600'
    default:
      return 'bg-blue-600 text-white hover:bg-blue-600'
  }
}

const NaturalLanguageInput: React.FC<NaturalLanguageInputProps> = memo(({
  selectedIncidentId,
  selectedIncident,
  onWorkflowCreated,
  onSwitchToDesigner
}) => {
  const { dispatch } = useAppContext()  // Only get dispatch, not state
  const [input, setInput] = useState('')
  const [isProcessing, setIsProcessing] = useState(false)
  const [parsedWorkflow, setParsedWorkflow] = useState<ParsedWorkflow | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isCreating, setIsCreating] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const [actionLibrary, setActionLibrary] = useState<Record<string, any>>({})
  const [incidentContext, setIncidentContext] = useState<IncidentContext | null>(null)
  const [loadingContext, setLoadingContext] = useState(false)

  useEffect(() => {
    let cancelled = false

    const loadActions = async () => {
      try {
        const result = await getAvailableResponseActions()
        if (!cancelled && result?.success && result.actions) {
          setActionLibrary(result.actions)
        }
      } catch (err) {
        console.error('Failed to load response actions for NLP preview:', err)
      }
    }

    loadActions()

    return () => {
      cancelled = true
    }
  }, [])

  // Fetch incident context when incident is selected
  useEffect(() => {
    let cancelled = false

    const loadIncidentContext = async () => {
      if (!selectedIncidentId) {
        setIncidentContext(null)
        return
      }

      setLoadingContext(true)
      try {
        const context = await getIncidentContextForNLP(selectedIncidentId)
        if (!cancelled) {
          setIncidentContext(context)
        }
      } catch (err) {
        console.error('Failed to load incident context:', err)
        if (!cancelled) {
          setIncidentContext(null)
        }
      } finally {
        if (!cancelled) {
          setLoadingContext(false)
        }
      }
    }

    loadIncidentContext()

    return () => {
      cancelled = true
    }
  }, [selectedIncidentId])

  // Debug: Log re-renders to understand why component is updating
  useEffect(() => {
    console.log('[NaturalLanguageInput] Component rendered/re-rendered')
  })

  // Sample prompts for user guidance
  const samplePrompts = [
    "Create a malware response workflow for incident #5 with host isolation and memory dumping",
    "Set up DDoS protection for the affected servers with rate limiting and traffic analysis",
    "Implement credential stuffing defense including password reset and MFA enforcement",
    "Deploy ransomware containment with network segmentation and backup verification",
    "Execute phishing response with email blocking and user awareness training"
  ]

  const buildParsedWorkflowFromResponse = (response: any): ParsedWorkflow => {
    const priority = (response?.priority || 'medium').toString().toLowerCase()
    const actions = (response?.actions || []).map((action: any) => {
      const registryEntry = actionLibrary?.[action.action_type]
      const friendlyName = registryEntry?.name || toTitleCase(action.action_type)
      const description = registryEntry?.description || `Auto-execute ${friendlyName}`
      const estimatedDuration = Number(
        registryEntry?.estimated_duration ??
        action?.timeout_seconds ??
        FALLBACK_ESTIMATED_DURATION
      )

      return {
        action_type: action.action_type,
        category: String(registryEntry?.category || action.category || 'other'),
        description,
        parameters: action.parameters || {},
        estimated_duration: estimatedDuration
      }
    })

    const threatType = response?.conditions?.threat_type

    return {
      suggested_playbook: buildSuggestedPlaybook(priority, threatType),
      confidence: typeof response?.confidence === 'number' ? response.confidence : 0,
      actions,
      explanation: response?.explanation || `Generated ${actions.length} workflow steps from your request.`,
      risk_assessment: getRiskProfile(priority),
      priority,
      approval_required: Boolean(response?.approval_required),
      target_ip: response?.target_ip ?? null,
      conditions: response?.conditions || {}
    }
  }

  const parseNaturalLanguage = async (text: string) => {
    const trimmed = text.trim()
    if (!trimmed) {
      return
    }

    setIsProcessing(true)
    setError(null)
    setParsedWorkflow(null)

    try {
      const result = await parseNlpWorkflow({
        text: trimmed,
        incident_id: selectedIncidentId ?? null,
        auto_execute: false
      })

      if (result?.success && Array.isArray(result.actions) && result.actions.length > 0) {
        setParsedWorkflow(buildParsedWorkflowFromResponse(result))
      } else {
        const fallbackWorkflow = createMockWorkflow(trimmed)
        if (result?.target_ip) {
          fallbackWorkflow.target_ip = result.target_ip
          fallbackWorkflow.conditions = {
            ...fallbackWorkflow.conditions,
            target_ip: result.target_ip,
            target_ips: result.conditions?.target_ips || [result.target_ip]
          }
        }

        const parserFeedback = result?.explanation || result?.message
        fallbackWorkflow.explanation = [
          'Recommended workflow template generated because the NLP parser could not identify actionable steps.',
          parserFeedback ? `Parser feedback:\n${parserFeedback}` : null
        ].filter(Boolean).join('\n\n')

        setParsedWorkflow(fallbackWorkflow)

        if (result && result.success === false) {
          setError(result.message || 'Could not identify any workflow actions. Please refine your request.')
        } else if (!result?.actions?.length) {
          setError('Parser could not identify actionable steps. Showing recommended template.')
        }
      }
    } catch (err) {
      console.error('Natural language parsing failed:', err)
      const fallbackWorkflow = createMockWorkflow(trimmed)
      setParsedWorkflow(fallbackWorkflow)
      setError(err instanceof Error ? err.message : 'Failed to parse workflow request')
    } finally {
      setIsProcessing(false)
    }
  }

  const createMockWorkflow = (text: string): ParsedWorkflow => {
    const lowerText = text.toLowerCase()

    let playbook = 'Custom Response Workflow'
    let actions: any[] = []
    let priority: string = 'medium'
    let threatType: string | undefined

    if (lowerText.includes('malware') || lowerText.includes('virus')) {
      playbook = 'Malware Response Workflow'
      priority = 'high'
      threatType = 'malware response'
      actions = [
        {
          action_type: 'isolate_host_advanced',
          category: 'endpoint',
          description: 'Isolate infected host from network',
          parameters: { isolation_level: 'strict' },
          estimated_duration: 300
        },
        {
          action_type: 'memory_dump_collection',
          category: 'forensics',
          description: 'Collect memory dump for analysis',
          parameters: { dump_type: 'full' },
          estimated_duration: 900
        }
      ]
    } else if (lowerText.includes('ddos') || lowerText.includes('flood')) {
      playbook = 'DDoS Protection Workflow'
      priority = 'high'
      threatType = 'network defense'
      actions = [
        {
          action_type: 'deploy_rate_limiting',
          category: 'network',
          description: 'Deploy rate limiting rules',
          parameters: { max_requests_per_minute: 100 },
          estimated_duration: 180
        },
        {
          action_type: 'block_ip_advanced',
          category: 'network',
          description: 'Block attacking IP ranges',
          parameters: { duration: 3600 },
          estimated_duration: 60
        }
      ]
    } else if (lowerText.includes('credential') || lowerText.includes('password')) {
      playbook = 'Credential Protection Workflow'
      priority = 'high'
      threatType = 'identity defense'
      actions = [
        {
          action_type: 'force_password_reset',
          category: 'identity',
          description: 'Force password reset for affected accounts',
          parameters: { scope: 'targeted_users' },
          estimated_duration: 300
        },
        {
          action_type: 'enforce_mfa',
          category: 'identity',
          description: 'Enforce multi-factor authentication',
          parameters: { grace_period: 24 },
          estimated_duration: 120
        }
      ]
    } else {
      actions = [
        {
          action_type: 'threat_intel_lookup',
          category: 'forensics',
          description: 'Perform threat intelligence lookup',
          parameters: {},
          estimated_duration: 60
        },
        {
          action_type: 'collect_logs',
          category: 'forensics',
          description: 'Collect relevant security logs',
          parameters: { time_range: 3600 },
          estimated_duration: 180
        }
      ]
    }

    return {
      suggested_playbook: threatType ? buildSuggestedPlaybook(priority, threatType) : playbook,
      confidence: 0.85,
      actions,
      explanation: `Based on your description, I've created a ${playbook.toLowerCase()} with ${actions.length} automated response actions.`,
      risk_assessment: getRiskProfile(priority),
      priority,
      approval_required: priority !== 'low',
      target_ip: null,
      conditions: threatType ? { threat_type: threatType } : {}
    }
  }

  const createWorkflow = async () => {
    if (!parsedWorkflow || !selectedIncidentId) return

    setIsCreating(true)
    setError(null)

    try {
      const workflowText = input.trim() || parsedWorkflow.explanation

      const result = await createNlpWorkflow({
        text: workflowText,
        incident_id: selectedIncidentId,
        auto_execute: false
      })

      if (result?.success) {
        dispatch(appActions.addWorkflow({
          id: result.workflow_db_id || 0,
          workflow_id: result.workflow_id || '',
          incident_id: selectedIncidentId,
          playbook_name: parsedWorkflow.suggested_playbook,
          status: 'pending',
          progress_percentage: 0,
          current_step: 0,
          total_steps: result.actions_created || parsedWorkflow.actions.length,
          created_at: new Date().toISOString(),
          approval_required: parsedWorkflow.approval_required,
          auto_executed: false
        }))

        onWorkflowCreated?.(result.workflow_id)

        setInput('')
        setParsedWorkflow(null)
      } else {
        throw new Error(result?.message || 'Failed to create workflow')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create workflow')
    } finally {
      setIsCreating(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Input Section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-blue-600" />
            Natural Language Workflow Creator
          </CardTitle>
          <CardDescription>
            Describe what you want to do in plain English, and I'll create a response workflow for you
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Incident Context Panel */}
          {selectedIncidentId && (
            <div className="border border-blue-200 rounded-lg overflow-hidden bg-gradient-to-br from-blue-50 to-indigo-50">
              {/* Header */}
              <div className="bg-blue-600 text-white p-3 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Target className="h-5 w-5" />
                  <span className="font-semibold">Incident Context #{selectedIncidentId}</span>
                </div>
                {loadingContext && <Loader2 className="h-4 w-4 animate-spin" />}
              </div>

              {/* Context Details */}
              {incidentContext ? (
                <div className="p-4 space-y-3">
                  {/* Summary */}
                  <div className="bg-white rounded-lg p-3 border border-blue-100">
                    <div className="flex items-start gap-2">
                      <AlertTriangle className={`h-5 w-5 flex-shrink-0 mt-0.5 ${
                        incidentContext.risk_score > 0.7 ? 'text-red-500' : 
                        incidentContext.risk_score > 0.4 ? 'text-yellow-500' : 'text-green-500'
                      }`} />
                      <div className="flex-1">
                        <p className="text-sm font-medium text-gray-700">{incidentContext.context_summary}</p>
                      </div>
                    </div>
                  </div>

                  {/* Quick Stats Grid */}
                  <div className="grid grid-cols-3 gap-2">
                    <div className="bg-white rounded-lg p-2 border border-gray-200">
                      <div className="text-xs text-gray-500">Source IP</div>
                      <div className="text-sm font-mono font-semibold text-gray-800">{incidentContext.src_ip}</div>
                    </div>
                    <div className="bg-white rounded-lg p-2 border border-gray-200">
                      <div className="text-xs text-gray-500">Risk Score</div>
                      <div className="text-sm font-semibold">
                        <Badge className={`${
                          incidentContext.risk_score > 0.7 ? 'bg-red-500' : 
                          incidentContext.risk_score > 0.4 ? 'bg-yellow-500' : 'bg-green-500'
                        } text-white`}>
                          {(incidentContext.risk_score * 100).toFixed(0)}%
                        </Badge>
                      </div>
                    </div>
                    <div className="bg-white rounded-lg p-2 border border-gray-200">
                      <div className="text-xs text-gray-500">Events</div>
                      <div className="text-sm font-semibold text-gray-800">{incidentContext.total_events}</div>
                    </div>
                  </div>

                  {/* Attack Patterns */}
                  {incidentContext.attack_patterns && incidentContext.attack_patterns.length > 0 && (
                    <div className="bg-white rounded-lg p-3 border border-gray-200">
                      <div className="text-xs text-gray-500 mb-2">Attack Patterns Detected</div>
                      <div className="flex flex-wrap gap-1">
                        {incidentContext.attack_patterns.map((pattern, idx) => (
                          <Badge key={idx} variant="outline" className="text-xs bg-purple-50 text-purple-700 border-purple-200">
                            {pattern}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* AI Context Notice */}
                  <div className="bg-gradient-to-r from-purple-50 to-blue-50 rounded-lg p-3 border border-purple-200">
                    <div className="flex items-start gap-2">
                      <Sparkles className="h-4 w-4 text-purple-600 mt-0.5" />
                      <div>
                        <div className="text-xs font-semibold text-purple-700 mb-1">AI Context Enabled</div>
                        <div className="text-xs text-gray-600">
                          The workflow generator will use this incident's context to create targeted response actions automatically.
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ) : loadingContext ? (
                <div className="p-4 text-center text-gray-500">
                  <Loader2 className="h-6 w-6 animate-spin mx-auto mb-2" />
                  <div className="text-sm">Loading incident context...</div>
                </div>
              ) : (
                <div className="p-4 text-center text-gray-500">
                  <div className="text-sm">No context available for this incident</div>
                </div>
              )}
            </div>
          )}

          {/* Text Input */}
          <div className="relative">
            <Textarea
              ref={textareaRef}
              placeholder="Describe your response workflow... (e.g., 'Create a malware response workflow with host isolation and forensic analysis')"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              className="min-h-[100px] resize-none pr-12"
              onKeyDown={(e) => {
                if (e.key === 'Enter' && e.ctrlKey && input.trim()) {
                  parseNaturalLanguage(input)
                }
              }}
            />
            <Button
              size="sm"
              className="absolute bottom-2 right-2"
              onClick={() => parseNaturalLanguage(input)}
              disabled={!input.trim() || isProcessing}
            >
              {isProcessing ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Sparkles className="h-4 w-4" />
              )}
              Parse
            </Button>
          </div>

          {/* Sample Prompts */}
          <div>
            <p className="text-sm font-medium mb-2">Try these examples:</p>
            <div className="flex flex-wrap gap-2">
              {samplePrompts.slice(0, 3).map((prompt, index) => (
                <Button
                  key={index}
                  variant="outline"
                  size="sm"
                  onClick={() => setInput(prompt)}
                  className="text-xs h-auto py-1 px-2"
                >
                  {prompt.substring(0, 50)}...
                </Button>
              ))}
            </div>
          </div>

          {/* Error Display */}
          {error && (
            <Alert variant="destructive">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Parsed Workflow Preview */}
      {parsedWorkflow && (
        <Card>
          <CardHeader>
            <CardTitle className="flex flex-wrap items-center gap-2">
              <CheckCircle className="h-5 w-5 text-green-600" />
              <span>Generated Workflow: {parsedWorkflow.suggested_playbook}</span>
              <Badge variant="secondary">
                {Math.round(parsedWorkflow.confidence * 100)}% confidence
              </Badge>
              <Badge className={getPriorityBadgeClass(parsedWorkflow.priority)}>
                {toTitleCase(parsedWorkflow.priority)} priority
              </Badge>
              <Badge variant={parsedWorkflow.approval_required ? 'destructive' : 'outline'}>
                {parsedWorkflow.approval_required ? 'Approval required' : 'Auto executable'}
              </Badge>
              {parsedWorkflow.target_ip && (
                <Badge variant="outline">
                  Target {parsedWorkflow.target_ip}
                </Badge>
              )}
            </CardTitle>
            <CardDescription className="whitespace-pre-line">
              {parsedWorkflow.explanation}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Actions */}
            <div>
              <h4 className="font-medium mb-2">Workflow Steps ({parsedWorkflow.actions.length})</h4>
              <div className="space-y-2">
                {parsedWorkflow.actions.map((action, index) => {
                  const IconComponent = categoryIcons[action.category as keyof typeof categoryIcons] || Target
                  return (
                    <div key={index} className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-gray-500">
                          {index + 1}.
                        </span>
                        <IconComponent className="h-4 w-4 text-blue-600" />
                      </div>
                      <div className="flex-1">
                        <p className="font-medium">{action.description}</p>
                        <p className="text-sm text-gray-600">
                          Category: {action.category} • Duration: ~{Math.round(action.estimated_duration / 60)}m
                        </p>
                      </div>
                      <Badge variant="outline">
                        {action.action_type}
                      </Badge>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* Risk Assessment */}
            <div className="p-3 bg-yellow-50 rounded-lg border border-yellow-200">
              <h4 className="font-medium mb-2 flex items-center gap-2">
                <AlertTriangle className="h-4 w-4 text-yellow-600" />
                Risk Assessment: {parsedWorkflow.risk_assessment.level.toUpperCase()}
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="font-medium mb-1">Concerns:</p>
                  <ul className="list-disc list-inside space-y-1">
                    {parsedWorkflow.risk_assessment.concerns.map((concern, index) => (
                      <li key={index}>{concern}</li>
                    ))}
                  </ul>
                </div>
                <div>
                  <p className="font-medium mb-1">Mitigations:</p>
                  <ul className="list-disc list-inside space-y-1">
                    {parsedWorkflow.risk_assessment.mitigations.map((mitigation, index) => (
                      <li key={index}>{mitigation}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>

            {/* Actions */}
            <div className="flex gap-3 pt-2">
              <Button
                onClick={createWorkflow}
                disabled={isCreating || !selectedIncidentId}
                className="flex items-center gap-2"
              >
                {isCreating ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Zap className="h-4 w-4" />
                )}
                Create Workflow
              </Button>

              <Button
                variant="outline"
                onClick={onSwitchToDesigner}
                className="flex items-center gap-2"
              >
                <MessageSquare className="h-4 w-4" />
                Open in Designer
              </Button>

              <Button
                variant="outline"
                onClick={() => {
                  setParsedWorkflow(null)
                  setInput('')
                }}
              >
                Start Over
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}, (prevProps, nextProps) => {
  // Custom comparison function - only re-render if props actually changed
  // This prevents re-renders when parent state changes for other reasons
  
  const shouldSkipRender = (
    prevProps.selectedIncidentId === nextProps.selectedIncidentId &&
    prevProps.onWorkflowCreated === nextProps.onWorkflowCreated &&
    prevProps.onSwitchToDesigner === nextProps.onSwitchToDesigner &&
    prevProps.selectedIncident === nextProps.selectedIncident
  )
  
  if (!shouldSkipRender) {
    console.log('[NaturalLanguageInput] Re-rendering due to prop changes:', {
      incidentIdChanged: prevProps.selectedIncidentId !== nextProps.selectedIncidentId,
      onWorkflowCreatedChanged: prevProps.onWorkflowCreated !== nextProps.onWorkflowCreated,
      onSwitchToDesignerChanged: prevProps.onSwitchToDesigner !== nextProps.onSwitchToDesigner,
      selectedIncidentChanged: prevProps.selectedIncident !== nextProps.selectedIncident
    })
  } else {
    console.log('[NaturalLanguageInput] Skipping re-render - props unchanged')
  }
  
  return shouldSkipRender
})

NaturalLanguageInput.displayName = 'NaturalLanguageInput'

export default NaturalLanguageInput
