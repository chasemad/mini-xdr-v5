'use client'

/**
 * Enterprise Workflow Orchestration Platform
 *
 * Comprehensive workflow management with AI-powered natural language interface,
 * visual drag-and-drop designer, pre-built playbook templates, and real-time execution monitoring.
 */

import React, { useState, useEffect, useMemo, useCallback } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Alert, AlertDescription } from "@/components/ui/alert"
import Link from 'next/link'
import {
  Workflow,
  Play,
  Save,
  Brain,
  Target,
  Activity,
  MessageSquare,
  Wifi,
  WifiOff,
  AlertTriangle,
  Sparkles,
  Zap,
  TrendingUp,
  CheckCircle2,
  Clock,
  ArrowRight,
  RefreshCw,
  Shield,
  Globe,
  Search,
  BarChart3,
  ChevronRight,
  ChevronDown,
  Bot,
  Settings,
  Power,
  PowerOff,
  Edit,
  Trash2,
  Plus,
  Copy
} from 'lucide-react'
import WorkflowDesigner from '../components/WorkflowDesigner'
import PlaybookTemplates from '../components/PlaybookTemplates'
import WorkflowExecutor from '../components/WorkflowExecutor'
import NaturalLanguageInput from '../components/NaturalLanguageInput'
import AutomationsPanel from '../components/AutomationsPanel'
import { useAppContext, appActions } from '../contexts/AppContext'
import { createDataService } from '../services/DataService'
import { listWorkflowTriggers, disableWorkflowTrigger, enableWorkflowTrigger } from '../lib/api'
import { DashboardLayout } from '@/components/DashboardLayout'

interface Incident {
  id: number
  src_ip: string
  reason: string
  status: string
  created_at: string
  risk_score?: number
}

interface ResponseWorkflow {
  id: number
  workflow_id: string
  incident_id: number
  playbook_name: string
  status: string
  progress_percentage: number
  total_steps: number
  created_at: string
}

interface WorkflowTrigger {
  id: number
  name: string
  description: string | null
  category: string
  enabled: boolean
  auto_execute: boolean
  priority: string
  conditions: Record<string, any>
  playbook_name: string
  workflow_steps: Array<{
    action_type: string
    parameters: Record<string, any>
    timeout_seconds?: number
    continue_on_failure?: boolean
  }>
  trigger_count: number
  success_count: number
  failure_count: number
  success_rate: number
  avg_response_time_ms: number
  last_triggered_at: string | null
  created_at: string
  updated_at: string
}

const WorkflowsPage: React.FC = () => {
  const { state, dispatch } = useAppContext()
  const [activeTab, setActiveTab] = useState('natural')
  const [dataService, setDataService] = useState<any>(null)
  const [triggers, setTriggers] = useState<WorkflowTrigger[]>([])
  const [triggersLoading, setTriggersLoading] = useState(false)

  // Calculate workflow stats from global state
  const workflowStats = {
    total: state.workflows.length,
    active: state.workflows.filter(w => ['pending', 'running'].includes(w.status)).length,
    completed: state.workflows.filter(w => w.status === 'completed').length,
    failed: state.workflows.filter(w => w.status === 'failed').length,
    templates: 8
  }

  // Initialize data service
  useEffect(() => {
    const service = createDataService({
      onIncidentsUpdate: (incidents) => {
        dispatch(appActions.setIncidents(incidents))

        // Auto-select first open incident if none selected
        if (!state.selectedIncident) {
          const openIncidents = incidents.filter((i: any) => i.status === 'open')
          if (openIncidents.length > 0) {
            dispatch(appActions.setSelectedIncident(openIncidents[0].id))
          }
        }
      },
      onWorkflowsUpdate: (workflows) => {
        dispatch(appActions.setWorkflows(workflows))
      },
      onError: (error, type) => {
        dispatch(appActions.setError(type, error))
      },
      onLoading: (loading, type) => {
        dispatch(appActions.setLoading(type, loading))
      }
    })

    setDataService(service)

    // Load initial data
    service.loadInitialData()

    // Start periodic refresh as fallback (uses default 45s interval with intelligent change detection)
    if (!state.websocket.connected) {
      service.startPeriodicRefresh()
    }

    return () => {
      service.cleanup()
    }
  }, [])

  // Fetch workflow triggers
  useEffect(() => {
    const fetchTriggers = async () => {
      setTriggersLoading(true)
      try {
        const response = await listWorkflowTriggers({}) // Get all triggers, not just honeypot
        // API returns array directly
        if (Array.isArray(response)) {
          setTriggers(response)
        }
      } catch (error) {
        console.error('Failed to fetch workflow triggers:', error)
      } finally {
        setTriggersLoading(false)
      }
    }

    // Fetch triggers on mount or when triggers tab is activated
    if (activeTab === 'triggers') {
      fetchTriggers()
    }
  }, [activeTab])

  // Handle trigger toggle
  const handleToggleTrigger = async (triggerId: number, currentlyEnabled: boolean) => {
    try {
      if (currentlyEnabled) {
        await disableWorkflowTrigger(triggerId)
      } else {
        await enableWorkflowTrigger(triggerId)
      }
      // Refresh triggers list
      const response = await listWorkflowTriggers({}) // Get all triggers
      if (Array.isArray(response)) {
        setTriggers(response)
      }
    } catch (error) {
      console.error('Failed to toggle trigger:', error)
    }
  }

  // Stop/start periodic refresh based on WebSocket status
  useEffect(() => {
    if (dataService) {
      if (state.websocket.connected) {
        dataService.stopPeriodicRefresh()
      } else {
        dataService.startPeriodicRefresh() // Uses default 45s interval with intelligent change detection
      }
    }
  }, [state.websocket.connected, dataService])

  // Handle workflow creation (memoized to prevent unnecessary re-renders)
  const handleWorkflowCreated = useCallback((workflowId: string) => {
    console.log('[WorkflowsPage] Workflow created:', workflowId)
    setActiveTab('executor')
  }, []) // Empty deps - this function doesn't depend on any state

  // Handle incident selection (memoized)
  const handleIncidentSelect = useCallback((incidentId: number) => {
    console.log('[WorkflowsPage] Incident selected:', incidentId)
    dispatch(appActions.setSelectedIncident(incidentId))
  }, [dispatch])

  // Load triggers function (for manual refresh)
  const loadTriggers = async () => {
    setTriggersLoading(true)
    try {
      const response = await listWorkflowTriggers({})
      if (Array.isArray(response)) {
        setTriggers(response)
      }
    } catch (error) {
      console.error('Failed to fetch workflow triggers:', error)
    } finally {
      setTriggersLoading(false)
    }
  }

  // Toggle trigger function (wrapper)
  const toggleTrigger = (triggerId: number, currentlyEnabled: boolean) => {
    handleToggleTrigger(triggerId, currentlyEnabled)
  }

  // Memoize selected incident to prevent unnecessary re-renders of child components
  // CRITICAL FIX: Only depend on selectedIncident ID, not incidents array
  const selectedIncidentObject = useMemo(() => {
    if (!state.selectedIncident) {
      console.log('[WorkflowsPage] No incident selected')
      return null
    }
    const incident = state.incidents.find(i => i.id === state.selectedIncident)
    // Return a stable object by only including the fields we need
    const result = incident ? {
      id: incident.id,
      status: incident.status,
      src_ip: incident.src_ip,
      reason: incident.reason
    } : null
    console.log('[WorkflowsPage] Memoized selectedIncidentObject:', result)
    return result
  }, [state.selectedIncident, state.incidents])  // Depend on selected ID and incidents array

  // Helper function to format condition display
  const formatTriggerConditions = (conditions: Record<string, any>) => {
    const items: Array<{ label: string; value: string }> = []

    if (conditions.event_type) {
      items.push({ label: 'Event Type', value: conditions.event_type })
    }
    if (conditions.threshold && conditions.window_seconds) {
      items.push({ label: 'Threshold', value: `≥ ${conditions.threshold} within ${conditions.window_seconds}s` })
    }
    if (conditions.pattern_match) {
      items.push({ label: 'Pattern Match', value: conditions.pattern_match })
    }
    if (conditions.risk_score_min) {
      items.push({ label: 'Risk Score', value: `≥ ${conditions.risk_score_min}` })
    }
    if (conditions.source) {
      items.push({ label: 'Source', value: conditions.source })
    }

    return items
  }

  // Helper function to format workflow steps
  const formatWorkflowSteps = (steps: Array<{action_type: string, parameters: Record<string, any>}>) => {
    const stepDescriptions: Record<string, {title: string, description: string}> = {
      'block_ip': { title: 'Block Attacker IP', description: 'Immediately blocks the source IP address at firewall level' },
      'create_incident': { title: 'Create Incident Case', description: 'Automatically creates documented incident for review' },
      'invoke_ai_agent': { title: 'AI Security Analysis', description: 'Analyzes attack patterns and provides threat intelligence' },
      'send_notification': { title: 'Alert Security Team', description: 'Sends notification to SOC analysts' },
      'analyze_payload': { title: 'Analyze Payload', description: 'Deep analysis of attack payload and techniques' },
      'isolate_file': { title: 'Isolate File', description: 'Quarantines suspicious files for analysis' },
    }

    return steps.map((step, index) => ({
      number: index + 1,
      ...stepDescriptions[step.action_type] || {
        title: step.action_type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
        description: 'Executes automated response action'
      }
    }))
  }

  // Helper function to get icon color based on trigger name
  const getTriggerIconColor = (name: string) => {
    if (name.toLowerCase().includes('ssh') || name.toLowerCase().includes('brute')) {
      return { bg: 'bg-red-500/10', border: 'border-red-500/30', icon: 'text-red-400' }
    }
    if (name.toLowerCase().includes('sql') || name.toLowerCase().includes('injection')) {
      return { bg: 'bg-orange-500/10', border: 'border-orange-500/30', icon: 'text-orange-400' }
    }
    if (name.toLowerCase().includes('malware') || name.toLowerCase().includes('payload')) {
      return { bg: 'bg-purple-500/10', border: 'border-purple-500/30', icon: 'text-purple-400' }
    }
    return { bg: 'bg-blue-500/10', border: 'border-blue-500/30', icon: 'text-blue-400' }
  }

  if (state.loading.incidents || state.loading.workflows) {
    return (
      <div className="min-h-screen bg-gray-950">
        <div className="container mx-auto p-6">
          <div className="flex items-center justify-center h-[calc(100vh-12rem)]">
            <div className="text-center">
              <RefreshCw className="w-12 h-12 text-blue-500 animate-spin mx-auto mb-4" />
              <span className="text-lg text-gray-300">Loading workflow orchestration system...</span>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <DashboardLayout breadcrumbs={[{ label: "Workflows" }]}>
      <div className="space-y-6">
          {/* Header Section */}
        <Card className="bg-gray-900 border-gray-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="p-3 bg-blue-500/10 rounded-lg border border-blue-500/20">
                  <Workflow className="h-8 w-8 text-blue-400" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-white flex items-center gap-3">
                    Workflow Automation Platform
                    {state.websocket.connected ? (
                      <Badge className="bg-green-500/10 text-green-400 border-green-500/30">
                        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse mr-2" />
                        Real-time
                      </Badge>
                    ) : (
                      <Badge className="bg-yellow-500/10 text-yellow-400 border-yellow-500/30">
                        <Clock className="w-3 h-3 mr-1" />
                        Polling
                      </Badge>
                    )}
                  </h1>
                  <p className="text-gray-400 text-sm mt-1 flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-blue-400" />
                    AI-powered response orchestration with natural language processing
                  </p>
                </div>
              </div>

              {/* Quick Stats Grid */}
              <div className="grid grid-cols-4 gap-3">
                <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-3 text-center">
                  <div className="text-2xl font-bold text-blue-400">{workflowStats.total}</div>
                  <div className="text-xs text-gray-400 mt-1">Total</div>
                </div>
                <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-3 text-center">
                  <div className="text-2xl font-bold text-green-400">{workflowStats.active}</div>
                  <div className="text-xs text-gray-400 mt-1">Active</div>
                </div>
                <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-3 text-center">
                  <div className="text-2xl font-bold text-purple-400">{workflowStats.completed}</div>
                  <div className="text-xs text-gray-400 mt-1">Completed</div>
                </div>
                <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-3 text-center">
                  <div className="text-2xl font-bold text-orange-400">{workflowStats.templates}</div>
                  <div className="text-xs text-gray-400 mt-1">Templates</div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Error Messages */}
        {(state.errors.incidents || state.errors.workflows) && (
          <Alert className="bg-red-500/10 border-red-500/30 text-red-400">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              {state.errors.incidents || state.errors.workflows}
            </AlertDescription>
          </Alert>
        )}

        {/* Compact Incident Context Selector */}
        <div className="bg-gray-800/50 border border-gray-700/50 rounded-lg p-3">
          <div className="flex items-center gap-3">
            <Target className="w-5 h-5 text-blue-400 flex-shrink-0" />
            <div className="flex-1 flex items-center gap-3">
              <span className="text-sm font-medium text-gray-300">Workflow Scope:</span>
              <select
                value={state.selectedIncident || 'none'}
                onChange={(e) => {
                  const value = e.target.value === 'none' ? null : parseInt(e.target.value)
                  handleIncidentSelect(value)
                }}
                className="flex-1 bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              >
                <option value="none">None - General Workflow (Always-On)</option>
                {state.incidents.filter(i => i.status === 'open').map(incident => (
                  <option key={incident.id} value={incident.id}>
                    Incident #{incident.id} - {incident.src_ip} - {incident.reason}
                  </option>
                ))}
              </select>
            </div>
            {state.selectedIncident ? (
              <Badge className="bg-blue-600 text-white border-blue-500 flex-shrink-0">
                <Sparkles className="w-3 h-3 mr-1" />
                Incident-Specific
              </Badge>
            ) : (
              <Badge className="bg-purple-600 text-white border-purple-500 flex-shrink-0">
                <Zap className="w-3 h-3 mr-1" />
                General Workflow
              </Badge>
            )}
          </div>
          {state.selectedIncident && (() => {
            const incident = state.incidents.find(i => i.id === state.selectedIncident)
            return incident ? (
              <div className="mt-3 pt-3 border-t border-gray-700/50 flex items-center gap-4 text-xs text-gray-400">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-yellow-400" />
                  <span className="font-medium text-white">Context:</span>
                  <span>{incident.reason}</span>
                </div>
                <div className="flex items-center gap-2">
                  <Globe className="w-4 h-4 text-blue-400" />
                  <span>{incident.src_ip}</span>
                </div>
                {incident.risk_score && (
                  <div className="flex items-center gap-2">
                    <BarChart3 className="w-4 h-4 text-red-400" />
                    <span>Risk: {(incident.risk_score * 100).toFixed(0)}%</span>
                  </div>
                )}
                <div className="ml-auto text-gray-500">
                  {incident.num_events || 0} events
                </div>
              </div>
            ) : null
          })()}
        </div>

        {/* Main Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-6 bg-gray-900 border border-gray-800 p-1">
            <TabsTrigger
              value="natural"
              className="flex items-center gap-2 data-[state=active]:bg-blue-600 data-[state=active]:text-white text-gray-400"
            >
              <MessageSquare className="h-4 w-4" />
              <span className="hidden sm:inline">Natural Language</span>
              <span className="sm:hidden">NLP</span>
            </TabsTrigger>
            <TabsTrigger
              value="designer"
              className="flex items-center gap-2 data-[state=active]:bg-blue-600 data-[state=active]:text-white text-gray-400"
            >
              <Workflow className="h-4 w-4" />
              <span className="hidden sm:inline">Designer</span>
            </TabsTrigger>
            <TabsTrigger
              value="templates"
              className="flex items-center gap-2 data-[state=active]:bg-blue-600 data-[state=active]:text-white text-gray-400"
            >
              <Save className="h-4 w-4" />
              <span className="hidden sm:inline">Templates</span>
            </TabsTrigger>
            <TabsTrigger
              value="executor"
              className="flex items-center gap-2 data-[state=active]:bg-blue-600 data-[state=active]:text-white text-gray-400"
            >
              <Play className="h-4 w-4" />
              <span className="hidden sm:inline">Executor</span>
            </TabsTrigger>
            <TabsTrigger
              value="analytics"
              className="flex items-center gap-2 data-[state=active]:bg-blue-600 data-[state=active]:text-white text-gray-400"
            >
              <Activity className="h-4 w-4" />
              <span className="hidden sm:inline">Analytics</span>
            </TabsTrigger>
            <TabsTrigger
              value="triggers"
              className="flex items-center gap-2 data-[state=active]:bg-blue-600 data-[state=active]:text-white text-gray-400"
            >
              <Settings className="h-4 w-4" />
              <span className="hidden sm:inline">Auto Triggers</span>
            </TabsTrigger>
          </TabsList>

          {/* Natural Language Tab - Enhanced with workflow creation and execution */}
          <TabsContent value="natural" className="space-y-4 mt-6">
            <Card className="bg-gray-900 border-gray-800">
              <CardHeader>
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-blue-500/10 rounded-lg border border-blue-500/20">
                    <Brain className="h-5 w-5 text-blue-400" />
                  </div>
                  <div>
                    <CardTitle className="text-white flex items-center gap-2">
                      AI-Powered Workflow Chat
                      <Badge className="bg-green-500/10 text-green-400 border-green-500/30">
                        <Sparkles className="w-3 h-3 mr-1" />
                        GPT-4 Enhanced
                      </Badge>
                    </CardTitle>
                    <CardDescription className="text-gray-400 mt-1">
                      Chat with AI to create workflows, execute single tasks, or get response recommendations
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <NaturalLanguageInput
                  selectedIncidentId={state.selectedIncident}
                  selectedIncident={selectedIncidentObject}
                  onWorkflowCreated={handleWorkflowCreated}
                />
              </CardContent>
            </Card>

            {/* Feature Highlights */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card className="bg-gray-900 border-blue-500/30">
                <CardContent className="p-4">
                  <Zap className="w-8 h-8 text-blue-400 mb-2" />
                  <h3 className="font-semibold text-white mb-1">Instant Execution</h3>
                  <p className="text-sm text-gray-400">Execute single tasks or create complex workflows instantly</p>
                </CardContent>
              </Card>
              <Card className="bg-gray-900 border-purple-500/30">
                <CardContent className="p-4">
                  <Brain className="w-8 h-8 text-purple-400 mb-2" />
                  <h3 className="font-semibold text-white mb-1">AI Understanding</h3>
                  <p className="text-sm text-gray-400">GPT-4 powered parsing with 90%+ accuracy</p>
                </CardContent>
              </Card>
              <Card className="bg-gray-900 border-green-500/30">
                <CardContent className="p-4">
                  <CheckCircle2 className="w-8 h-8 text-green-400 mb-2" />
                  <h3 className="font-semibold text-white mb-1">Safety First</h3>
                  <p className="text-sm text-gray-400">Automatic approval workflows for critical actions</p>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Designer Tab */}
          <TabsContent value="designer" className="space-y-4 mt-6">
            <Card className="bg-gray-900 border-gray-800">
              <CardHeader>
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-purple-500/10 rounded-lg border border-purple-500/20">
                    <Workflow className="h-5 w-5 text-purple-400" />
                  </div>
                  <div>
                    <CardTitle className="text-white">Visual Workflow Designer</CardTitle>
                    <CardDescription className="text-gray-400 mt-1">
                      Drag-and-drop interface with 68 pre-built response actions
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <WorkflowDesigner
                  incidentId={state.selectedIncident}
                  onWorkflowCreated={handleWorkflowCreated}
                />
              </CardContent>
            </Card>
          </TabsContent>

          {/* Templates Tab */}
          <TabsContent value="templates" className="space-y-4 mt-6">
            <PlaybookTemplates
              selectedIncidentId={state.selectedIncident}
              onSelectTemplate={(template) => {
                console.log('Template selected:', template)
                setActiveTab('designer')
              }}
            />
          </TabsContent>

          {/* Executor Tab */}
          <TabsContent value="executor" className="space-y-4 mt-6">
            <Card className="bg-gray-900 border-gray-800">
              <CardHeader>
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-green-500/10 rounded-lg border border-green-500/20">
                    <Play className="h-5 w-5 text-green-400" />
                  </div>
                  <div>
                    <CardTitle className="text-white">Workflow Execution Monitor</CardTitle>
                    <CardDescription className="text-gray-400 mt-1">
                      Real-time workflow execution with progress tracking
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <WorkflowExecutor
                  workflows={state.workflows}
                  onWorkflowUpdate={() => {
                    if (dataService) {
                      dataService.refreshWorkflows()
                    }
                  }}
                />
              </CardContent>
            </Card>
          </TabsContent>

          {/* Analytics Tab */}
          <TabsContent value="analytics" className="space-y-4 mt-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <Card className="bg-gray-900 border-blue-500/30">
                <CardContent className="p-6 text-center">
                  <TrendingUp className="w-8 h-8 text-blue-400 mx-auto mb-3" />
                  <div className="text-3xl font-bold text-blue-400 mb-1">{workflowStats.total}</div>
                  <div className="text-sm text-gray-400">Total Workflows</div>
                </CardContent>
              </Card>
              <Card className="bg-gray-900 border-green-500/30">
                <CardContent className="p-6 text-center">
                  <Activity className="w-8 h-8 text-green-400 mx-auto mb-3" />
                  <div className="text-3xl font-bold text-green-400 mb-1">{workflowStats.active}</div>
                  <div className="text-sm text-gray-400">Active Now</div>
                </CardContent>
              </Card>
              <Card className="bg-gray-900 border-purple-500/30">
                <CardContent className="p-6 text-center">
                  <CheckCircle2 className="w-8 h-8 text-purple-400 mx-auto mb-3" />
                  <div className="text-3xl font-bold text-purple-400 mb-1">{workflowStats.completed}</div>
                  <div className="text-sm text-gray-400">Completed</div>
                </CardContent>
              </Card>
              <Card className="bg-gray-900 border-orange-500/30">
                <CardContent className="p-6 text-center">
                  <Save className="w-8 h-8 text-orange-400 mx-auto mb-3" />
                  <div className="text-3xl font-bold text-orange-400 mb-1">{workflowStats.templates}</div>
                  <div className="text-sm text-gray-400">Templates</div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Auto Triggers Tab */}
          <TabsContent value="triggers" className="space-y-4 mt-6">
            <Card className="bg-gray-900 border-gray-800">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Settings className="h-5 w-5 text-blue-400" />
                  Automation & Triggers Management
                </CardTitle>
                <CardDescription>
                  Manage automated workflows, AI suggestions, and trigger settings all in one place
                </CardDescription>
              </CardHeader>
              <CardContent>
                <AutomationsPanel />
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </DashboardLayout>
  )
}

export default WorkflowsPage
