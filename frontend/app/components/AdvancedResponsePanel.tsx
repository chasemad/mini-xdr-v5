'use client'

/**
 * Advanced Response Panel Component
 * 
 * Provides enterprise-grade response capabilities with multi-vector actions,
 * workflow orchestration, and real-time monitoring.
 */

import React, { useState, useEffect, useCallback } from 'react'
import { 
  getAvailableResponseActions, 
  createResponseWorkflow, 
  listResponseWorkflows,
  executeSingleResponseAction
} from '@/app/lib/api'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ScrollArea } from "@/components/ui/scroll-area"
import { 
  Shield, 
  Zap, 
  Network, 
  Server, 
  Mail, 
  Cloud, 
  Key, 
  Database,
  AlertTriangle,
  CheckCircle,
  Clock,
  Play,
  Pause,
  RotateCcw,
  Settings,
  Activity,
  Target,
  Brain
} from 'lucide-react'

interface ResponseAction {
  id: string
  name: string
  category: string
  description: string
  safety_level: string
  estimated_duration: number
  rollback_supported: boolean
  parameters: string[]
}

interface ResponseWorkflow {
  id: number
  workflow_id: string
  incident_id: number
  playbook_name: string
  status: string
  progress_percentage: number
  current_step: number
  total_steps: number
  created_at: string
  completed_at?: string
  approval_required: boolean
  auto_executed: boolean
  success_rate?: number
}

interface WorkflowStep {
  action_type: string
  parameters: Record<string, any>
  timeout_seconds?: number
  continue_on_failure?: boolean
  retry_count?: number
  max_retries?: number
}

interface AdvancedResponsePanelProps {
  incidentId?: number
  onWorkflowCreated?: (workflowId: string) => void
  onActionExecuted?: (actionId: string, result: any) => void
}

const AdvancedResponsePanel: React.FC<AdvancedResponsePanelProps> = ({
  incidentId,
  onWorkflowCreated,
  onActionExecuted
}) => {
  // State management
  const [availableActions, setAvailableActions] = useState<Record<string, ResponseAction>>({})
  const [activeWorkflows, setActiveWorkflows] = useState<ResponseWorkflow[]>([])
  const [selectedActions, setSelectedActions] = useState<string[]>([])
  const [workflowName, setWorkflowName] = useState('')
  const [isCreatingWorkflow, setIsCreatingWorkflow] = useState(false)
  const [isExecutingAction, setIsExecutingAction] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState('actions')
  const [actionCategory, setActionCategory] = useState<string>('all')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Category icons mapping
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

  // Safety level colors
  const safetyColors = {
    low: 'bg-green-100 text-green-800 border-green-200',
    medium: 'bg-yellow-100 text-yellow-800 border-yellow-200',
    high: 'bg-red-100 text-red-800 border-red-200'
  }

  // Status colors
  const statusColors = {
    pending: 'bg-gray-100 text-gray-800',
    running: 'bg-blue-100 text-blue-800',
    completed: 'bg-green-100 text-green-800',
    failed: 'bg-red-100 text-red-800',
    cancelled: 'bg-orange-100 text-orange-800',
    awaiting_approval: 'bg-purple-100 text-purple-800'
  }

  // Load available actions
  const loadAvailableActions = useCallback(async (category?: string) => {
    try {
      // Get real available actions from API
      const data = await getAvailableResponseActions(category)
      
      if (data.success) {
        setAvailableActions(data.actions || {})
        setError(null)
      } else {
        throw new Error(data.error || 'Failed to load actions')
      }
    } catch (err) {
      console.error('Failed to load actions:', err)
      setError(err instanceof Error ? err.message : 'Unknown error')
      
      // Set empty actions on error
      setAvailableActions({})
    }
  }, [])

  // Load active workflows
  const loadActiveWorkflows = useCallback(async () => {
    if (!incidentId) return

    try {
      const data = await listResponseWorkflows({ incident_id: incidentId })
      if (data.success) {
        setActiveWorkflows(data.workflows || [])
      } else {
        throw new Error(data.error || 'Failed to load workflows')
      }
    } catch (err) {
      console.error('Failed to load workflows:', err)
      setActiveWorkflows([])
    }
  }, [incidentId])

  // No more helper functions needed - using real API data

  // Create workflow
  const createWorkflow = async () => {
    if (!incidentId || selectedActions.length === 0 || !workflowName.trim()) {
      setError('Please select actions and provide a workflow name')
      return
    }

    setIsCreatingWorkflow(true)
    setError(null)

    try {
      const steps: WorkflowStep[] = selectedActions.map(actionType => ({
        action_type: actionType,
        parameters: {
          target: incidentId,
          reason: `Workflow: ${workflowName}`
        },
        timeout_seconds: 300,
        continue_on_failure: false,
        max_retries: 3
      }))

      // Create real workflow via API
      const result = await createResponseWorkflow({
        incident_id: incidentId,
        playbook_name: workflowName,
        steps: steps,
        auto_execute: false,
        priority: 'medium'
      })

      if (result.success) {
        onWorkflowCreated?.(result.workflow_id)
        setSelectedActions([])
        setWorkflowName('')
        setActiveTab('workflows')
        
        // Reload workflows to show the new one
        loadActiveWorkflows()
      } else {
        throw new Error(result.error || 'Failed to create workflow')
      }

    } catch (err) {
      console.error('Failed to create workflow:', err)
      setError(err instanceof Error ? err.message : 'Failed to create workflow')
    } finally {
      setIsCreatingWorkflow(false)
    }
  }

  // Execute single action
  const executeSingleAction = async (actionType: string) => {
    if (!incidentId) return

    setIsExecutingAction(actionType)
    setError(null)

    try {
      // Execute real action via API
      const result = await executeSingleResponseAction({
        action_type: actionType,
        incident_id: incidentId,
        parameters: {
          target: incidentId,
          reason: 'Manual execution from Advanced Response Panel'
        }
      })

      onActionExecuted?.(actionType, result)
      
      if (!result.success) {
        setError(result.error || `Failed to execute ${actionType}`)
      }
      
    } catch (err) {
      console.error('Failed to execute action:', err)
      setError(err instanceof Error ? err.message : 'Failed to execute action')
    } finally {
      setIsExecutingAction(null)
    }
  }

  // Initialize component
  useEffect(() => {
    const initialize = async () => {
      setLoading(true)
      await Promise.all([
        loadAvailableActions(actionCategory),
        loadActiveWorkflows()
      ])
      setLoading(false)
    }

    initialize()
  }, [loadAvailableActions, loadActiveWorkflows, actionCategory])

  // Filter actions by category
  const filteredActions = Object.entries(availableActions).filter(([_, action]) => 
    actionCategory === 'all' || action.category === actionCategory
  )

  if (loading) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Advanced Response System
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-32">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <span className="ml-2">Loading response capabilities...</span>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="w-full space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Advanced Response System
            <Badge variant="outline" className="ml-auto">
              {Object.keys(availableActions).length} Actions Available
            </Badge>
          </CardTitle>
          <CardDescription>
            Enterprise-grade response capabilities with workflow orchestration and safety controls
          </CardDescription>
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
              <TabsTrigger value="actions">Response Actions</TabsTrigger>
              <TabsTrigger value="workflows">Active Workflows</TabsTrigger>
              <TabsTrigger value="analytics">Analytics</TabsTrigger>
            </TabsList>

            <TabsContent value="actions" className="space-y-4">
              {/* Category Filter */}
              <div className="flex flex-wrap gap-2">
                <Button
                  variant={actionCategory === 'all' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setActionCategory('all')}
                >
                  All Categories
                </Button>
                {Object.keys(categoryIcons).map(category => {
                  const Icon = categoryIcons[category as keyof typeof categoryIcons]
                  return (
                    <Button
                      key={category}
                      variant={actionCategory === category ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => setActionCategory(category)}
                      className="flex items-center gap-1"
                    >
                      <Icon className="h-3 w-3" />
                      {category.charAt(0).toUpperCase() + category.slice(1)}
                    </Button>
                  )
                })}
              </div>

              {/* Action Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {filteredActions.map(([actionType, action]) => {
                  const Icon = categoryIcons[action.category as keyof typeof categoryIcons] || Shield
                  const isSelected = selectedActions.includes(actionType)
                  const isExecuting = isExecutingAction === actionType

                  return (
                    <Card 
                      key={actionType} 
                      className={`cursor-pointer transition-all ${
                        isSelected ? 'ring-2 ring-blue-500 bg-blue-50' : 'hover:shadow-md'
                      }`}
                      onClick={() => {
                        if (isSelected) {
                          setSelectedActions(prev => prev.filter(a => a !== actionType))
                        } else {
                          setSelectedActions(prev => [...prev, actionType])
                        }
                      }}
                    >
                      <CardContent className="p-4">
                        <div className="flex items-start justify-between mb-2">
                          <Icon className="h-5 w-5 text-blue-600" />
                          <Badge 
                            variant="outline" 
                            className={safetyColors[action.safety_level as keyof typeof safetyColors]}
                          >
                            {action.safety_level}
                          </Badge>
                        </div>
                        
                        <h4 className="font-semibold text-sm mb-1">{action.name}</h4>
                        <p className="text-xs text-gray-600 mb-2">{action.description}</p>
                        
                        <div className="flex items-center justify-between text-xs text-gray-500">
                          <span className="flex items-center gap-1">
                            <Clock className="h-3 w-3" />
                            {Math.floor(action.estimated_duration / 60)}m
                          </span>
                          {action.rollback_supported && (
                            <span className="flex items-center gap-1 text-green-600">
                              <RotateCcw className="h-3 w-3" />
                              Rollback
                            </span>
                          )}
                        </div>
                        
                        <div className="mt-3 pt-3 border-t flex gap-2">
                          <Button
                            size="sm"
                            variant="outline"
                            className="flex-1"
                            onClick={(e) => {
                              e.stopPropagation()
                              executeSingleAction(actionType)
                            }}
                            disabled={isExecuting || !incidentId}
                          >
                            {isExecuting ? (
                              <div className="flex items-center gap-1">
                                <div className="animate-spin h-3 w-3 border border-current border-t-transparent rounded-full"></div>
                                Executing
                              </div>
                            ) : (
                              <div className="flex items-center gap-1">
                                <Play className="h-3 w-3" />
                                Execute
                              </div>
                            )}
                          </Button>
                        </div>
                      </CardContent>
                    </Card>
                  )
                })}
              </div>

              {/* Workflow Creation */}
              {selectedActions.length > 0 && (
                <Card className="border-blue-200 bg-blue-50">
                  <CardContent className="p-4">
                    <div className="flex items-center gap-2 mb-3">
                      <Brain className="h-5 w-5 text-blue-600" />
                      <h4 className="font-semibold">Create Response Workflow</h4>
                      <Badge variant="outline">
                        {selectedActions.length} actions selected
                      </Badge>
                    </div>
                    
                    <div className="flex gap-2">
                      <input
                        type="text"
                        placeholder="Workflow name (e.g., 'Malware Response')"
                        value={workflowName}
                        onChange={(e) => setWorkflowName(e.target.value)}
                        className="flex-1 px-3 py-2 border rounded-md text-sm"
                        disabled={isCreatingWorkflow}
                      />
                      <Button
                        onClick={createWorkflow}
                        disabled={isCreatingWorkflow || !workflowName.trim() || !incidentId}
                      >
                        {isCreatingWorkflow ? (
                          <div className="flex items-center gap-1">
                            <div className="animate-spin h-4 w-4 border border-current border-t-transparent rounded-full"></div>
                            Creating
                          </div>
                        ) : (
                          <div className="flex items-center gap-1">
                            <Zap className="h-4 w-4" />
                            Create Workflow
                          </div>
                        )}
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              )}
            </TabsContent>

            <TabsContent value="workflows" className="space-y-4">
              <div className="flex items-center justify-between">
                <h4 className="font-semibold">Active Workflows</h4>
                <Button variant="outline" size="sm" onClick={loadActiveWorkflows}>
                  <Activity className="h-4 w-4 mr-1" />
                  Refresh
                </Button>
              </div>

              {activeWorkflows.length === 0 ? (
                <Card>
                  <CardContent className="p-8 text-center">
                    <Activity className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                    <h4 className="font-medium text-gray-900 mb-2">No Active Workflows</h4>
                    <p className="text-gray-600">Create a workflow from the Response Actions tab to get started.</p>
                  </CardContent>
                </Card>
              ) : (
                <div className="space-y-3">
                  {activeWorkflows.map((workflow) => (
                    <Card key={workflow.workflow_id}>
                      <CardContent className="p-4">
                        <div className="flex items-start justify-between mb-3">
                          <div>
                            <h4 className="font-semibold">{workflow.playbook_name}</h4>
                            <p className="text-sm text-gray-600">
                              Workflow ID: {workflow.workflow_id}
                            </p>
                          </div>
                          <Badge className={statusColors[workflow.status as keyof typeof statusColors]}>
                            {workflow.status.replace('_', ' ')}
                          </Badge>
                        </div>
                        
                        <div className="space-y-2">
                          <div className="flex items-center justify-between text-sm">
                            <span>Progress</span>
                            <span>{workflow.current_step} / {workflow.total_steps} steps</span>
                          </div>
                          <Progress value={workflow.progress_percentage} className="h-2" />
                        </div>
                        
                        <div className="mt-3 pt-3 border-t flex items-center justify-between text-xs text-gray-500">
                          <span>Created: {new Date(workflow.created_at).toLocaleString()}</span>
                          {workflow.approval_required && (
                            <Badge variant="outline" className="text-orange-600 border-orange-200">
                              Approval Required
                            </Badge>
                          )}
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </TabsContent>

            <TabsContent value="analytics">
              <Card>
                <CardContent className="p-8 text-center">
                  <Settings className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <h4 className="font-medium text-gray-900 mb-2">Analytics Dashboard</h4>
                  <p className="text-gray-600">
                    Response analytics and performance metrics will be available here.
                  </p>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  )
}

export default AdvancedResponsePanel
