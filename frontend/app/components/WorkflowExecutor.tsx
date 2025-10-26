'use client'

/**
 * Workflow Executor Component
 * 
 * Real-time execution and monitoring of response workflows.
 * Provides live progress tracking, step-by-step execution monitoring,
 * and interactive control capabilities.
 */

import React, { useState, useEffect, useCallback } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Play,
  Pause,
  Square,
  RotateCcw,
  CheckCircle,
  XCircle,
  Clock,
  AlertTriangle,
  Eye,
  Activity,
  Zap,
  RefreshCw,
  TrendingUp,
  Target,
  Shield,
  Network,
  Server,
  Mail,
  Cloud,
  Key,
  Database,
  Edit,
  Trash2
} from 'lucide-react'
import { 
  executeResponseWorkflow,
  cancelWorkflow,
  getWorkflowStatus,
  getWorkflowActions
} from '@/app/lib/api'

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
  execution_time_ms?: number
}

interface WorkflowAction {
  action_id: string
  action_type: string
  action_name: string
  status: string
  result_data?: any
  error_details?: any
  completed_at?: string
  confidence_score?: number
}

interface WorkflowExecutorProps {
  workflows: ResponseWorkflow[]
  onWorkflowUpdate: () => void
}

const WorkflowExecutor: React.FC<WorkflowExecutorProps> = ({
  workflows,
  onWorkflowUpdate
}) => {
  const [selectedWorkflow, setSelectedWorkflow] = useState<ResponseWorkflow | null>(null)
  const [workflowActions, setWorkflowActions] = useState<WorkflowAction[]>([])
  const [isExecuting, setIsExecuting] = useState<string | null>(null)
  const [isCancelling, setIsCancelling] = useState<string | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [refreshInterval, setRefreshInterval] = useState(5000) // 5 seconds

  // Category icons
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

  // Status colors
  const statusColors = {
    pending: 'bg-gray-100 text-gray-800',
    running: 'bg-blue-100 text-blue-800',
    completed: 'bg-green-100 text-green-800',
    failed: 'bg-red-100 text-red-800',
    cancelled: 'bg-orange-100 text-orange-800',
    awaiting_approval: 'bg-purple-100 text-purple-800'
  }

  // Load workflow actions when workflow is selected
  const loadWorkflowActions = useCallback(async (workflowId: string) => {
    try {
      const data = await getWorkflowActions(workflowId)
      if (data.success) {
        setWorkflowActions(data.actions || [])
      }
    } catch (error) {
      console.error('Failed to load workflow actions:', error)
      setWorkflowActions([])
    }
  }, [])

  // Auto-refresh workflow status
  useEffect(() => {
    if (!autoRefresh || !selectedWorkflow) return

    const interval = setInterval(async () => {
      try {
        const data = await getWorkflowStatus(selectedWorkflow.workflow_id)
        if (data.success) {
          // Update the selected workflow with new status
          setSelectedWorkflow(prev => prev ? { ...prev, ...data } : null)
          onWorkflowUpdate()
        }
      } catch (error) {
        console.error('Failed to refresh workflow status:', error)
      }
    }, refreshInterval)

    return () => clearInterval(interval)
  }, [autoRefresh, selectedWorkflow, refreshInterval, onWorkflowUpdate])

  // Load actions when workflow is selected
  useEffect(() => {
    if (selectedWorkflow) {
      loadWorkflowActions(selectedWorkflow.workflow_id)
    }
  }, [selectedWorkflow, loadWorkflowActions])

  // Execute workflow
  const executeWorkflow = async (workflow: ResponseWorkflow) => {
    setIsExecuting(workflow.workflow_id)
    try {
      const result = await executeResponseWorkflow(workflow.id)
      if (result.success) {
        onWorkflowUpdate()
        setSelectedWorkflow(prev => prev ? { ...prev, status: 'running' } : null)
      } else {
        alert(`Failed to execute workflow: ${result.error}`)
      }
    } catch (error) {
      console.error('Failed to execute workflow:', error)
      alert('Failed to execute workflow')
    } finally {
      setIsExecuting(null)
    }
  }

  // Cancel workflow
  const cancelWorkflowExecution = async (workflow: ResponseWorkflow) => {
    setIsCancelling(workflow.workflow_id)
    try {
      const result = await cancelWorkflow(workflow.workflow_id)
      if (result.success) {
        onWorkflowUpdate()
        setSelectedWorkflow(prev => prev ? { ...prev, status: 'cancelled' } : null)
      } else {
        alert(`Failed to cancel workflow: ${result.error}`)
      }
    } catch (error) {
      console.error('Failed to cancel workflow:', error)
      alert('Failed to cancel workflow')
    } finally {
      setIsCancelling(null)
    }
  }

  // Delete workflow
  const deleteWorkflow = async (workflow: ResponseWorkflow) => {
    if (!confirm(`Are you sure you want to delete workflow "${workflow.playbook_name}"?`)) {
      return
    }

    try {
      const response = await fetch(`http://localhost:8000/api/response/workflows/${workflow.workflow_id}`, {
        method: 'DELETE',
        headers: {
          'x-api-key': process.env.NEXT_PUBLIC_API_KEY || 'demo-minixdr-api-key'
        }
      })

      if (response.ok) {
        onWorkflowUpdate()
        if (selectedWorkflow?.workflow_id === workflow.workflow_id) {
          setSelectedWorkflow(null)
        }
      } else {
        alert('Failed to delete workflow')
      }
    } catch (error) {
      console.error('Failed to delete workflow:', error)
      alert('Failed to delete workflow')
    }
  }

  // Edit workflow (placeholder)
  const editWorkflow = (workflow: ResponseWorkflow) => {
    alert('Edit functionality coming soon! You can cancel this workflow and create a new one with updated parameters.')
  }

  // Format duration
  const formatDuration = (ms?: number): string => {
    if (!ms) return 'N/A'
    if (ms < 1000) return `${ms}ms`
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
    return `${(ms / 60000).toFixed(1)}m`
  }

  // Get action icon
  const getActionIcon = (actionType: string, category: string) => {
    const Icon = categoryIcons[category as keyof typeof categoryIcons] || Shield
    return Icon
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Workflow List */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg">Active Workflows</CardTitle>
            <div className="flex items-center gap-2">
              <Button
                size="sm"
                variant="outline"
                onClick={() => setAutoRefresh(!autoRefresh)}
              >
                <RefreshCw className={`h-3 w-3 ${autoRefresh ? 'animate-spin' : ''}`} />
                {autoRefresh ? 'Auto' : 'Manual'}
              </Button>
            </div>
          </div>
        </CardHeader>
        
        <CardContent>
          <ScrollArea className="h-96">
            <div className="space-y-3">
              {workflows.map(workflow => (
                <Card 
                  key={workflow.workflow_id}
                  className={`cursor-pointer transition-all ${
                    selectedWorkflow?.workflow_id === workflow.workflow_id 
                      ? 'ring-2 ring-blue-500 bg-blue-50' 
                      : 'hover:shadow-md'
                  }`}
                  onClick={() => setSelectedWorkflow(workflow)}
                >
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <h4 className="font-semibold text-sm">{workflow.playbook_name}</h4>
                        <p className="text-xs text-gray-600">
                          Incident #{workflow.incident_id} • {workflow.workflow_id}
                        </p>
                      </div>
                      <Badge className={statusColors[workflow.status as keyof typeof statusColors]}>
                        {workflow.status.replace('_', ' ')}
                      </Badge>
                    </div>
                    
                    {/* Progress */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-xs">
                        <span>Progress</span>
                        <span>{workflow.current_step} / {workflow.total_steps} steps</span>
                      </div>
                      <Progress value={workflow.progress_percentage} className="h-1" />
                    </div>
                    
                    {/* Metrics */}
                    <div className="mt-2 flex items-center justify-between text-xs text-gray-500">
                      <span>Created: {new Date(workflow.created_at).toLocaleTimeString()}</span>
                      {workflow.success_rate !== undefined && (
                        <span className="text-green-600">{Math.round(workflow.success_rate * 100)}% success</span>
                      )}
                    </div>
                    
                    {/* Quick Actions */}
                    <div className="mt-3 flex gap-2">
                      {workflow.status === 'pending' && (
                        <>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={(e) => {
                              e.stopPropagation()
                              executeWorkflow(workflow)
                            }}
                            disabled={isExecuting === workflow.workflow_id}
                            title="Execute Workflow"
                          >
                            {isExecuting === workflow.workflow_id ? (
                              <div className="animate-spin h-3 w-3 border border-current border-t-transparent rounded-full"></div>
                            ) : (
                              <Play className="h-3 w-3" />
                            )}
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={(e) => {
                              e.stopPropagation()
                              editWorkflow(workflow)
                            }}
                            title="Edit Workflow"
                          >
                            <Edit className="h-3 w-3" />
                          </Button>
                        </>
                      )}

                      {workflow.status === 'running' && (
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={(e) => {
                            e.stopPropagation()
                            cancelWorkflowExecution(workflow)
                          }}
                          disabled={isCancelling === workflow.workflow_id}
                          title="Cancel Workflow"
                        >
                          {isCancelling === workflow.workflow_id ? (
                            <div className="animate-spin h-3 w-3 border border-current border-t-transparent rounded-full"></div>
                          ) : (
                            <Square className="h-3 w-3" />
                          )}
                        </Button>
                      )}

                      {/* Delete button - available for all non-running workflows */}
                      {workflow.status !== 'running' && (
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={(e) => {
                            e.stopPropagation()
                            deleteWorkflow(workflow)
                          }}
                          title="Delete Workflow"
                          className="text-red-600 hover:text-red-700 hover:bg-red-50"
                        >
                          <Trash2 className="h-3 w-3" />
                        </Button>
                      )}
                    </div>
                  </CardContent>
                </Card>
              ))}
              
              {workflows.length === 0 && (
                <div className="text-center py-8">
                  <Activity className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <h4 className="font-medium text-gray-900 mb-2">No Workflows Found</h4>
                  <p className="text-gray-600">Create a workflow using the Designer tab to get started.</p>
                </div>
              )}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>

      {/* Workflow Details */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">
            {selectedWorkflow ? 'Workflow Details' : 'Select Workflow'}
          </CardTitle>
          {selectedWorkflow && (
            <CardDescription>
              Real-time execution monitoring and control
            </CardDescription>
          )}
        </CardHeader>
        
        <CardContent>
          {selectedWorkflow ? (
            <div className="space-y-4">
              {/* Workflow Header */}
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="font-semibold">{selectedWorkflow.playbook_name}</h3>
                  <p className="text-sm text-gray-600">
                    Incident #{selectedWorkflow.incident_id} • {selectedWorkflow.workflow_id}
                  </p>
                </div>
                <Badge className={statusColors[selectedWorkflow.status as keyof typeof statusColors]}>
                  {selectedWorkflow.status.replace('_', ' ')}
                </Badge>
              </div>

              {/* Overall Progress */}
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-medium">Overall Progress</span>
                  <span>{selectedWorkflow.current_step} / {selectedWorkflow.total_steps}</span>
                </div>
                <Progress value={selectedWorkflow.progress_percentage} className="h-2" />
              </div>

              {/* Execution Metrics */}
              <div className="grid grid-cols-2 gap-4 p-3 bg-gray-50 rounded-lg">
                <div className="text-center">
                  <div className="text-lg font-bold text-blue-600">
                    {selectedWorkflow.success_rate !== undefined 
                      ? Math.round(selectedWorkflow.success_rate * 100) + '%'
                      : 'N/A'
                    }
                  </div>
                  <div className="text-xs text-gray-600">Success Rate</div>
                </div>
                <div className="text-center">
                  <div className="text-lg font-bold text-orange-600">
                    {formatDuration(selectedWorkflow.execution_time_ms)}
                  </div>
                  <div className="text-xs text-gray-600">Execution Time</div>
                </div>
              </div>

              {/* Action Steps */}
              <div>
                <h4 className="font-semibold text-sm mb-3 flex items-center gap-2">
                  <Zap className="h-4 w-4" />
                  Workflow Steps
                </h4>
                
                <ScrollArea className="h-48">
                  <div className="space-y-2">
                    {workflowActions.map((action, index) => {
                      const Icon = getActionIcon(action.action_type, 'network') // Default category
                      const isActive = selectedWorkflow.current_step === index + 1
                      const isCompleted = index < selectedWorkflow.current_step
                      
                      return (
                        <div 
                          key={action.action_id}
                          className={`p-3 border rounded-lg ${
                            isActive ? 'border-blue-300 bg-blue-50' :
                            isCompleted ? 'border-green-300 bg-green-50' :
                            'border-gray-200'
                          }`}
                        >
                          <div className="flex items-start justify-between">
                            <div className="flex items-center gap-2">
                              <div className={`p-1 rounded ${
                                isCompleted ? 'bg-green-100' :
                                isActive ? 'bg-blue-100' :
                                'bg-gray-100'
                              }`}>
                                <Icon className={`h-3 w-3 ${
                                  isCompleted ? 'text-green-600' :
                                  isActive ? 'text-blue-600' :
                                  'text-gray-600'
                                }`} />
                              </div>
                              
                              <div>
                                <div className="font-medium text-sm">
                                  {action.action_name || action.action_type}
                                </div>
                                <div className="text-xs text-gray-600">
                                  Step {index + 1} • {action.action_type}
                                </div>
                              </div>
                            </div>
                            
                            <div className="flex items-center gap-2">
                              {action.status === 'completed' && (
                                <CheckCircle className="h-4 w-4 text-green-600" />
                              )}
                              {action.status === 'failed' && (
                                <XCircle className="h-4 w-4 text-red-600" />
                              )}
                              {action.status === 'running' && (
                                <div className="animate-spin h-4 w-4 border border-current border-t-transparent rounded-full text-blue-600"></div>
                              )}
                              
                              <Badge variant="outline" className="text-xs">
                                {action.status}
                              </Badge>
                            </div>
                          </div>
                          
                          {/* Action Results */}
                          {action.result_data && (
                            <div className="mt-2 p-2 bg-white rounded text-xs">
                              <div className="font-medium mb-1">Result:</div>
                              {action.result_data.detail && (
                                <div className="text-gray-600">{action.result_data.detail}</div>
                              )}
                              {action.confidence_score && (
                                <div className="text-green-600 mt-1">
                                  Confidence: {Math.round(action.confidence_score * 100)}%
                                </div>
                              )}
                            </div>
                          )}
                          
                          {/* Error Details */}
                          {action.error_details && (
                            <div className="mt-2 p-2 bg-red-50 border border-red-200 rounded text-xs">
                              <div className="font-medium text-red-800 mb-1">Error:</div>
                              <div className="text-red-700">{action.error_details.error}</div>
                            </div>
                          )}
                        </div>
                      )
                    })}
                    
                    {workflowActions.length === 0 && selectedWorkflow && (
                      <div className="text-center py-4">
                        <AlertTriangle className="h-8 w-8 text-gray-400 mx-auto mb-2" />
                        <p className="text-sm text-gray-600">No action details available</p>
                      </div>
                    )}
                  </div>
                </ScrollArea>
              </div>

              {/* Control Buttons */}
              <div className="flex gap-2 pt-3 border-t">
                {selectedWorkflow.status === 'pending' && (
                  <Button
                    onClick={() => executeWorkflow(selectedWorkflow)}
                    disabled={isExecuting === selectedWorkflow.workflow_id}
                    className="flex-1"
                  >
                    {isExecuting === selectedWorkflow.workflow_id ? (
                      <div className="flex items-center gap-1">
                        <div className="animate-spin h-4 w-4 border border-current border-t-transparent rounded-full"></div>
                        Executing
                      </div>
                    ) : (
                      <div className="flex items-center gap-1">
                        <Play className="h-4 w-4" />
                        Execute Workflow
                      </div>
                    )}
                  </Button>
                )}
                
                {selectedWorkflow.status === 'running' && (
                  <Button
                    variant="destructive"
                    onClick={() => cancelWorkflowExecution(selectedWorkflow)}
                    disabled={isCancelling === selectedWorkflow.workflow_id}
                    className="flex-1"
                  >
                    {isCancelling === selectedWorkflow.workflow_id ? (
                      <div className="flex items-center gap-1">
                        <div className="animate-spin h-4 w-4 border border-current border-t-transparent rounded-full"></div>
                        Cancelling
                      </div>
                    ) : (
                      <div className="flex items-center gap-1">
                        <Square className="h-4 w-4" />
                        Cancel Workflow
                      </div>
                    )}
                  </Button>
                )}
                
                <Button
                  variant="outline"
                  onClick={() => loadWorkflowActions(selectedWorkflow.workflow_id)}
                  className="flex-1"
                >
                  <RefreshCw className="h-4 w-4 mr-1" />
                  Refresh
                </Button>
              </div>
            </div>
          ) : (
            <div className="text-center py-12">
              <Activity className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h4 className="font-medium text-gray-900 mb-2">Select a Workflow</h4>
              <p className="text-gray-600">
                Choose a workflow from the list to monitor its execution and control its progress.
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Real-time Activity Feed */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Live Activity Feed
          </CardTitle>
          <CardDescription>
            Real-time workflow execution events and system notifications
          </CardDescription>
        </CardHeader>
        
        <CardContent>
          <ScrollArea className="h-96">
            <div className="space-y-2">
              {/* Mock activity feed - in production would come from real-time events */}
              {[
                {
                  id: 1,
                  timestamp: new Date(Date.now() - 30000),
                  type: 'workflow_started',
                  message: 'Malware Response workflow initiated',
                  severity: 'info'
                },
                {
                  id: 2,
                  timestamp: new Date(Date.now() - 25000),
                  type: 'action_completed',
                  message: 'Host isolation completed successfully',
                  severity: 'success'
                },
                {
                  id: 3,
                  timestamp: new Date(Date.now() - 20000),
                  type: 'action_started',
                  message: 'Memory dump collection in progress',
                  severity: 'info'
                },
                {
                  id: 4,
                  timestamp: new Date(Date.now() - 15000),
                  type: 'action_completed',
                  message: 'IP blocking rules deployed',
                  severity: 'success'
                },
                {
                  id: 5,
                  timestamp: new Date(Date.now() - 10000),
                  type: 'workflow_completed',
                  message: 'DDoS mitigation workflow completed (94% success)',
                  severity: 'success'
                }
              ].map(event => {
                const severityColors = {
                  info: 'border-l-blue-500 bg-blue-50',
                  success: 'border-l-green-500 bg-green-50',
                  warning: 'border-l-yellow-500 bg-yellow-50',
                  error: 'border-l-red-500 bg-red-50'
                }
                
                return (
                  <div 
                    key={event.id}
                    className={`p-2 border-l-4 rounded-r ${severityColors[event.severity as keyof typeof severityColors]}`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="text-sm">{event.message}</div>
                      <div className="text-xs text-gray-500">
                        {event.timestamp.toLocaleTimeString()}
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  )
}

export default WorkflowExecutor













