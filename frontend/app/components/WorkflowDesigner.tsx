'use client'

/**
 * Visual Workflow Designer Component
 * 
 * Drag-and-drop interface for creating response workflows using React Flow.
 * Provides real-time validation, execution tracking, and template management.
 */

import React, { useState, useCallback, useEffect, useMemo } from 'react'
import {
  ReactFlow,
  Node,
  Edge,
  Connection,
  useNodesState,
  useEdgesState,
  addEdge,
  Controls,
  Background,
  BackgroundVariant,
  MiniMap,
  Panel,
  NodeTypes,
  EdgeTypes,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { 
  getAvailableResponseActions, 
  createResponseWorkflow,
  executeResponseWorkflow
} from '@/app/lib/api'
import { 
  Shield, 
  Network, 
  Server, 
  Mail, 
  Cloud, 
  Key, 
  Database,
  AlertTriangle,
  CheckCircle,
  Play,
  Save,
  Trash2,
  Plus,
  Zap,
  Brain,
  Clock,
  Target
} from 'lucide-react'

import ActionNodeLibrary from './ActionNodeLibrary'

// Custom node types for different action categories
const nodeTypes: NodeTypes = {
  actionNode: ({ data }: { data: any }) => (
    <div className={`px-4 py-2 rounded-lg border-2 min-w-32 ${
      data.category === 'network' ? 'bg-blue-50 border-blue-300' :
      data.category === 'endpoint' ? 'bg-green-50 border-green-300' :
      data.category === 'email' ? 'bg-purple-50 border-purple-300' :
      data.category === 'cloud' ? 'bg-orange-50 border-orange-300' :
      data.category === 'identity' ? 'bg-yellow-50 border-yellow-300' :
      data.category === 'data' ? 'bg-gray-50 border-gray-300' :
      'bg-white border-gray-300'
    }`}>
      <div className="flex items-center gap-2 mb-1">
        {data.icon && <data.icon className="h-4 w-4" />}
        <div className="font-semibold text-sm">{data.name}</div>
      </div>
      <div className="text-xs text-gray-600">{data.description}</div>
      <div className="flex items-center justify-between mt-2">
        <Badge variant="outline" className="text-xs">
          {data.safety_level}
        </Badge>
        <div className="flex items-center gap-1 text-xs text-gray-500">
          <Clock className="h-3 w-3" />
          {Math.floor(data.estimated_duration / 60)}m
        </div>
      </div>
    </div>
  ),
  startNode: ({ data }: { data: any }) => (
    <div className="px-4 py-2 rounded-full bg-green-100 border-2 border-green-300">
      <div className="flex items-center gap-2">
        <Play className="h-4 w-4 text-green-600" />
        <span className="font-semibold text-green-800">Start</span>
      </div>
    </div>
  ),
  endNode: ({ data }: { data: any }) => (
    <div className="px-4 py-2 rounded-full bg-red-100 border-2 border-red-300">
      <div className="flex items-center gap-2">
        <Target className="h-4 w-4 text-red-600" />
        <span className="font-semibold text-red-800">End</span>
      </div>
    </div>
  )
}

interface WorkflowDesignerProps {
  incidentId: number | null
  onWorkflowCreated?: (workflowId: string) => void
}

const WorkflowDesigner: React.FC<WorkflowDesignerProps> = ({
  incidentId,
  onWorkflowCreated
}) => {
  // React Flow state
  const [nodes, setNodes, onNodesChange] = useNodesState([])
  const [edges, setEdges, onEdgesChange] = useEdgesState([])
  
  // Component state
  const [availableActions, setAvailableActions] = useState<Record<string, any>>({})
  const [workflowName, setWorkflowName] = useState('')
  const [isCreating, setIsCreating] = useState(false)
  const [isExecuting, setIsExecuting] = useState(false)
  const [validation, setValidation] = useState<{ valid: boolean; errors: string[] }>({
    valid: false,
    errors: []
  })
  const [showLibrary, setShowLibrary] = useState(true)

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

  // Load available actions
  useEffect(() => {
    const loadActions = async () => {
      try {
        const data = await getAvailableResponseActions()
        if (data.success) {
          setAvailableActions(data.actions || {})
        }
      } catch (error) {
        console.error('Failed to load actions:', error)
      }
    }

    loadActions()
  }, [])

  // Initialize with start and end nodes
  useEffect(() => {
    if (nodes.length === 0) {
      const initialNodes: Node[] = [
        {
          id: 'start',
          type: 'startNode',
          position: { x: 100, y: 100 },
          data: { label: 'Start' },
          deletable: false,
        },
        {
          id: 'end',
          type: 'endNode',
          position: { x: 600, y: 400 },
          data: { label: 'End' },
          deletable: false,
        }
      ]
      setNodes(initialNodes)
    }
  }, [nodes.length, setNodes])

  // Handle connections between nodes
  const onConnect = useCallback(
    (params: Connection) => {
      setEdges((eds) => addEdge(params, eds))
      validateWorkflow()
    },
    [setEdges]
  )

  // Handle dropping action from library
  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault()

      const actionType = event.dataTransfer.getData('application/reactflow')
      if (!actionType) return

      const action = availableActions[actionType]
      if (!action) return

      // Calculate drop position
      const rect = (event.target as Element).closest('.react-flow')?.getBoundingClientRect()
      if (!rect) return

      const position = {
        x: event.clientX - rect.left - 100,
        y: event.clientY - rect.top - 50,
      }

      // Create new action node
      const newNode: Node = {
        id: `action-${Date.now()}`,
        type: 'actionNode',
        position,
        data: {
          ...action,
          actionType,
          icon: categoryIcons[action.category as keyof typeof categoryIcons] || Shield
        },
      }

      setNodes((nds) => nds.concat(newNode))
      validateWorkflow()
    },
    [availableActions, setNodes, categoryIcons]
  )

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    event.dataTransfer.dropEffect = 'move'
  }, [])

  // Validate workflow logic
  const validateWorkflow = useCallback(() => {
    const actionNodes = nodes.filter(n => n.type === 'actionNode')
    const errors: string[] = []

    if (actionNodes.length === 0) {
      errors.push('Workflow must contain at least one action')
    }

    if (!workflowName.trim()) {
      errors.push('Workflow name is required')
    }

    if (!incidentId) {
      errors.push('Please select an incident')
    }

    // Check if all action nodes are connected
    const connectedNodes = new Set<string>()
    edges.forEach(edge => {
      connectedNodes.add(edge.source)
      connectedNodes.add(edge.target)
    })

    const disconnectedActions = actionNodes.filter(node => !connectedNodes.has(node.id))
    if (disconnectedActions.length > 0) {
      errors.push(`${disconnectedActions.length} action(s) are not connected to the workflow`)
    }

    setValidation({
      valid: errors.length === 0,
      errors
    })
  }, [nodes, edges, workflowName, incidentId])

  // Re-validate when dependencies change
  useEffect(() => {
    validateWorkflow()
  }, [validateWorkflow])

  // Create workflow from visual design
  const createWorkflow = async () => {
    if (!validation.valid || !incidentId) return

    setIsCreating(true)
    try {
      // Convert visual workflow to API format
      const actionNodes = nodes.filter(n => n.type === 'actionNode')
      const steps = actionNodes.map((node, index) => ({
        action_type: node.data.actionType,
        parameters: {
          target: incidentId,
          reason: `Visual workflow: ${workflowName}`,
          step_order: index + 1
        },
        timeout_seconds: node.data.estimated_duration || 300,
        continue_on_failure: false,
        max_retries: 3
      }))

      const result = await createResponseWorkflow({
        incident_id: incidentId,
        playbook_name: workflowName,
        steps: steps,
        auto_execute: false,
        priority: 'medium'
      })

      if (result.success) {
        onWorkflowCreated?.(result.workflow_id)
        
        // Reset designer
        setWorkflowName('')
        setNodes([
          {
            id: 'start',
            type: 'startNode',
            position: { x: 100, y: 100 },
            data: { label: 'Start' },
            deletable: false,
          },
          {
            id: 'end',
            type: 'endNode',
            position: { x: 600, y: 400 },
            data: { label: 'End' },
            deletable: false,
          }
        ])
        setEdges([])
        
        alert(`Workflow "${workflowName}" created successfully!`)
      } else {
        alert(`Failed to create workflow: ${result.error}`)
      }
    } catch (error) {
      console.error('Failed to create workflow:', error)
      alert('Failed to create workflow')
    } finally {
      setIsCreating(false)
    }
  }

  // Execute workflow immediately
  const executeWorkflow = async () => {
    if (!validation.valid || !incidentId) return

    setIsExecuting(true)
    try {
      // First create the workflow
      const actionNodes = nodes.filter(n => n.type === 'actionNode')
      const steps = actionNodes.map((node, index) => ({
        action_type: node.data.actionType,
        parameters: {
          target: incidentId,
          reason: `Visual workflow execution: ${workflowName}`,
          step_order: index + 1
        },
        timeout_seconds: node.data.estimated_duration || 300,
        continue_on_failure: false,
        max_retries: 3
      }))

      const createResult = await createResponseWorkflow({
        incident_id: incidentId,
        playbook_name: workflowName || 'Quick Execute Workflow',
        steps: steps,
        auto_execute: true,
        priority: 'high'
      })

      if (createResult.success) {
        alert(`Workflow executed successfully! Workflow ID: ${createResult.workflow_id}`)
        onWorkflowCreated?.(createResult.workflow_id)
      } else {
        alert(`Failed to execute workflow: ${createResult.error}`)
      }
    } catch (error) {
      console.error('Failed to execute workflow:', error)
      alert('Failed to execute workflow')
    } finally {
      setIsExecuting(false)
    }
  }

  return (
    <div className="h-96 border rounded-lg relative">
      {/* Action Library Panel */}
      {showLibrary && (
        <div className="absolute top-4 left-4 z-10 w-80">
          <ActionNodeLibrary 
            actions={availableActions}
            onClose={() => setShowLibrary(false)}
          />
        </div>
      )}

      {/* Workflow Canvas */}
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onDrop={onDrop}
        onDragOver={onDragOver}
        nodeTypes={nodeTypes}
        className="bg-gray-50"
      >
        <Background variant={BackgroundVariant.Dots} />
        <Controls />
        <MiniMap />
        
        {/* Control Panel */}
        <Panel position="top-right">
          <Card className="w-80">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <Brain className="h-4 w-4" />
                Workflow Controls
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {/* Workflow Name */}
              <div>
                <label className="text-xs font-medium text-gray-600">Workflow Name</label>
                <input
                  type="text"
                  value={workflowName}
                  onChange={(e) => setWorkflowName(e.target.value)}
                  placeholder="Enter workflow name..."
                  className="w-full mt-1 px-2 py-1 border rounded text-sm"
                />
              </div>

              {/* Validation Status */}
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  {validation.valid ? (
                    <CheckCircle className="h-4 w-4 text-green-600" />
                  ) : (
                    <AlertTriangle className="h-4 w-4 text-red-600" />
                  )}
                  <span className="text-xs font-medium">
                    {validation.valid ? 'Workflow Valid' : 'Validation Errors'}
                  </span>
                </div>
                
                {validation.errors.length > 0 && (
                  <div className="space-y-1">
                    {validation.errors.map((error, index) => (
                      <div key={index} className="text-xs text-red-600 bg-red-50 p-1 rounded">
                        {error}
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Action Buttons */}
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={createWorkflow}
                  disabled={!validation.valid || isCreating}
                  className="flex-1"
                >
                  {isCreating ? (
                    <div className="flex items-center gap-1">
                      <div className="animate-spin h-3 w-3 border border-current border-t-transparent rounded-full"></div>
                      Creating
                    </div>
                  ) : (
                    <div className="flex items-center gap-1">
                      <Save className="h-3 w-3" />
                      Save
                    </div>
                  )}
                </Button>
                
                <Button
                  size="sm"
                  onClick={executeWorkflow}
                  disabled={!validation.valid || isExecuting}
                  className="flex-1"
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

              {/* Workflow Stats */}
              <div className="pt-2 border-t">
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="text-center">
                    <div className="font-bold text-blue-600">
                      {nodes.filter(n => n.type === 'actionNode').length}
                    </div>
                    <div className="text-gray-500">Actions</div>
                  </div>
                  <div className="text-center">
                    <div className="font-bold text-green-600">
                      {edges.length}
                    </div>
                    <div className="text-gray-500">Connections</div>
                  </div>
                </div>
              </div>

              {/* Toggle Library */}
              <Button
                size="sm"
                variant="outline"
                onClick={() => setShowLibrary(!showLibrary)}
                className="w-full"
              >
                {showLibrary ? 'Hide' : 'Show'} Action Library
              </Button>
            </CardContent>
          </Card>
        </Panel>
      </ReactFlow>
    </div>
  )
}

export default WorkflowDesigner

