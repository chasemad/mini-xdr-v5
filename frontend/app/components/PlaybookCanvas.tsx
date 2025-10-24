'use client'

/**
 * Playbook Canvas Component
 * 
 * Advanced drag-and-drop interface for visual workflow creation with
 * conditional logic, branching, and real-time validation.
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
  useReactFlow,
  MarkerType,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { 
  Plus, 
  Save, 
  Play, 
  Trash2,
  Copy,
  Undo,
  Redo,
  ZoomIn,
  ZoomOut,
  Maximize,
  Settings,
  CheckCircle,
  AlertTriangle,
  Clock,
  Zap,
  Brain,
  GitBranch,
  Pause,
  RotateCcw,
  Shield,
  Network,
  Server,
  Mail,
  Cloud,
  Key,
  Database,
  Target
} from 'lucide-react'
import { validateWorkflow, createResponseWorkflow } from '@/app/lib/api'

// Enhanced node types for visual workflow design
const nodeTypes: NodeTypes = {
  startNode: ({ data, selected }: { data: any; selected: boolean }) => (
    <div className={`px-4 py-2 rounded-full border-2 ${
      selected ? 'border-blue-500 shadow-lg' : 'border-green-300'
    } bg-green-100`}>
      <div className="flex items-center gap-2">
        <Play className="h-4 w-4 text-green-600" />
        <span className="font-semibold text-green-800">Start</span>
      </div>
    </div>
  ),
  
  endNode: ({ data, selected }: { data: any; selected: boolean }) => (
    <div className={`px-4 py-2 rounded-full border-2 ${
      selected ? 'border-blue-500 shadow-lg' : 'border-red-300'
    } bg-red-100`}>
      <div className="flex items-center gap-2">
        <Target className="h-4 w-4 text-red-600" />
        <span className="font-semibold text-red-800">End</span>
      </div>
    </div>
  ),
  
  actionNode: ({ data, selected }: { data: any; selected: boolean }) => {
    const categoryColors = {
      network: 'bg-blue-50 border-blue-300',
      endpoint: 'bg-green-50 border-green-300',
      email: 'bg-purple-50 border-purple-300',
      cloud: 'bg-orange-50 border-orange-300',
      identity: 'bg-yellow-50 border-yellow-300',
      data: 'bg-gray-50 border-gray-300',
      compliance: 'bg-indigo-50 border-indigo-300',
      forensics: 'bg-pink-50 border-pink-300'
    }
    
    return (
      <div className={`px-4 py-3 rounded-lg border-2 min-w-48 max-w-64 ${
        selected ? 'border-blue-500 shadow-lg' : categoryColors[data.category as keyof typeof categoryColors] || 'border-gray-300'
      }`}>
        <div className="flex items-center gap-2 mb-2">
          {data.icon && <data.icon className="h-4 w-4" />}
          <div className="font-semibold text-sm">{data.name}</div>
          {data.validation_status === 'error' && (
            <AlertTriangle className="h-3 w-3 text-red-500" />
          )}
          {data.validation_status === 'warning' && (
            <AlertTriangle className="h-3 w-3 text-yellow-500" />
          )}
        </div>
        
        <div className="text-xs text-gray-600 mb-2 line-clamp-2">
          {data.description}
        </div>
        
        <div className="flex items-center justify-between">
          <Badge variant="outline" className="text-xs">
            {data.safety_level}
          </Badge>
          <div className="flex items-center gap-1 text-xs text-gray-500">
            <Clock className="h-3 w-3" />
            {Math.floor(data.estimated_duration / 60)}m
          </div>
        </div>
        
        {data.rollback_supported && (
          <div className="flex items-center gap-1 mt-1">
            <RotateCcw className="h-3 w-3 text-green-500" />
            <span className="text-xs text-green-600">Rollback available</span>
          </div>
        )}
      </div>
    )
  },
  
  conditionNode: ({ data, selected }: { data: any; selected: boolean }) => (
    <div className={`px-3 py-2 rounded-lg border-2 min-w-32 ${
      selected ? 'border-blue-500 shadow-lg' : 'border-yellow-300'
    } bg-yellow-50`}>
      <div className="flex items-center gap-2 mb-1">
        <GitBranch className="h-4 w-4 text-yellow-600" />
        <span className="font-semibold text-sm text-yellow-800">Condition</span>
      </div>
      <div className="text-xs text-gray-600">
        {data.condition || 'If/Then logic'}
      </div>
    </div>
  ),
  
  waitNode: ({ data, selected }: { data: any; selected: boolean }) => (
    <div className={`px-3 py-2 rounded-lg border-2 min-w-32 ${
      selected ? 'border-blue-500 shadow-lg' : 'border-gray-300'
    } bg-gray-50`}>
      <div className="flex items-center gap-2 mb-1">
        <Pause className="h-4 w-4 text-gray-600" />
        <span className="font-semibold text-sm text-gray-800">Wait</span>
      </div>
      <div className="text-xs text-gray-600">
        {data.duration || '30s'}
      </div>
    </div>
  ),
  
  approvalNode: ({ data, selected }: { data: any; selected: boolean }) => (
    <div className={`px-3 py-2 rounded-lg border-2 min-w-32 ${
      selected ? 'border-blue-500 shadow-lg' : 'border-purple-300'
    } bg-purple-50`}>
      <div className="flex items-center gap-2 mb-1">
        <Shield className="h-4 w-4 text-purple-600" />
        <span className="font-semibold text-sm text-purple-800">Approval</span>
      </div>
      <div className="text-xs text-gray-600">
        {data.approver || 'Manager approval'}
      </div>
    </div>
  )
}

// Enhanced edge types
const edgeTypes: EdgeTypes = {
  conditional: {
    type: 'smoothstep',
    markerEnd: {
      type: MarkerType.ArrowClosed,
      color: '#3B82F6'
    },
    style: { strokeWidth: 2, stroke: '#3B82F6' },
    labelStyle: { fontSize: 10, backgroundColor: '#EFF6FF' }
  }
}

interface PlaybookCanvasProps {
  incidentId: number | null
  onWorkflowSaved?: (workflowId: string) => void
  onValidationChange?: (isValid: boolean, errors: string[]) => void
  initialNodes?: Node[]
  initialEdges?: Edge[]
}

const PlaybookCanvas: React.FC<PlaybookCanvasProps> = ({
  incidentId,
  onWorkflowSaved,
  onValidationChange,
  initialNodes = [],
  initialEdges = []
}) => {
  // React Flow state
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges)
  const { setViewport, getViewport, zoomIn, zoomOut, fitView } = useReactFlow()
  
  // Canvas state
  const [selectedNodes, setSelectedNodes] = useState<string[]>([])
  const [selectedEdges, setSelectedEdges] = useState<string[]>([])
  const [canvasMode, setCanvasMode] = useState<'design' | 'simulate' | 'execute'>('design')
  const [validation, setValidation] = useState<{ valid: boolean; errors: string[]; warnings: string[] }>({
    valid: false,
    errors: [],
    warnings: []
  })
  const [workflowName, setWorkflowName] = useState('')
  const [isSaving, setIsSaving] = useState(false)
  const [isValidating, setIsValidating] = useState(false)
  
  // History for undo/redo
  const [history, setHistory] = useState<{ nodes: Node[]; edges: Edge[] }[]>([])
  const [historyIndex, setHistoryIndex] = useState(-1)

  // Initialize default nodes if empty
  useEffect(() => {
    if (nodes.length === 0 && initialNodes.length === 0) {
      const defaultNodes: Node[] = [
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
      setNodes(defaultNodes)
    }
  }, [nodes.length, initialNodes.length, setNodes])

  // Save to history for undo/redo
  const saveToHistory = useCallback(() => {
    const newHistory = history.slice(0, historyIndex + 1)
    newHistory.push({ nodes: [...nodes], edges: [...edges] })
    setHistory(newHistory)
    setHistoryIndex(newHistory.length - 1)
  }, [nodes, edges, history, historyIndex])

  // Undo/Redo functionality
  const undo = useCallback(() => {
    if (historyIndex > 0) {
      const previousState = history[historyIndex - 1]
      setNodes(previousState.nodes)
      setEdges(previousState.edges)
      setHistoryIndex(historyIndex - 1)
    }
  }, [history, historyIndex, setNodes, setEdges])

  const redo = useCallback(() => {
    if (historyIndex < history.length - 1) {
      const nextState = history[historyIndex + 1]
      setNodes(nextState.nodes)
      setEdges(nextState.edges)
      setHistoryIndex(historyIndex + 1)
    }
  }, [history, historyIndex, setNodes, setEdges])

  // Handle connections with conditional logic support
  const onConnect = useCallback(
    (params: Connection) => {
      const newEdge = {
        ...params,
        type: 'conditional',
        id: `edge-${params.source}-${params.target}-${Date.now()}`,
        data: { condition: 'success' }
      }
      setEdges((eds) => addEdge(newEdge, eds))
      saveToHistory()
      validateWorkflowDesign()
    },
    [setEdges, saveToHistory]
  )

  // Handle node drag end
  const onNodeDragStop = useCallback(() => {
    saveToHistory()
  }, [saveToHistory])

  // Validate workflow design
  const validateWorkflowDesign = useCallback(async () => {
    setIsValidating(true)
    
    try {
      const actionNodes = nodes.filter(n => n.type === 'actionNode')
      const errors: string[] = []
      const warnings: string[] = []

      // Basic validation
      if (actionNodes.length === 0) {
        errors.push('Workflow must contain at least one action')
      }

      if (!workflowName.trim()) {
        errors.push('Workflow name is required')
      }

      if (!incidentId) {
        errors.push('Please select an incident')
      }

      // Check connectivity
      const connectedNodes = new Set<string>()
      edges.forEach(edge => {
        connectedNodes.add(edge.source)
        connectedNodes.add(edge.target)
      })

      const disconnectedActions = actionNodes.filter(node => !connectedNodes.has(node.id))
      if (disconnectedActions.length > 0) {
        warnings.push(`${disconnectedActions.length} action(s) are disconnected`)
      }

      // Check for cycles
      const hasCycles = detectCycles(nodes, edges)
      if (hasCycles) {
        errors.push('Workflow contains circular dependencies')
      }

      // Validate with backend if no basic errors
      if (errors.length === 0 && actionNodes.length > 0) {
        try {
          const steps = actionNodes.map(node => ({
            action_type: node.data.actionType,
            parameters: node.data.parameters || {}
          }))

          const validationResult = await validateWorkflow({ steps })
          
          if (validationResult.success && validationResult.validation) {
            if (!validationResult.validation.valid) {
              errors.push(...validationResult.validation.errors)
            }
            warnings.push(...validationResult.validation.warnings || [])
          }
        } catch (validationError) {
          warnings.push('Backend validation temporarily unavailable')
        }
      }

      const validationState = {
        valid: errors.length === 0,
        errors,
        warnings
      }

      setValidation(validationState)
      onValidationChange?.(validationState.valid, validationState.errors)

    } catch (error) {
      console.error('Validation error:', error)
      setValidation({
        valid: false,
        errors: ['Validation system error'],
        warnings: []
      })
    } finally {
      setIsValidating(false)
    }
  }, [nodes, edges, workflowName, incidentId, onValidationChange])

  // Re-validate when dependencies change
  useEffect(() => {
    const debounceTimer = setTimeout(validateWorkflowDesign, 500)
    return () => clearTimeout(debounceTimer)
  }, [validateWorkflowDesign])

  // Handle drop from node library
  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault()

      const nodeData = event.dataTransfer.getData('application/json')
      if (!nodeData) return

      try {
        const { type, data: nodeDataParsed } = JSON.parse(nodeData)

        const rect = (event.target as Element).closest('.react-flow')?.getBoundingClientRect()
        if (!rect) return

        const position = {
          x: event.clientX - rect.left - 100,
          y: event.clientY - rect.top - 50,
        }

        let newNode: Node

        switch (type) {
          case 'action':
            newNode = {
              id: `action-${Date.now()}`,
              type: 'actionNode',
              position,
              data: nodeDataParsed,
            }
            break
            
          case 'condition':
            newNode = {
              id: `condition-${Date.now()}`,
              type: 'conditionNode',
              position,
              data: { condition: 'if success', label: 'Condition' },
            }
            break
            
          case 'wait':
            newNode = {
              id: `wait-${Date.now()}`,
              type: 'waitNode',
              position,
              data: { duration: '30s', label: 'Wait' },
            }
            break
            
          case 'approval':
            newNode = {
              id: `approval-${Date.now()}`,
              type: 'approvalNode',
              position,
              data: { approver: 'manager', label: 'Approval Required' },
            }
            break
            
          default:
            return
        }

        setNodes((nds) => nds.concat(newNode))
        saveToHistory()
      } catch (error) {
        console.error('Error processing dropped node:', error)
      }
    },
    [setNodes, saveToHistory]
  )

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    event.dataTransfer.dropEffect = 'move'
  }, [])

  // Save workflow
  const saveWorkflow = async () => {
    if (!validation.valid || !incidentId) return

    setIsSaving(true)
    try {
      const actionNodes = nodes.filter(n => n.type === 'actionNode')
      const steps = actionNodes.map((node, index) => ({
        action_type: node.data.actionType,
        parameters: {
          target: incidentId,
          reason: `Canvas workflow: ${workflowName}`,
          step_order: index + 1,
          node_id: node.id,
          position: node.position
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
        onWorkflowSaved?.(result.workflow_id)
        alert(`Workflow "${workflowName}" saved successfully!`)
      } else {
        alert(`Failed to save workflow: ${result.error}`)
      }
    } catch (error) {
      console.error('Failed to save workflow:', error)
      alert('Failed to save workflow')
    } finally {
      setIsSaving(false)
    }
  }

  // Clear canvas
  const clearCanvas = () => {
    const defaultNodes: Node[] = [
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
    setNodes(defaultNodes)
    setEdges([])
    setWorkflowName('')
    saveToHistory()
  }

  // Get validation summary
  const validationSummary = useMemo(() => {
    const totalIssues = validation.errors.length + validation.warnings.length
    return {
      status: validation.valid ? 'valid' : 'invalid',
      totalIssues,
      severity: validation.errors.length > 0 ? 'error' : validation.warnings.length > 0 ? 'warning' : 'none'
    }
  }, [validation])

  return (
    <div className="h-full flex flex-col">
      {/* Canvas Toolbar */}
      <div className="flex items-center justify-between p-4 border-b bg-gray-50">
        <div className="flex items-center gap-2">
          {/* Canvas Controls */}
          <Button size="sm" variant="outline" onClick={undo} disabled={historyIndex <= 0}>
            <Undo className="h-3 w-3" />
          </Button>
          <Button size="sm" variant="outline" onClick={redo} disabled={historyIndex >= history.length - 1}>
            <Redo className="h-3 w-3" />
          </Button>
          
          <div className="w-px h-6 bg-gray-300 mx-2" />
          
          <Button size="sm" variant="outline" onClick={() => zoomIn()}>
            <ZoomIn className="h-3 w-3" />
          </Button>
          <Button size="sm" variant="outline" onClick={() => zoomOut()}>
            <ZoomOut className="h-3 w-3" />
          </Button>
          <Button size="sm" variant="outline" onClick={() => fitView()}>
            <Maximize className="h-3 w-3" />
          </Button>
          
          <div className="w-px h-6 bg-gray-300 mx-2" />
          
          <Button size="sm" variant="outline" onClick={clearCanvas}>
            <Trash2 className="h-3 w-3" />
            Clear
          </Button>
        </div>

        <div className="flex items-center gap-4">
          {/* Workflow Name */}
          <input
            type="text"
            value={workflowName}
            onChange={(e) => setWorkflowName(e.target.value)}
            placeholder="Enter workflow name..."
            className="px-3 py-1 border rounded text-sm min-w-48"
          />
          
          {/* Validation Status */}
          <div className="flex items-center gap-2">
            {isValidating ? (
              <div className="animate-spin h-4 w-4 border border-current border-t-transparent rounded-full"></div>
            ) : validationSummary.status === 'valid' ? (
              <CheckCircle className="h-4 w-4 text-green-500" />
            ) : (
              <AlertTriangle className="h-4 w-4 text-red-500" />
            )}
            
            <Badge 
              variant="outline" 
              className={
                validationSummary.severity === 'error' ? 'border-red-300 text-red-700' :
                validationSummary.severity === 'warning' ? 'border-yellow-300 text-yellow-700' :
                'border-green-300 text-green-700'
              }
            >
              {validationSummary.totalIssues === 0 ? 'Valid' : `${validationSummary.totalIssues} issues`}
            </Badge>
          </div>

          {/* Action Buttons */}
          <Button
            size="sm"
            onClick={saveWorkflow}
            disabled={!validation.valid || isSaving}
          >
            {isSaving ? (
              <div className="flex items-center gap-1">
                <div className="animate-spin h-3 w-3 border border-current border-t-transparent rounded-full"></div>
                Saving
              </div>
            ) : (
              <div className="flex items-center gap-1">
                <Save className="h-3 w-3" />
                Save
              </div>
            )}
          </Button>
        </div>
      </div>

      {/* Main Canvas Area */}
      <div className="flex-1 relative">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onNodeDragStop={onNodeDragStop}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          defaultEdgeOptions={{
            type: 'conditional',
            animated: canvasMode === 'simulate'
          }}
          className="bg-white"
          fitView
        >
          <Background variant={BackgroundVariant.Dots} />
          <Controls />
          <MiniMap 
            nodeColor={(node) => {
              switch (node.type) {
                case 'startNode': return '#10B981'
                case 'endNode': return '#EF4444'
                case 'actionNode': return '#3B82F6'
                case 'conditionNode': return '#F59E0B'
                case 'waitNode': return '#6B7280'
                case 'approvalNode': return '#8B5CF6'
                default: return '#6B7280'
              }
            }}
          />
          
          {/* Canvas Mode Selector */}
          <Panel position="top-left">
            <Card className="w-48">
              <CardContent className="p-3">
                <div className="text-xs font-medium mb-2">Canvas Mode</div>
                <div className="flex gap-1">
                  {[
                    { key: 'design', label: 'Design', icon: Settings },
                    { key: 'simulate', label: 'Simulate', icon: Play },
                    { key: 'execute', label: 'Execute', icon: Zap }
                  ].map(({ key, label, icon: Icon }) => (
                    <Button
                      key={key}
                      size="sm"
                      variant={canvasMode === key ? 'default' : 'outline'}
                      onClick={() => setCanvasMode(key as any)}
                      className="flex-1 text-xs"
                    >
                      <Icon className="h-3 w-3" />
                    </Button>
                  ))}
                </div>
              </CardContent>
            </Card>
          </Panel>

          {/* Node Palette */}
          <Panel position="top-right">
            <Card className="w-64">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Node Palette</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {[
                  { type: 'condition', label: 'Condition', icon: GitBranch, color: 'bg-yellow-100' },
                  { type: 'wait', label: 'Wait/Delay', icon: Pause, color: 'bg-gray-100' },
                  { type: 'approval', label: 'Approval', icon: Shield, color: 'bg-purple-100' }
                ].map(({ type, label, icon: Icon, color }) => (
                  <div
                    key={type}
                    draggable
                    onDragStart={(e) => {
                      e.dataTransfer.setData('application/json', JSON.stringify({
                        type,
                        data: { label }
                      }))
                    }}
                    className={`p-2 border rounded cursor-move hover:shadow-sm transition-shadow ${color}`}
                  >
                    <div className="flex items-center gap-2">
                      <Icon className="h-3 w-3" />
                      <span className="text-xs font-medium">{label}</span>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          </Panel>

          {/* Validation Panel */}
          {(validation.errors.length > 0 || validation.warnings.length > 0) && (
            <Panel position="bottom-left">
              <Card className="w-80">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm flex items-center gap-2">
                    {validation.errors.length > 0 ? (
                      <AlertTriangle className="h-4 w-4 text-red-500" />
                    ) : (
                      <AlertTriangle className="h-4 w-4 text-yellow-500" />
                    )}
                    Validation Issues
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  {validation.errors.map((error, index) => (
                    <div key={`error-${index}`} className="p-2 bg-red-50 border border-red-200 rounded text-xs">
                      <div className="flex items-center gap-1">
                        <AlertTriangle className="h-3 w-3 text-red-500" />
                        <span className="text-red-700 font-medium">Error:</span>
                      </div>
                      <div className="text-red-600 mt-1">{error}</div>
                    </div>
                  ))}
                  
                  {validation.warnings.map((warning, index) => (
                    <div key={`warning-${index}`} className="p-2 bg-yellow-50 border border-yellow-200 rounded text-xs">
                      <div className="flex items-center gap-1">
                        <AlertTriangle className="h-3 w-3 text-yellow-500" />
                        <span className="text-yellow-700 font-medium">Warning:</span>
                      </div>
                      <div className="text-yellow-600 mt-1">{warning}</div>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </Panel>
          )}

          {/* Canvas Stats */}
          <Panel position="bottom-right">
            <Card className="w-48">
              <CardContent className="p-3">
                <div className="text-xs font-medium mb-2">Canvas Stats</div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="text-center">
                    <div className="font-bold text-blue-600">
                      {nodes.filter(n => n.type === 'actionNode').length}
                    </div>
                    <div className="text-gray-500">Actions</div>
                  </div>
                  <div className="text-center">
                    <div className="font-bold text-green-600">{edges.length}</div>
                    <div className="text-gray-500">Connections</div>
                  </div>
                  <div className="text-center">
                    <div className="font-bold text-purple-600">
                      {nodes.filter(n => n.type === 'conditionNode').length}
                    </div>
                    <div className="text-gray-500">Conditions</div>
                  </div>
                  <div className="text-center">
                    <div className="font-bold text-orange-600">
                      {nodes.filter(n => n.type === 'approvalNode').length}
                    </div>
                    <div className="text-gray-500">Approvals</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </Panel>
        </ReactFlow>
      </div>
    </div>
  )
}

// Helper function to detect cycles in the workflow
function detectCycles(nodes: Node[], edges: Edge[]): boolean {
  const adjacencyList: Record<string, string[]> = {}
  const visited = new Set<string>()
  const recursionStack = new Set<string>()

  // Build adjacency list
  nodes.forEach(node => {
    adjacencyList[node.id] = []
  })

  edges.forEach(edge => {
    if (adjacencyList[edge.source]) {
      adjacencyList[edge.source].push(edge.target)
    }
  })

  // DFS to detect cycles
  function hasCycleDFS(nodeId: string): boolean {
    if (recursionStack.has(nodeId)) return true
    if (visited.has(nodeId)) return false

    visited.add(nodeId)
    recursionStack.add(nodeId)

    for (const neighbor of adjacencyList[nodeId] || []) {
      if (hasCycleDFS(neighbor)) return true
    }

    recursionStack.delete(nodeId)
    return false
  }

  // Check each node
  for (const node of nodes) {
    if (!visited.has(node.id)) {
      if (hasCycleDFS(node.id)) return true
    }
  }

  return false
}

export default PlaybookCanvas












