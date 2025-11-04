/**
 * Workflow Approval Panel Component
 * Provides interface for approving/rejecting response workflows
 */

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  CheckCircle,
  XCircle,
  Clock,
  AlertTriangle,
  User,
  Shield,
  Zap,
  Eye
} from 'lucide-react'
import { apiUrl } from '../utils/api'

interface WorkflowApprovalPanelProps {
  incidentId: number
}

interface WorkflowForApproval {
  id: number
  workflow_id: string
  playbook_name: string
  total_steps: number
  risk_level: string
  estimated_duration: number
  created_at: string
  steps: Array<{
    action_type: string
    parameters: Record<string, any>
    risk_level: string
  }>
  impact_assessment: {
    affected_systems: string[]
    reversibility: string
    risk_score: number
  }
}

export default function WorkflowApprovalPanel({ incidentId }: WorkflowApprovalPanelProps) {
  const [pendingWorkflows, setPendingWorkflows] = useState<WorkflowForApproval[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [processingId, setProcessingId] = useState<string | null>(null)

  // Load pending workflows
  useEffect(() => {
    loadPendingWorkflows()
  }, [incidentId])

  const loadPendingWorkflows = async () => {
    try {
      setLoading(true)
      const response = await fetch(apiUrl(`/api/response/workflows?status=awaiting_approval&incident_id=${incidentId}`), {
        headers: {
          'x-api-key': process.env.NEXT_PUBLIC_API_KEY || 'demo-minixdr-api-key'
        }
      })

      if (!response.ok) {
        throw new Error(`Failed to load workflows: ${response.statusText}`)
      }

      const data = await response.json()
      setPendingWorkflows(data.workflows || [])
      setError(null)
    } catch (err) {
      console.error('Failed to load pending workflows:', err)
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  const approveWorkflow = async (workflowId: string) => {
    try {
      setProcessingId(workflowId)

      const response = await fetch(apiUrl(`/api/response/workflows/${workflowId}/approve`), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': process.env.NEXT_PUBLIC_API_KEY || 'demo-minixdr-api-key'
        },
        body: JSON.stringify({
          approved_by: 'SOC Analyst',
          approval_reason: 'Reviewed and approved for execution'
        })
      })

      if (!response.ok) {
        throw new Error(`Approval failed: ${response.statusText}`)
      }

      await loadPendingWorkflows() // Refresh the list

    } catch (err) {
      console.error('Approval failed:', err)
      setError(err instanceof Error ? err.message : 'Approval failed')
    } finally {
      setProcessingId(null)
    }
  }

  const rejectWorkflow = async (workflowId: string) => {
    try {
      setProcessingId(workflowId)

      const response = await fetch(apiUrl(`/api/response/workflows/${workflowId}/reject`), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': process.env.NEXT_PUBLIC_API_KEY || 'demo-minixdr-api-key'
        },
        body: JSON.stringify({
          rejected_by: 'SOC Analyst',
          rejection_reason: 'Manual review determined action not necessary'
        })
      })

      if (!response.ok) {
        throw new Error(`Rejection failed: ${response.statusText}`)
      }

      await loadPendingWorkflows() // Refresh the list

    } catch (err) {
      console.error('Rejection failed:', err)
      setError(err instanceof Error ? err.message : 'Rejection failed')
    } finally {
      setProcessingId(null)
    }
  }

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'high': return 'bg-red-100 text-red-800'
      case 'medium': return 'bg-yellow-100 text-yellow-800'
      case 'low': return 'bg-green-100 text-green-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  if (loading) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock className="w-5 h-5 text-blue-400" />
            Workflow Approvals
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
            <span className="ml-3 text-gray-400">Loading pending approvals...</span>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Shield className="w-5 h-5 text-blue-400" />
          Workflow Approvals
          {pendingWorkflows.length > 0 && (
            <Badge variant="destructive" className="ml-2">
              {pendingWorkflows.length} Pending
            </Badge>
          )}
        </CardTitle>
        <CardDescription>
          Review and approve high-impact response workflows before execution
        </CardDescription>
      </CardHeader>
      <CardContent>
        {error && (
          <Alert className="mb-4">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {pendingWorkflows.length === 0 ? (
          <div className="text-center py-8 text-gray-400">
            <CheckCircle className="w-12 h-12 mx-auto mb-4 text-green-400" />
            <p>No workflows pending approval</p>
            <p className="text-sm">All response actions have been reviewed</p>
          </div>
        ) : (
          <ScrollArea className="h-96">
            <div className="space-y-4">
              {pendingWorkflows.map((workflow) => (
                <div key={workflow.workflow_id} className="border border-gray-700 rounded-lg p-4 bg-gray-800/50">
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <h4 className="font-semibold text-white mb-1">{workflow.playbook_name}</h4>
                      <p className="text-sm text-gray-400">Workflow ID: {workflow.workflow_id}</p>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge className={getRiskColor(workflow.risk_level)}>
                        {workflow.risk_level?.toUpperCase() || 'MEDIUM'} RISK
                      </Badge>
                      <Badge variant="outline">
                        {workflow.total_steps} Steps
                      </Badge>
                    </div>
                  </div>

                  <div className="mb-4">
                    <h5 className="text-sm font-medium text-gray-300 mb-2">Planned Actions:</h5>
                    <div className="space-y-2">
                      {workflow.steps?.map((step, idx) => (
                        <div key={idx} className="flex items-center gap-3 text-sm">
                          <div className="w-6 h-6 rounded-full bg-blue-600 text-white flex items-center justify-center text-xs">
                            {idx + 1}
                          </div>
                          <span className="text-gray-300">{step.action_type.replace(/_/g, ' ').toUpperCase()}</span>
                          <Badge size="sm" className={getRiskColor(step.risk_level || 'medium')}>
                            {step.risk_level || 'medium'}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </div>

                  {workflow.impact_assessment && (
                    <div className="mb-4 p-3 bg-gray-900/50 rounded border border-gray-600">
                      <h5 className="text-sm font-medium text-gray-300 mb-2">Impact Assessment:</h5>
                      <div className="grid grid-cols-2 gap-3 text-sm">
                        <div>
                          <span className="text-gray-400">Risk Score:</span>
                          <span className="ml-2 font-medium">{workflow.impact_assessment.risk_score || 'N/A'}</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Reversibility:</span>
                          <span className="ml-2 font-medium">{workflow.impact_assessment.reversibility || 'N/A'}</span>
                        </div>
                      </div>
                      {workflow.impact_assessment.affected_systems?.length > 0 && (
                        <div className="mt-2">
                          <span className="text-gray-400">Affected Systems:</span>
                          <span className="ml-2 text-sm">{workflow.impact_assessment.affected_systems.join(', ')}</span>
                        </div>
                      )}
                    </div>
                  )}

                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 text-sm text-gray-400">
                      <User className="w-4 h-4" />
                      <span>Created {new Date(workflow.created_at).toLocaleString()}</span>
                    </div>

                    <div className="flex items-center gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => rejectWorkflow(workflow.workflow_id)}
                        disabled={processingId === workflow.workflow_id}
                        className="border-red-600 text-red-400 hover:bg-red-600/10"
                      >
                        <XCircle className="w-4 h-4 mr-1" />
                        Reject
                      </Button>

                      <Button
                        size="sm"
                        onClick={() => approveWorkflow(workflow.workflow_id)}
                        disabled={processingId === workflow.workflow_id}
                        className="bg-green-600 hover:bg-green-700"
                      >
                        <CheckCircle className="w-4 h-4 mr-1" />
                        {processingId === workflow.workflow_id ? 'Processing...' : 'Approve & Execute'}
                      </Button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>
        )}
      </CardContent>
    </Card>
  )
}
