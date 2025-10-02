/**
 * Data Service
 * Centralized service for data management with real-time updates
 */

import { getIncidents, listResponseWorkflows, createResponseWorkflow, executeResponseWorkflow } from '../lib/api'

interface DataServiceCallbacks {
  onIncidentsUpdate?: (incidents: any[]) => void
  onWorkflowsUpdate?: (workflows: any[]) => void
  onError?: (error: string, type: 'incidents' | 'workflows') => void
  onLoading?: (loading: boolean, type: 'incidents' | 'workflows') => void
}

export class DataService {
  private callbacks: DataServiceCallbacks = {}
  private refreshInterval: NodeJS.Timeout | null = null
  private isRefreshing = false
  private lastIncidents: any[] = []
  private lastWorkflows: any[] = []
  private incidentsLoaded = false
  private workflowsLoaded = false

  constructor(callbacks: DataServiceCallbacks = {}) {
    this.callbacks = callbacks
  }

  /**
   * Compare two arrays to check if data has changed significantly
   * Uses intelligent comparison to prevent unnecessary UI updates
   */
  private hasDataChanged(oldData: any[], newData: any[], dataType: 'incidents' | 'workflows'): boolean {
    // Length change is always significant
    if (oldData.length !== newData.length) {
      console.log(`[DataService] ${dataType} count changed: ${oldData.length} → ${newData.length}`)
      return true
    }

    // Empty arrays are equal
    if (oldData.length === 0) return false

    // Create maps by ID for efficient comparison
    const oldMap = new Map()
    const newMap = new Map()

    if (dataType === 'workflows') {
      oldData.forEach(item => oldMap.set(item.workflow_id || item.id, item))
      newData.forEach(item => newMap.set(item.workflow_id || item.id, item))

      // Check each workflow for significant changes
      for (const [id, newItem] of newMap) {
        const oldItem = oldMap.get(id)

        // New workflow appeared
        if (!oldItem) {
          console.log(`[DataService] New workflow detected: ${id}`)
          return true
        }

        // Check for significant changes only
        if (oldItem.status !== newItem.status) {
          console.log(`[DataService] Workflow ${id} status changed: ${oldItem.status} → ${newItem.status}`)
          return true
        }

        if (oldItem.current_step !== newItem.current_step) {
          console.log(`[DataService] Workflow ${id} step changed: ${oldItem.current_step} → ${newItem.current_step}`)
          return true
        }

        // Only update if progress changed by more than 10% to reduce churn
        const oldProgress = oldItem.progress_percentage || 0
        const newProgress = newItem.progress_percentage || 0
        const progressDiff = Math.abs(oldProgress - newProgress)

        if (progressDiff > 10) {
          console.log(`[DataService] Workflow ${id} progress changed significantly: ${oldProgress.toFixed(1)}% → ${newProgress.toFixed(1)}%`)
          return true
        }
      }

      // Check if any workflow was removed
      for (const [id] of oldMap) {
        if (!newMap.has(id)) {
          console.log(`[DataService] Workflow ${id} was removed`)
          return true
        }
      }

      console.log(`[DataService] No significant workflow changes detected`)
      return false
    }

    if (dataType === 'incidents') {
      oldData.forEach(item => oldMap.set(item.id, item))
      newData.forEach(item => newMap.set(item.id, item))

      // Check each incident for changes
      for (const [id, newItem] of newMap) {
        const oldItem = oldMap.get(id)

        // New incident appeared
        if (!oldItem) {
          console.log(`[DataService] New incident detected: ${id}`)
          return true
        }

        // Check for status changes
        if (oldItem.status !== newItem.status) {
          console.log(`[DataService] Incident ${id} status changed: ${oldItem.status} → ${newItem.status}`)
          return true
        }

        // Check for escalation level changes
        if (oldItem.escalation_level !== newItem.escalation_level) {
          console.log(`[DataService] Incident ${id} escalation changed: ${oldItem.escalation_level} → ${newItem.escalation_level}`)
          return true
        }

        // Check for significant risk score changes (more than 0.1 difference)
        const oldRisk = oldItem.risk_score || 0
        const newRisk = newItem.risk_score || 0
        if (Math.abs(oldRisk - newRisk) > 0.1) {
          console.log(`[DataService] Incident ${id} risk score changed: ${oldRisk} → ${newRisk}`)
          return true
        }
      }

      // Check if any incident was removed
      for (const [id] of oldMap) {
        if (!newMap.has(id)) {
          console.log(`[DataService] Incident ${id} was removed`)
          return true
        }
      }

      console.log(`[DataService] No significant incident changes detected`)
      return false
    }

    // Fallback: shouldn't reach here
    return false
  }

  /**
   * Load initial data
   */
  async loadInitialData() {
    await Promise.all([
      this.refreshIncidents(),
      this.refreshWorkflows()
    ])
  }

  /**
   * Refresh incidents from API
   */
  async refreshIncidents() {
    const shouldShowLoading = !this.incidentsLoaded
    try {
      if (shouldShowLoading) {
        this.callbacks.onLoading?.(true, 'incidents')
      }
      const response = await getIncidents()

      let incidents: any[] = []

      // Handle response - incidents endpoint returns array directly
      if (Array.isArray(response)) {
        incidents = response
      } else if (response.success) {
        incidents = response.incidents || []
      } else {
        throw new Error(response.error || 'Failed to fetch incidents')
      }

      // Only update if data has changed significantly
      const hasChanged = this.hasDataChanged(this.lastIncidents, incidents, 'incidents')
      console.log('[DataService] refreshIncidents - hasChanged:', hasChanged)
      
      this.lastIncidents = incidents
      this.incidentsLoaded = true

      if (hasChanged) {
        console.log('[DataService] ✅ Calling onIncidentsUpdate callback')
        this.callbacks.onIncidentsUpdate?.(incidents)
      } else {
        console.log('[DataService] ⏭️  Skipping onIncidentsUpdate - no significant changes')
      }

      this.callbacks.onError?.(null, 'incidents')
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      this.callbacks.onError?.(errorMessage, 'incidents')
    } finally {
      if (shouldShowLoading) {
        this.callbacks.onLoading?.(false, 'incidents')
      }
    }
  }

  /**
   * Refresh workflows from API
   */
  async refreshWorkflows() {
    const shouldShowLoading = !this.workflowsLoaded
    try {
      if (shouldShowLoading) {
        this.callbacks.onLoading?.(true, 'workflows')
      }
      const response = await listResponseWorkflows()

      let workflows: any[] = []

      // Handle response - workflows endpoint returns {success, workflows}
      if (response.success) {
        workflows = response.workflows || []
      } else if (Array.isArray(response)) {
        // Fallback: handle array response
        workflows = response
      } else {
        throw new Error(response.error || 'Failed to fetch workflows')
      }

      // Only update if data has changed significantly
      const hasChanged = this.hasDataChanged(this.lastWorkflows, workflows, 'workflows')
      console.log('[DataService] refreshWorkflows - hasChanged:', hasChanged)
      
      this.lastWorkflows = workflows
      this.workflowsLoaded = true

      if (hasChanged) {
        console.log('[DataService] ✅ Calling onWorkflowsUpdate callback')
        this.callbacks.onWorkflowsUpdate?.(workflows)
      } else {
        console.log('[DataService] ⏭️  Skipping onWorkflowsUpdate - no significant changes')
      }

      this.callbacks.onError?.(null, 'workflows')
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      this.callbacks.onError?.(errorMessage, 'workflows')
    } finally {
      if (shouldShowLoading) {
        this.callbacks.onLoading?.(false, 'workflows')
      }
    }
  }

  /**
   * Create a new workflow
   */
  async createWorkflow(workflowData: {
    incident_id: number
    playbook_name: string
    steps: any[]
    auto_execute?: boolean
    priority?: string
  }) {
    try {
      const response = await createResponseWorkflow(workflowData)

      if (response.success) {
        // Refresh workflows to get the latest data
        await this.refreshWorkflows()
        return response
      } else {
        throw new Error(response.error || 'Failed to create workflow')
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      this.callbacks.onError?.(errorMessage, 'workflows')
      throw error
    }
  }

  /**
   * Execute a workflow
   */
  async executeWorkflow(workflowDbId: number, executedBy = 'analyst') {
    try {
      const response = await executeResponseWorkflow({
        workflow_db_id: workflowDbId,
        executed_by: executedBy
      })

      if (response.success) {
        // Refresh workflows to get updated status
        await this.refreshWorkflows()
        return response
      } else {
        throw new Error(response.error || 'Failed to execute workflow')
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      this.callbacks.onError?.(errorMessage, 'workflows')
      throw error
    }
  }

  /**
   * Start periodic refresh (fallback for when WebSocket is not available)
   *
   * Polling is necessary because WebSocket is currently disabled. This ensures:
   * - New incidents are detected and displayed
   * - Workflow progress and status updates are shown
   * - UI stays in sync with backend state
   *
   * Default interval: 45s with intelligent change detection that only triggers
   * UI updates when significant changes occur (new items, status changes, >10% progress)
   * This prevents UI disruption from minor progress updates.
   */
  startPeriodicRefresh(intervalMs = 45000) {
    if (this.refreshInterval) {
      this.stopPeriodicRefresh()
    }

    this.refreshInterval = setInterval(async () => {
      if (!this.isRefreshing) {
        this.isRefreshing = true
        try {
          await Promise.all([
            this.refreshIncidents(),
            this.refreshWorkflows()
          ])
        } catch (error) {
          console.error('Periodic refresh failed:', error)
        } finally {
          this.isRefreshing = false
        }
      }
    }, intervalMs)
  }

  /**
   * Stop periodic refresh
   */
  stopPeriodicRefresh() {
    if (this.refreshInterval) {
      clearInterval(this.refreshInterval)
      this.refreshInterval = null
    }
  }

  /**
   * Update callbacks
   */
  updateCallbacks(callbacks: Partial<DataServiceCallbacks>) {
    this.callbacks = { ...this.callbacks, ...callbacks }
  }

  /**
   * Cleanup
   */
  cleanup() {
    this.stopPeriodicRefresh()
    this.callbacks = {}
  }
}

/**
 * Factory function to create a data service
 */
export function createDataService(callbacks: DataServiceCallbacks = {}) {
  return new DataService(callbacks)
}
