'use client'

/**
 * Global Application Context
 * Manages application-wide state for workflows, incidents, and real-time updates
 */

import React, { createContext, useContext, useReducer, useEffect, ReactNode } from 'react'
// import { useWorkflowWebSocket, useIncidentWebSocket } from '../hooks/useWebSocket'

// Types
interface Incident {
  id: number
  src_ip: string
  reason: string
  status: string
  created_at: string
  escalation_level?: string
  risk_score?: number
}

interface Workflow {
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
  approval_required?: boolean
  auto_executed?: boolean
}

interface AppState {
  incidents: Incident[]
  workflows: Workflow[]
  selectedIncident: number | null
  selectedWorkflow: string | null
  loading: {
    incidents: boolean
    workflows: boolean
  }
  errors: {
    incidents: string | null
    workflows: string | null
  }
  websocket: {
    connected: boolean
    error: string | null
  }
  lastUpdated: {
    incidents: number | null
    workflows: number | null
  }
}

// Actions
type AppAction =
  | { type: 'SET_INCIDENTS'; payload: Incident[] }
  | { type: 'ADD_INCIDENT'; payload: Incident }
  | { type: 'UPDATE_INCIDENT'; payload: Partial<Incident> & { id: number } }
  | { type: 'SET_WORKFLOWS'; payload: Workflow[] }
  | { type: 'ADD_WORKFLOW'; payload: Workflow }
  | { type: 'UPDATE_WORKFLOW'; payload: Partial<Workflow> & { workflow_id: string } }
  | { type: 'SET_SELECTED_INCIDENT'; payload: number | null }
  | { type: 'SET_SELECTED_WORKFLOW'; payload: string | null }
  | { type: 'SET_LOADING'; payload: { type: 'incidents' | 'workflows'; loading: boolean } }
  | { type: 'SET_ERROR'; payload: { type: 'incidents' | 'workflows'; error: string | null } }
  | { type: 'SET_WEBSOCKET_STATUS'; payload: { connected: boolean; error: string | null } }
  | { type: 'SET_LAST_UPDATED'; payload: { type: 'incidents' | 'workflows'; timestamp: number } }

// Initial state
const initialState: AppState = {
  incidents: [],
  workflows: [],
  selectedIncident: null,
  selectedWorkflow: null,
  loading: {
    incidents: false,
    workflows: false
  },
  errors: {
    incidents: null,
    workflows: null
  },
  websocket: {
    connected: false,
    error: null
  },
  lastUpdated: {
    incidents: null,
    workflows: null
  }
}

// Reducer
function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'SET_INCIDENTS':
      // CRITICAL FIX: Only create new state if data actually changed
      // This prevents unnecessary re-renders when polling returns same data
      if (state.incidents === action.payload) {
        console.log('[AppContext] SET_INCIDENTS: Data unchanged, returning same state')
        return state  // Return same object reference = no re-render
      }
      console.log('[AppContext] SET_INCIDENTS: Data changed, creating new state')
      return {
        ...state,
        incidents: action.payload,
        lastUpdated: { ...state.lastUpdated, incidents: Date.now() }
      }

    case 'ADD_INCIDENT':
      console.log('[AppContext] ADD_INCIDENT: Adding new incident')
      return {
        ...state,
        incidents: [action.payload, ...state.incidents],
        lastUpdated: { ...state.lastUpdated, incidents: Date.now() }
      }

    case 'UPDATE_INCIDENT':
      console.log('[AppContext] UPDATE_INCIDENT: Updating incident', action.payload.id)
      return {
        ...state,
        incidents: state.incidents.map(incident =>
          incident.id === action.payload.id
            ? { ...incident, ...action.payload }
            : incident
        ),
        lastUpdated: { ...state.lastUpdated, incidents: Date.now() }
      }

    case 'SET_WORKFLOWS':
      // CRITICAL FIX: Only create new state if data actually changed
      if (state.workflows === action.payload) {
        console.log('[AppContext] SET_WORKFLOWS: Data unchanged, returning same state')
        return state  // Return same object reference = no re-render
      }
      console.log('[AppContext] SET_WORKFLOWS: Data changed, creating new state')
      return {
        ...state,
        workflows: action.payload,
        lastUpdated: { ...state.lastUpdated, workflows: Date.now() }
      }

    case 'ADD_WORKFLOW':
      return {
        ...state,
        workflows: [action.payload, ...state.workflows],
        lastUpdated: { ...state.lastUpdated, workflows: Date.now() }
      }

    case 'UPDATE_WORKFLOW':
      return {
        ...state,
        workflows: state.workflows.map(workflow =>
          workflow.workflow_id === action.payload.workflow_id
            ? { ...workflow, ...action.payload }
            : workflow
        ),
        lastUpdated: { ...state.lastUpdated, workflows: Date.now() }
      }

    case 'SET_SELECTED_INCIDENT':
      return { ...state, selectedIncident: action.payload }

    case 'SET_SELECTED_WORKFLOW':
      return { ...state, selectedWorkflow: action.payload }

    case 'SET_LOADING': {
      const { type, loading } = action.payload
      if (state.loading[type] === loading) {
        return state
      }
      return {
        ...state,
        loading: { ...state.loading, [type]: loading }
      }
    }

    case 'SET_ERROR': {
      const { type, error } = action.payload
      if (state.errors[type] === error) {
        return state
      }
      return {
        ...state,
        errors: { ...state.errors, [type]: error }
      }
    }

    case 'SET_WEBSOCKET_STATUS':
      return {
        ...state,
        websocket: action.payload
      }

    case 'SET_LAST_UPDATED':
      return {
        ...state,
        lastUpdated: { ...state.lastUpdated, [action.payload.type]: action.payload.timestamp }
      }

    default:
      return state
  }
}

// Context
const AppContext = createContext<{
  state: AppState
  dispatch: React.Dispatch<AppAction>
} | null>(null)

// Provider
interface AppProviderProps {
  children: ReactNode
}

export function AppProvider({ children }: AppProviderProps) {
  const [state, dispatch] = useReducer(appReducer, initialState)

  // WebSocket connections temporarily disabled to fix connection leak
  /*
  const workflowWS = useWorkflowWebSocket({
    onMessage: (message) => {
      // ... workflow message handling
    },
    onConnect: () => {
      dispatch({
        type: 'SET_WEBSOCKET_STATUS',
        payload: { connected: true, error: null }
      })
    },
    onDisconnect: () => {
      dispatch({
        type: 'SET_WEBSOCKET_STATUS',
        payload: { connected: false, error: null }
      })
    },
    onError: () => {
      dispatch({
        type: 'SET_WEBSOCKET_STATUS',
        payload: { connected: false, error: 'WebSocket connection failed' }
      })
    }
  })

  const incidentWS = useIncidentWebSocket({
    onMessage: (message) => {
      // ... incident message handling
    }
  })
  */

  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  )
}

// Hook to use the context
export function useAppContext() {
  const context = useContext(AppContext)
  if (!context) {
    throw new Error('useAppContext must be used within an AppProvider')
  }
  return context
}

// Action creators for common operations
export const appActions = {
  setIncidents: (incidents: Incident[]) => ({ type: 'SET_INCIDENTS' as const, payload: incidents }),
  addIncident: (incident: Incident) => ({ type: 'ADD_INCIDENT' as const, payload: incident }),
  updateIncident: (incident: Partial<Incident> & { id: number }) => ({ type: 'UPDATE_INCIDENT' as const, payload: incident }),
  setWorkflows: (workflows: Workflow[]) => ({ type: 'SET_WORKFLOWS' as const, payload: workflows }),
  addWorkflow: (workflow: Workflow) => ({ type: 'ADD_WORKFLOW' as const, payload: workflow }),
  updateWorkflow: (workflow: Partial<Workflow> & { workflow_id: string }) => ({ type: 'UPDATE_WORKFLOW' as const, payload: workflow }),
  setSelectedIncident: (id: number | null) => ({ type: 'SET_SELECTED_INCIDENT' as const, payload: id }),
  setSelectedWorkflow: (id: string | null) => ({ type: 'SET_SELECTED_WORKFLOW' as const, payload: id }),
  setLoading: (type: 'incidents' | 'workflows', loading: boolean) => ({ type: 'SET_LOADING' as const, payload: { type, loading } }),
  setError: (type: 'incidents' | 'workflows', error: string | null) => ({ type: 'SET_ERROR' as const, payload: { type, error } }),
}
