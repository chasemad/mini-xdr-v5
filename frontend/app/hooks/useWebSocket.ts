/**
 * WebSocket Hook for Real-Time Updates
 * Provides real-time communication with the backend for workflow and incident updates
 */

import { useEffect, useRef, useState, useCallback } from 'react'
import { resolveApiBaseUrl } from '@/app/utils/api'

interface WebSocketMessage {
  type: string
  data?: any
  timestamp?: number
}

interface UseWebSocketOptions {
  onMessage?: (message: WebSocketMessage) => void
  onConnect?: () => void
  onDisconnect?: () => void
  onError?: (error: Event) => void
  reconnectInterval?: number
  maxReconnectAttempts?: number
}

interface UseWebSocketReturn {
  isConnected: boolean
  error: string | null
  sendMessage: (message: any) => void
  disconnect: () => void
  reconnect: () => void
}

export function useWebSocket(url: string, options: UseWebSocketOptions = {}): UseWebSocketReturn {
  const {
    onMessage,
    onConnect,
    onDisconnect,
    onError,
    reconnectInterval = 3000,
    maxReconnectAttempts = 5
  } = options

  const [isConnected, setIsConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectAttemptsRef = useRef(0)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)

  // Store callbacks in refs to avoid dependency changes
  const onMessageRef = useRef(onMessage)
  const onConnectRef = useRef(onConnect)
  const onDisconnectRef = useRef(onDisconnect)
  const onErrorRef = useRef(onError)

  // Update refs when callbacks change
  useEffect(() => {
    onMessageRef.current = onMessage
    onConnectRef.current = onConnect
    onDisconnectRef.current = onDisconnect
    onErrorRef.current = onError
  })

  const connect = useCallback(() => {
    try {
      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onopen = () => {
        setIsConnected(true)
        setError(null)
        reconnectAttemptsRef.current = 0
        onConnectRef.current?.()
      }

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data)
          onMessageRef.current?.(message)
        } catch (err) {
          console.error('Failed to parse WebSocket message:', event.data)
        }
      }

      ws.onclose = (event) => {
        setIsConnected(false)
        wsRef.current = null
        onDisconnectRef.current?.()

        // Auto-reconnect if not intentionally closed
        if (event.code !== 1000 && reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current += 1
          setError(`Connection lost. Reconnecting... (${reconnectAttemptsRef.current}/${maxReconnectAttempts})`)

          reconnectTimeoutRef.current = setTimeout(() => {
            connect()
          }, reconnectInterval)
        } else if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
          setError('Max reconnection attempts reached. Please refresh the page.')
        }
      }

      ws.onerror = (event) => {
        setError('WebSocket connection error')
        onErrorRef.current?.(event)
      }

    } catch (err) {
      setError('Failed to establish WebSocket connection')
      console.error('WebSocket connection failed:', err)
    }
  }, [url, reconnectInterval, maxReconnectAttempts])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }

    if (wsRef.current) {
      wsRef.current.close(1000, 'Intentional disconnect')
      wsRef.current = null
    }

    setIsConnected(false)
    setError(null)
  }, [])

  const reconnect = useCallback(() => {
    disconnect()
    reconnectAttemptsRef.current = 0
    connect()
  }, [connect, disconnect])

  const sendMessage = useCallback((message: any) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(typeof message === 'string' ? message : JSON.stringify(message))
    } else {
      console.warn('WebSocket is not connected')
    }
  }, [])

  // Initialize connection on mount
  useEffect(() => {
    connect()

    // Cleanup on unmount
    return () => {
      disconnect()
    }
  }, []) // Remove dependencies to prevent reconnection loops

  // Ping-pong for connection health
  useEffect(() => {
    if (isConnected) {
      const pingInterval = setInterval(() => {
        sendMessage('ping')
      }, 30000) // Ping every 30 seconds

      return () => clearInterval(pingInterval)
    }
  }, [isConnected, sendMessage])

  return {
    isConnected,
    error,
    sendMessage,
    disconnect,
    reconnect
  }
}

/**
 * Specialized hook for workflow updates
 */
const buildWebSocketUrl = (path: string): string => {
  const rawBase = resolveApiBaseUrl()
  const baseToUse = rawBase === null ? 'http://localhost:8000' : rawBase
  const normalizedBase = baseToUse.trim().replace(/\/+$/, '')
  const normalizedPath = path.startsWith('/') ? path : `/${path}`

  if (!normalizedBase) {
    return normalizedPath
  }

  const wsBase = normalizedBase
    .replace(/^https:/, 'wss:')
    .replace(/^http:/, 'ws:')

  return `${wsBase}${normalizedPath}`
}

export function useWorkflowWebSocket(options: UseWebSocketOptions = {}) {
  const wsUrl = buildWebSocketUrl('/ws/workflows')

  return useWebSocket(wsUrl, options)
}

/**
 * Specialized hook for incident updates
 */
export function useIncidentWebSocket(options: UseWebSocketOptions = {}) {
  const wsUrl = buildWebSocketUrl('/ws/incidents')

  return useWebSocket(wsUrl, options)
}

/**
 * General purpose WebSocket hook
 */
export function useGeneralWebSocket(options: UseWebSocketOptions = {}) {
  const wsUrl = buildWebSocketUrl('/ws/general')

  return useWebSocket(wsUrl, options)
}
