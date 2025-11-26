/**
 * Phase 4.1: Threat Data Integration Layer
 * Real-time data fetching and processing for 3D visualization components
 */

import { ThreatPoint, AttackPath, ThreatDataProcessor } from './three-helpers'
import { TimelineEvent } from '../app/visualizations/3d-timeline'

// API response interfaces
interface DistributedAPIResponse {
  status: string
  message: string
  data?: Record<string, unknown>
  timestamp: number
}

interface ThreatIntelResponse {
  threats: RawThreatData[]
  total_count: number
  last_updated: number
  sources: string[]
  real_data?: boolean
}

interface RawThreatData {
  id: string
  ip_address?: string
  latitude: number
  longitude: number
  country: string
  country_code: string
  threat_type: string
  confidence: number
  severity: number // 1-4 scale
  first_seen: number
  last_seen: number
  source: string
  tags?: string[]
  metadata?: {
    asn?: string
    isp?: string
    malware_family?: string
    attack_technique?: string
    mitre_attack_id?: string
  }
}

interface IncidentResponse {
  incidents: RawIncidentData[]
  total_count: number
  time_range: { start: number, end: number }
}

interface RawIncidentData {
  id: string
  timestamp: number
  title: string
  description: string
  severity: string
  status: string
  attack_vectors: string[]
  affected_assets: string[]
  source_ip?: string
  target_ip?: string
  location_data?: {
    source_country?: string
    target_country?: string
    source_lat?: number
    source_lng?: number
    target_lat?: number
    target_lng?: number
  }
  mitre_attack?: {
    technique_id: string
    technique_name: string
    tactic: string
  }
}

// Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || '${API_BASE}'
const DEFAULT_FETCH_INTERVAL = 30000 // 30 seconds
const MAX_RETRY_ATTEMPTS = 3
const RETRY_DELAY = 1000 // 1 second

// Validate configuration
if (!API_BASE_URL) {
  console.error('🚨 API_BASE_URL is not configured. Please set NEXT_PUBLIC_API_URL environment variable.')
}

console.log('🔧 ThreatDataService configuration:', {
  API_BASE_URL,
  MAX_RETRY_ATTEMPTS,
  RETRY_DELAY
})

// Threat data service class
export class ThreatDataService {
  private static instance: ThreatDataService
  private cache: Map<string, { data: any, timestamp: number }> = new Map()
  private subscriptions: Map<string, ((data: any) => void)[]> = new Map()
  private intervals: Map<string, NodeJS.Timeout> = new Map()

  private constructor() {}

  static getInstance(): ThreatDataService {
    if (!ThreatDataService.instance) {
      ThreatDataService.instance = new ThreatDataService()
    }
    return ThreatDataService.instance
  }

  /**
   * Fetch real-time threat intelligence data
   */
  async fetchThreatIntelligence(): Promise<ThreatPoint[]> {
    try {
      console.log('🔄 Fetching threat intelligence from:', `${API_BASE_URL}/api/intelligence/threats`)
      const response = await this.fetchWithRetry('/api/intelligence/threats')
      const data: ThreatIntelResponse = await response.json()

      console.log('✅ Received threat data:', {
        total: data.threats?.length || 0,
        sources: data.sources,
        realData: data.real_data,
        sample: data.threats?.slice(0, 2)
      })

      const processedThreats = data.threats
        .filter(threat => threat && threat.id) // Filter out invalid entries
        .map(threat => this.convertRawThreatToPoint(threat))
        .filter(threat => {
          // Only include threats with valid coordinates (not 0,0 which is center of globe)
          const hasValidCoords = threat.latitude !== 0 || threat.longitude !== 0
          if (!hasValidCoords) {
            console.log('⚠️ Skipping threat with invalid coordinates:', threat.id, threat.country)
          }
          return hasValidCoords
        })

      console.log('🎯 Processed threats:', {
        total: processedThreats.length,
        withCoords: processedThreats.filter(t => t.latitude !== 0 || t.longitude !== 0).length,
        sample: processedThreats.slice(0, 2)
      })

      // No test data - only show real honeypot data
      console.log('📊 Using only real threat data from honeypot')

      return processedThreats
    } catch (error) {
      console.error('❌ Failed to fetch threat intelligence:', error)
      console.log('📊 No fallback data - returning empty array until real honeypot data is available')
      return [] // Return empty array instead of mock data
    }
  }

  /**
   * Fetch distributed system status and threat data
   */
  async fetchDistributedThreats(): Promise<ThreatPoint[]> {
    try {
      console.log('🌐 Fetching distributed threats...')

      // Try both endpoints but handle individual failures gracefully
      const results = await Promise.allSettled([
        this.fetchWithRetry('/api/distributed/status'),
        this.fetchWithRetry('/api/intelligence/distributed-threats')
      ])

      let status = null
      let threats = null

      // Handle status endpoint result
      if (results[0].status === 'fulfilled') {
        try {
          status = await results[0].value.json()
          console.log('✅ Distributed status fetched successfully')
        } catch (e) {
          console.warn('⚠️ Failed to parse distributed status JSON:', e)
        }
      } else {
        console.warn('⚠️ Distributed status endpoint failed:', results[0].reason)
      }

      // Handle threats endpoint result
      if (results[1].status === 'fulfilled') {
        try {
          threats = await results[1].value.json()
          console.log('✅ Distributed threats fetched successfully')
        } catch (e) {
          console.warn('⚠️ Failed to parse distributed threats JSON:', e)
        }
      } else {
        console.warn('⚠️ Distributed threats endpoint failed:', results[1].reason)
      }

      // Return what we can, even if one endpoint failed
      if (!threats?.threats) {
        console.log('📦 No distributed threats available, returning empty array')
        return []
      }

      // Enhance threats with distributed node information if status is available
      return threats.threats.map((threat: RawThreatData) => {
        const baseThreat = this.convertRawThreatToPoint(threat)

        if (status?.active_nodes) {
          return {
            ...baseThreat,
            metadata: {
              ...threat.metadata,
              distributed_source: status.active_nodes.find((node: any) =>
                node.region === this.getRegionFromCountry(threat.country_code)
              )?.node_id
            }
          }
        }

        return baseThreat
      }) || []

    } catch (error) {
      console.error('❌ Failed to fetch distributed threats:', error)
      return []
    }
  }

  /**
   * Fetch incident timeline data
   */
  async fetchIncidentTimeline(timeRange?: { start: number, end: number }): Promise<TimelineEvent[]> {
    try {
      const params = new URLSearchParams()
      if (timeRange) {
        params.set('start_time', timeRange.start.toString())
        params.set('end_time', timeRange.end.toString())
      }

      const response = await this.fetchWithRetry(`/api/incidents/timeline?${params}`)
      const data: IncidentResponse = await response.json()

      return data.incidents.map(incident => this.convertIncidentToTimelineEvent(incident))
    } catch (error) {
      console.error('Failed to fetch incident timeline:', error)
      console.log('📊 No mock timeline data - returning empty array until real data is available')
      return [] // Return empty array instead of mock data
    }
  }

  /**
   * Fetch attack path data
   */
  async fetchAttackPaths(): Promise<AttackPath[]> {
    try {
      const response = await this.fetchWithRetry('/api/incidents/attack-paths')
      const data = await response.json()

      return data.attack_paths
        ?.filter((path: any) => path && path.id && path.source && path.target) // Filter out invalid entries
        .map((path: any) => ({
          id: path.id,
          source: this.convertRawThreatToPoint(path.source),
          target: this.convertRawThreatToPoint(path.target),
          progress: path.progress || 0,
          type: path.attack_type || 'lateral_movement',
          isActive: path.is_active || false
        })) || []

    } catch (error) {
      console.error('Failed to fetch attack paths:', error)
      return []
    }
  }

  /**
   * Fetch federated learning insights for threat intelligence
   */
  async fetchFederatedInsights(): Promise<{
    global_threats: ThreatPoint[],
    predictions: Record<string, unknown>[],
    model_performance: Record<string, unknown>
  }> {
    try {
      console.log('🧠 Fetching federated insights...')
      const response = await this.fetchWithRetry('/api/federated/insights')
      const data = await response.json()

      console.log('✅ Federated insights received:', {
        hasGlobalThreats: !!data.global_threat_landscape,
        threatCount: data.global_threat_landscape?.length || 0,
        hasPredictions: !!data.threat_predictions,
        predictionCount: data.threat_predictions?.length || 0,
        hasPerformance: !!data.model_performance
      })

      return {
        global_threats: data.global_threat_landscape?.map((threat: RawThreatData) =>
          this.convertRawThreatToPoint(threat)
        ) || [],
        predictions: data.threat_predictions || [],
        model_performance: data.model_performance || {}
      }
    } catch (error) {
      console.error('❌ Federated insights endpoint failed:', {
        error: error instanceof Error ? error.message : String(error),
        endpoint: '/api/federated/insights'
      })

      console.log('🔄 Falling back to empty federated data')
      return {
        global_threats: [],
        predictions: [],
        model_performance: {
          status: 'unavailable',
          reason: 'endpoint_failed',
          fallback: true
        }
      }
    }
  }

  /**
   * Subscribe to real-time updates
   */
  subscribeToUpdates(
    dataType: 'threats' | 'incidents' | 'attacks' | 'federated',
    callback: (data: ThreatPoint[] | TimelineEvent[] | AttackPath[] | Record<string, unknown>) => void,
    interval: number = DEFAULT_FETCH_INTERVAL
  ): () => void {
    if (!this.subscriptions.has(dataType)) {
      this.subscriptions.set(dataType, [])
    }

    this.subscriptions.get(dataType)!.push(callback)

    // Set up polling interval
    if (!this.intervals.has(dataType)) {
      const intervalId = setInterval(async () => {
        let data

        switch (dataType) {
          case 'threats':
            data = await this.fetchThreatIntelligence()
            break
          case 'incidents':
            data = await this.fetchIncidentTimeline()
            break
          case 'attacks':
            data = await this.fetchAttackPaths()
            break
          case 'federated':
            data = await this.fetchFederatedInsights()
            break
        }

        // Notify all subscribers
        this.subscriptions.get(dataType)?.forEach(cb => cb(data))
      }, interval)

      this.intervals.set(dataType, intervalId)
    }

    // Return unsubscribe function
    return () => {
      const subs = this.subscriptions.get(dataType)
      if (subs) {
        const index = subs.indexOf(callback)
        if (index > -1) {
          subs.splice(index, 1)
        }

        if (subs.length === 0) {
          const intervalId = this.intervals.get(dataType)
          if (intervalId) {
            clearInterval(intervalId)
            this.intervals.delete(dataType)
          }
          this.subscriptions.delete(dataType)
        }
      }
    }
  }

  /**
   * Convert raw API threat data to ThreatPoint format
   */
  private convertRawThreatToPoint(rawThreat: RawThreatData): ThreatPoint {
    if (!rawThreat) {
      console.warn('Missing rawThreat in convertRawThreatToPoint, using defaults')
      return {
        id: 'unknown-' + Date.now(),
        latitude: 0,
        longitude: 0,
        intensity: 0.5,
        type: 'exploit',
        country: 'Unknown',
        timestamp: Date.now(),
        details: {
          source: 'Unknown',
          confidence: 0,
          severity: 'medium'
        }
      }
    }

    return {
      id: rawThreat.id || 'unknown-' + Date.now(),
      latitude: rawThreat.latitude || 0,
      longitude: rawThreat.longitude || 0,
      intensity: Math.min((rawThreat.severity || 2) / 4, 1), // Normalize to 0-1
      type: this.mapThreatType(rawThreat.threat_type),
      country: rawThreat.country || 'Unknown',
      timestamp: rawThreat.last_seen || Date.now(),
      details: {
        source: rawThreat.source || 'Unknown',
        confidence: rawThreat.confidence || 0,
        severity: this.mapSeverityLevel(rawThreat.severity || 2)
      }
    }
  }

  /**
   * Convert incident data to timeline event
   */
  private convertIncidentToTimelineEvent(incident: RawIncidentData): TimelineEvent {
    return {
      id: incident.id,
      timestamp: incident.timestamp,
      type: this.mapIncidentType(incident.title),
      title: incident.title,
      description: incident.description,
      severity: incident.severity as 'low' | 'medium' | 'high' | 'critical',
      relatedThreats: this.extractThreatsFromIncident(incident),
      metadata: {
        source: incident.source_ip,
        target: incident.target_ip,
        technique: incident.mitre_attack?.technique_name,
        mitre_attack_id: incident.mitre_attack?.technique_id
      }
    }
  }

  /**
   * HTTP fetch with retry logic
   */
  private async fetchWithRetry(endpoint: string, attempts = 0): Promise<Response> {
    try {
      // Ensure API_BASE_URL is defined and normalize endpoint
      const baseUrl = API_BASE_URL.endsWith('/') ? API_BASE_URL.slice(0, -1) : API_BASE_URL
      const normalizedEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`
      const url = `${baseUrl}${normalizedEndpoint}`

      console.log(`🔄 Attempt ${attempts + 1}/${MAX_RETRY_ATTEMPTS + 1}: Fetching ${url}`)

      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': process.env.NEXT_PUBLIC_API_KEY || 'demo-minixdr-api-key',
        },
        signal: AbortSignal.timeout(10000) // 10 second timeout
      })

      console.log(`📡 Response: ${response.status} ${response.statusText} for ${endpoint}`)

      if (!response.ok) {
        const errorText = await response.text().catch(() => 'Unable to read error response')

        // Handle specific error types with more context
        if (response.status === 500) {
          throw new Error(`Server Error (500): The ${endpoint} endpoint encountered an internal error. This is likely due to missing dependencies or configuration issues on the backend.`)
        } else if (response.status === 404) {
          throw new Error(`Not Found (404): The ${endpoint} endpoint does not exist on the server.`)
        } else if (response.status === 403) {
          throw new Error(`Forbidden (403): Access denied to ${endpoint}. Check API authentication.`)
        }

        throw new Error(`HTTP ${response.status}: ${response.statusText} - ${errorText}`)
      }

      return response
    } catch (error) {
      // Enhanced error logging with more context
      if (error instanceof TypeError && error.message.includes('fetch')) {
        console.error(`🚨 Network Error for ${endpoint}:`, {
          possibleCauses: [
            'Backend server is not running',
            'CORS configuration issue',
            'Network connectivity problem',
            'Invalid URL configuration'
          ],
          endpoint,
          url: `${API_BASE_URL}${endpoint}`,
          attempt: attempts + 1
        })
      } else {
        console.error(`❌ Fetch failed for ${endpoint} (attempt ${attempts + 1}):`, {
          error: error instanceof Error ? error.message : String(error),
          endpoint,
          url: `${API_BASE_URL}${endpoint}`
        })
      }

      if (attempts < MAX_RETRY_ATTEMPTS) {
        const delay = RETRY_DELAY * Math.pow(2, attempts) // Exponential backoff
        console.log(`⏳ Retrying ${endpoint} in ${delay}ms...`)
        await new Promise(resolve => setTimeout(resolve, delay))
        return this.fetchWithRetry(endpoint, attempts + 1)
      }
      throw error
    }
  }

  /**
   * Map threat types from API to visualization types
   */
  private mapThreatType(apiType: string | undefined): ThreatPoint['type'] {
    // Handle missing or null values silently
    if (!apiType || apiType === 'undefined' || apiType === 'null') {
      return 'exploit' // Default without console warning
    }

    const typeMap: { [key: string]: ThreatPoint['type'] } = {
      'malware': 'malware',
      'botnet': 'botnet',
      'phishing': 'phishing',
      'ddos': 'ddos',
      'exploit': 'exploit',
      'brute_force': 'exploit',
      'sql_injection': 'exploit',
      'xss': 'exploit',
      'reconnaissance': 'exploit',
      'cowrie.login.failed': 'exploit',
      'cowrie.session.connect': 'exploit'
    }

    const cleanType = String(apiType).toLowerCase().trim()
    return typeMap[cleanType] || 'exploit'
  }

  /**
   * Map severity numbers to labels
   */
  private mapSeverityLevel(severity: number): 'low' | 'medium' | 'high' | 'critical' {
    if (severity >= 4) return 'critical'
    if (severity >= 3) return 'high'
    if (severity >= 2) return 'medium'
    return 'low'
  }

  /**
   * Map incident titles to timeline event types
   */
  private mapIncidentType(title: string): TimelineEvent['type'] {
    const lowerTitle = title.toLowerCase()

    if (lowerTitle.includes('blocked') || lowerTitle.includes('prevented')) {
      return 'attack_blocked'
    }
    if (lowerTitle.includes('lateral') || lowerTitle.includes('movement')) {
      return 'lateral_movement'
    }
    if (lowerTitle.includes('breach') || lowerTitle.includes('data')) {
      return 'data_breach'
    }
    if (lowerTitle.includes('mitig') || lowerTitle.includes('contain')) {
      return 'mitigation'
    }

    return 'threat_detected'
  }

  /**
   * Extract threat points from incident location data
   */
  private extractThreatsFromIncident(incident: RawIncidentData): ThreatPoint[] {
    const threats: ThreatPoint[] = []

    if (incident.location_data?.source_lat && incident.location_data?.source_lng) {
      threats.push({
        id: `${incident.id}_source`,
        latitude: incident.location_data.source_lat,
        longitude: incident.location_data.source_lng,
        intensity: this.getSeverityIntensity(incident.severity),
        type: 'exploit',
        country: incident.location_data.source_country || 'Unknown',
        timestamp: incident.timestamp,
        details: {
          source: incident.source_ip,
          confidence: 0.8,
          severity: incident.severity as 'low' | 'medium' | 'high' | 'critical'
        }
      })
    }

    return threats
  }

  /**
   * Get region from country code for distributed node mapping
   */
  private getRegionFromCountry(countryCode: string): string {
    const regionMap: { [key: string]: string } = {
      'US': 'us-west-1',
      'GB': 'eu-west-1',
      'DE': 'eu-central-1',
      'JP': 'ap-northeast-1',
      'CN': 'ap-east-1',
      'AU': 'ap-southeast-2',
      'BR': 'sa-east-1',
      'IN': 'ap-south-1'
    }

    return regionMap[countryCode] || 'us-east-1'
  }

  /**
   * Convert severity string to intensity number
   */
  private getSeverityIntensity(severity: string): number {
    const severityMap: { [key: string]: number } = {
      'low': 0.25,
      'medium': 0.5,
      'high': 0.75,
      'critical': 1.0
    }

    return severityMap[severity.toLowerCase()] || 0.5
  }

  /**
   * Generate mock data for development/fallback
   */
  private getMockThreatData(): ThreatPoint[] {
    console.log('🚨 Fallback to mock data - API might be failing')
    const mockThreats: ThreatPoint[] = []
    const countries = [
      { code: 'CN', name: 'China', lat: 35.8617, lng: 104.1954 },
      { code: 'RU', name: 'Russia', lat: 61.5240, lng: 105.3188 },
      { code: 'US', name: 'United States', lat: 39.8283, lng: -98.5795 },
      { code: 'KP', name: 'North Korea', lat: 40.3399, lng: 127.5101 },
      { code: 'IR', name: 'Iran', lat: 32.4279, lng: 53.6880 }
    ]

    const threatTypes: ThreatPoint['type'][] = ['malware', 'botnet', 'phishing', 'ddos', 'exploit']

    for (let i = 0; i < 20; i++) {
      const country = countries[Math.floor(Math.random() * countries.length)]
      const type = threatTypes[Math.floor(Math.random() * threatTypes.length)]

      mockThreats.push({
        id: `mock-threat-${i}`,
        latitude: country.lat + (Math.random() - 0.5) * 10,
        longitude: country.lng + (Math.random() - 0.5) * 10,
        intensity: Math.random(),
        type,
        country: country.name,
        timestamp: Date.now() - Math.random() * 3600000,
        details: {
          source: 'Mock Data',
          confidence: Math.random(),
          severity: ['low', 'medium', 'high', 'critical'][Math.floor(Math.random() * 4)] as 'low' | 'medium' | 'high' | 'critical'
        }
      })
    }

    return mockThreats
  }

  /**
   * Generate mock timeline data
   */
  private getMockTimelineData(): TimelineEvent[] {
    const events: TimelineEvent[] = []
    const now = Date.now()
    const eventTypes: TimelineEvent['type'][] = [
      'threat_detected', 'attack_blocked', 'lateral_movement', 'data_breach', 'mitigation'
    ]

    for (let i = 0; i < 15; i++) {
      const type = eventTypes[Math.floor(Math.random() * eventTypes.length)]

      events.push({
        id: `mock-event-${i}`,
        timestamp: now - Math.random() * 7200000, // Last 2 hours
        type,
        title: `${type.replace('_', ' ').toUpperCase()} Event`,
        description: `Mock ${type} event for demonstration`,
        severity: ['low', 'medium', 'high', 'critical'][Math.floor(Math.random() * 4)] as 'low' | 'medium' | 'high' | 'critical',
        relatedThreats: [],
        metadata: {
          source: '192.168.1.' + Math.floor(Math.random() * 255),
          technique: 'T10' + Math.floor(Math.random() * 99).toString().padStart(2, '0')
        }
      })
    }

    return events.sort((a, b) => a.timestamp - b.timestamp)
  }

  /**
   * Cleanup resources
   */
  dispose(): void {
    this.intervals.forEach(interval => clearInterval(interval))
    this.intervals.clear()
    this.subscriptions.clear()
    this.cache.clear()
  }
}

// Export singleton instance
export const threatDataService = ThreatDataService.getInstance()

// Utility hooks for React components
export const useThreatData = () => {
  return {
    fetchThreats: () => threatDataService.fetchThreatIntelligence(),
    fetchDistributedThreats: () => threatDataService.fetchDistributedThreats(),
    fetchTimeline: (timeRange?: { start: number, end: number }) =>
      threatDataService.fetchIncidentTimeline(timeRange),
    fetchAttackPaths: () => threatDataService.fetchAttackPaths(),
    fetchFederatedInsights: () => threatDataService.fetchFederatedInsights(),
    subscribe: (
      dataType: 'threats' | 'incidents' | 'attacks' | 'federated',
      callback: (data: ThreatPoint[] | TimelineEvent[] | AttackPath[] | Record<string, unknown>) => void,
      interval?: number
    ) => threatDataService.subscribeToUpdates(dataType, callback, interval)
  }
}
