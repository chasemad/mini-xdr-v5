'use client'

/**
 * Phase 4.1: 3D Visualization Dashboard
 * Immersive cybersecurity threat visualization with real-time data integration
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/src/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/src/components/ui/tabs"
import { Button } from "@/src/components/ui/button"
import { Badge } from "@/src/components/ui/badge"
import { Progress } from "@/src/components/ui/progress"
import { ScrollArea } from "@/src/components/ui/scroll-area"
import dynamic from 'next/dynamic'

// Dynamic imports for 3D components to prevent SSR issues
const ThreatGlobe = dynamic(() => import('./threat-globe'), { 
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-96 bg-gray-900 text-white">
      <div className="text-center">
        <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-4"></div>
        <div>Loading 3D Threat Globe...</div>
      </div>
    </div>
  )
})

const Attack3DTimeline = dynamic(() => import('./3d-timeline'), { 
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-96 bg-gray-900 text-white">
      <div className="text-center">
        <div className="animate-spin w-8 h-8 border-4 border-purple-500 border-t-transparent rounded-full mx-auto mb-4"></div>
        <div>Loading 3D Timeline...</div>
      </div>
    </div>
  )
})

import { ThreatPoint, AttackPath, PerformanceMetrics } from '../../lib/three-helpers'
import { TimelineEvent } from './3d-timeline'
import { threatDataService, useThreatData } from '../../lib/threat-data'
import { 
  Globe, 
  Timeline, 
  Activity, 
  Shield, 
  AlertTriangle, 
  TrendingUp,
  Settings,
  Maximize2,
  RotateCcw,
  Pause,
  Play,
  Zap,
  BarChart3,
  Target,
  Search,
  Brain,
  Users,
  FileText,
  Home,
  Menu,
  X
} from 'lucide-react'
import Link from 'next/link'

// Dashboard state interface
interface DashboardState {
  threats: ThreatPoint[]
  attackPaths: AttackPath[]
  timelineEvents: TimelineEvent[]
  federatedInsights: any
  isLoading: {
    threats: boolean
    timeline: boolean
    attacks: boolean
    federated: boolean
  }
  error: string | null
  lastUpdated: number
}

// Performance and statistics interface
interface DashboardStats {
  totalThreats: number
  criticalThreats: number
  activeAttacks: number
  blockedAttacks: number
  globalCoverage: number
  averageResponseTime: number
  performance: PerformanceMetrics | null
}

// Visualization settings
interface VisualizationSettings {
  globeOpacity: number
  autoRotate: boolean
  showCountryLabels: boolean
  showCountryOutlines: boolean
  showAttackPaths: boolean
  showPerformanceHUD: boolean
  animationSpeed: number
  updateInterval: number
  threatFilter: 'all' | 'critical' | 'high' | 'medium' | 'low'
  regionFilter: string
}

const VisualizationDashboard: React.FC = () => {
  // State management
  const [dashboardState, setDashboardState] = useState<DashboardState>({
    threats: [],
    attackPaths: [],
    timelineEvents: [],
    federatedInsights: {},
    isLoading: {
      threats: true,
      timeline: true,
      attacks: true,
      federated: true
    },
    error: null,
    lastUpdated: 0
  })

  const [settings, setSettings] = useState<VisualizationSettings>({
    globeOpacity: 0.8,
    autoRotate: true,
    showCountryLabels: false,  // Disable initially to debug
    showCountryOutlines: false,  // Disable initially to prevent crash
    showAttackPaths: true,
    showPerformanceHUD: true,  // Enable to monitor performance
    animationSpeed: 1.0,
    updateInterval: 30000,
    threatFilter: 'all',
    regionFilter: 'all'
  })

  const [activeView, setActiveView] = useState<'globe' | 'timeline' | 'split'>('globe')
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  // Data fetching hooks
  const { 
    fetchThreats, 
    fetchDistributedThreats, 
    fetchTimeline, 
    fetchAttackPaths, 
    fetchFederatedInsights,
    subscribe 
  } = useThreatData()

  // Calculate dashboard statistics
  const dashboardStats = useMemo<DashboardStats>(() => {
    const { threats, attackPaths, timelineEvents, performance } = dashboardState
    
    return {
      totalThreats: threats.length,
      criticalThreats: threats.filter(t => t.details.severity === 'critical').length,
      activeAttacks: attackPaths.filter(a => a.isActive).length,
      blockedAttacks: timelineEvents.filter(e => e.type === 'attack_blocked').length,
      globalCoverage: Math.min(new Set(threats.map(t => t.country)).size * 5, 100),
      averageResponseTime: performance?.frameTime || 0,
      performance
    }
  }, [dashboardState])

  // Filter threats based on settings
  const filteredThreats = useMemo(() => {
    let filtered = dashboardState.threats

    // Severity filter
    if (settings.threatFilter !== 'all') {
      filtered = filtered.filter(threat => threat.details.severity === settings.threatFilter)
    }

    // Region filter (simplified)
    if (settings.regionFilter !== 'all') {
      const regionCountries: { [key: string]: string[] } = {
        'americas': ['United States', 'Canada', 'Brazil', 'Mexico'],
        'europe': ['United Kingdom', 'Germany', 'France', 'Russia'],
        'asia': ['China', 'Japan', 'South Korea', 'India'],
        'africa': ['South Africa', 'Egypt', 'Nigeria'],
        'oceania': ['Australia', 'New Zealand']
      }

      const countries = regionCountries[settings.regionFilter] || []
      filtered = filtered.filter(threat => countries.includes(threat.country))
    }

    return filtered
  }, [dashboardState.threats, settings.threatFilter, settings.regionFilter])

  // Initialize data fetching
  useEffect(() => {
    const loadInitialData = async () => {
      try {
        setDashboardState(prev => ({
          ...prev,
          isLoading: { threats: true, timeline: true, attacks: true, federated: true }
        }))

        // Use the threat data service directly to avoid unstable function references
        const [threats, distributedThreats, timeline, attacks, federated] = await Promise.all([
          threatDataService.fetchThreatIntelligence(),
          threatDataService.fetchDistributedThreats(),
          threatDataService.fetchIncidentTimeline({ start: Date.now() - 3600000, end: Date.now() }),
          threatDataService.fetchAttackPaths(),
          threatDataService.fetchFederatedInsights()
        ])

        // Combine regular and distributed threats
        const allThreats = [...threats, ...distributedThreats]
        
        console.log('📊 Dashboard data loaded:', {
          regularThreats: threats.length,
          distributedThreats: distributedThreats.length,
          totalThreats: allThreats.length,
          attackPaths: attacks.length,
          timelineEvents: timeline.length,
          sampleThreat: allThreats[0]
        })

        setDashboardState(prev => ({
          ...prev,
          threats: allThreats,
          attackPaths: attacks,
          timelineEvents: timeline,
          federatedInsights: federated,
          isLoading: { threats: false, timeline: false, attacks: false, federated: false },
          lastUpdated: Date.now(),
          error: null
        }))
      } catch (error) {
        console.error('Failed to load dashboard data:', error)
        setDashboardState(prev => ({
          ...prev,
          error: error instanceof Error ? error.message : 'Failed to load data',
          isLoading: { threats: false, timeline: false, attacks: false, federated: false }
        }))
      }
    }

    loadInitialData()
  }, []) // Empty dependency array since we're using the stable threatDataService singleton

  // Set up real-time subscriptions with stable dependencies
  const stableUpdateInterval = settings.updateInterval
  const stableIsPaused = isPaused
  
  useEffect(() => {
    if (stableIsPaused) return

    // Use the threat data service directly to avoid unstable function references
    const unsubscribes = [
      threatDataService.subscribeToUpdates('threats', (threats: ThreatPoint[]) => {
        setDashboardState(prev => ({
          ...prev,
          threats,
          lastUpdated: Date.now()
        }))
      }, stableUpdateInterval),

      threatDataService.subscribeToUpdates('attacks', (attacks: AttackPath[]) => {
        setDashboardState(prev => ({
          ...prev,
          attackPaths: attacks,
          lastUpdated: Date.now()
        }))
      }, stableUpdateInterval),

      threatDataService.subscribeToUpdates('incidents', (events: TimelineEvent[]) => {
        setDashboardState(prev => ({
          ...prev,
          timelineEvents: events,
          lastUpdated: Date.now()
        }))
      }, stableUpdateInterval)
    ]

    return () => {
      unsubscribes.forEach(unsub => unsub())
    }
  }, [stableUpdateInterval, stableIsPaused]) // Remove subscribe dependency

  // Handle threat click
  const handleThreatClick = useCallback((threat: ThreatPoint) => {
    console.log('Threat clicked:', threat)
    // TODO: Show threat details modal
  }, [])

  // Handle timeline event click
  const handleEventClick = useCallback((event: TimelineEvent) => {
    console.log('Event clicked:', event)
    // TODO: Show event details modal
  }, [])

  // Handle performance metrics update with throttling to prevent infinite loops
  const handlePerformanceUpdate = useCallback((metrics: PerformanceMetrics) => {
    setDashboardState(prev => {
      // Only update if performance metrics have changed significantly to prevent infinite loops
      const significantFpsChange = !prev.performance || Math.abs(prev.performance.fps - metrics.fps) > 1.0
      const significantDrawCallChange = !prev.performance || Math.abs(prev.performance.drawCalls - metrics.drawCalls) > 10
      const significantTriangleChange = !prev.performance || Math.abs(prev.performance.triangles - metrics.triangles) > 1000
      
      if (significantFpsChange || significantDrawCallChange || significantTriangleChange) {
        return {
          ...prev,
          performance: metrics
        }
      }
      return prev
    })
  }, [])

  // Toggle fullscreen
  const toggleFullscreen = useCallback(() => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen()
      setIsFullscreen(true)
    } else {
      document.exitFullscreen()
      setIsFullscreen(false)
    }
  }, [])

  // Render threat severity badge
  const renderSeverityBadge = (severity: string) => {
    const colors = {
      critical: 'bg-red-500 text-white',
      high: 'bg-orange-500 text-white',
      medium: 'bg-yellow-500 text-black',
      low: 'bg-green-500 text-white'
    }
    
    return (
      <Badge className={colors[severity as keyof typeof colors]}>
        {severity.toUpperCase()}
      </Badge>
    )
  }

  // Navigation menu items
  const navigationItems = [
    { href: '/', label: 'SOC Dashboard', icon: Home, description: 'Main command center' },
    { href: '/incidents', label: 'Incidents', icon: AlertTriangle, description: 'Active security incidents' },
    { href: '/visualizations', label: '3D Visualization', icon: Globe, description: 'Interactive threat globe', active: true },
    { href: '/analytics', label: 'Analytics', icon: BarChart3, description: 'ML monitoring & insights' },
    { href: '/intelligence', label: 'Threat Intel', icon: Brain, description: 'Intelligence feeds' },
    { href: '/hunt', label: 'Threat Hunting', icon: Target, description: 'Proactive hunting' },
    { href: '/investigations', label: 'Investigations', icon: Search, description: 'Case management' },
    { href: '/agents', label: 'AI Agents', icon: Users, description: 'Agent coordination' },
    { href: '/settings', label: 'Settings', icon: Settings, description: 'System configuration' }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white flex">
      {/* Sidebar Navigation */}
      <div className={`fixed inset-y-0 left-0 z-50 w-80 bg-gray-900/95 backdrop-blur-sm border-r border-gray-700 transform transition-transform duration-300 ease-in-out ${
        sidebarOpen ? 'translate-x-0' : '-translate-x-full'
      } lg:translate-x-0 lg:static lg:inset-0`}>
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-gray-700">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <Shield className="w-5 h-5 text-white" />
              </div>
              <div>
                <h2 className="font-bold text-lg">Mini-XDR</h2>
                <p className="text-xs text-gray-400">Security Operations</p>
              </div>
            </div>
            <button
              onClick={() => setSidebarOpen(false)}
              className="lg:hidden p-2 hover:bg-gray-800 rounded-lg"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Navigation */}
          <nav className="flex-1 p-4 space-y-2 overflow-y-auto">
            {navigationItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-colors group ${
                  item.active 
                    ? 'bg-blue-600/20 text-blue-300 border border-blue-500/30' 
                    : 'hover:bg-gray-800/50 text-gray-300 hover:text-white'
                }`}
              >
                <item.icon className={`w-5 h-5 ${item.active ? 'text-blue-400' : 'text-gray-400 group-hover:text-white'}`} />
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-sm">{item.label}</div>
                  <div className="text-xs text-gray-500 truncate">{item.description}</div>
                </div>
              </Link>
            ))}
          </nav>

          {/* Status Footer */}
          <div className="p-4 border-t border-gray-700">
            <div className="flex items-center gap-2 text-xs">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-gray-400">System Status: Operational</span>
            </div>
            <div className="text-xs text-gray-500 mt-1">
              {dashboardStats.totalThreats} active threats detected
            </div>
          </div>
        </div>
      </div>

      {/* Overlay for mobile */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main Content */}
      <div className={`flex-1 ${isFullscreen ? 'p-0' : 'p-6'} lg:ml-0`}>
        {/* Header */}
        {!isFullscreen && (
          <div className="mb-6">
            <div className="flex items-center justify-between">
              {/* Mobile menu button */}
              <button
                onClick={() => setSidebarOpen(true)}
                className="lg:hidden p-2 hover:bg-gray-800 rounded-lg mr-4"
              >
                <Menu className="w-6 h-6" />
              </button>
              
              <div>
              <h1 className="text-3xl font-bold flex items-center gap-2">
                <Globe className="w-8 h-8 text-blue-400" />
                3D Threat Visualization
              </h1>
              <p className="text-gray-300 mt-1">
                Real-time cybersecurity intelligence with immersive 3D visualization
              </p>
              
              {/* Purpose and Usage Guide */}
              <div className="mt-6 bg-gradient-to-r from-blue-600/10 to-purple-600/10 border border-blue-500/20 rounded-lg p-6">
                <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                  <Globe className="w-5 h-5 text-blue-400" />
                  SOC Analyst Dashboard Guide
                </h2>
                <div className="grid md:grid-cols-2 gap-6 text-sm text-gray-300">
                  <div>
                    <h3 className="font-semibold text-blue-400 mb-2">What You're Seeing:</h3>
                    <ul className="space-y-1 list-disc list-inside">
                      <li>Real incidents from your network (last 7 days)</li>
                      <li>Geographic threat distribution globally</li>
                      <li>Attack timeline with 3D progression</li>
                      <li>Live threat intelligence correlation</li>
                      <li>Performance metrics and system health</li>
                    </ul>
                  </div>
                  <div>
                    <h3 className="font-semibold text-purple-400 mb-2">How to Use:</h3>
                    <ul className="space-y-1 list-disc list-inside">
                      <li>Click threat points for detailed analysis</li>
                      <li>Use timeline to replay attack sequences</li>
                      <li>Filter by severity/region for focus areas</li>
                      <li>Track attack patterns across geography</li>
                      <li>Monitor real-time threat emergence</li>
                    </ul>
                  </div>
                </div>
                <div className="mt-4 flex items-center gap-4 text-xs">
                  <div className="flex items-center gap-2 text-green-400">
                    <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                    Connected to live incident database ({dashboardStats.totalThreats} active threats)
                  </div>
                  {dashboardStats.performance && (
                    <div className="text-blue-400">
                      Rendering at {dashboardStats.performance.fps.toFixed(1)} FPS
                    </div>
                  )}
                </div>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              <div className="text-right">
                <div className="text-sm text-gray-400">Last Updated</div>
                <div className="text-sm">
                  {dashboardState.lastUpdated ? 
                    new Date(dashboardState.lastUpdated).toLocaleTimeString() : 
                    'Never'
                  }
                </div>
              </div>
              
              <Button 
                variant="outline" 
                size="sm"
                onClick={() => setIsPaused(!isPaused)}
              >
                {isPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
                {isPaused ? 'Resume' : 'Pause'}
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Statistics Cards */}
      {!isFullscreen && (
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 mb-6">
          <Card className="bg-gray-800 border-gray-700">
            <CardContent className="p-4">
              <div className="flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-red-400" />
                <div>
                  <div className="text-2xl font-bold">{dashboardStats.totalThreats}</div>
                  <div className="text-xs text-gray-400">Total Threats</div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gray-800 border-gray-700">
            <CardContent className="p-4">
              <div className="flex items-center gap-2">
                <Zap className="w-5 h-5 text-orange-400" />
                <div>
                  <div className="text-2xl font-bold">{dashboardStats.criticalThreats}</div>
                  <div className="text-xs text-gray-400">Critical</div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gray-800 border-gray-700">
            <CardContent className="p-4">
              <div className="flex items-center gap-2">
                <Activity className="w-5 h-5 text-yellow-400" />
                <div>
                  <div className="text-2xl font-bold">{dashboardStats.activeAttacks}</div>
                  <div className="text-xs text-gray-400">Active Attacks</div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gray-800 border-gray-700">
            <CardContent className="p-4">
              <div className="flex items-center gap-2">
                <Shield className="w-5 h-5 text-green-400" />
                <div>
                  <div className="text-2xl font-bold">{dashboardStats.blockedAttacks}</div>
                  <div className="text-xs text-gray-400">Blocked</div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gray-800 border-gray-700">
            <CardContent className="p-4">
              <div className="flex items-center gap-2">
                <Globe className="w-5 h-5 text-blue-400" />
                <div>
                  <div className="text-2xl font-bold">{dashboardStats.globalCoverage}%</div>
                  <div className="text-xs text-gray-400">Coverage</div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gray-800 border-gray-700">
            <CardContent className="p-4">
              <div className="flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-purple-400" />
                <div>
                  <div className="text-2xl font-bold">
                    {dashboardStats.performance?.fps?.toFixed(1) || '0.0'}
                  </div>
                  <div className="text-xs text-gray-400">FPS</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Main Content */}
      <div className="flex gap-6">
        {/* Visualization Panel */}
        <div className={`${isFullscreen ? 'w-full' : 'flex-1'}`}>
          <Card className="bg-gray-800 border-gray-700 h-full">
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <Globe className="w-5 h-5" />
                  Interactive Visualization
                </CardTitle>
                <CardDescription>
                  3D threat intelligence and attack timeline visualization
                </CardDescription>
              </div>
              
              <div className="flex items-center gap-2">
                <Tabs value={activeView} onValueChange={(v) => setActiveView(v as any)}>
                  <TabsList className="grid w-48 grid-cols-3">
                    <TabsTrigger value="globe">Globe</TabsTrigger>
                    <TabsTrigger value="timeline">Timeline</TabsTrigger>
                    <TabsTrigger value="split">Split</TabsTrigger>
                  </TabsList>
                </Tabs>
                
                <Button variant="outline" size="sm" onClick={toggleFullscreen}>
                  <Maximize2 className="w-4 h-4" />
                </Button>
              </div>
            </CardHeader>

            <CardContent className="p-0">
              {activeView === 'globe' && (
                <ThreatGlobe
                  threats={filteredThreats}
                  attackPaths={dashboardState.attackPaths}
                  onThreatClick={handleThreatClick}
                  onPerformanceUpdate={handlePerformanceUpdate}
                  showCountryLabels={settings.showCountryLabels}
                  showCountryOutlines={settings.showCountryOutlines}
                  showAttackPaths={settings.showAttackPaths}
                  showPerformanceHUD={settings.showPerformanceHUD}
                  globeOpacity={settings.globeOpacity}
                  autoRotate={settings.autoRotate}
                  animationSpeed={settings.animationSpeed}
                  height={isFullscreen ? window.innerHeight : 600}
                  className="rounded-b-lg overflow-hidden"
                />
              )}

              {activeView === 'timeline' && (
                <Attack3DTimeline
                  events={dashboardState.timelineEvents}
                  timeRange={{
                    start: Date.now() - 3600000, // Last hour
                    end: Date.now()
                  }}
                  onEventClick={handleEventClick}
                  autoPlay={!isPaused}
                  playbackSpeed={settings.animationSpeed}
                  showEventLabels={true}
                  showConnections={true}
                  height={isFullscreen ? window.innerHeight : 600}
                  className="rounded-b-lg overflow-hidden"
                />
              )}

              {activeView === 'split' && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-2 p-2">
                  <ThreatGlobe
                    threats={filteredThreats}
                    attackPaths={dashboardState.attackPaths}
                    onThreatClick={handleThreatClick}
                    showCountryLabels={false}
                    showCountryOutlines={settings.showCountryOutlines}
                    showAttackPaths={settings.showAttackPaths}
                    showPerformanceHUD={false}  // Hide in split view for space
                    globeOpacity={settings.globeOpacity}
                    autoRotate={settings.autoRotate}
                    animationSpeed={settings.animationSpeed}
                    height={300}
                    className="rounded overflow-hidden"
                  />
                  <Attack3DTimeline
                    events={dashboardState.timelineEvents}
                    timeRange={{
                      start: Date.now() - 3600000,
                      end: Date.now()
                    }}
                    onEventClick={handleEventClick}
                    autoPlay={!isPaused}
                    playbackSpeed={settings.animationSpeed}
                    showEventLabels={false}
                    showConnections={true}
                    height={300}
                    className="rounded overflow-hidden"
                  />
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Control Panel */}
        {!isFullscreen && (
          <div className="w-80">
            <div className="space-y-4">
              {/* Settings Card */}
              <Card className="bg-gray-800 border-gray-700">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Settings className="w-5 h-5" />
                    Visualization Settings
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* Globe Opacity */}
                  <div>
                    <label className="text-sm font-medium block mb-2">
                      Globe Opacity: {Math.round(settings.globeOpacity * 100)}%
                    </label>
                    <input
                      type="range"
                      min={0.1}
                      max={1}
                      step={0.1}
                      value={settings.globeOpacity}
                      onChange={(e) => 
                        setSettings(prev => ({ ...prev, globeOpacity: Number(e.target.value) }))
                      }
                      className="w-full"
                    />
                  </div>

                  {/* Animation Speed */}
                  <div>
                    <label className="text-sm font-medium block mb-2">
                      Animation Speed: {settings.animationSpeed}x
                    </label>
                    <input
                      type="range"
                      min={0.1}
                      max={5}
                      step={0.1}
                      value={settings.animationSpeed}
                      onChange={(e) => 
                        setSettings(prev => ({ ...prev, animationSpeed: Number(e.target.value) }))
                      }
                      className="w-full"
                    />
                  </div>

                  {/* Filters */}
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Threat Severity</label>
                    <select
                      value={settings.threatFilter}
                      onChange={(e) => 
                        setSettings(prev => ({ ...prev, threatFilter: e.target.value as any }))
                      }
                      className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                    >
                      <option value="all">All Severities</option>
                      <option value="critical">Critical Only</option>
                      <option value="high">High Only</option>
                      <option value="medium">Medium Only</option>
                      <option value="low">Low Only</option>
                    </select>
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium">Region</label>
                    <select
                      value={settings.regionFilter}
                      onChange={(e) => 
                        setSettings(prev => ({ ...prev, regionFilter: e.target.value }))
                      }
                      className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                    >
                      <option value="all">All Regions</option>
                      <option value="americas">Americas</option>
                      <option value="europe">Europe</option>
                      <option value="asia">Asia</option>
                      <option value="africa">Africa</option>
                      <option value="oceania">Oceania</option>
                    </select>
                  </div>

                  {/* Toggle Options */}
                  <div className="space-y-3">
                    <div className="border-t border-gray-600 pt-3">
                      <h4 className="text-sm font-semibold text-blue-400 mb-2">Animation Controls</h4>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <div>
                        <label className="text-sm font-medium">Auto Rotate Globe</label>
                        <div className="text-xs text-gray-400">Continuously rotate the globe</div>
                      </div>
                      <input
                        type="checkbox"
                        checked={settings.autoRotate}
                        onChange={(e) => 
                          setSettings(prev => ({ ...prev, autoRotate: e.target.checked }))
                        }
                        className="rounded"
                      />
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <div>
                        <label className="text-sm font-medium">Country Labels</label>
                        <div className="text-xs text-gray-400">Show country names on globe</div>
                      </div>
                      <input
                        type="checkbox"
                        checked={settings.showCountryLabels}
                        onChange={(e) => 
                          setSettings(prev => ({ ...prev, showCountryLabels: e.target.checked }))
                        }
                        className="rounded"
                      />
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <div>
                        <label className="text-sm font-medium">Attack Paths</label>
                        <div className="text-xs text-gray-400">Animated attack flow lines</div>
                      </div>
                      <input
                        type="checkbox"
                        checked={settings.showAttackPaths}
                        onChange={(e) => 
                          setSettings(prev => ({ ...prev, showAttackPaths: e.target.checked }))
                        }
                        className="rounded"
                      />
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <div>
                        <label className="text-sm font-medium">Performance HUD</label>
                        <div className="text-xs text-gray-400">Show FPS and rendering stats</div>
                      </div>
                      <input
                        type="checkbox"
                        checked={settings.showPerformanceHUD}
                        onChange={(e) => 
                          setSettings(prev => ({ ...prev, showPerformanceHUD: e.target.checked }))
                        }
                        className="rounded"
                      />
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <div>
                        <label className="text-sm font-medium">Pause Updates</label>
                        <div className="text-xs text-gray-400">Stop real-time data updates</div>
                      </div>
                      <input
                        type="checkbox"
                        checked={isPaused}
                        onChange={(e) => setIsPaused(e.target.checked)}
                        className="rounded"
                      />
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Performance Card */}
              {dashboardStats.performance && (
                <Card className="bg-gray-800 border-gray-700">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Activity className="w-5 h-5" />
                      Performance Metrics
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>FPS:</span>
                      <span className={dashboardStats.performance.fps < 30 ? 'text-red-400' : 'text-green-400'}>
                        {dashboardStats.performance.fps.toFixed(1)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Draw Calls:</span>
                      <span>{dashboardStats.performance.drawCalls}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Triangles:</span>
                      <span>{dashboardStats.performance.triangles.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Geometries:</span>
                      <span>{dashboardStats.performance.geometries}</span>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Recent Threats */}
              <Card className="bg-gray-800 border-gray-700">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <AlertTriangle className="w-5 h-5" />
                    Recent Threats
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  <ScrollArea className="h-64">
                    <div className="p-4 space-y-3">
                      {filteredThreats.slice(0, 10).map((threat) => (
                        <div
                          key={threat.id}
                          className="flex items-center justify-between p-2 rounded bg-gray-700 hover:bg-gray-600 cursor-pointer"
                          onClick={() => handleThreatClick(threat)}
                        >
                          <div className="flex-1 min-w-0">
                            <div className="font-medium text-sm truncate">
                              {threat.type.toUpperCase()}
                            </div>
                            <div className="text-xs text-gray-400 truncate">
                              {threat.country}
                            </div>
                          </div>
                          <div className="flex items-center gap-2">
                            {renderSeverityBadge(threat.details.severity)}
                            <div className="text-xs text-gray-400">
                              {Math.round(threat.intensity * 100)}%
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
              </div>
            </div>
          )}
        </div>

        {/* Error Display */}
        {dashboardState.error && (
          <div className="fixed bottom-4 right-4 bg-red-600 text-white p-4 rounded-lg shadow-lg max-w-md">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-5 h-5" />
              <div>
                <div className="font-medium">Error</div>
                <div className="text-sm">{dashboardState.error}</div>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setDashboardState(prev => ({ ...prev, error: null }))}
              >
                ✕
              </Button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default VisualizationDashboard
