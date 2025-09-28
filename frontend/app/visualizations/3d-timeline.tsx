'use client'

/**
 * Phase 4.1: 3D Attack Timeline Visualization
 * Real-time attack progression timeline with WebGL-accelerated animations
 */

import React, { useRef, useEffect, useState, useMemo, useCallback } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Text, Html, Line } from '@react-three/drei'
import * as THREE from 'three'
import { 
  ThreatPoint, 
  AttackPath, 
  PerformanceMetrics,
  AnimationHelper,
  ThreatDataProcessor,
  threatMaterials,
  geometryPool
} from '../../lib/three-helpers'

// Timeline event interface
interface TimelineEvent {
  id: string
  timestamp: number
  type: 'threat_detected' | 'attack_blocked' | 'lateral_movement' | 'data_breach' | 'mitigation'
  title: string
  description: string
  severity: 'low' | 'medium' | 'high' | 'critical'
  position?: THREE.Vector3
  duration?: number
  relatedThreats: ThreatPoint[]
  metadata: {
    source?: string
    target?: string
    technique?: string
    mitre_attack_id?: string
  }
}

interface Attack3DTimelineProps {
  events?: TimelineEvent[]
  timeRange?: { start: number, end: number }
  currentTime?: number
  playbackSpeed?: number
  onEventClick?: (event: TimelineEvent) => void
  onTimeChange?: (time: number) => void
  autoPlay?: boolean
  showEventLabels?: boolean
  showConnections?: boolean
}

// Timeline axis component
const TimelineAxis: React.FC<{
  timeRange: { start: number, end: number }
  currentTime: number
  height: number
}> = ({ timeRange, currentTime, height }) => {
  const axisRef = useRef<THREE.Group>(null)
  const duration = timeRange.end - timeRange.start
  
  // Time markers
  const timeMarkers = useMemo(() => {
    const markers = []
    const numMarkers = 10
    
    for (let i = 0; i <= numMarkers; i++) {
      const t = i / numMarkers
      const time = timeRange.start + duration * t
      const y = -height/2 + height * t
      
      markers.push({
        position: new THREE.Vector3(0, y, 0),
        time,
        label: new Date(time).toLocaleTimeString()
      })
    }
    
    return markers
  }, [timeRange, duration, height])
  
  // Current time indicator position
  const currentTimeY = useMemo(() => {
    const progress = (currentTime - timeRange.start) / duration
    return -height/2 + height * Math.max(0, Math.min(1, progress))
  }, [currentTime, timeRange, duration, height])

  return (
    <group ref={axisRef}>
      {/* Main axis line */}
      <Line
        points={[[0, -height/2, 0], [0, height/2, 0]]}
        color="white"
        lineWidth={2}
      />
      
      {/* Time markers */}
      {timeMarkers.map((marker, index) => (
        <group key={index} position={marker.position}>
          <Line
            points={[[-0.1, 0, 0], [0.1, 0, 0]]}
            color="white"
            lineWidth={1}
          />
          <Text
            position={[0.2, 0, 0]}
            fontSize={0.05}
            color="white"
            anchorX="left"
            anchorY="middle"
          >
            {marker.label}
          </Text>
        </group>
      ))}
      
      {/* Current time indicator */}
      <group position={[0, currentTimeY, 0]}>
        <mesh>
          <sphereGeometry args={[0.03, 8, 8]} />
          <meshBasicMaterial color="red" />
        </mesh>
        <Line
          points={[[-0.3, 0, 0], [3, 0, 0]]}
          color="red"
          lineWidth={3}
        />
      </group>
    </group>
  )
}

// 3D event visualization component
const TimelineEvents: React.FC<{
  events: TimelineEvent[]
  timeRange: { start: number, end: number }
  currentTime: number
  height: number
  onEventClick?: (event: TimelineEvent) => void
  showLabels: boolean
}> = ({ events, timeRange, currentTime, height, onEventClick, showLabels }) => {
  const groupRef = useRef<THREE.Group>(null)
  const duration = timeRange.end - timeRange.start

  // Event colors by type
  const eventColors = useMemo(() => ({
    threat_detected: new THREE.Color(0xff4444),
    attack_blocked: new THREE.Color(0x44ff44),
    lateral_movement: new THREE.Color(0xffaa44),
    data_breach: new THREE.Color(0xff44ff),
    mitigation: new THREE.Color(0x44aaff)
  }), [])

  // Animate events as they become active
  useFrame((state) => {
    if (!groupRef.current) return
    
    const time = state.clock.getElapsedTime()
    
    groupRef.current.children.forEach((child, index) => {
      if (child instanceof THREE.Group && events[index]) {
        const event = events[index]
        const eventProgress = (currentTime - event.timestamp) / 5000 // 5 second fade
        const isActive = currentTime >= event.timestamp && eventProgress <= 1
        
        if (isActive) {
          // Pulse animation for active events
          const pulse = AnimationHelper.pulseScale(time + index * 0.2, 3)
          child.scale.setScalar(pulse)
          
          // Fade out older events
          const opacity = Math.max(0, 1 - eventProgress)
          child.children.forEach(mesh => {
            if (mesh instanceof THREE.Mesh && mesh.material instanceof THREE.MeshBasicMaterial) {
              mesh.material.opacity = opacity
            }
          })
        } else {
          child.scale.setScalar(0.01)
        }
      }
    })
  })

  return (
    <group ref={groupRef}>
      {events.map((event, index) => {
        // Position event along timeline
        const progress = (event.timestamp - timeRange.start) / duration
        const y = -height/2 + height * Math.max(0, Math.min(1, progress))
        
        // Spread events in 3D space based on severity and type
        const severityRadius = {
          low: 0.5,
          medium: 1.0,
          high: 1.5,
          critical: 2.0
        }[event.severity]
        
        const angle = (index * 137.5) * (Math.PI / 180) // Golden angle distribution
        const x = Math.cos(angle) * severityRadius
        const z = Math.sin(angle) * severityRadius
        
        const color = eventColors[event.type]
        const isActive = currentTime >= event.timestamp
        
        return (
          <group key={event.id} position={[x, y, z]}>
            {/* Main event sphere */}
            <mesh
              onClick={() => onEventClick?.(event)}
              userData={{ event }}
            >
              <sphereGeometry args={[0.05, 16, 8]} />
              <meshBasicMaterial 
                color={color}
                transparent
                opacity={isActive ? 0.8 : 0.3}
              />
            </mesh>
            
            {/* Glow effect for critical events */}
            {event.severity === 'critical' && (
              <mesh scale={[2, 2, 2]}>
                <sphereGeometry args={[0.05, 8, 4]} />
                <meshBasicMaterial 
                  color={color}
                  transparent
                  opacity={isActive ? 0.3 : 0.1}
                  side={THREE.BackSide}
                />
              </mesh>
            )}
            
            {/* Connection line to axis */}
            <Line
              points={[[0, 0, 0], [-x, 0, -z]]}
              color={color.getStyle()}
              lineWidth={1}
              transparent
              opacity={isActive ? 0.4 : 0.1}
            />
            
            {/* Event label */}
            {showLabels && isActive && (
              <Html position={[0.1, 0.1, 0]} distanceFactor={10}>
                <div className="bg-black/80 text-white p-2 rounded text-xs max-w-48">
                  <div className="font-semibold text-yellow-400">{event.title}</div>
                  <div className="text-xs opacity-75 mt-1">{event.description}</div>
                  <div className="text-xs mt-1">
                    <span className={`inline-block w-2 h-2 rounded-full mr-1 ${
                      event.severity === 'critical' ? 'bg-red-500' :
                      event.severity === 'high' ? 'bg-orange-500' :
                      event.severity === 'medium' ? 'bg-yellow-500' :
                      'bg-green-500'
                    }`}></span>
                    {event.severity.toUpperCase()}
                  </div>
                </div>
              </Html>
            )}
          </group>
        )
      })}
    </group>
  )
}

// Attack progression connections
const AttackConnections: React.FC<{
  events: TimelineEvent[]
  currentTime: number
  show: boolean
}> = ({ events, currentTime, show }) => {
  const connections = useMemo(() => {
    const conns = []
    
    // Connect related events in chronological order
    const sortedEvents = [...events].sort((a, b) => a.timestamp - b.timestamp)
    
    for (let i = 1; i < sortedEvents.length; i++) {
      const currentEvent = sortedEvents[i]
      const prevEvent = sortedEvents[i - 1]
      
      // Connect events that share threat indicators
      const hasSharedThreats = currentEvent.relatedThreats.some(threat =>
        prevEvent.relatedThreats.some(prevThreat => 
          threat.country === prevThreat.country || threat.type === prevThreat.type
        )
      )
      
      if (hasSharedThreats) {
        conns.push({ from: prevEvent, to: currentEvent })
      }
    }
    
    return conns
  }, [events])

  if (!show) return null

  return (
    <group>
      {connections.map((conn, index) => {
        const isVisible = currentTime >= conn.to.timestamp
        
        if (!isVisible) return null
        
        // This would require calculating positions similar to TimelineEvents
        // Simplified for now - you would get the actual 3D positions of the events
        
        return (
          <Line
            key={`${conn.from.id}-${conn.to.id}`}
            points={[[0, 0, 0], [0, 1, 0]]} // Placeholder - replace with actual positions
            color="cyan"
            lineWidth={2}
            transparent
            opacity={0.6}
          />
        )
      })}
    </group>
  )
}

// Playback controls
const TimelineControls: React.FC<{
  currentTime: number
  timeRange: { start: number, end: number }
  isPlaying: boolean
  playbackSpeed: number
  onTimeChange: (time: number) => void
  onPlayPause: () => void
  onSpeedChange: (speed: number) => void
  onReset: () => void
}> = ({
  currentTime,
  timeRange,
  isPlaying,
  playbackSpeed,
  onTimeChange,
  onPlayPause,
  onSpeedChange,
  onReset
}) => {
  const progress = (currentTime - timeRange.start) / (timeRange.end - timeRange.start)

  return (
    <Html position={[0, -3, 0]} transform={false}>
      <div className="bg-black/80 text-white p-4 rounded-lg min-w-96">
        <div className="flex items-center gap-4 mb-4">
          <button
            onClick={onPlayPause}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded"
          >
            {isPlaying ? '⏸ Pause' : '▶ Play'}
          </button>
          
          <button
            onClick={onReset}
            className="px-3 py-2 bg-gray-600 hover:bg-gray-700 rounded"
          >
            ⏮ Reset
          </button>
          
          <select
            value={playbackSpeed}
            onChange={(e) => onSpeedChange(Number(e.target.value))}
            className="px-2 py-1 bg-gray-700 rounded"
          >
            <option value={0.5}>0.5x</option>
            <option value={1}>1x</option>
            <option value={2}>2x</option>
            <option value={5}>5x</option>
            <option value={10}>10x</option>
          </select>
          
          <span className="text-sm opacity-75">
            {new Date(currentTime).toLocaleTimeString()}
          </span>
        </div>
        
        <div className="flex items-center gap-2">
          <span className="text-xs opacity-75 min-w-16">
            {new Date(timeRange.start).toLocaleTimeString()}
          </span>
          
          <input
            type="range"
            min={0}
            max={1}
            step={0.001}
            value={progress}
            onChange={(e) => {
              const newProgress = Number(e.target.value)
              const newTime = timeRange.start + (timeRange.end - timeRange.start) * newProgress
              onTimeChange(newTime)
            }}
            className="flex-1"
          />
          
          <span className="text-xs opacity-75 min-w-16">
            {new Date(timeRange.end).toLocaleTimeString()}
          </span>
        </div>
      </div>
    </Html>
  )
}

// Main 3D timeline scene
const Timeline3DScene: React.FC<Attack3DTimelineProps> = ({
  events = [],
  timeRange = { start: Date.now() - 3600000, end: Date.now() },
  currentTime = Date.now(),
  playbackSpeed = 1,
  onEventClick,
  onTimeChange,
  autoPlay = false,
  showEventLabels = true,
  showConnections = true
}) => {
  const [isPlaying, setIsPlaying] = useState(autoPlay)
  const [internalTime, setInternalTime] = useState(currentTime)
  const [internalSpeed, setInternalSpeed] = useState(playbackSpeed)
  
  const timelineHeight = 4

  // Auto-advance timeline when playing
  useFrame((state, delta) => {
    if (isPlaying && internalTime < timeRange.end) {
      const newTime = internalTime + (delta * 1000 * internalSpeed)
      setInternalTime(Math.min(newTime, timeRange.end))
      onTimeChange?.(Math.min(newTime, timeRange.end))
    }
  })

  const handlePlayPause = useCallback(() => {
    setIsPlaying(!isPlaying)
  }, [isPlaying])

  const handleTimeChange = useCallback((time: number) => {
    setInternalTime(time)
    onTimeChange?.(time)
  }, [onTimeChange])

  const handleSpeedChange = useCallback((speed: number) => {
    setInternalSpeed(speed)
  }, [])

  const handleReset = useCallback(() => {
    setInternalTime(timeRange.start)
    setIsPlaying(false)
    onTimeChange?.(timeRange.start)
  }, [timeRange.start, onTimeChange])

  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.4} />
      <pointLight position={[5, 5, 5]} intensity={0.8} />
      <pointLight position={[-5, -5, -5]} intensity={0.4} />
      
      {/* Controls */}
      <OrbitControls
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        target={[0, 0, 0]}
        minDistance={2}
        maxDistance={15}
      />
      
      {/* Timeline axis */}
      <TimelineAxis
        timeRange={timeRange}
        currentTime={internalTime}
        height={timelineHeight}
      />
      
      {/* Timeline events */}
      <TimelineEvents
        events={events}
        timeRange={timeRange}
        currentTime={internalTime}
        height={timelineHeight}
        onEventClick={onEventClick}
        showLabels={showEventLabels}
      />
      
      {/* Attack connections */}
      <AttackConnections
        events={events}
        currentTime={internalTime}
        show={showConnections}
      />
      
      {/* Playback controls */}
      <TimelineControls
        currentTime={internalTime}
        timeRange={timeRange}
        isPlaying={isPlaying}
        playbackSpeed={internalSpeed}
        onTimeChange={handleTimeChange}
        onPlayPause={handlePlayPause}
        onSpeedChange={handleSpeedChange}
        onReset={handleReset}
      />
    </>
  )
}

// Main component export
const Attack3DTimeline: React.FC<Attack3DTimelineProps & {
  className?: string
  height?: number
}> = ({ 
  className = "", 
  height = 600,
  ...sceneProps 
}) => {
  const [error, setError] = useState<string | null>(null)

  const handleError = useCallback((error: Error) => {
    console.error('Attack3DTimeline Error:', error)
    setError(error.message)
  }, [])

  if (error) {
    return (
      <div className={`flex items-center justify-center bg-gray-900 text-white ${className}`}
           style={{ height }}>
        <div className="text-center p-8">
          <div className="text-red-400 mb-4">⚠ 3D Timeline Error</div>
          <div className="text-sm opacity-70">{error}</div>
          <button 
            onClick={() => setError(null)}
            className="mt-4 px-4 py-2 bg-blue-600 rounded hover:bg-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className={`relative ${className}`} style={{ height }}>
      <Canvas
        camera={{ position: [3, 0, 3], fov: 60 }}
        gl={{ 
          antialias: true, 
          alpha: true,
          powerPreference: 'high-performance'
        }}
        // onError removed due to type compatibility issues
      >
        <Timeline3DScene {...sceneProps} />
      </Canvas>
      
      <div className="absolute top-4 right-4 text-white/70 text-sm">
        ⏰ 3D Attack Timeline
      </div>
    </div>
  )
}

export default Attack3DTimeline
export type { Attack3DTimelineProps, TimelineEvent }
