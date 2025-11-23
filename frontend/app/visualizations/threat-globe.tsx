'use client'

/**
 * Phase 4.1: Interactive 3D Threat Globe
 * Real-time cybersecurity threat visualization with WebGL optimization
 */

import React, { useRef, useEffect, useState, useMemo } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Stars, Text, Html, Line } from '@react-three/drei'
import * as THREE from 'three'
import {
  ThreatPoint,
  AttackPath,
  PerformanceMetrics,
  geoProjection,
  threatMaterials,
  geometryPool,
  PerformanceMonitor,
  AnimationHelper,
  ThreatDataProcessor
} from '../../lib/three-helpers'

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { cn } from "@/lib/utils"

// ... existing imports ...

// Component props interface
interface ThreatGlobeProps {
  threats?: ThreatPoint[]
  attackPaths?: AttackPath[]
  onThreatClick?: (threat: ThreatPoint) => void
  onPerformanceUpdate?: (metrics: PerformanceMetrics) => void
  showCountryLabels?: boolean
  showCountryOutlines?: boolean
  showAttackPaths?: boolean
  showPerformanceHUD?: boolean
  showHeatmap?: boolean
  globeOpacity?: number
  autoRotate?: boolean
  animationSpeed?: number
  className?: string
  height?: number
}

// ... existing helper components (DetailedCountryOutlines, GlobeCore, ThreatPoints, AttackPaths) ...

// Performance monitoring HUD
const PerformanceHUD: React.FC<{
  metrics: PerformanceMetrics | null
}> = ({ metrics }) => {
  if (!metrics) return null

  return (
    <Html position={[-3, 2.5, 0]} transform={false}>
      <Card className="w-48 bg-black/80 border-gray-600 text-white text-xs backdrop-blur-sm">
        <CardHeader className="p-3 pb-2">
          <CardTitle className="text-blue-400 text-xs font-mono text-center">Performance</CardTitle>
        </CardHeader>
        <CardContent className="p-3 pt-0 space-y-1 font-mono">
          <div className="flex justify-between">
            <span className="text-gray-300">FPS:</span>
            <span className="text-green-400">{metrics.fps.toFixed(1)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-300">Frame:</span>
            <span className="text-yellow-400">{metrics.frameTime.toFixed(1)}ms</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-300">Draw Calls:</span>
            <span className="text-purple-400">{metrics.drawCalls}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-300">Triangles:</span>
            <span className="text-orange-400">{metrics.triangles.toLocaleString()}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-300">Geometries:</span>
            <span className="text-cyan-400">{metrics.geometries}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-300">Materials:</span>
            <span className="text-pink-400">{metrics.materials}</span>
          </div>
        </CardContent>
      </Card>
    </Html>
  )
}

// ... existing CountryOutlines and CountryLabels components ...

// Main scene component with performance monitoring
const ThreatGlobeScene: React.FC<ThreatGlobeProps> = ({
  threats = [],
  attackPaths = [],
  onThreatClick,
  onPerformanceUpdate,
  showCountryLabels = false,
  showCountryOutlines = true,
  showAttackPaths = true,
  showPerformanceHUD = false,
  globeOpacity = 0.6,
  autoRotate = true,
  animationSpeed = 1.0
}) => {
  // ... existing implementation ...
  const { gl } = useThree()
  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics | null>(null)
  const performanceMonitor = useMemo(() => new PerformanceMonitor(gl), [gl])

  // Update performance metrics
  useFrame(() => {
    performanceMonitor.update()
    const metrics = performanceMonitor.getMetrics()
    setPerformanceMetrics(metrics)
    onPerformanceUpdate?.(metrics)
  })

  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.3} />
      <pointLight position={[10, 10, 10]} intensity={0.8} />
      <pointLight position={[-10, -10, -10]} intensity={0.4} />

      {/* Controls */}
      <OrbitControls
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        autoRotate={autoRotate}
        autoRotateSpeed={0.5}
        minDistance={3}
        maxDistance={10}
      />

      {/* Background stars */}
      <Stars
        radius={100}
        depth={50}
        count={5000}
        factor={4}
        saturation={0}
        fade
        speed={1}
      />

      {/* Globe */}
      <GlobeCore opacity={globeOpacity} />

      {/* Country outlines */}
      <DetailedCountryOutlines show={showCountryOutlines} />

      {/* Threat visualizations */}
      <ThreatPoints
        threats={threats}
        onThreatClick={onThreatClick}
        animationSpeed={animationSpeed}
      />

      {/* Attack paths */}
      {showAttackPaths && (
        <AttackPaths
          attackPaths={attackPaths}
          animationSpeed={animationSpeed}
        />
      )}

      {/* Country labels */}
      <CountryLabels show={showCountryLabels} />

      {/* Performance HUD - only show if enabled */}
      {showPerformanceHUD && <PerformanceHUD metrics={performanceMetrics} />}
    </>
  )
}

// Main component export
const ThreatGlobe: React.FC<ThreatGlobeProps> = ({
  className = "",
  height = 600,
  ...sceneProps
}) => {
  const [error, setError] = useState<string | null>(null)
  const [isClient, setIsClient] = useState(false)

  // Ensure client-side only rendering
  useEffect(() => {
    setIsClient(true)
  }, [])

  // Error boundary
  // Don't render on server-side
  if (!isClient) {
    return (
      <Card className={cn("flex items-center justify-center bg-gray-900 text-white border-0", className)} style={{ height }}>
        <div className="text-center">
          <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-4"></div>
          <div>Loading 3D Threat Globe...</div>
        </div>
      </Card>
    )
  }

  if (error) {
    return (
      <Card className={cn("flex items-center justify-center bg-gray-900 text-white border-0", className)} style={{ height }}>
        <div className="text-center p-8">
          <div className="text-red-400 mb-4">⚠ WebGL Error</div>
          <div className="text-sm opacity-70">{error}</div>
          <Button
            onClick={() => setError(null)}
            variant="secondary"
            className="mt-4"
          >
            Retry
          </Button>
        </div>
      </Card>
    )
  }

  try {
    return (
      <Card className={cn("relative border-0 overflow-hidden", className)} style={{ height }}>
        <Canvas
          camera={{ position: [0, 0, 5], fov: 50 }}
          gl={{
            antialias: true,
            alpha: false,
            powerPreference: 'high-performance',
            preserveDrawingBuffer: true // Prevent context loss
          }}
          onCreated={({ gl }) => {
            const context = gl.getContext()
            if (!context) {
              return
            }

            console.log('🎨 WebGL Context Created:', {
              vendor: context.getParameter(context.VENDOR),
              version: context.getParameter(context.VERSION),
              maxTextures: context.getParameter(context.MAX_TEXTURE_IMAGE_UNITS)
            });
          }}
          style={{ background: '#0a0a0f' }}
        >
          <ThreatGlobeScene {...sceneProps} />
        </Canvas>

        {/* Debug indicator */}
        <div className="absolute top-4 right-4 text-green-400 text-sm bg-black/50 p-2 rounded backdrop-blur-sm border border-white/10">
          🌍 Canvas Active ({sceneProps.threats?.length || 0} threats)
        </div>

        {/* Simple test to verify rendering */}
        <div className="absolute bottom-4 left-4 text-white/50 text-xs bg-black/50 p-2 rounded backdrop-blur-sm">
          WebGL Status: Ready | Threats: {sceneProps.threats?.length || 0}
        </div>
      </Card>
    )
  } catch (error) {
    console.error('ThreatGlobe render error:', error)
    return (
      <Card className={cn("flex items-center justify-center bg-gray-900 text-white border-0", className)} style={{ height }}>
        <div className="text-center p-8">
          <div className="text-red-400 mb-4">⚠ Render Error</div>
          <div className="text-sm opacity-70">Failed to initialize 3D globe</div>
          <div className="text-xs opacity-50 mt-2">WebGL may not be supported</div>
        </div>
      </Card>
    )
  }
}

export default ThreatGlobe
export type { ThreatGlobeProps }
