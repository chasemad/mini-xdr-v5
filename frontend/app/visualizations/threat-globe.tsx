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
}

// Add to imports

// Simplified country outlines to prevent WebGL overload
const DetailedCountryOutlines: React.FC<{ show: boolean; radius?: number }> = ({ show, radius = 2.01 }) => {
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (!show) return;
    
    setIsLoading(true);
    console.log('🌍 Loading country outlines...');
    
    fetch('/world-countries-detailed.geojson')
      .then(res => res.json())
      .then(data => {
        console.log('📊 GeoJSON loaded, features:', data.features?.length);
        
        const positions: number[] = [];
        let totalPoints = 0;
        
        // Limit to prevent WebGL overload
        const maxFeatures = 50; // Limit countries to prevent crash
        const features = data.features.slice(0, maxFeatures);
        
        features.forEach((feature: any, featureIndex: number) => {
          if (totalPoints > 10000) return; // Hard limit on points
          
          const geom = feature.geometry;
          if (!geom || !geom.coordinates) return;
          
          const polys = geom.type === 'MultiPolygon' ? geom.coordinates : [geom.coordinates];
          
          polys.forEach((poly: any) => {
            if (totalPoints > 10000) return;
            
            poly.forEach((ring: any) => {
              if (!Array.isArray(ring) || ring.length < 2) return;
              
              // Simplify by taking every nth point to reduce complexity
              const step = Math.max(1, Math.floor(ring.length / 20));
              
              for (let i = 0; i < ring.length - step; i += step) {
                if (totalPoints > 10000) break;
                
                const [lng1, lat1] = ring[i];
                const [lng2, lat2] = ring[i + step];
                
                if (typeof lng1 !== 'number' || typeof lat1 !== 'number') continue;
                if (typeof lng2 !== 'number' || typeof lat2 !== 'number') continue;
                
                const p1 = geoProjection.latLngToVector3(lat1, lng1, radius);
                const p2 = geoProjection.latLngToVector3(lat2, lng2, radius);
                
                positions.push(p1.x, p1.y, p1.z, p2.x, p2.y, p2.z);
                totalPoints += 2;
              }
            });
          });
        });
        
        console.log('✅ Country outline points generated:', totalPoints);
        
        if (positions.length > 0) {
          const geo = new THREE.BufferGeometry();
          geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
          setGeometry(geo);
        }
        
        setIsLoading(false);
      })
      .catch(error => {
        console.error('❌ Failed to load GeoJSON:', error);
        setIsLoading(false);
      });
  }, [show, radius]);

  if (!show) return null;
  
  if (isLoading) {
    console.log('⏳ Country outlines still loading...');
    return null;
  }
  
  if (!geometry) {
    console.log('⚠️ No country outline geometry available');
    return null;
  }

  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial color="#ffffff" transparent opacity={0.2} />
    </lineSegments>
  );
};

// Globe mesh component with Earth texture
const GlobeCore: React.FC<{ opacity: number }> = ({ opacity }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [texture, setTexture] = useState<THREE.Texture | null>(null);

  // Load Earth texture using standard Three.js TextureLoader
  useEffect(() => {
    const loader = new THREE.TextureLoader();
    loader.load(
      'https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/earth_atmos_2048.jpg',
      (loadedTexture) => {
        setTexture(loadedTexture);
      },
      undefined,
      (error) => {
        console.warn('Failed to load Earth texture, using fallback color:', error);
        // Fallback will be handled by the material
      }
    );
  }, []);

  const globeMaterial = useMemo(() => {
    if (texture) {
      return new THREE.MeshPhongMaterial({
        map: texture,
        transparent: true,
        opacity,
        shininess: 30,
        side: THREE.FrontSide
      });
    } else {
      // Fallback to a nice Earth-like color scheme
      return new THREE.MeshPhongMaterial({
        color: new THREE.Color(0x2563eb), // Ocean blue
        transparent: true,
        opacity,
        shininess: 30,
        side: THREE.FrontSide
      });
    }
  }, [texture, opacity]);

  return (
    <mesh ref={meshRef} material={globeMaterial}>
      <sphereGeometry args={[2, 64, 32]} />
    </mesh>
  );
};

// Threat point visualization component
const ThreatPoints: React.FC<{
  threats: ThreatPoint[]
  onThreatClick?: (threat: ThreatPoint) => void
  animationSpeed: number
}> = ({ threats, onThreatClick, animationSpeed }) => {
  const groupRef = useRef<THREE.Group>(null)
  
  // Debug logging
  useEffect(() => {
    console.log('🎯 ThreatPoints component received:', {
      totalThreats: threats.length,
      withCoords: threats.filter(t => t.latitude !== 0 || t.longitude !== 0).length,
      sample: threats.slice(0, 3).map(t => ({ 
        id: t.id, 
        lat: t.latitude, 
        lng: t.longitude, 
        country: t.country,
        type: t.type
      }))
    })
  }, [threats])
  
  useFrame((state) => {
    if (!groupRef.current) return
    
    // Animate threat intensities with pulsing effect
    groupRef.current.children.forEach((child, index) => {
      if (child instanceof THREE.Mesh && threats[index]) {
        const threat = threats[index]
        const time = state.clock.getElapsedTime() * animationSpeed
        const pulseScale = AnimationHelper.pulseScale(time + index * 0.1, 2)
        
        child.scale.setScalar(pulseScale * (0.5 + threat.intensity * 0.5))
        
        // Update material opacity based on intensity
        if (child.material instanceof THREE.MeshBasicMaterial) {
          child.material.opacity = 0.4 + threat.intensity * 0.6
        }
      }
    })
  })

  return (
    <group ref={groupRef}>
      {threats.map((threat, index) => {
        const position = geoProjection.latLngToVector3(
          threat.latitude, 
          threat.longitude, 
          2.05 // Slightly above globe surface
        )
        
        const material = threatMaterials.getThreatMaterial(threat.type, threat.intensity)
        
        return (
          <mesh
            key={threat.id}
            position={[position.x, position.y, position.z]}
            geometry={geometryPool.getSphereGeometry()}
            material={material}
            onClick={() => onThreatClick?.(threat)}
            userData={{ threat }}
          >
            {/* Add glow effect for high-intensity threats */}
            {threat.intensity > 0.7 && (
              <mesh
                geometry={geometryPool.getSphereGeometry()}
                material={threatMaterials.getGlowMaterial(
                  new THREE.Color(0xff4444), 
                  threat.intensity
                )}
                scale={[2, 2, 2]}
              />
            )}
            
            {/* Tooltip on hover */}
            <Html distanceFactor={20} position={[0, 0.1, 0]}>
              <div className="hidden group-hover:block bg-black/80 text-white p-2 rounded text-xs whitespace-nowrap">
                <div><strong>{threat.type.toUpperCase()}</strong></div>
                <div>Country: {threat.country}</div>
                <div>Intensity: {Math.round(threat.intensity * 100)}%</div>
                <div>Severity: {threat.details.severity}</div>
              </div>
            </Html>
          </mesh>
        )
      })}
    </group>
  )
}

// Attack path visualization component
const AttackPaths: React.FC<{
  attackPaths: AttackPath[]
  animationSpeed: number
}> = ({ attackPaths, animationSpeed }) => {
  const groupRef = useRef<THREE.Group>(null)

  useFrame((state) => {
    if (!groupRef.current) return
    
    const time = state.clock.getElapsedTime() * animationSpeed
    
    groupRef.current.children.forEach((child, index) => {
      if (child instanceof THREE.Line && attackPaths[index]) {
        const path = attackPaths[index]
        
        if (path.isActive) {
          // Animate attack progress
          const progress = (Math.sin(time + index) * 0.5 + 0.5)
          path.progress = progress
          
          // Update line opacity based on activity
          if (child.material instanceof THREE.LineBasicMaterial) {
            child.material.opacity = 0.3 + progress * 0.7
          }
        }
      }
    })
  })

  return (
    <group ref={groupRef}>
      {attackPaths.map((attackPath, index) => {
        const sourcePos = geoProjection.latLngToVector3(
          attackPath.source.latitude,
          attackPath.source.longitude,
          2.02
        )
        const targetPos = geoProjection.latLngToVector3(
          attackPath.target.latitude,
          attackPath.target.longitude,
          2.02
        )
        
        const curve = AnimationHelper.createAttackCurve(sourcePos, targetPos)
        const points = curve.getPoints(50)
        const geometry = new THREE.BufferGeometry().setFromPoints(points)
        
        const material = new THREE.LineBasicMaterial({
          color: new THREE.Color(0xff8800),
          transparent: true,
          opacity: attackPath.isActive ? 0.8 : 0.3,
          linewidth: 2
        })
        
        return (
          <Line
            key={attackPath.id}
            points={[
              [attackPath.source.latitude, attackPath.source.longitude, 1.01],
              [attackPath.target.latitude, attackPath.target.longitude, 1.01]
            ]}
            color="#ff4444"
            lineWidth={2}
          />
        )
      })}
    </group>
  )
}

// Performance monitoring HUD
const PerformanceHUD: React.FC<{
  metrics: PerformanceMetrics | null
}> = ({ metrics }) => {
  if (!metrics) return null

  return (
    <Html position={[-3, 2.5, 0]} transform={false}>
      <div className="bg-black/80 text-white p-4 rounded-lg font-mono text-xs min-w-48 border border-gray-600">
        <div className="text-blue-400 font-semibold mb-2 text-center">Performance</div>
        <div className="space-y-1">
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
        </div>
      </div>
    </Html>
  )
}

// Simple country outlines component with basic shapes
const CountryOutlines: React.FC<{ show: boolean }> = ({ show }) => {
  if (!show) return null

  // Simple country boundaries for major countries
  const simpleCountries = [
    { name: 'United States', bounds: [[-125, 48], [-66, 48], [-66, 25], [-125, 25], [-125, 48]] },
    { name: 'China', bounds: [[73, 53], [135, 53], [135, 18], [73, 18], [73, 53]] },
    { name: 'Russia', bounds: [[19, 78], [180, 78], [180, 41], [19, 41], [19, 78]] },
    { name: 'Brazil', bounds: [[-74, 5], [-35, 5], [-35, -34], [-74, -34], [-74, 5]] },
    { name: 'Australia', bounds: [[113, -10], [154, -10], [154, -44], [113, -44], [113, -10]] },
    { name: 'Canada', bounds: [[-141, 84], [-52, 84], [-52, 42], [-141, 42], [-141, 84]] },
    { name: 'India', bounds: [[68, 37], [97, 37], [97, 6], [68, 6], [68, 37]] },
    { name: 'Argentina', bounds: [[-74, -22], [-53, -22], [-53, -55], [-74, -55], [-74, -22]] }
  ]

  return (
    <group>
      {simpleCountries.map((country, index) => {
        const points = country.bounds.map(([lng, lat]) => 
          geoProjection.latLngToVector3(lat, lng, 2.01)
        )
        
        const geometry = new THREE.BufferGeometry().setFromPoints(points)
        
        return (
          <line key={index}>
            <bufferGeometry attach="geometry" {...geometry} />
            <lineBasicMaterial 
              attach="material" 
              color="#ffffff" 
              transparent
              opacity={0.8}
            />
          </line>
        )
      })}
    </group>
  )
}

// Country labels component
const CountryLabels: React.FC<{ show: boolean }> = ({ show }) => {
  if (!show) return null

  const majorCountries = [
    { code: 'US', name: 'United States' },
    { code: 'CN', name: 'China' },
    { code: 'RU', name: 'Russia' },
    { code: 'GB', name: 'United Kingdom' },
    { code: 'DE', name: 'Germany' },
    { code: 'JP', name: 'Japan' },
    { code: 'IN', name: 'India' },
    { code: 'BR', name: 'Brazil' }
  ]

  return (
    <group>
      {majorCountries.map((country) => {
        const center = geoProjection.getCountryCenter(country.code)
        const position = geoProjection.latLngToVector3(
          center.lat, 
          center.lng, 
          2.2
        )
        
        return (
          <Text
            key={country.code}
            position={[position.x, position.y, position.z]}
            fontSize={0.08}
            color="white"
            anchorX="center"
            anchorY="middle"
          >
            {country.name}
          </Text>
        )
      })}
    </group>
  )
}

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
const ThreatGlobe: React.FC<ThreatGlobeProps & {
  className?: string
  height?: number
}> = ({ 
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
      <div className={`flex items-center justify-center bg-gray-900 text-white ${className}`}
           style={{ height }}>
        <div className="text-center">
          <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-4"></div>
          <div>Loading 3D Threat Globe...</div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className={`flex items-center justify-center bg-gray-900 text-white ${className}`}
           style={{ height }}>
        <div className="text-center p-8">
          <div className="text-red-400 mb-4">⚠ WebGL Error</div>
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

  try {
    return (
      <div className={`relative ${className}`} style={{ height }}>
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
        <div className="absolute top-4 right-4 text-green-400 text-sm bg-black/50 p-2 rounded">
          🌍 Canvas Active ({sceneProps.threats?.length || 0} threats)
        </div>
        
        {/* Simple test to verify rendering */}
        <div className="absolute bottom-4 left-4 text-white/50 text-xs">
          WebGL Status: Ready | Threats: {sceneProps.threats?.length || 0}
        </div>
      </div>
    )
  } catch (error) {
    console.error('ThreatGlobe render error:', error)
    return (
      <div className={`flex items-center justify-center bg-gray-900 text-white ${className}`}
           style={{ height }}>
        <div className="text-center p-8">
          <div className="text-red-400 mb-4">⚠ Render Error</div>
          <div className="text-sm opacity-70">Failed to initialize 3D globe</div>
          <div className="text-xs opacity-50 mt-2">WebGL may not be supported</div>
        </div>
      </div>
    )
  }
}

export default ThreatGlobe
export type { ThreatGlobeProps }
