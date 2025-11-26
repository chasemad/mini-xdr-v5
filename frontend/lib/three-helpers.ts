/**
 * Phase 4.1: Three.js Utilities for 3D Threat Visualization
 * Optimized helpers for WebGL rendering, geographic projections, and performance monitoring
 */

import * as THREE from 'three'
import { geoNaturalEarth1 } from 'd3-geo'

// Performance monitoring interface
export interface PerformanceMetrics {
  fps: number
  frameTime: number
  geometries: number
  materials: number
  textures: number
  drawCalls: number
  vertices: number
  triangles: number
  memoryUsage: {
    programs: number
    geometries: number
    textures: number
  }
}

// Threat visualization data types
export interface ThreatPoint {
  id: string
  latitude: number
  longitude: number
  intensity: number // 0-1 scale
  type: 'malware' | 'ddos' | 'phishing' | 'botnet' | 'exploit'
  country: string
  timestamp: number
  details: {
    source?: string
    target?: string
    confidence: number
    severity: 'low' | 'medium' | 'high' | 'critical'
  }
}

export interface AttackPath {
  id: string
  source: ThreatPoint
  target: ThreatPoint
  progress: number // 0-1 animation progress
  type: 'lateral_movement' | 'data_exfiltration' | 'command_control' | 'reconnaissance'
  isActive: boolean
}

// Geographic projection utilities
export class GeoProjection {
  private projection = geoNaturalEarth1()

  constructor() {
    // Configure projection for our globe (radius 2 units)
    this.projection
      .scale(100)
      .translate([0, 0])
  }

  /**
   * Convert lat/lng to 3D sphere coordinates
   */
  latLngToVector3(lat: number, lng: number, radius: number = 2): THREE.Vector3 {
    const phi = (90 - lat) * (Math.PI / 180)
    const theta = (lng + 180) * (Math.PI / 180)

    const x = radius * Math.sin(phi) * Math.cos(theta)
    const z = radius * Math.sin(phi) * Math.sin(theta)
    const y = radius * Math.cos(phi)

    return new THREE.Vector3(x, y, z)
  }

  /**
   * Project coordinates for 2D overlay elements
   */
  project(lat: number, lng: number): [number, number] | null {
    return this.projection([lng, lat])
  }

  /**
   * Get country center coordinates
   */
  getCountryCenter(countryCode: string): { lat: number, lng: number } {
    // Approximate country centers for major countries
    const countryCenters: { [key: string]: { lat: number, lng: number } } = {
      'US': { lat: 39.8283, lng: -98.5795 },
      'CN': { lat: 35.8617, lng: 104.1954 },
      'RU': { lat: 61.5240, lng: 105.3188 },
      'GB': { lat: 55.3781, lng: -3.4360 },
      'DE': { lat: 51.1657, lng: 10.4515 },
      'FR': { lat: 46.6034, lng: 1.8883 },
      'JP': { lat: 36.2048, lng: 138.2529 },
      'KR': { lat: 35.9078, lng: 127.7669 },
      'IN': { lat: 20.5937, lng: 78.9629 },
      'BR': { lat: -14.2350, lng: -51.9253 },
      'AU': { lat: -25.2744, lng: 133.7751 },
      'CA': { lat: 56.1304, lng: -106.3468 },
      'MX': { lat: 23.6345, lng: -102.5528 },
      'AR': { lat: -38.4161, lng: -63.6167 },
      'ZA': { lat: -30.5595, lng: 22.9375 },
      'EG': { lat: 26.8206, lng: 30.8025 },
      'TR': { lat: 38.9637, lng: 35.2433 },
      'IR': { lat: 32.4279, lng: 53.6880 },
      'SA': { lat: 23.8859, lng: 45.0792 },
      'AE': { lat: 23.4241, lng: 53.8478 },
      'SG': { lat: 1.3521, lng: 103.8198 },
      'TH': { lat: 15.8700, lng: 100.9925 },
      'VN': { lat: 14.0583, lng: 108.2772 },
      'ID': { lat: -0.7893, lng: 113.9213 },
      'MY': { lat: 4.2105, lng: 101.9758 },
      'PH': { lat: 12.8797, lng: 121.7740 }
    }

    return countryCenters[countryCode] || { lat: 0, lng: 0 }
  }
}

// Material factory for consistent threat visualization
export class ThreatMaterialFactory {
  private static instance: ThreatMaterialFactory
  private materials: Map<string, THREE.Material> = new Map()

  static getInstance(): ThreatMaterialFactory {
    if (!ThreatMaterialFactory.instance) {
      ThreatMaterialFactory.instance = new ThreatMaterialFactory()
    }
    return ThreatMaterialFactory.instance
  }

  /**
   * Get material for threat type with intensity-based colors
   */
  getThreatMaterial(type: ThreatPoint['type'], intensity: number): THREE.Material {
    const key = `${type}-${Math.round(intensity * 10)}`

    if (!this.materials.has(key)) {
      const colors = {
        malware: new THREE.Color(0xff4444),
        ddos: new THREE.Color(0xff8844),
        phishing: new THREE.Color(0xffaa44),
        botnet: new THREE.Color(0xaa44ff),
        exploit: new THREE.Color(0xff44aa)
      }

      const baseColor = colors[type] || new THREE.Color(0xff4444)
      const color = baseColor.clone().multiplyScalar(0.5 + intensity * 0.5)

      const material = new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0.3 + intensity * 0.7,
        side: THREE.DoubleSide
      })

      this.materials.set(key, material)
    }

    return this.materials.get(key)!
  }

  /**
   * Get glowing effect material for high-intensity threats
   */
  getGlowMaterial(color: THREE.Color, intensity: number): THREE.ShaderMaterial {
    return new THREE.ShaderMaterial({
      uniforms: {
        color: { value: color },
        intensity: { value: intensity },
        time: { value: 0 }
      },
      vertexShader: `
        varying vec3 vNormal;
        varying vec3 vPosition;
        void main() {
          vNormal = normalize(normalMatrix * normal);
          vPosition = position;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform vec3 color;
        uniform float intensity;
        uniform float time;
        varying vec3 vNormal;
        varying vec3 vPosition;

        void main() {
          float pulse = sin(time * 3.0) * 0.5 + 0.5;
          float glow = pow(max(0.0, dot(vNormal, vec3(0, 0, 1))), 2.0);
          vec3 finalColor = color * (intensity + pulse * 0.3);
          gl_FragColor = vec4(finalColor, glow * intensity);
        }
      `,
      transparent: true,
      side: THREE.BackSide,
      blending: THREE.AdditiveBlending
    })
  }

  /**
   * Clean up materials to prevent memory leaks
   */
  dispose(): void {
    this.materials.forEach(material => material.dispose())
    this.materials.clear()
  }
}

// Geometry pools for performance optimization
export class GeometryPool {
  private static instance: GeometryPool
  private sphereGeometry: THREE.SphereGeometry
  private cylinderGeometry: THREE.CylinderGeometry
  private planeGeometry: THREE.PlaneGeometry

  private constructor() {
    this.sphereGeometry = new THREE.SphereGeometry(0.02, 8, 6)
    this.cylinderGeometry = new THREE.CylinderGeometry(0.001, 0.002, 1, 6)
    this.planeGeometry = new THREE.PlaneGeometry(0.1, 0.1)
  }

  static getInstance(): GeometryPool {
    if (!GeometryPool.instance) {
      GeometryPool.instance = new GeometryPool()
    }
    return GeometryPool.instance
  }

  getSphereGeometry(): THREE.SphereGeometry {
    return this.sphereGeometry
  }

  getCylinderGeometry(): THREE.CylinderGeometry {
    return this.cylinderGeometry
  }

  getPlaneGeometry(): THREE.PlaneGeometry {
    return this.planeGeometry
  }

  dispose(): void {
    this.sphereGeometry.dispose()
    this.cylinderGeometry.dispose()
    this.planeGeometry.dispose()
  }
}

// Performance monitor class
export class PerformanceMonitor {
  private renderer: THREE.WebGLRenderer
  private startTime: number = Date.now()
  private frames: number = 0
  private lastFrameTime: number = Date.now()

  constructor(renderer: THREE.WebGLRenderer) {
    this.renderer = renderer
  }

  /**
   * Update frame counters
   */
  update(): void {
    this.frames++
    this.lastFrameTime = Date.now()
  }

  /**
   * Get current performance metrics
   */
  getMetrics(): PerformanceMetrics {
    const currentTime = Date.now()
    const elapsedTime = (currentTime - this.startTime) / 1000
    const fps = this.frames / elapsedTime
    const frameTime = currentTime - this.lastFrameTime

    const info = this.renderer.info

    return {
      fps: Math.round(fps * 10) / 10,
      frameTime,
      geometries: info.memory.geometries,
      materials: info.programs?.length || 0,
      textures: info.memory.textures,
      drawCalls: info.render.calls,
      vertices: info.render.points + info.render.lines * 2,
      triangles: info.render.triangles,
      memoryUsage: {
        programs: info.programs?.length || 0,
        geometries: info.memory.geometries,
        textures: info.memory.textures
      }
    }
  }

  /**
   * Reset performance counters
   */
  reset(): void {
    this.startTime = Date.now()
    this.frames = 0
    this.renderer.info.reset()
  }
}

// Animation utilities
export class AnimationHelper {
  /**
   * Create smooth attack path animation curve
   */
  static createAttackCurve(source: THREE.Vector3, target: THREE.Vector3): THREE.CatmullRomCurve3 {
    const midpoint = source.clone().add(target).multiplyScalar(0.5)
    const height = Math.max(source.distanceTo(target) * 0.3, 0.5)

    // Create arc that goes above the globe surface
    midpoint.normalize().multiplyScalar(2 + height)

    return new THREE.CatmullRomCurve3([source, midpoint, target])
  }

  /**
   * Ease in-out function for smooth animations
   */
  static easeInOutCubic(t: number): number {
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2
  }

  /**
   * Pulse animation for threat indicators
   */
  static pulseScale(time: number, frequency: number = 2): number {
    return 1 + Math.sin(time * frequency) * 0.3
  }

  /**
   * Color lerp with HSL interpolation
   */
  static lerpColor(color1: THREE.Color, color2: THREE.Color, t: number): THREE.Color {
    const hsl1 = { h: 0, s: 0, l: 0 }
    const hsl2 = { h: 0, s: 0, l: 0 }

    color1.getHSL(hsl1)
    color2.getHSL(hsl2)

    const h = hsl1.h + (hsl2.h - hsl1.h) * t
    const s = hsl1.s + (hsl2.s - hsl1.s) * t
    const l = hsl1.l + (hsl2.l - hsl1.l) * t

    return new THREE.Color().setHSL(h, s, l)
  }
}

// Utility functions for threat data processing
export class ThreatDataProcessor {
  /**
   * Convert API threat data to ThreatPoint format
   */
  static processApiThreat(apiData: Record<string, unknown>): ThreatPoint {
    return {
      id: (apiData.id as string) || `threat-${Date.now()}-${Math.random()}`,
      latitude: (apiData.latitude as number) || 0,
      longitude: (apiData.longitude as number) || 0,
      intensity: Math.min(Math.max((apiData.intensity as number) || 0.5, 0), 1),
      type: (apiData.type as 'malware' | 'ddos' | 'phishing' | 'botnet' | 'exploit') || 'exploit',
      country: (apiData.country as string) || 'UNKNOWN',
      timestamp: (apiData.timestamp as number) || Date.now(),
      details: {
        source: (apiData.source as string) || '',
        target: (apiData.target as string) || '',
        confidence: Math.min(Math.max((apiData.confidence as number) || 0.5, 0), 1),
        severity: (apiData.severity as 'low' | 'medium' | 'high' | 'critical') || 'medium'
      }
    }
  }

  /**
   * Group threats by geographic region for clustering
   */
  static clusterThreatsByRegion(threats: ThreatPoint[], threshold: number = 5): ThreatPoint[][] {
    const clusters: ThreatPoint[][] = []
    const processed = new Set<string>()

    threats.forEach(threat => {
      if (processed.has(threat.id)) return

      const cluster = [threat]
      processed.add(threat.id)

      // Find nearby threats
      threats.forEach(other => {
        if (processed.has(other.id)) return

        const distance = Math.sqrt(
          Math.pow(threat.latitude - other.latitude, 2) +
          Math.pow(threat.longitude - other.longitude, 2)
        )

        if (distance < threshold) {
          cluster.push(other)
          processed.add(other.id)
        }
      })

      clusters.push(cluster)
    })

    return clusters
  }

  /**
   * Calculate threat intensity heatmap
   */
  static generateHeatmapData(threats: ThreatPoint[], resolution: number = 50): number[][] {
    const heatmap: number[][] = Array(resolution).fill(null).map(() => Array(resolution).fill(0))

    threats.forEach(threat => {
      const x = Math.floor(((threat.longitude + 180) / 360) * resolution)
      const y = Math.floor(((90 - threat.latitude) / 180) * resolution)

      if (x >= 0 && x < resolution && y >= 0 && y < resolution) {
        heatmap[y][x] += threat.intensity
      }
    })

    return heatmap
  }
}

// Export singleton instances
export const geoProjection = new GeoProjection()
export const threatMaterials = ThreatMaterialFactory.getInstance()
export const geometryPool = GeometryPool.getInstance()
