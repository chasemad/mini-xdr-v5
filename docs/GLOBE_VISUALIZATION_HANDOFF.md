# Interactive Threat Globe Visualization - Development Handoff

## Current Status & Context

We have been working on fixing the Interactive Threat Globe visualization in a Mini-XDR cybersecurity system. The globe is supposed to display real-time cyber threats detected from honeypot VMs, but we encountered rendering issues and had to simplify the implementation.

## What We're Building

An interactive 3D globe that visualizes cybersecurity threats in real-time, including:
- **Real threat data** from honeypot VM incidents
- **Geographic visualization** of attack sources and targets
- **Interactive controls** (rotate, zoom, pan)
- **Country boundaries** showing actual world geography
- **Threat indicators** (colored dots/points for different attack types)
- **Attack paths** (animated lines between source and target)
- **Performance monitoring** and optimization

## Current Implementation Status

### ✅ Working Components
- **Basic 3D globe rendering** with Three.js and React Three Fiber
- **Interactive controls** (OrbitControls working)
- **Canvas setup** with proper WebGL initialization
- **Error handling** and fallback states
- **Performance monitoring** infrastructure

### ⚠️ Current Issues
- **Country outlines** appear as rectangles instead of actual country shapes
- **Complex GeoJSON processing** was causing black screen crashes
- **Performance issues** with thousands of individual geometries
- **Currently simplified** to just a blue sphere for stability

## Key Files & Locations

### Main Visualization Component
- **File**: `/Users/chasemad/Desktop/mini-xdr/frontend/app/visualizations/threat-globe.tsx`
- **Status**: Currently shows a simple blue sphere (working but simplified)
- **Original Features**: Had complex country outlines, threat points, attack paths

### GeoJSON Data
- **File**: `/Users/chasemad/Desktop/mini-xdr/frontend/public/world-countries-detailed.geojson` (247KB)
- **Status**: Downloaded detailed world boundaries, but processing causes crashes
- **Issue**: Creating thousands of individual line geometries overwhelms WebGL

### Supporting Files
- **Three.js Helpers**: `/Users/chasemad/Desktop/mini-xdr/frontend/lib/three-helpers.ts`
- **API Integration**: `/Users/chasemad/Desktop/mini-xdr/frontend/lib/api.ts`
- **Threat Data**: `/Users/chasemad/Desktop/mini-xdr/frontend/lib/threat-data.ts`

## Technical Architecture

### Frontend Stack
- **React** with TypeScript
- **Three.js** with React Three Fiber (@react-three/fiber)
- **React Three Drei** for additional components
- **Next.js** framework

### Data Flow (Original Design)
1. **Honeypot VMs** detect real cyber attacks
2. **Backend API** processes and classifies threats
3. **Frontend** fetches threat data via REST/WebSocket
4. **Globe component** renders threats geographically
5. **Real-time updates** show new attacks as they happen

## Previous Working Features (Before Issues)

### Threat Visualization
- **ThreatPoint interface**: `{ id, latitude, longitude, intensity, type, country, timestamp, details }`
- **Attack types**: malware, ddos, phishing, botnet, exploit
- **Intensity-based coloring**: Red for high severity, yellow for medium, etc.
- **Pulsing animations**: Threat points pulse to show activity

### Geographic Features
- **Country outlines**: White lines showing country boundaries
- **Country labels**: Text labels for major countries
- **Attack paths**: Curved lines showing attack source → target
- **Geo projection**: Lat/lng coordinates converted to 3D sphere positions

### Performance Features
- **Geometry pooling**: Reused geometries for better performance
- **Material factory**: Shared materials to reduce draw calls
- **Performance monitoring**: FPS, memory usage, draw calls tracking
- **Adaptive quality**: Reduced detail on slower devices

## Specific Problems Encountered

### 1. Country Outline Rendering Issue
**Problem**: Country outlines appeared as rectangles instead of actual country shapes
**Root Cause**: Original GeoJSON had simplified rectangular bounding boxes
**Attempted Fix**: Downloaded detailed GeoJSON with proper country boundaries
**New Issue**: Detailed data created thousands of geometries, causing crashes

### 2. Performance/Crash Issues
**Problem**: Globe goes black after loading, especially after adding detailed countries
**Root Cause**: Creating individual `<line>` components for each country polygon ring
**Scale**: 200+ countries × multiple polygons × multiple rings = thousands of meshes
**Impact**: Overwhelmed WebGL renderer, caused context loss

### 3. Current Workaround
**Status**: Simplified to basic blue sphere for stability
**Missing**: Country outlines, threat points, real data integration
**Need**: Efficient way to render country boundaries without performance issues

## What Needs To Be Done Next

### Immediate Priority: Fix Country Outlines
1. **Combine geometries**: Merge all country outlines into single BufferGeometry
2. **Use LineSegments**: Instead of individual line components
3. **Optimize data**: Simplify polygon complexity for performance
4. **Test incrementally**: Add countries back gradually

### Secondary: Restore Threat Visualization
1. **Re-enable threat points**: Add back ThreatPoints component
2. **Connect real data**: Integrate with honeypot VM data feed
3. **Add attack paths**: Animated lines between threat source/target
4. **Performance monitoring**: Ensure smooth 60fps rendering

### Future Enhancements
1. **Texture mapping**: Use actual Earth satellite imagery
2. **Advanced animations**: Particle systems for attack visualization
3. **Clustering**: Group nearby threats to reduce visual clutter
4. **Time-based filtering**: Show threats from different time periods

## Key Code Patterns & Examples

### Threat Point Structure
```typescript
interface ThreatPoint {
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
```

### Geographic Projection
```typescript
// Convert lat/lng to 3D sphere coordinates
latLngToVector3(lat: number, lng: number, radius: number = 2): THREE.Vector3 {
  const phi = (90 - lat) * (Math.PI / 180)
  const theta = (lng + 180) * (Math.PI / 180)
  const x = radius * Math.sin(phi) * Math.cos(theta)
  const z = radius * Math.sin(phi) * Math.sin(theta)
  const y = radius * Math.cos(phi)
  return new THREE.Vector3(x, y, z)
}
```

## Current Working State

The visualization currently shows:
- ✅ Blue interactive 3D sphere
- ✅ Mouse controls (rotate, zoom, pan)
- ✅ Proper lighting and materials
- ✅ Debug indicators showing "Canvas Active"
- ❌ No country outlines
- ❌ No threat data
- ❌ No real-time updates

## Testing & Verification

### To Test Current State
1. Navigate to `/visualizations` page
2. Click "Globe" tab
3. Should see blue interactive sphere
4. Mouse controls should work smoothly
5. No console errors

### When Adding Features Back
1. **Monitor performance**: Watch for FPS drops
2. **Check console**: Look for WebGL context loss errors
3. **Test incremental**: Add one feature at a time
4. **Verify data flow**: Ensure real threat data displays correctly

## Environment Setup

### Development Server
- **Frontend**: `cd frontend && npm run dev` (port 3000)
- **Backend**: `cd backend && python -m uvicorn main:app --reload` (port 8000)
- **Database**: SQLite at `backend/xdr.db`

### Key Dependencies
- `@react-three/fiber`: 3D rendering
- `@react-three/drei`: Three.js utilities
- `three`: Core 3D library
- `d3-geo`: Geographic projections

## Success Criteria

When complete, the globe should:
1. **Display actual country outlines** (not rectangles)
2. **Show real threat data** from honeypot VMs
3. **Render smoothly** at 60fps
4. **Handle real-time updates** without crashes
5. **Look professional** and Earth-like

## Contact & Context

This is a cybersecurity threat visualization for a Mini-XDR system. The globe is the centerpiece of the SOC analyst dashboard, showing real-time cyber attacks detected by honeypot infrastructure. Performance and visual accuracy are both critical for operational use.
