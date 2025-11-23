# Incident V2 Details Page - Design Document

**Date**: November 21, 2025
**Status**: Design Phase - No Frontend Changes Yet
**Phase**: Phase 2 Integration

---

## Overview

This document outlines the design for the Incident V2 Details page, which will display all Phase 2 ML and AI agent data without any frontend code changes yet. This design ensures all new backend capabilities are properly wired to the UI.

---

## Backend Data Available

### 1. Enhanced Incident Data Structure

The backend now provides the following data for each incident via `GET /api/incidents/{id}`:

```json
{
  "id": 1,
  "src_ip": "192.168.1.100",
  "status": "open",
  "escalation_level": "high",
  "risk_score": 0.85,
  "threat_category": "malware_botnet",

  // Phase 1: ML Detection Data
  "ml_confidence": 0.87,
  "containment_method": "ml_driven",
  "containment_confidence": 0.87,

  // Phase 1: Council of Models Data
  "council_verdict": "THREAT",
  "council_reasoning": "High confidence malware detection with lateral movement patterns",
  "council_confidence": 0.92,
  "routing_path": ["ml_predict", "council_verify", "gemini_judge"],
  "api_calls_made": ["gemini", "grok", "openai"],
  "processing_time_ms": 1250,
  "gemini_analysis": { /* detailed analysis */ },
  "grok_intel": { /* threat intelligence */ },
  "openai_remediation": { /* automated response scripts */ },

  // Phase 2: Advanced Features (in triage_note.indicators)
  "triage_note": {
    "summary": "Malware/Botnet detected from 192.168.1.100",
    "confidence": 0.92,
    "anomaly_score": 0.85,
    "threat_class": 5,
    "event_count": 45,
    "indicators": {
      "enhanced_model_prediction": {
        "class_probabilities": [0.01, 0.03, 0.02, 0.04, 0.05, 0.85, 0.00],
        "uncertainty_score": 0.15,
        "explanation": "High C2 traffic detected",
        "feature_importance": { /* top features */ }
      },
      "phase2_advanced_features": {
        "feature_count": 100,
        "features_extracted": true,
        "feature_dimensions": "100D (79 base + 21 advanced)"
      }
    },
    "council_verified": true,
    "council_verdict": "THREAT"
  }
}
```

### 2. Agent Coordination Data

Available via `GET /api/agents/incidents/{incident_id}/coordination`:

```json
{
  "incident_id": 1,
  "coordination_status": "council_verified",
  "participating_agents": ["attribution", "containment", "forensics"],
  "agent_decisions": {
    "attribution": {
      "threat_actor": "APT29",
      "confidence": 0.78,
      "tactics": ["initial_access", "lateral_movement"],
      "iocs_identified": 12
    },
    "containment": {
      "actions_taken": ["isolate_host", "block_c2"],
      "effectiveness": 0.92,
      "status": "active"
    },
    "forensics": {
      "evidence_collected": ["memory_dump", "disk_image"],
      "timeline_events": 45,
      "suspicious_processes": 3
    }
  },
  "coordination_timeline": [
    {
      "timestamp": "2025-11-21T08:00:00Z",
      "event": "council_verification",
      "details": "High confidence malware detection",
      "verdict": "THREAT"
    },
    {
      "timestamp": "2025-11-21T08:00:05Z",
      "event": "agent_coordination_initiated",
      "agents": ["attribution", "containment", "forensics"]
    }
  ],
  "recommendations": [
    "Immediate containment and forensic analysis required",
    "Block C2 domains: evil.com, malicious.net",
    "Isolate affected systems: host-01, host-02"
  ]
}
```

### 3. Individual Agent Data

Available via `GET /api/agents/ai/{agent_name}/status` and `/decisions`:

- **Attribution Agent** (`/api/agents/ai/attribution/status`)
- **Containment Agent** (`/api/agents/ai/containment/status`)
- **Forensics Agent** (`/api/agents/ai/forensics/status`)
- **Deception Agent** (`/api/agents/ai/deception/status`)

Each returns:
```json
{
  "agent_name": "attribution",
  "status": "operational",
  "decisions_count": 42,
  "performance_metrics": {
    "accuracy": 0.87,
    "avg_response_time_ms": 450,
    "decisions_today": 12
  },
  "last_active_timestamp": "2025-11-21T08:00:00Z"
}
```

---

## UI Component Design

### Page Structure

```
┌─────────────────────────────────────────────────────────────┐
│ Incident #1 - Malware/Botnet Detection                     │
│ Status: Open | Risk: High (0.85) | ML Confidence: 92%      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ [Tab: Overview] [Tab: Council Analysis] [Tab: Agents]       │
│ [Tab: Timeline] [Tab: Evidence] [Tab: Response Actions]     │
└─────────────────────────────────────────────────────────────┘

Tab Content (see below for each tab)
```

---

## Tab 1: Overview

**Data Sources**:
- `GET /api/incidents/{id}` - incident.triage_note
- Phase 1 ML detection data
- Phase 2 advanced features metadata

**Components Needed** (shadcn-ui):

1. **Threat Summary Card** (`Card`, `CardHeader`, `CardContent`)
   - Threat type, source IP, confidence
   - Risk score with progress bar (`Progress`)
   - Escalation level badge (`Badge`)

2. **ML Detection Details** (`Accordion` with 3 sections)
   - **Section 1: Base Prediction**
     - 7-class probability chart (`Recharts` bar chart)
     - Confidence score with uncertainty
     - Threat class label

   - **Section 2: Phase 2 Advanced Features**
     - Feature extraction status (`Badge` - success/failure)
     - Feature dimensions (100D indicator)
     - Top contributing features (`Table`)

   - **Section 3: Model Metadata**
     - Model version
     - Inference time
     - Feature store cache hit/miss

3. **Event Statistics** (`Card`)
   - Event count
   - Time span
   - Unique ports/IPs
   - Failed login attempts

**Example Layout**:
```tsx
<div className="grid gap-4">
  {/* Threat Summary */}
  <Card>
    <CardHeader>
      <div className="flex items-center justify-between">
        <CardTitle>Malware/Botnet Detection</CardTitle>
        <Badge variant="destructive">High Risk</Badge>
      </div>
    </CardHeader>
    <CardContent>
      <div className="space-y-4">
        <div>
          <Label>ML Confidence</Label>
          <Progress value={92} className="mt-2" />
          <p className="text-sm text-muted-foreground">92% (Uncertainty: 8%)</p>
        </div>
        {/* More metrics */}
      </div>
    </CardContent>
  </Card>

  {/* ML Detection Accordion */}
  <Accordion type="multiple" defaultValue={["base"]}>
    <AccordionItem value="base">
      <AccordionTrigger>Base ML Prediction</AccordionTrigger>
      <AccordionContent>
        {/* Probability chart */}
      </AccordionContent>
    </AccordionItem>

    <AccordionItem value="phase2">
      <AccordionTrigger>
        Phase 2 Advanced Features
        <Badge className="ml-2">100D</Badge>
      </AccordionTrigger>
      <AccordionContent>
        {/* Feature details */}
      </AccordionContent>
    </AccordionItem>
  </Accordion>
</div>
```

---

## Tab 2: Council Analysis

**Data Sources**:
- `GET /api/incidents/{id}` - council_* fields
- Gemini, Grok, OpenAI analysis data

**Components Needed**:

1. **Council Verdict Banner** (`Alert`)
   - Verdict: THREAT/FALSE_POSITIVE/INVESTIGATE
   - Council confidence
   - Routing path visualization

2. **Routing Path** (`Separator` with custom styling)
   - Visual flow: ML → Council → Gemini Judge → Verdict
   - Time spent at each stage
   - API calls made

3. **Three-Column Analysis** (`Tabs` or `Card` grid)
   - **Gemini Judge** (deep reasoning)
     - Reasoning text
     - Confidence adjustment
     - Threat indicators found

   - **Grok Intel** (threat intelligence)
     - TTP mapping
     - Similar attacks
     - Threat actor attribution

   - **OpenAI Remediation** (response scripts)
     - Automated response plan
     - Recommended actions
     - Containment scripts

4. **Performance Metrics** (`Card`)
   - Total processing time
   - API costs (estimated)
   - Cache hit/miss

**Example Layout**:
```tsx
<div className="space-y-4">
  {/* Council Verdict */}
  <Alert variant="destructive">
    <ShieldAlert className="h-4 w-4" />
    <AlertTitle>Council Verdict: THREAT</AlertTitle>
    <AlertDescription>
      Confidence: 92% | Processing Time: 1.25s | Path: ML → Council → Gemini
    </AlertDescription>
  </Alert>

  {/* Routing Path Visualization */}
  <Card>
    <CardHeader>
      <CardTitle>Decision Path</CardTitle>
    </CardHeader>
    <CardContent>
      {/* SVG or div-based flow diagram */}
      <RoutingPathVisualization path={incident.routing_path} />
    </CardContent>
  </Card>

  {/* Analysis Grid */}
  <div className="grid md:grid-cols-3 gap-4">
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center">
          <Sparkles className="mr-2 h-4 w-4" />
          Gemini Judge
        </CardTitle>
      </CardHeader>
      <CardContent>
        {/* Deep reasoning */}
      </CardContent>
    </Card>

    {/* Grok and OpenAI cards */}
  </div>
</div>
```

---

## Tab 3: AI Agents

**Data Sources**:
- `GET /api/agents/incidents/{id}/coordination`
- Individual agent endpoints

**Components Needed**:

1. **Agent Coordination Status** (`Card`)
   - Coordination status badge
   - Number of participating agents
   - Overall recommendation

2. **Agent Cards Grid** (`Card` grid with icons)
   - **Attribution Agent**
     - Threat actor identified
     - Confidence score
     - TTPs mapped
     - IOCs found

   - **Containment Agent**
     - Actions taken
     - Effectiveness score
     - Status (active/completed)
     - Systems isolated

   - **Forensics Agent**
     - Evidence collected
     - Timeline events
     - Suspicious processes
     - Files analyzed

   - **Deception Agent** (if active)
     - Honeytokens deployed
     - Attacker interactions
     - Intelligence gathered

3. **Coordination Timeline** (`Timeline` component or custom)
   - Events in chronological order
   - Agent activation times
   - Decision points
   - Actions executed

4. **Agent Performance Comparison** (`Recharts` radar chart)
   - Compare confidence scores
   - Response times
   - Decision counts

**Example Layout**:
```tsx
<div className="space-y-6">
  {/* Coordination Status */}
  <Card>
    <CardHeader>
      <CardTitle>Agent Coordination</CardTitle>
      <Badge>Council Verified</Badge>
    </CardHeader>
    <CardContent>
      <p>{coordinationData.participating_agents.length} agents participated</p>
      <Alert className="mt-4">
        <AlertTitle>Recommendations</AlertTitle>
        <ul>
          {coordinationData.recommendations.map(rec => (
            <li key={rec}>• {rec}</li>
          ))}
        </ul>
      </Alert>
    </CardContent>
  </Card>

  {/* Agent Cards */}
  <div className="grid md:grid-cols-2 gap-4">
    {/* Attribution */}
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center">
          <Target className="mr-2 h-4 w-4" />
          Attribution
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          <div>
            <Label>Threat Actor</Label>
            <p className="font-semibold">APT29</p>
          </div>
          <div>
            <Label>Confidence</Label>
            <Progress value={78} />
          </div>
          <div>
            <Label>TTPs</Label>
            <div className="flex gap-2">
              <Badge>Initial Access</Badge>
              <Badge>Lateral Movement</Badge>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>

    {/* Other agent cards */}
  </div>

  {/* Coordination Timeline */}
  <Card>
    <CardHeader>
      <CardTitle>Coordination Timeline</CardTitle>
    </CardHeader>
    <CardContent>
      <CoordinationTimeline events={coordinationData.coordination_timeline} />
    </CardContent>
  </Card>
</div>
```

---

## Tab 4: Timeline

**Data Sources**:
- Coordination timeline
- Event logs
- Agent action history

**Components Needed**:

1. **Interactive Timeline** (Custom component with `Separator`)
   - Vertical timeline with timestamps
   - Event cards along timeline
   - Color-coded by event type:
     - Blue: Detection
     - Orange: Council verification
     - Red: Agent coordination
     - Green: Response actions

2. **Event Filters** (`Select`, `Checkbox`)
   - Filter by event type
   - Filter by agent
   - Date/time range

3. **Event Detail Modal** (`Dialog`)
   - Full event details
   - Related events
   - Context

---

## Tab 5: Evidence

**Data Sources**:
- Forensics agent data
- IOCs from attribution
- Network logs

**Components Needed**:

1. **IOC Table** (`Table` with `DataTable`)
   - Type (IP, domain, hash, etc.)
   - Value
   - Confidence
   - Source (agent that found it)
   - Actions (copy, block, search)

2. **Evidence Cards** (`Card` grid)
   - Memory dumps
   - Disk images
   - Network captures
   - Log files

3. **File Analysis** (`Accordion`)
   - File hashes
   - VirusTotal links
   - Static analysis results

---

## Tab 6: Response Actions

**Data Sources**:
- Containment agent actions
- OpenAI remediation scripts
- Manual response history

**Components Needed**:

1. **Quick Actions Bar** (`Button` group)
   - Isolate host
   - Block IP
   - Quarantine file
   - Deploy honeypot

2. **Automated Response Plan** (`Card` with `Checkbox`)
   - Action items from OpenAI
   - Execution status
   - Manual approval required

3. **Action History Table** (`Table`)
   - Timestamp
   - Action taken
   - Status (pending/success/failed)
   - Agent/user responsible

4. **Response Effectiveness** (`Card` with chart)
   - Effectiveness score
   - Time to contain
   - Actions successful/failed

---

## Additional UI Locations

### 1. Dashboard Widgets

**Location**: `/app/page.tsx` (Dashboard)

**New Widgets**:

1. **Phase 2 Performance Widget** (`Card`)
   - Feature store cache hit rate
   - Advanced features usage
   - Training samples collected
   - Next retrain countdown

2. **Agent Coordination Widget** (`Card`)
   - Active agents count
   - Recent coordinations
   - Avg coordination time

3. **ML Confidence Trend** (`Recharts` line chart)
   - ML confidence over time
   - Council override rate
   - Accuracy trend

**shadcn-ui Components**: `Card`, `Progress`, `Badge`, Line chart from `Recharts`

---

### 2. Agents Page

**Location**: `/app/agents/page.tsx`

**Enhancements**:

1. **Agent Status Cards** (update existing)
   - Add performance metrics
   - Add decision count
   - Add last active timestamp

2. **Coordination Hub Card** (new)
   - Total coordinations
   - Active agents
   - Coordination metrics

**shadcn-ui Components**: `Card`, `Badge`, `Table`, `Progress`

---

### 3. Intelligence Page

**Location**: `/app/intelligence/page.tsx`

**New Sections**:

1. **Feature Store Analytics** (`Card`)
   - Cache hit/miss ratio
   - Most frequently extracted features
   - Performance impact

2. **Training Data Collection** (`Card`)
   - Samples collected today
   - Training data quality
   - Next retrain ETA

**shadcn-ui Components**: `Card`, `Progress`, Bar chart from `Recharts`

---

### 4. Settings Page

**Location**: `/app/settings/page.tsx`

**New Settings Sections**:

1. **Phase 2 Configuration** (`Accordion`)
   - Feature store TTL
   - Advanced features toggle
   - Training data collection

2. **Agent Coordination Settings** (`Form`)
   - Enable/disable coordination
   - Coordination threshold
   - Agent priorities

**shadcn-ui Components**: `Accordion`, `Switch`, `Slider`, `Form` components

---

## shadcn-ui Components Needed

### Already Installed
- `Card`, `CardHeader`, `CardContent`
- `Badge`
- `Button`
- `Alert`, `AlertTitle`, `AlertDescription`
- `Accordion`, `AccordionItem`, `AccordionTrigger`, `AccordionContent`

### Need to Install

```bash
# For incident v2 details page
npx shadcn@latest add progress       # ML confidence bars
npx shadcn@latest add separator      # Routing path visualization
npx shadcn@latest add tabs           # Tab navigation
npx shadcn@latest add label          # Form labels
npx shadcn@latest add select         # Filters
npx shadcn@latest add checkbox       # Multi-select filters
npx shadcn@latest add dialog         # Event detail modals
npx shadcn@latest add switch         # Settings toggles
npx shadcn@latest add slider         # Settings sliders
npx shadcn@latest add form           # Settings forms

# For data visualization (already have recharts)
# No additional installation needed - use Recharts library
```

---

## Data Flow Diagram

```
┌─────────────────────┐
│   User Navigates    │
│   to Incident #1    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│  Frontend: /incidents/incident/[id]     │
│  - Fetches incident data                │
│  - Fetches coordination data            │
│  - Fetches agent statuses               │
└──────────┬──────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│  Backend API Calls                                   │
│  1. GET /api/incidents/1                             │
│  2. GET /api/agents/incidents/1/coordination         │
│  3. GET /api/agents/ai/attribution/status            │
│  4. GET /api/agents/ai/containment/status            │
│  5. GET /api/agents/ai/forensics/status              │
└──────────┬───────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│  Backend Processing                                  │
│  - Intelligent Detection (Phase 2 integrated)        │
│    - Feature store caching ✅                        │
│    - Advanced features extraction ✅                 │
│    - Training data collection ✅                     │
│  - Council orchestration                             │
│  - Agent coordination                                │
└──────────┬───────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│  Database                                            │
│  - Incidents table (with Phase 2 fields)             │
│  - Training samples table (Phase 2) ✅               │
│  - Events table                                      │
└──────────┬───────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│  External Services                                   │
│  - Redis (feature store cache) ✅                    │
│  - ML models (enhanced with Phase 2) ✅              │
│  - Gemini, Grok, OpenAI (Council)                    │
│  - Agent coordination hub                            │
└──────────────────────────────────────────────────────┘
```

---

## Implementation Priority

### Phase 1: Core Incident Details (High Priority)
1. Overview tab with ML detection details
2. Council Analysis tab
3. Agents tab with coordination data
4. Install required shadcn-ui components

### Phase 2: Enhanced Features (Medium Priority)
1. Timeline tab with visualization
2. Evidence tab with IOC table
3. Response Actions tab

### Phase 3: Dashboard & Other Pages (Low Priority)
1. Dashboard widgets
2. Agents page enhancements
3. Intelligence page additions
4. Settings page updates

---

## API Endpoints Summary

**Incident Data**:
- `GET /api/incidents/{id}` - Full incident details with Phase 2 data

**Agent Coordination**:
- `GET /api/agents/coordination-hub/status` - Overall coordination status
- `GET /api/agents/ai/{agent_name}/status` - Individual agent status
- `GET /api/agents/ai/{agent_name}/decisions` - Agent decision history
- `GET /api/agents/incidents/{incident_id}/coordination` - Incident-specific coordination

**All endpoints are now live** ✅

---

## Next Steps

1. **Frontend Development** (when ready):
   - Install shadcn-ui components listed above
   - Implement tabs structure
   - Create reusable components (AgentCard, CoordinationTimeline, etc.)
   - Add TypeScript interfaces for all API responses

2. **Testing**:
   - Test API endpoints with real data
   - Verify all Phase 2 fields are populated
   - Check agent coordination responses

3. **Documentation**:
   - API endpoint documentation (OpenAPI/Swagger)
   - Component usage guide
   - State management approach

---

## Questions & Decisions

1. **State Management**: Should we use React Context, Zustand, or Redux for agent coordination state?
   - **Recommendation**: React Context for now, migrate to Zustand if performance issues

2. **Real-time Updates**: Should agent coordination updates be real-time (WebSocket) or polling?
   - **Recommendation**: Polling every 5s for now, WebSocket later

3. **Chart Library**: Continue with Recharts or switch to another?
   - **Recommendation**: Stick with Recharts (already installed)

4. **Mobile Responsive**: Priority for mobile view?
   - **Recommendation**: Desktop-first, tablet next, mobile last (security analyst tool)

---

## Status

✅ **Phase 2 Backend Integration**: Complete
✅ **API Endpoints**: Live
✅ **Design Document**: Complete
⏳ **Frontend Implementation**: Pending (awaiting user approval)
⏳ **Testing**: Pending
⏳ **Documentation**: In progress
