# Dashboard Guide

The SOC dashboard (`frontend/app/page.tsx`) provides analysts with comprehensive real-time visibility into incidents, AI-powered analysis, and automated response capabilities.

## Layout

The dashboard is organized into three main tabs: Threat Overview, Active Incidents, and Response Actions.

### Threat Overview Tab
1. **Key Metrics & Recent Activity** – Top-level display of Total Incidents, High Priority count, Containment stats, and AI Detection rates, alongside the most recent activity feed.
2. **Live Threat Intelligence** – Central monitoring view with Live Event Stream, Attack Surface details, Top Attack Types, and Response Metrics.
3. **Phase 2 Intelligence** – Bottom section tracking ML model performance, Feature Store metrics, Agent Hub status, and Training Data collection.

### Active Incidents Tab
1. **Filters & Search** – Advanced filtering by severity, status, and search queries.
2. **Incident List** – Detailed cards for each incident with quick actions (Block IP, Isolate Host) and analysis links.

### General Features
1. **Auto-Refresh** – Configurable refresh intervals with live updates.
2. **AI Assistant** – Integrated Copilot context for incident analysis.

## Incident Details

Clicking an incident opens `frontend/app/incidents/incident/[id]/page.tsx`, which provides comprehensive incident analysis and response capabilities:

- **AI Analysis Hub** – Integrated AI analysis with threat assessment, recommendations, and automated plan generation (`/api/incidents/{id}/ai-analysis`).
- **Threat Status Bar** – Real-time threat intelligence and status indicators with automated updates.
- **Unified Response Timeline** – Complete action history with agent coordination and workflow execution tracking.
- **Tactical Decision Center** – Advanced response orchestration with multi-agent coordination, approval workflows, and automated containment.
- **Evidence & Forensics** – Comprehensive evidence collection with agent-assisted forensic analysis and timeline reconstruction.
- **IOC Analysis** – Automated indicator extraction, correlation, and threat intelligence enrichment.
- **Workflow Designer** – Visual workflow creation with drag-and-drop interface and NLP-to-workflow conversion.
- **Attack Path Visualization** – Graph-based attack path analysis with predictive modeling.

## Configuration & Setup

- **API Configuration**: Ensure `NEXT_PUBLIC_API_KEY` and `NEXT_PUBLIC_API_BASE` match backend settings for proper authentication and connectivity.
- **WebSocket Connectivity**: Real-time updates via `useIncidentRealtime` hook with automatic fallback to polling on connection issues.
- **Agent Integration**: Configure agent endpoints and credentials for full AI assistant functionality.
- **Theme & Customization**: Dashboard supports dark/light themes and customizable metric displays.

## Advanced Features

- **Bulk Operations**: Select multiple incidents for bulk analysis, containment, or workflow execution.
- **Export & Reporting**: Generate comprehensive incident reports with AI analysis and timeline data.
- **Collaboration**: Real-time collaboration features for incident response teams.
- **Integration APIs**: RESTful APIs for third-party tool integration and automation.

## Troubleshooting

- **Authentication Issues**: Verify API key configuration and backend connectivity.
- **WebSocket Failures**: Check network policies and WebSocket endpoint availability.
- **Performance Issues**: Monitor browser console for API call performance and optimize filter usage.
- **Feature Access**: Ensure user roles and permissions are correctly configured for advanced features.

Update this guide when new UI components, features, or workflows are added.
