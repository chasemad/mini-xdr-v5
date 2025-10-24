# Automation Designer

The automation designer provides a comprehensive visual workflow creation and management system implemented in `frontend/app/components/WorkflowDesigner.tsx` using `@xyflow/react`.

## Core Concepts

- **Node Library**: `frontend/app/components/ActionNodeLibrary.tsx` loads available actions from `/api/response/actions` and `/api/orchestrator/status`, mapping them to draggable nodes with safety levels, duration estimates, and approval requirements.
- **Canvas**: `ReactFlow` canvas in `frontend/app/components/PlaybookCanvas.tsx` supports start/end nodes, conditional branches, parallel execution paths, and approval workflows.
- **NLP Integration**: Natural language input via `frontend/app/components/NaturalLanguageInput.tsx` converts text descriptions to workflow structures using `/api/workflows/nlp/parse`.
- **Validation Engine**: Advanced client-side validation ensures workflow logic, approval chains, and execution dependencies are properly configured.
- **Execution Engine**: Workflows support immediate execution, scheduled execution, approval-based execution, and monitoring via `ResponseImpactMonitor` and real-time status updates.

## Usage Methods

### Visual Workflow Designer

1. **Access Designer**: Navigate to incident details or automation section to access the workflow designer.
2. **Select Actions**: Browse the action library with categorized actions (containment, forensics, intelligence, etc.) and drag nodes onto the canvas.
3. **Connect Logic**: Create execution flows with conditional branches, parallel paths, and approval gates using the visual canvas.
4. **Configure Properties**: Set action parameters, approval requirements, timeout values, and error handling for each node.
5. **Validate & Save**: Run validation checks and save the workflow with versioning and approval routing options.

### Natural Language Creation

1. **Describe Workflow**: Use the NLP input interface to describe desired automation in natural language.
2. **AI Processing**: The system parses the description using `/api/workflows/nlp/parse` and generates a structured workflow.
3. **Review & Edit**: Review the AI-generated workflow and make adjustments using the visual editor.
4. **Approval & Execution**: Submit for approval if required, then execute with monitoring and rollback capabilities.

### Execution & Monitoring

1. **Execute Workflow**: Run workflows immediately, schedule for later, or submit for approval-based execution.
2. **Real-time Monitoring**: Track execution progress, agent coordination, and impact metrics in real-time.
3. **Error Handling**: Monitor for failures and automatically trigger remediation workflows.
4. **Audit & Reporting**: Review execution history, performance metrics, and generate compliance reports.

## Extending Actions

- **Backend Implementation**: Define new actions in `backend/app/advanced_response_engine.py` or agent-specific modules.
- **Metadata Configuration**: Ensure actions return comprehensive metadata including category, safety_level, estimated_duration, approval requirements, and rollback capabilities.
- **Agent Integration**: For agent-backed actions, implement in the appropriate agent module (`backend/app/agents/`) with proper HMAC authentication.
- **Policy Integration**: Update policy definitions in `policies/` for approval workflows and access controls.
- **UI Registration**: Actions automatically appear in the node library after backend registration.

## Advanced Features

- **Conditional Logic**: Support for if/then/else branches based on action results or external conditions.
- **Parallel Execution**: Run multiple actions simultaneously with synchronization points.
- **Approval Workflows**: Route workflows through organizational approval chains.
- **Error Handling**: Automatic retry logic, alternative action paths, and failure notifications.
- **Template Library**: Pre-built workflow templates for common security scenarios.

## Troubleshooting

- **Action Library Empty**: Verify `/api/response/actions` endpoint and API authentication. Check backend logs for action registration failures.
- **Workflow Validation Errors**: Use browser dev tools to inspect validation payloads. Ensure all required nodes are connected and configured.
- **Execution Failures**: Check workflow status via `/api/response/workflows/{id}/status` and review agent coordination logs.
- **Canvas Issues**: Validate React Flow configuration after updates. Check for node type conflicts and canvas state corruption.
- **NLP Parsing Issues**: Verify `/api/workflows/nlp/parse` connectivity and ensure natural language descriptions are well-formed.

Update this guide when new workflow features, action types, or UI components are added.
