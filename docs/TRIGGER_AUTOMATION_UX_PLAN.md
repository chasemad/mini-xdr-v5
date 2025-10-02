# Trigger Automation & Workflow Management UX Blueprint

## Goals
- Give analysts a single pane to review, edit, or retire workflows generated through NLP or manually created triggers.
- Make it obvious when an NLP suggestion has graduated into an automated trigger, including version history and audit metadata.
- Provide streamlined CRUD controls with guardrails (confirmation dialogs, approval flags) to avoid accidental destructive automation.
- Reflect end-to-end NLP coverage so analysts understand when a chat-derived workflow is ready for automation and where parser confidence gaps remain.
- Support request-type differentiation (Response vs Investigation vs Automation) so each automation is contextualized and routed to the right agents/tools.

## Navigation & Placement
- **Location:** rename the current "Auto Triggers" tab to **"Automation & Triggers"**.
- **Sub-tabs:**
  1. **Active Automations** – default view listing every enabled trigger or scheduled workflow (NLP, manual, imported).
  2. **NLP Suggestions** – queue of parsed workflows awaiting analyst review/approval.
  3. **Archived** – disabled or historical automations for audit reference.
  4. **Coverage Insights** *(optional analytics view)* – parser accuracy, fallback counts, request-type distribution.

## Active Automations View
- **Table columns:**
  - Status badge (Enabled, Paused, Awaiting Approval, Error)
  - Trigger Name (clickable → detail drawer)
  - Source (`NLP`, `Manual`, `Template`, `API`)
  - Attached Incident / Scope (incident id, tags, asset groups)
  - Execution Mode (Auto, Approval Required, Simulation)
  - Last Run timestamp & outcome (Success/Failed/Skipped)
  - Owner / Last Editor
  - Confidence / Version (e.g., NLP parse confidence, workflow version #)
  - Request Type (Response, Investigation, Automation, Reporting)
- **Row actions:** `View`, `Pause/Resume`, `Edit`, `Clone`, `Archive`.
- **Bulk actions:** Pause, Resume, Archive (with multi-select).

### Detail Drawer / Page
- High-level summary: description, creation date, last editor, originating NLP prompt (with link to chat transcript).
- Workflow steps displayed in order with categories, estimated durations, linked response actions, and mapped agents/tools.
- Approval requirements & guardrails, including escalation contacts and policy references (e.g., “Requires SOC Tier 2 approval”).
- Request type badge (Response / Investigation / Automation) plus parser confidence and fallback status.
- Change history timeline (creation, edits, execution events) with diff previews for workflow modifications.
- Buttons: `Edit Workflow`, `Open in Designer`, `View Run History`, `Promote/Demote` (move between tabs), `Export JSON`.
- Secondary panel: **Agent Readiness** showing which agents (Containment, Forensics, Reporting) are required and whether credentials/integrations are configured.

## NLP Suggestions Tab
- Feed of recently generated NLP workflows awaiting disposition.
- Each card/table row shows:
  - Natural language prompt
  - Detected request type (Response / Investigation / Automation / Reporting / Q&A)
  - Parsed confidence, priority, approval flag, fallback indicator
  - Suggested steps preview (collapsible)
  - Buttons: `Approve & Automate`, `Convert to Manual Workflow`, `Dismiss`.
- Selecting `Approve & Automate` launches a modal to confirm trigger metadata (name, scope, schedule, auto-execution).
- Modal includes:
  - Editable metadata form (owner, tags, linked incident types)
  - Execution policy presets (Auto, Requires Approval, Simulation)
  - Trigger scope selection (incident filters, external webhook, schedule).
  - Parser diagnostics: matched actions, missing actions (if fallback), recommended synonyms to watch.
- Provide bulk triage actions (Approve/Dismiss multiple) filtered by request type or confidence thresholds.

## Archived Tab
- Read-only list with filters (date, status, owner, source).
- Actions limited to `Restore` (re-enable) or `Delete Permanently` (with warning if audit policy forbids).
- Display retirement reason (manual disable, policy change, superseded by new version).
- Link to automation health metrics prior to archival.

## Editing / Creation Flow
1. **Metadata Step** – name, description, source prompt reference, ownership, tags.
2. **Workflow Designer** – leverage existing canvas with ability to tweak steps, add conditions, and preview agent assignments; highlight unsupported actions.
3. **Execution Policy** – set approval requirements, throttling (max runs per hour/day), maintenance windows, simulation/rollback options.
4. **Automation Rules** – choose trigger type (incident attribute, schedule, external webhook) and define condition builder with AND/OR logic, thresholds, and time windows.
5. **Testing & Validation** – run dry-run simulation against historical incidents; surface parser coverage checks and dependency validation results.
6. **Review & Confirm** – diff summary of changes, risk warnings, final enable toggle, plus optional change justification comment.

## Additional Enhancements
- **Trigger Health Indicators:** show success rate, last failure reason; highlight high-risk automations with pending approvals.
- **Search & Filters:** multi-facet filtering (category, action type, priority, owner, source, tags).
- **Audit Export:** generate PDF/CSV report of automation inventory for compliance.
- **Role-based Controls:** integrate with RBAC so only authorized roles can enable auto execution or delete triggers.
- **Parser Coverage Dashboard:** integrate stats from `docs/NLP_PARSER_COVERAGE.md` – show action detection coverage by category, fallback percentages, and top missing phrases.
- **Request-Type Analytics:** chart distribution of Response vs Investigation vs Automation prompts, average approval time, and agent utilization.
- **Coach Marks & Knowledge Base Links:** contextual help guiding analysts on best practices (e.g., when to require approval, how to refine prompts).
- **Runbook Attachments:** allow linking to documentation/KB articles per automation entry.

## API / Backend Considerations
- Extend `WorkflowTrigger` model & routes to expose:
  - `source` (`nlp`, `manual`, `template`, `external`)
  - `source_prompt`, `parser_confidence`, `parser_version`, `request_type`
  - `owner`, `last_editor`, `last_run`, `last_run_status`
  - `version` / revision history (link to new `workflow_trigger_versions` table)
- Add endpoints for bulk state changes (pause/resume/archive) and tag management.
- Persist NLP prompt + parser output alongside workflow definition for traceability; include fallback template metadata if parser failed.
- Capture change history (who edited, when, what changed) for audit trail – consider `workflow_trigger_audit` table.
- Introduce simulation endpoint (`POST /api/triggers/{id}/simulate`) for dry runs.
- Provide search API with query parameters (tags, request_type, min_confidence, agent dependency).
- Store agent/tool dependencies per workflow step to validate availability before enabling automation.
- Log parser coverage metrics on backend to feed coverage dashboard.

## NLP Workflow Parser Alignment
- Maintain mapping table (see `docs/NLP_PARSER_COVERAGE.md`) that ties phrases → actions and request types.
- In the UI, surface parser diagnostics (confidence, matched actions, missed keywords) so analysts can refine prompts or request parser updates.
- Track fallback occurrences and allow analysts to flag false negatives directly from the suggestion card (feeds a backlog of parser enhancements).
- Provide “Suggest parser update” button that pre-populates issue template with prompt, expected actions, and fallback output.
- Display request-type determination logic (Response/Investigation/etc.) using badges across UI surfaces. When parser can’t classify, show `Unclassified` and route to manual review.
- Integrate agent capability matrix: for each action, show required agent and status (configured/needs credentials/unsupported). Highlight unsupported actions before automation approval.

## NLP Chat Workflow End-to-End Handling
1. **Prompt Intake** – Chat UI captures raw request, attaches session metadata (user, incident context).
2. **Parser Pass** – `parse_workflow_from_natural_language` extracts actions, priority, approval needs, request type, confidence.
3. **Fallback & Template** – if actions < threshold or confidence low, use curated template but mark as `Fallback` with recommended rephrasing.
4. **Agent Validation** – cross-check actions against available agents/tools; raise warnings if capability missing.
5. **Suggestion Queue** – push result (parsed workflow + diagnostics) into NLP Suggestions tab for analyst disposition.
6. **Automation Decision** – analyst approves (→ trigger), converts to manual workflow, or dismisses. All decisions recorded for training data.
7. **Continuous Learning** – aggregate dismissed prompts / missing actions to expand parser coverage and agent capabilities.

## Request Type Handling
- **Incident Response:** default flow; produce workflow actions, highlight containment steps, require approval when destructive.
- **Investigation/Hunt:** parse relevant search/hunt actions, surface recommended queries, optionally convert to investigation playbooks.
- **Automation Configuration:** detect phrases like “every time/whenever” to prompt trigger creation wizard directly.
- **Reporting / Analytics:** route to analytics agents; offer export or summary generation rather than workflow automation.
- **Informational Q&A:** delegate to knowledge base or LLM answer; do not create workflows.

UI surfaces (chat history, suggestion cards, automation tables) should display request type and provide recommended next action (e.g., “Convert to investigation playbook”).

## Next Steps
1. Confirm terminology and navigation with stakeholders; align on request-type badges and automation statuses.
2. Break down backend changes (models/migrations) and frontend components into epics with dependencies (parser logging, trigger CRUD enhancements, UI views).
3. Prototype table views, detail drawer, and suggestion cards in Figma (or preferred tool) incorporating diagnostics, agent readiness, and request-type badges.
4. Establish parser coverage review cadence using metrics surfaced in Coverage Insights tab.
