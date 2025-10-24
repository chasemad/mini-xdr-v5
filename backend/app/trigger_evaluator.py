"""
Automatic Workflow Trigger Evaluation Engine
Evaluates trigger conditions and executes workflows automatically
"""
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from sqlalchemy import select, and_, func, update
from sqlalchemy.ext.asyncio import AsyncSession

from .models import WorkflowTrigger, Incident, Event, ResponseWorkflow
from .advanced_response_engine import get_response_engine

logger = logging.getLogger(__name__)


class TriggerEvaluator:
    """Evaluates workflow triggers and executes matching workflows"""

    def __init__(self):
        self.response_engine = None
        self._cooldown_tracker = {}  # Track last execution time per trigger
        self._daily_counter = {}  # Track daily execution count per trigger

    async def _get_response_engine(self):
        """Lazy load response engine"""
        if not self.response_engine:
            self.response_engine = await get_response_engine()
        return self.response_engine

    async def evaluate_triggers_for_incident(
        self,
        db: AsyncSession,
        incident: Incident,
        events: List[Event] = None
    ) -> List[str]:
        """
        Evaluate all enabled triggers for a newly created incident

        Args:
            db: Database session
            incident: The incident to evaluate
            events: Related events that triggered the incident

        Returns:
            List of workflow IDs that were executed
        """
        executed_workflows = []

        try:
            # Get all enabled triggers
            query = select(WorkflowTrigger).where(
                WorkflowTrigger.enabled == True
            ).order_by(WorkflowTrigger.priority.desc())

            result = await db.execute(query)
            triggers = result.scalars().all()

            logger.info(f"Evaluating {len(triggers)} enabled triggers for incident #{incident.id}")

            for trigger in triggers:
                try:
                    # Check if trigger conditions match
                    if await self._evaluate_conditions(trigger, incident, events):
                        logger.info(f"Trigger '{trigger.name}' matched incident #{incident.id}")

                        # Check rate limits and cooldown
                        if await self._check_rate_limits(trigger):
                            # Execute the workflow
                            workflow_id = await self._execute_trigger_workflow(
                                db, trigger, incident
                            )

                            if workflow_id:
                                executed_workflows.append(workflow_id)

                                # Update trigger metrics
                                await self._update_trigger_metrics(
                                    db, trigger, success=True
                                )
                            else:
                                await self._update_trigger_metrics(
                                    db, trigger, success=False
                                )
                        else:
                            logger.info(f"Trigger '{trigger.name}' rate limited - skipping")

                except Exception as e:
                    logger.error(f"Error evaluating trigger '{trigger.name}': {e}")
                    await self._update_trigger_metrics(db, trigger, success=False)
                    continue

            await db.commit()

        except Exception as e:
            logger.error(f"Error in trigger evaluation: {e}")

        return executed_workflows

    async def _evaluate_conditions(
        self,
        trigger: WorkflowTrigger,
        incident: Incident,
        events: List[Event] = None
    ) -> bool:
        """
        Evaluate if trigger conditions match the incident

        Conditions format:
        {
            "event_type": "cowrie.login.failed",
            "threshold": 6,
            "window_seconds": 60,
            "source": "honeypot",
            "risk_score_min": 0.5,
            "pattern_match": "brute"
        }
        """
        try:
            conditions = trigger.conditions or {}

            incident_reason = (incident.reason or "").lower()
            incident_category = (incident.threat_category or "").lower()
            incident_severity = (incident.escalation_level or "medium").lower()

            # Optional threat category matching
            if "threat_category" in conditions:
                required = conditions["threat_category"]
                if not isinstance(required, (list, tuple, set)):
                    required = [required]
                required = {str(value).lower() for value in required}
                if incident_category not in required:
                    logger.debug(
                        "Trigger '%s': threat_category '%s' not in %s",
                        trigger.name,
                        incident_category,
                        required,
                    )
                    return False

            if "threat_category_in" in conditions:
                required = conditions["threat_category_in"]
                required = {str(value).lower() for value in required}
                if incident_category not in required:
                    logger.debug(
                        "Trigger '%s': threat_category '%s' not in %s",
                        trigger.name,
                        incident_category,
                        required,
                    )
                    return False

            if "threat_category_pattern" in conditions:
                pattern = str(conditions["threat_category_pattern"]).lower()
                if pattern not in incident_category and pattern not in incident_reason:
                    logger.debug(
                        "Trigger '%s': threat pattern '%s' not found",
                        trigger.name,
                        pattern,
                    )
                    return False

            # Escalation / severity matching
            severity_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}

            if "escalation_level" in conditions:
                required = conditions["escalation_level"]
                if not isinstance(required, (list, tuple, set)):
                    required = [required]
                required = {str(value).lower() for value in required}
                if incident_severity not in required:
                    logger.debug(
                        "Trigger '%s': escalation level '%s' not in %s",
                        trigger.name,
                        incident_severity,
                        required,
                    )
                    return False

            if "escalation_level_min" in conditions:
                min_level = str(conditions["escalation_level_min"]).lower()
                incident_rank = severity_order.get(incident_severity, 1)
                required_rank = severity_order.get(min_level, 1)
                if incident_rank < required_rank:
                    logger.debug(
                        "Trigger '%s': escalation level '%s' below minimum '%s'",
                        trigger.name,
                        incident_severity,
                        min_level,
                    )
                    return False

            # Check event type (if events provided)
            if events and "event_type" in conditions:
                event_type = conditions["event_type"]
                matching_events = [e for e in events if e.eventid == event_type]

                # Check threshold
                if "threshold" in conditions:
                    threshold = conditions["threshold"]
                    if len(matching_events) < threshold:
                        logger.debug(f"Trigger '{trigger.name}': Event count {len(matching_events)} < threshold {threshold}")
                        return False

            # Check risk score minimum
            if "risk_score_min" in conditions:
                min_score = conditions["risk_score_min"]
                incident_score = incident.risk_score or 0.0
                if incident_score < min_score:
                    logger.debug(f"Trigger '{trigger.name}': Risk score {incident_score} < minimum {min_score}")
                    return False

            # Check pattern match in incident reason
            if "pattern_match" in conditions:
                pattern = conditions["pattern_match"].lower()
                if pattern not in incident.reason.lower():
                    logger.debug(f"Trigger '{trigger.name}': Pattern '{pattern}' not in incident reason")
                    return False

            # Check source (honeypot vs production)
            if "source" in conditions:
                source = conditions["source"]
                # For now, all incidents are from honeypot
                # In production, add a source field to Incident model
                if source != "honeypot":
                    logger.debug(f"Trigger '{trigger.name}': Source mismatch")
                    return False

            logger.info(f"✓ Trigger '{trigger.name}' conditions matched")
            return True

        except Exception as e:
            logger.error(f"Error evaluating conditions for trigger '{trigger.name}': {e}")
            return False

    async def _check_rate_limits(self, trigger: WorkflowTrigger) -> bool:
        """
        Check if trigger is within rate limits

        Returns True if execution is allowed, False if rate limited
        """
        try:
            trigger_key = f"trigger_{trigger.id}"
            now = datetime.now(timezone.utc)

            # Check cooldown
            if trigger_key in self._cooldown_tracker:
                last_execution = self._cooldown_tracker[trigger_key]
                time_since_last = (now - last_execution).total_seconds()

                if time_since_last < trigger.cooldown_seconds:
                    logger.debug(f"Trigger '{trigger.name}' in cooldown: {int(time_since_last)}s / {trigger.cooldown_seconds}s")
                    return False

            # Check daily limit
            daily_key = f"{trigger_key}_{now.date()}"
            current_count = self._daily_counter.get(daily_key, 0)

            if current_count >= trigger.max_triggers_per_day:
                logger.warning(f"Trigger '{trigger.name}' hit daily limit: {current_count}/{trigger.max_triggers_per_day}")
                return False

            # Update counters
            self._cooldown_tracker[trigger_key] = now
            self._daily_counter[daily_key] = current_count + 1

            # Clean old daily counters (older than 2 days)
            cutoff_date = (now - timedelta(days=2)).date()
            old_keys = [k for k in self._daily_counter.keys() if k.split('_')[-1] < str(cutoff_date)]
            for key in old_keys:
                del self._daily_counter[key]

            return True

        except Exception as e:
            logger.error(f"Error checking rate limits: {e}")
            return True  # Allow execution on error to avoid blocking

    def _resolve_template_variables(
        self,
        steps: List[Dict[str, Any]],
        incident: Incident,
        events: List[Event] = None
    ) -> List[Dict[str, Any]]:
        """
        Resolve template variables in workflow step parameters

        Supports variables like:
        - {source_ip} or event.source_ip -> incident.src_ip
        - {incident_id} -> incident.id
        - {severity} -> incident.severity
        """
        resolved_steps = []

        # Build context for template variable resolution
        context = {
            "source_ip": incident.src_ip,
            "incident_id": incident.id,
            "severity": incident.escalation_level or "medium",
            "escalation_level": incident.escalation_level or "medium",
            "risk_score": incident.risk_score or 0.0,
            "threat_category": incident.threat_category or "unknown",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        for step in steps:
            resolved_step = step.copy()
            parameters = step.get("parameters", {})
            resolved_params = {}

            def _resolve_value(value):
                """Recursively resolve template variables within parameter values"""
                if isinstance(value, str):
                    if value.startswith("event."):
                        var_name = value.split(".", 1)[1]
                        return context.get(var_name, value)
                    if value.startswith("{") and value.endswith("}"):
                        var_name = value[1:-1]
                        return context.get(var_name, value)
                    return value
                if isinstance(value, list):
                    return [_resolve_value(item) for item in value]
                if isinstance(value, dict):
                    return {k: _resolve_value(v) for k, v in value.items()}
                return value

            for param_key, param_value in parameters.items():
                resolved_params[param_key] = _resolve_value(param_value)

            resolved_step["parameters"] = resolved_params
            resolved_steps.append(resolved_step)

        return resolved_steps

    async def _execute_trigger_workflow(
        self,
        db: AsyncSession,
        trigger: WorkflowTrigger,
        incident: Incident
    ) -> Optional[str]:
        """
        Execute the workflow associated with this trigger

        Returns:
            Workflow ID if successful, None otherwise
        """
        try:
            start_time = datetime.now(timezone.utc)

            logger.info(f"🚀 Executing workflow for trigger '{trigger.name}' on incident #{incident.id}")

            # Get response engine
            engine = await self._get_response_engine()

            # Resolve template variables in workflow steps
            resolved_steps = self._resolve_template_variables(
                trigger.workflow_steps,
                incident
            )

            logger.info(f"Resolved {len(resolved_steps)} workflow steps with incident context (IP: {incident.src_ip})")

            # Create workflow from trigger definition
            # Convert priority string to ResponsePriority enum
            from .advanced_response_engine import ResponsePriority
            priority_map = {
                "low": ResponsePriority.LOW,
                "medium": ResponsePriority.MEDIUM,
                "high": ResponsePriority.HIGH,
                "critical": ResponsePriority.CRITICAL
            }
            priority_enum = priority_map.get(trigger.priority.lower(), ResponsePriority.MEDIUM)

            # Create workflow in database - call with positional arguments
            workflow = await engine.create_workflow(
                incident_id=incident.id,
                playbook_name=trigger.playbook_name,
                steps=resolved_steps,  # Use resolved steps instead of raw trigger.workflow_steps
                auto_execute=trigger.auto_execute,
                priority=priority_enum,
                db_session=db
            )

            if not workflow:
                logger.error(f"Failed to create workflow from trigger '{trigger.name}'")
                return None

            workflow_id = workflow.get("workflow_id")
            workflow_db_id = workflow.get("workflow_db_id")

            logger.info(f"✓ Created workflow {workflow_id} (DB ID: {workflow_db_id})")

            # Check if workflow was already executed (when auto_execute=True, create_workflow executes it)
            if trigger.auto_execute and workflow.get("status") in ["running", "completed"]:
                logger.info(f"✓ Workflow {workflow_id} was auto-executed during creation (status: {workflow.get('status')})")
            elif trigger.auto_execute:
                # This shouldn't happen now, but keep as fallback
                try:
                    execution_result = await engine.execute_workflow(
                        workflow_db_id,
                        db,
                        executed_by=f"auto_trigger:{trigger.name}"
                    )

                    if execution_result.get("status") == "running":
                        logger.info(f"✓ Workflow {workflow_id} execution started")
                    else:
                        logger.warning(f"Workflow {workflow_id} execution status: {execution_result.get('status')}")

                except Exception as e:
                    logger.error(f"Failed to execute workflow {workflow_id}: {e}")
                    return workflow_id  # Still return ID even if execution failed
            else:
                logger.info(f"⏸️  Workflow {workflow_id} created but not auto-executed (requires manual approval)")

            # Calculate response time
            response_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            # Store response time for metrics update
            trigger.last_response_time_ms = response_time_ms

            return workflow_id

        except Exception as e:
            logger.error(f"Error executing workflow for trigger '{trigger.name}': {e}")
            return None

    async def _update_trigger_metrics(
        self,
        db: AsyncSession,
        trigger: WorkflowTrigger,
        success: bool
    ):
        """Update trigger performance metrics"""
        try:
            # Increment counters
            trigger.trigger_count += 1

            if success:
                trigger.success_count += 1
            else:
                trigger.failure_count += 1

            # Calculate success rate
            if trigger.trigger_count > 0:
                trigger.success_rate = (trigger.success_count / trigger.trigger_count) * 100

            # Update average response time (exponential moving average)
            if hasattr(trigger, 'last_response_time_ms') and trigger.last_response_time_ms:
                if trigger.avg_response_time_ms == 0:
                    trigger.avg_response_time_ms = trigger.last_response_time_ms
                else:
                    # EMA with alpha=0.3 (gives more weight to recent values)
                    alpha = 0.3
                    trigger.avg_response_time_ms = (
                        alpha * trigger.last_response_time_ms +
                        (1 - alpha) * trigger.avg_response_time_ms
                    )

            # Update last triggered timestamp
            trigger.last_triggered_at = datetime.now(timezone.utc)

            # Save to database
            await db.flush()

            logger.debug(f"Updated metrics for trigger '{trigger.name}': "
                        f"{trigger.success_count}/{trigger.trigger_count} success, "
                        f"avg response time: {trigger.avg_response_time_ms:.1f}ms")

        except Exception as e:
            logger.error(f"Error updating trigger metrics: {e}")


# Global trigger evaluator instance
trigger_evaluator = TriggerEvaluator()
