"""
AI Copilot Handler for Mini-XDR
Intelligent conversation manager with intent analysis, parameter collection, and confirmation flows
"""

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .models import Event, Incident, ResponseWorkflow

logger = logging.getLogger(__name__)


class CopilotIntent(str, Enum):
    """Types of user intents the copilot can detect"""

    GENERAL_QUESTION = "general_question"  # Asking for information/analysis
    ACTION_REQUEST = "action_request"  # Wants to execute actions/create workflows
    CLARIFICATION_NEEDED = "clarification_needed"  # Ambiguous or missing info
    CONFIRMATION_PROVIDED = "confirmation_provided"  # User approving/rejecting action


class ResponseType(str, Enum):
    """Types of responses the copilot can return"""

    ANSWER = "answer"  # Direct answer to question
    FOLLOW_UP = "follow_up"  # Needs more information
    CONFIRMATION_REQUIRED = "confirmation_required"  # Ready to execute, needs approval
    EXECUTION_RESULT = "execution_result"  # Action completed


@dataclass
class CopilotResponse:
    """Standardized copilot response format"""

    response_type: ResponseType
    message: str
    confidence: float

    # For follow-up questions
    follow_up_questions: Optional[List[str]] = None
    suggested_options: Optional[Dict[str, List[str]]] = None

    # For confirmation prompts
    action_plan: Optional[Dict[str, Any]] = None
    affected_resources: Optional[List[str]] = None
    risk_level: Optional[str] = None
    estimated_duration: Optional[str] = None

    # For execution results
    workflow_id: Optional[str] = None
    workflow_db_id: Optional[int] = None
    execution_details: Optional[Dict[str, Any]] = None

    # Conversation state
    conversation_id: Optional[str] = None
    pending_action_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        result = {
            "response_type": self.response_type.value,
            "message": self.message,
            "confidence": self.confidence,
        }

        # Add optional fields if present
        if self.follow_up_questions:
            result["follow_up_questions"] = self.follow_up_questions
        if self.suggested_options:
            result["suggested_options"] = self.suggested_options
        if self.action_plan:
            result["action_plan"] = self.action_plan
        if self.affected_resources:
            result["affected_resources"] = self.affected_resources
        if self.risk_level:
            result["risk_level"] = self.risk_level
        if self.estimated_duration:
            result["estimated_duration"] = self.estimated_duration
        if self.workflow_id:
            result["workflow_id"] = self.workflow_id
        if self.workflow_db_id:
            result["workflow_db_id"] = self.workflow_db_id
        if self.execution_details:
            result["execution_details"] = self.execution_details
        if self.conversation_id:
            result["conversation_id"] = self.conversation_id
        if self.pending_action_id:
            result["pending_action_id"] = self.pending_action_id

        return result


class CopilotRequestAnalyzer:
    """Analyzes user requests to determine intent and extract parameters"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.openai_client = None
        self.initialized = False

    async def initialize(self):
        """Initialize OpenAI client"""
        try:
            if settings.llm_provider == "openai" and settings.openai_api_key:
                import openai

                self.openai_client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
                self.initialized = True
                self.logger.info("CopilotRequestAnalyzer initialized with OpenAI")
            else:
                self.logger.warning("OpenAI not configured, using fallback analysis")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI: {e}")

    async def analyze_intent(
        self,
        query: str,
        incident_data: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[CopilotIntent, Dict[str, Any]]:
        """
        Analyze user query to determine intent and extract information

        Returns:
            Tuple of (intent, extracted_params)
        """

        # Quick keyword-based detection for common patterns
        query_lower = query.lower()

        # Check for confirmation responses
        confirmation_keywords = [
            "yes",
            "approve",
            "confirm",
            "go ahead",
            "proceed",
            "do it",
            "execute",
        ]
        rejection_keywords = ["no", "cancel", "stop", "don't", "reject", "abort"]

        if any(keyword in query_lower for keyword in confirmation_keywords):
            return CopilotIntent.CONFIRMATION_PROVIDED, {"approved": True}
        if any(keyword in query_lower for keyword in rejection_keywords):
            return CopilotIntent.CONFIRMATION_PROVIDED, {"approved": False}

        # Check for action keywords
        action_keywords = [
            "block",
            "isolate",
            "quarantine",
            "ban",
            "deploy",
            "capture",
            "terminate",
            "kill",
            "disable",
            "revoke",
            "reset",
            "alert",
            "notify",
            "contain",
            "encrypt",
            "backup",
            "scan",
            "delete",
            "unblock",
            "restore",
            "enable",
        ]

        has_action_keyword = any(keyword in query_lower for keyword in action_keywords)

        # Use AI for more sophisticated analysis if available
        if self.initialized and self.openai_client:
            return await self._ai_analyze_intent(
                query, incident_data, chat_history, has_action_keyword
            )
        else:
            # Fallback to keyword-based analysis
            return self._fallback_analyze_intent(query, has_action_keyword)

    async def _ai_analyze_intent(
        self,
        query: str,
        incident_data: Optional[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, Any]]],
        has_action_keyword: bool,
    ) -> Tuple[CopilotIntent, Dict[str, Any]]:
        """Use OpenAI to analyze intent"""

        try:
            # Build context for AI
            context_parts = [f"User query: {query}"]

            if incident_data:
                context_parts.append(
                    f"Context: Analyzing incident #{incident_data.get('id')} from IP {incident_data.get('src_ip')}"
                )

            if chat_history:
                recent_history = chat_history[-3:]  # Last 3 messages
                history_str = "\n".join(
                    [
                        f"{'User' if msg.get('type') == 'user' else 'AI'}: {msg.get('content', '')[:100]}"
                        for msg in recent_history
                    ]
                )
                context_parts.append(f"Recent conversation:\n{history_str}")

            context = "\n\n".join(context_parts)

            prompt = f"""You are an AI security analyst assistant. Analyze the user's request and determine their intent.

{context}

Classify the intent as one of:
1. GENERAL_QUESTION - User is asking for information, analysis, or explanation
2. ACTION_REQUEST - User wants to execute security actions or create workflows
3. CLARIFICATION_NEEDED - Query is ambiguous or lacks required information

For ACTION_REQUEST, extract:
- target_ip: IP address to act on (if mentioned)
- action_types: List of actions requested (block_ip, isolate_host, etc.)
- duration: Time duration if mentioned
- urgency: low/medium/high/critical

Respond in JSON format:
{{
    "intent": "GENERAL_QUESTION|ACTION_REQUEST|CLARIFICATION_NEEDED",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "extracted_params": {{
        "target_ip": "x.x.x.x or null",
        "action_types": ["action1", "action2"],
        "duration": "duration or null",
        "urgency": "low|medium|high|critical"
    }}
}}"""

            response = await self.openai_client.chat.completions.create(
                model=settings.openai_model or "gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500,
            )

            # Parse AI response
            content = response.choices[0].message.content

            # Extract JSON from response
            import re

            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group(0))

                intent_str = analysis.get("intent", "GENERAL_QUESTION")
                intent = CopilotIntent(intent_str.lower())

                extracted_params = analysis.get("extracted_params", {})
                extracted_params["confidence"] = analysis.get("confidence", 0.7)
                extracted_params["reasoning"] = analysis.get("reasoning", "")

                return intent, extracted_params
            else:
                self.logger.warning(
                    "Failed to parse AI intent analysis, using fallback"
                )
                return self._fallback_analyze_intent(query, has_action_keyword)

        except Exception as e:
            self.logger.error(f"AI intent analysis failed: {e}")
            return self._fallback_analyze_intent(query, has_action_keyword)

    def _fallback_analyze_intent(
        self, query: str, has_action_keyword: bool
    ) -> Tuple[CopilotIntent, Dict[str, Any]]:
        """Fallback keyword-based intent analysis"""

        query_lower = query.lower()

        # Question patterns
        question_keywords = [
            "what",
            "why",
            "how",
            "when",
            "where",
            "who",
            "explain",
            "tell me",
            "show me",
            "describe",
            "status",
            "information",
            "details",
        ]

        has_question_pattern = any(
            keyword in query_lower for keyword in question_keywords
        )

        if has_action_keyword and not has_question_pattern:
            return CopilotIntent.ACTION_REQUEST, {
                "confidence": 0.7,
                "reasoning": "Detected action keyword",
            }
        elif has_question_pattern:
            return CopilotIntent.GENERAL_QUESTION, {
                "confidence": 0.8,
                "reasoning": "Detected question pattern",
            }
        else:
            # Ambiguous
            return CopilotIntent.CLARIFICATION_NEEDED, {
                "confidence": 0.5,
                "reasoning": "Query intent unclear",
            }


class ParameterCollector:
    """Identifies missing parameters and generates follow-up questions"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.openai_client = None
        self.initialized = False

    async def initialize(self):
        """Initialize OpenAI client"""
        try:
            if settings.llm_provider == "openai" and settings.openai_api_key:
                import openai

                self.openai_client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
                self.initialized = True
                self.logger.info("ParameterCollector initialized with OpenAI")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI: {e}")

    async def collect_missing_parameters(
        self,
        query: str,
        action_types: List[str],
        current_params: Dict[str, Any],
        incident_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        Identify missing parameters and generate follow-up questions

        Returns:
            Tuple of (questions, suggested_options)
        """

        if self.initialized and self.openai_client:
            return await self._ai_generate_questions(
                query, action_types, current_params, incident_data
            )
        else:
            return self._fallback_generate_questions(
                action_types, current_params, incident_data
            )

    async def _ai_generate_questions(
        self,
        query: str,
        action_types: List[str],
        current_params: Dict[str, Any],
        incident_data: Optional[Dict[str, Any]],
    ) -> Tuple[List[str], Dict[str, List[str]]]:
        """Use AI to generate intelligent follow-up questions"""

        try:
            context_parts = [
                f"User wants to: {query}",
                f"Detected actions: {', '.join(action_types)}",
                f"Current parameters: {json.dumps(current_params, indent=2)}",
            ]

            if incident_data:
                context_parts.append(
                    f"Incident context: IP {incident_data.get('src_ip')}, Severity: {incident_data.get('escalation_level', 'unknown')}"
                )

            context = "\n".join(context_parts)

            prompt = f"""You are a security analyst helping collect information for incident response actions.

{context}

The user wants to execute security actions but some parameters might be missing. Generate 1-3 follow-up questions to collect missing critical information.

For each action type, consider what parameters are needed:
- block_ip: IP address, duration, reason
- isolate_host: host identifier, isolation level
- alert_analysts: urgency, priority
- reset_passwords: which accounts, scope

Generate friendly, specific questions. Provide suggested options where appropriate.

Respond in JSON format:
{{
    "questions": ["Question 1?", "Question 2?"],
    "suggested_options": {{
        "duration": ["1 hour", "24 hours", "permanent"],
        "isolation_level": ["full", "network-only", "partial"]
    }}
}}"""

            response = await self.openai_client.chat.completions.create(
                model=settings.openai_model or "gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400,
            )

            content = response.choices[0].message.content

            # Extract JSON
            import re

            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                questions = result.get("questions", [])
                suggested_options = result.get("suggested_options", {})
                return questions, suggested_options
            else:
                return self._fallback_generate_questions(
                    action_types, current_params, incident_data
                )

        except Exception as e:
            self.logger.error(f"AI question generation failed: {e}")
            return self._fallback_generate_questions(
                action_types, current_params, incident_data
            )

    def _fallback_generate_questions(
        self,
        action_types: List[str],
        current_params: Dict[str, Any],
        incident_data: Optional[Dict[str, Any]],
    ) -> Tuple[List[str], Dict[str, List[str]]]:
        """Fallback question generation based on action types"""

        questions = []
        suggested_options = {}

        # Check for missing IP if needed
        if any(
            action in action_types
            for action in ["block_ip", "unblock_ip", "threat_intel_lookup"]
        ):
            if not current_params.get("target_ip") and not (
                incident_data and incident_data.get("src_ip")
            ):
                questions.append("Which IP address should I target for this action?")

        # Check for duration
        if any(action in action_types for action in ["block_ip", "isolate_host"]):
            if not current_params.get("duration"):
                questions.append("How long should this action be in effect?")
                suggested_options["duration"] = [
                    "1 hour",
                    "24 hours",
                    "7 days",
                    "Permanent",
                ]

        # Check for isolation level
        if "isolate_host" in action_types:
            if not current_params.get("isolation_level"):
                questions.append("What level of host isolation should I apply?")
                suggested_options["isolation_level"] = [
                    "Full isolation",
                    "Network-only",
                    "Partial",
                ]

        # Generic question if nothing specific
        if not questions:
            questions.append(
                "Can you provide more details about how you'd like me to execute this action?"
            )

        return questions, suggested_options


class ConfirmationGenerator:
    """Generates detailed confirmation prompts with risk assessment"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def generate_confirmation(
        self, workflow_intent: Any, incident_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a detailed confirmation prompt

        Args:
            workflow_intent: Parsed workflow intent from NLPWorkflowParser
            incident_data: Context about the incident

        Returns:
            Dictionary with confirmation details
        """

        actions = workflow_intent.actions if hasattr(workflow_intent, "actions") else []

        # Build action summary
        action_summary = []
        affected_resources = []

        for action in actions:
            action_type = action.get("action_type", "unknown")
            params = action.get("parameters", {})

            # Create human-readable action description
            desc = self._describe_action(action_type, params)
            action_summary.append(desc)

            # Extract affected resources
            if params.get("ip"):
                affected_resources.append(f"IP: {params['ip']}")
            if params.get("target_ip"):
                affected_resources.append(f"IP: {params['target_ip']}")
            if params.get("host"):
                affected_resources.append(f"Host: {params['host']}")

        # Add incident IP if available
        if incident_data and incident_data.get("src_ip"):
            src_ip = incident_data["src_ip"]
            if f"IP: {src_ip}" not in affected_resources:
                affected_resources.append(f"IP: {src_ip}")

        # Assess risk level
        risk_level = self._assess_risk_level(actions, incident_data)

        # Estimate duration
        estimated_duration = self._estimate_duration(actions)

        return {
            "action_summary": action_summary,
            "affected_resources": list(set(affected_resources)),  # Deduplicate
            "risk_level": risk_level,
            "estimated_duration": estimated_duration,
            "total_actions": len(actions),
            "requires_approval": workflow_intent.approval_required
            if hasattr(workflow_intent, "approval_required")
            else True,
        }

    def _describe_action(self, action_type: str, params: Dict[str, Any]) -> str:
        """Create human-readable action description"""

        descriptions = {
            "block_ip": f"Block IP address {params.get('ip') or params.get('target_ip', 'from incident')}",
            "unblock_ip": f"Unblock IP address {params.get('ip') or params.get('target_ip', 'from incident')}",
            "isolate_host": f"Isolate host {params.get('host', 'from incident')} - {params.get('isolation_level', 'full')} isolation",
            "un_isolate_host": f"Remove host isolation for {params.get('host', 'from incident')}",
            "reset_passwords": f"Reset passwords for affected accounts",
            "threat_intel_lookup": f"Perform threat intelligence lookup",
            "deploy_waf_rules": f"Deploy WAF rules for protection",
            "alert_security_analysts": f"Send alert to security team",
            "create_incident_case": f"Create formal incident case",
            "capture_network_traffic": f"Capture network traffic for analysis",
            "hunt_similar_attacks": f"Hunt for similar attack patterns",
            "check_database_integrity": f"Check database integrity",
        }

        return descriptions.get(action_type, f"Execute {action_type.replace('_', ' ')}")

    def _assess_risk_level(
        self, actions: List[Dict[str, Any]], incident_data: Optional[Dict[str, Any]]
    ) -> str:
        """Assess risk level of the proposed actions"""

        # High-risk actions
        high_risk_actions = {
            "reset_passwords",
            "delete_malicious_files",
            "terminate_process",
        }
        # Medium-risk actions
        medium_risk_actions = {
            "block_ip",
            "isolate_host",
            "deploy_waf_rules",
            "deploy_firewall_rules",
        }
        # Low-risk actions
        low_risk_actions = {
            "threat_intel_lookup",
            "alert_security_analysts",
            "create_incident_case",
            "capture_network_traffic",
        }

        action_types = [a.get("action_type", "") for a in actions]

        if any(action in high_risk_actions for action in action_types):
            return "high"
        elif any(action in medium_risk_actions for action in action_types):
            # Check incident severity
            if incident_data and incident_data.get("escalation_level") in [
                "critical",
                "high",
            ]:
                return "medium"
            return "medium"
        elif any(action in low_risk_actions for action in action_types):
            return "low"
        else:
            return "medium"  # Default to medium for unknown actions

    def _estimate_duration(self, actions: List[Dict[str, Any]]) -> str:
        """Estimate how long actions will take"""

        total_actions = len(actions)

        if total_actions <= 2:
            return "< 1 minute"
        elif total_actions <= 5:
            return "1-2 minutes"
        else:
            return f"~{total_actions // 3} minutes"


class AICopilotHandler:
    """Main handler for AI copilot interactions"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.request_analyzer = CopilotRequestAnalyzer()
        self.parameter_collector = ParameterCollector()
        self.confirmation_generator = ConfirmationGenerator()

        # Store pending actions for confirmation
        self.pending_actions: Dict[str, Dict[str, Any]] = {}

    async def initialize(self):
        """Initialize all components"""
        await self.request_analyzer.initialize()
        await self.parameter_collector.initialize()
        self.logger.info("AICopilotHandler initialized")

    async def handle_request(
        self,
        query: str,
        incident_id: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        conversation_id: Optional[str] = None,
        pending_action_id: Optional[str] = None,
        db_session: Optional[AsyncSession] = None,
    ) -> CopilotResponse:
        """
        Main entry point for handling copilot requests

        Args:
            query: User's query/message
            incident_id: Optional incident ID for context
            context: Additional context (incident_data, chat_history, etc.)
            conversation_id: ID for multi-turn conversations
            pending_action_id: ID of pending action if this is a confirmation response
            db_session: Database session

        Returns:
            CopilotResponse with appropriate response type and data
        """

        try:
            # Extract context data
            incident_data = context.get("incident_data") if context else None
            chat_history = context.get("chat_history") if context else None

            # Fetch incident data from DB if ID provided but no data
            if incident_id and not incident_data and db_session:
                incident = await db_session.get(Incident, incident_id)
                if incident:
                    incident_data = {
                        "id": incident.id,
                        "src_ip": incident.src_ip,
                        "reason": incident.reason,
                        "status": incident.status,
                        "escalation_level": incident.escalation_level,
                        "risk_score": incident.risk_score,
                    }

            # Analyze intent
            intent, extracted_params = await self.request_analyzer.analyze_intent(
                query, incident_data, chat_history
            )

            self.logger.info(
                f"Detected intent: {intent.value} with confidence {extracted_params.get('confidence', 0)}"
            )

            # Handle based on intent
            if intent == CopilotIntent.CONFIRMATION_PROVIDED:
                return await self._handle_confirmation(
                    query, extracted_params, pending_action_id, incident_id, db_session
                )

            elif intent == CopilotIntent.GENERAL_QUESTION:
                return await self._handle_general_question(
                    query, incident_id, incident_data, chat_history, db_session
                )

            elif intent == CopilotIntent.ACTION_REQUEST:
                return await self._handle_action_request(
                    query, incident_id, incident_data, extracted_params, db_session
                )

            elif intent == CopilotIntent.CLARIFICATION_NEEDED:
                return CopilotResponse(
                    response_type=ResponseType.FOLLOW_UP,
                    message="I'd be happy to help! Could you clarify what you'd like me to do? I can:\n\n• Answer questions about incidents and threats\n• Execute security actions (block IPs, isolate hosts, etc.)\n• Create response workflows\n• Investigate security events",
                    confidence=0.6,
                    follow_up_questions=["What would you like to know or do?"],
                    conversation_id=conversation_id or str(uuid.uuid4()),
                )

            else:
                # Fallback
                return CopilotResponse(
                    response_type=ResponseType.ANSWER,
                    message="I'm not sure how to help with that request. Try asking a specific question or requesting a security action.",
                    confidence=0.3,
                )

        except Exception as e:
            self.logger.error(f"Error handling copilot request: {e}", exc_info=True)
            return CopilotResponse(
                response_type=ResponseType.ANSWER,
                message=f"I encountered an error processing your request: {str(e)}",
                confidence=0.0,
            )

    async def _handle_confirmation(
        self,
        query: str,
        extracted_params: Dict[str, Any],
        pending_action_id: Optional[str],
        incident_id: Optional[int],
        db_session: Optional[AsyncSession],
    ) -> CopilotResponse:
        """Handle user confirmation (approve/reject)"""

        approved = extracted_params.get("approved", False)

        if not pending_action_id or pending_action_id not in self.pending_actions:
            return CopilotResponse(
                response_type=ResponseType.ANSWER,
                message="I don't have any pending actions to confirm. Please make a new request.",
                confidence=0.8,
            )

        pending_data = self.pending_actions[pending_action_id]

        if not approved:
            # User rejected the action
            del self.pending_actions[pending_action_id]
            return CopilotResponse(
                response_type=ResponseType.ANSWER,
                message="Understood. I've cancelled that action. Let me know if you'd like to do something else!",
                confidence=1.0,
            )

        # User approved - execute the workflow
        try:
            workflow_intent = pending_data["workflow_intent"]

            # Create workflow in database
            workflow = ResponseWorkflow(
                workflow_id=f"chat_{uuid.uuid4().hex[:12]}",
                incident_id=incident_id,
                playbook_name=pending_data.get(
                    "playbook_name", f"Chat Workflow: {query[:50]}..."
                ),
                steps=workflow_intent.actions,
                approval_required=False,  # Already approved
                auto_executed=False,
                total_steps=len(workflow_intent.actions),
                ai_confidence=workflow_intent.confidence,
            )

            db_session.add(workflow)
            await db_session.commit()
            await db_session.refresh(workflow)

            # Clean up pending action
            del self.pending_actions[pending_action_id]

            # ✨ NEW: Execute the workflow immediately
            self.logger.info(f"Executing workflow {workflow.id} after user approval")

            try:
                from .advanced_response_engine import get_response_engine

                response_engine = await get_response_engine()

                execution_result = await response_engine.execute_workflow(
                    workflow_db_id=workflow.id,
                    db_session=db_session,
                    executed_by="copilot_user_approved",
                )

                # Format execution results for user
                success_count = 0
                failed_count = 0
                action_results = []

                for result in execution_result.get("results", []):
                    action_type = result.get("action_type", "Unknown action")
                    # Clean up action name for display
                    action_name = action_type.replace("_", " ").title()

                    if result.get("success"):
                        success_count += 1
                        detail = result.get("result", {}).get("detail", "")
                        action_results.append(
                            f"✅ **{action_name}** - Success\n   └─ {detail[:100] if detail else 'Completed successfully'}"
                        )
                    else:
                        failed_count += 1
                        error = result.get("error", "Unknown error")
                        action_results.append(
                            f"❌ **{action_name}** - Failed\n   └─ {error[:100]}"
                        )

                # Build response message
                if failed_count == 0:
                    message = f"✅ **All Actions Executed Successfully!**\n\n"
                elif success_count == 0:
                    message = f"❌ **Workflow Execution Failed**\n\n"
                else:
                    message = f"⚠️ **Workflow Partially Completed**\n\n"

                message += "**Results:**\n" + "\n\n".join(action_results)

                message += f"\n\n**Summary:**"
                message += (
                    f"\n• ✅ Successful: {success_count}/{len(workflow_intent.actions)}"
                )
                if failed_count > 0:
                    message += (
                        f"\n• ❌ Failed: {failed_count}/{len(workflow_intent.actions)}"
                    )
                message += f"\n• ⏱️ Execution Time: {execution_result.get('execution_time_ms', 0)}ms"
                message += f"\n• 📋 Workflow ID: {workflow.workflow_id}"

                return CopilotResponse(
                    response_type=ResponseType.EXECUTION_RESULT,
                    message=message,
                    confidence=1.0 if failed_count == 0 else 0.7,
                    workflow_id=workflow.workflow_id,
                    workflow_db_id=workflow.id,
                    execution_details={
                        "success": execution_result.get("success", False),
                        "results": execution_result.get("results", []),
                        "execution_time_ms": execution_result.get(
                            "execution_time_ms", 0
                        ),
                        "steps_completed": execution_result.get("steps_completed", 0),
                        "total_steps": execution_result.get(
                            "total_steps", len(workflow_intent.actions)
                        ),
                        "success_count": success_count,
                        "failed_count": failed_count,
                    },
                )

            except Exception as exec_error:
                self.logger.error(f"Workflow execution failed: {exec_error}")
                return CopilotResponse(
                    response_type=ResponseType.ANSWER,
                    message=f"⚠️ **Workflow created but execution encountered an error:**\n\n{str(exec_error)}\n\n**Workflow ID:** {workflow.workflow_id} (ID: {workflow.id})\n\nYou can retry execution from the workflows page or try a different approach.",
                    confidence=0.5,
                    workflow_id=workflow.workflow_id,
                    workflow_db_id=workflow.id,
                )

        except Exception as e:
            self.logger.error(f"Failed to create workflow after confirmation: {e}")
            return CopilotResponse(
                response_type=ResponseType.ANSWER,
                message=f"I encountered an error creating the workflow: {str(e)}",
                confidence=0.0,
            )

    async def _handle_general_question(
        self,
        query: str,
        incident_id: Optional[int],
        incident_data: Optional[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, Any]]],
        db_session: Optional[AsyncSession],
    ) -> CopilotResponse:
        """Handle general information questions"""

        # Import the existing contextual analysis function
        from .main import _generate_contextual_analysis, _recent_events_for_ip

        # Check if this is an incident-specific question or general knowledge
        incident_specific_keywords = [
            "this incident",
            "the incident",
            "incident #",
            "incident id",
            "this attack",
            "the attack",
            "this ip",
            "the ip",
            "ioc",
            "indicator",
            "timeline",
            "events from",
        ]

        is_incident_specific = any(
            keyword in query.lower() for keyword in incident_specific_keywords
        )

        # If question is incident-specific and we have incident data
        if is_incident_specific and incident_id and incident_data and db_session:
            try:
                incident_obj = await db_session.get(Incident, incident_id)
                if incident_obj:
                    recent_events = await _recent_events_for_ip(
                        db_session, incident_obj.src_ip
                    )

                    # Generate contextual response
                    response_message = await _generate_contextual_analysis(
                        query, incident_obj, recent_events, incident_data
                    )

                    return CopilotResponse(
                        response_type=ResponseType.ANSWER,
                        message=response_message,
                        confidence=0.85,
                    )
            except Exception as e:
                self.logger.error(f"Error generating contextual analysis: {e}")

        # For general knowledge questions, use OpenAI directly
        if self.request_analyzer.initialized and self.request_analyzer.openai_client:
            try:
                return await self._answer_with_ai(query, incident_data, chat_history)
            except Exception as e:
                self.logger.error(f"AI answer generation failed: {e}")

        # Fallback: provide helpful response based on query content
        return await self._fallback_general_answer(query, incident_data)

    async def _answer_with_ai(
        self,
        query: str,
        incident_data: Optional[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, Any]]],
    ) -> CopilotResponse:
        """Use OpenAI to answer general questions"""

        try:
            # Build context
            context_parts = []

            if incident_data:
                context_parts.append(
                    f"Context: User is viewing incident #{incident_data.get('id')} "
                    f"from IP {incident_data.get('src_ip')} "
                    f"(severity: {incident_data.get('escalation_level', 'unknown')})"
                )

            if chat_history:
                recent = chat_history[-3:]
                history_str = "\n".join(
                    [
                        f"{'User' if msg.get('type') == 'user' else 'Assistant'}: {msg.get('content', '')[:150]}"
                        for msg in recent
                    ]
                )
                context_parts.append(f"Recent conversation:\n{history_str}")

            context = (
                "\n\n".join(context_parts)
                if context_parts
                else "No additional context."
            )

            system_prompt = """You are an expert cybersecurity analyst assistant helping SOC analysts.

Your role is to:
1. Answer questions about cybersecurity concepts, threats, and best practices
2. Provide actionable advice for incident response
3. Explain technical security topics clearly
4. When relevant, suggest specific security actions the analyst can take

Keep responses:
- Concise but informative (2-4 paragraphs max)
- Focused on practical security operations
- Professional but conversational
- Action-oriented when appropriate

If the user asks about a general security topic, provide a clear explanation.
If they ask about a specific incident, reference the context provided."""

            user_prompt = f"""Question: {query}

{context}

Provide a helpful, accurate response."""

            response = (
                await self.request_analyzer.openai_client.chat.completions.create(
                    model=settings.openai_model or "gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.7,
                    max_tokens=500,
                )
            )

            answer = response.choices[0].message.content

            return CopilotResponse(
                response_type=ResponseType.ANSWER,
                message=answer or "I'm having trouble generating a response right now.",
                confidence=0.9,
            )

        except Exception as e:
            self.logger.error(f"OpenAI answer generation failed: {e}")
            raise

    async def _fallback_general_answer(
        self, query: str, incident_data: Optional[Dict[str, Any]]
    ) -> CopilotResponse:
        """Provide fallback answers for common questions without OpenAI"""

        query_lower = query.lower()

        # Common security questions - knowledge base
        if "ransomware" in query_lower:
            return CopilotResponse(
                response_type=ResponseType.ANSWER,
                message="""**Ransomware Attack Overview:**

Ransomware is a type of malicious software that encrypts a victim's files or locks their systems, making them inaccessible. Attackers then demand a ransom payment (usually in cryptocurrency) in exchange for the decryption key.

**Common Attack Vectors:**
• Phishing emails with malicious attachments
• Exploiting unpatched vulnerabilities
• Compromised Remote Desktop Protocol (RDP)
• Malicious downloads from compromised websites

**Response Actions:**
1. Immediately isolate affected systems
2. Preserve evidence for forensics
3. Assess the scope of infection
4. Check backups and restoration options
5. Report to appropriate authorities

Would you like me to help you create a response workflow for a potential ransomware incident?""",
                confidence=0.8,
            )

        elif any(
            word in query_lower for word in ["ddos", "dos attack", "denial of service"]
        ):
            return CopilotResponse(
                response_type=ResponseType.ANSWER,
                message="""**DDoS Attack Overview:**

A Distributed Denial of Service (DDoS) attack overwhelms a target system, service, or network with a flood of Internet traffic, making it unavailable to legitimate users.

**Common Types:**
• Volume-based attacks (floods)
• Protocol attacks (SYN floods)
• Application layer attacks (HTTP floods)

**Detection Signs:**
• Unusual traffic spikes
• Service slowdowns or outages
• High number of requests from single IPs or regions

**Immediate Response:**
1. Enable DDoS protection/filtering
2. Block malicious IP ranges
3. Scale infrastructure if possible
4. Contact your ISP or CDN provider

Need help analyzing traffic patterns or blocking attacking IPs?""",
                confidence=0.8,
            )

        elif "phishing" in query_lower:
            return CopilotResponse(
                response_type=ResponseType.ANSWER,
                message="""**Phishing Attack Overview:**

Phishing is a social engineering attack where attackers impersonate legitimate organizations to trick victims into revealing sensitive information or downloading malware.

**Common Indicators:**
• Suspicious sender addresses
• Urgent or threatening language
• Unexpected attachments or links
• Requests for credentials or financial info
• Poor spelling/grammar

**Response Actions:**
1. Quarantine suspicious emails
2. Block sender domains
3. Check for credential compromise
4. Alert affected users
5. Review email gateway logs

Would you like me to help you quarantine emails or block suspicious senders?""",
                confidence=0.8,
            )

        # If we can't provide a specific answer
        if incident_data:
            return CopilotResponse(
                response_type=ResponseType.ANSWER,
                message=f"""I can help you with:

• **General Security Questions:** Ask about threats, attacks, or security concepts
• **Incident Analysis:** Currently viewing incident #{incident_data.get('id')} - ask specific questions about it
• **Response Actions:** Tell me what security actions you'd like to take

What would you like to know or do?""",
                confidence=0.6,
            )
        else:
            return CopilotResponse(
                response_type=ResponseType.ANSWER,
                message="""I'm your cybersecurity assistant! I can help with:

• **Security Concepts:** Explain threats, attacks, and defenses
• **Incident Response:** Analyze threats and recommend actions
• **Workflow Creation:** Execute security actions like blocking IPs, isolating hosts, etc.

You can also select an incident from the dashboard to get specific analysis.

What would you like to know?""",
                confidence=0.6,
            )

    async def _handle_action_request(
        self,
        query: str,
        incident_id: Optional[int],
        incident_data: Optional[Dict[str, Any]],
        extracted_params: Dict[str, Any],
        db_session: Optional[AsyncSession],
    ) -> CopilotResponse:
        """Handle action/workflow creation requests"""

        # Import NLP workflow parser
        from .nlp_workflow_parser import parse_workflow_from_natural_language

        try:
            # Parse workflow from natural language
            workflow_intent, explanation = await parse_workflow_from_natural_language(
                query, incident_id, db_session
            )

            # Check if we have all required parameters
            if (
                workflow_intent.clarification_needed
                or len(workflow_intent.actions) == 0
            ):
                # Generate follow-up questions
                action_types = [
                    a.get("action_type", "") for a in workflow_intent.actions
                ]
                (
                    questions,
                    suggested_options,
                ) = await self.parameter_collector.collect_missing_parameters(
                    query, action_types, extracted_params, incident_data
                )

                # Build helpful message
                message_parts = []

                if workflow_intent.unsupported_actions:
                    message_parts.append(
                        f"⚠️ **Note:** Some capabilities aren't available: {', '.join(workflow_intent.unsupported_actions)}"
                    )

                if workflow_intent.recommendations:
                    message_parts.append("\n💡 **Suggestions:**")
                    for rec in workflow_intent.recommendations:
                        message_parts.append(f"• {rec}")

                message_parts.append(
                    "\nTo help you better, I need some additional information:"
                )

                message = "\n".join(message_parts)

                return CopilotResponse(
                    response_type=ResponseType.FOLLOW_UP,
                    message=message,
                    confidence=workflow_intent.confidence,
                    follow_up_questions=questions,
                    suggested_options=suggested_options,
                    conversation_id=str(uuid.uuid4()),
                )

            # We have a valid workflow - generate confirmation
            confirmation_details = self.confirmation_generator.generate_confirmation(
                workflow_intent, incident_data
            )

            # Store pending action
            action_id = str(uuid.uuid4())
            self.pending_actions[action_id] = {
                "workflow_intent": workflow_intent,
                "explanation": explanation,
                "playbook_name": f"Chat Workflow: {query[:50]}...",
                "created_at": datetime.utcnow().isoformat(),
            }

            # Build confirmation message
            message = f"I'm ready to execute the following actions:\n\n"

            for i, action_desc in enumerate(confirmation_details["action_summary"], 1):
                message += f"{i}. {action_desc}\n"

            if confirmation_details["affected_resources"]:
                message += f"\n**Affected Resources:**\n"
                for resource in confirmation_details["affected_resources"]:
                    message += f"• {resource}\n"

            message += f"\n**Risk Level:** {confirmation_details['risk_level'].upper()}"
            message += f"\n**Estimated Duration:** {confirmation_details['estimated_duration']}"
            message += f"\n\n**Do you want to proceed with these actions?**"

            return CopilotResponse(
                response_type=ResponseType.CONFIRMATION_REQUIRED,
                message=message,
                confidence=workflow_intent.confidence,
                action_plan=confirmation_details,
                affected_resources=confirmation_details["affected_resources"],
                risk_level=confirmation_details["risk_level"],
                estimated_duration=confirmation_details["estimated_duration"],
                pending_action_id=action_id,
            )

        except Exception as e:
            self.logger.error(f"Error handling action request: {e}", exc_info=True)
            return CopilotResponse(
                response_type=ResponseType.ANSWER,
                message=f"I encountered an error processing your action request: {str(e)}",
                confidence=0.0,
            )


# Global copilot handler instance
copilot_handler = AICopilotHandler()


async def get_copilot_handler() -> AICopilotHandler:
    """Get the global copilot handler instance"""
    if not hasattr(copilot_handler, "_initialized"):
        await copilot_handler.initialize()
        copilot_handler._initialized = True
    return copilot_handler
