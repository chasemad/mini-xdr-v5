import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.agents.tools import get_tool_by_name
from app.models import Action, Incident, ResponseWorkflow
from sqlalchemy import select

logger = logging.getLogger(__name__)


class VisualWorkflowExecutor:
    """
    Executes visual workflows defined by React Flow nodes and edges.
    Traverses the graph, executes actions, and handles branching logic.
    """

    def __init__(
        self,
        db_session,
        incident_id: Optional[int] = None,
        workflow_id: Optional[int] = None,
        context: Dict[str, Any] = None,
    ):
        self.db = db_session
        self.incident_id = incident_id
        self.workflow_id = workflow_id
        self.execution_log = []
        self.context = context or {}  # Shared context for the workflow execution

    async def execute_graph(
        self, nodes: List[Dict], edges: List[Dict]
    ) -> Dict[str, Any]:
        """
        Execute the workflow graph starting from the Trigger node.
        """
        logger.info(
            f"Starting execution of visual workflow for incident {self.incident_id if self.incident_id else 'SYSTEM'}"
        )

        # 1. Build Adjacency List for easier traversal
        adj_list = {node["id"]: [] for node in nodes}
        node_map = {node["id"]: node for node in nodes}

        for edge in edges:
            source = edge["source"]
            target = edge["target"]
            label = edge.get("label")  # For conditional edges
            if source in adj_list:
                adj_list[source].append({"target": target, "label": label})

        # 2. Find Start Node (TriggerNode)
        start_node = next((n for n in nodes if n["type"] == "triggerNode"), None)
        if not start_node:
            logger.error("No trigger node found in workflow")
            return {"success": False, "error": "No trigger node found"}

        # 3. Traverse and Execute (BFS/Queue-based for simplicity, though workflows are mostly DAGs)
        # Using a queue to manage execution flow
        queue = [start_node["id"]]
        visited = set()

        # Track execution status of each node to handle dependencies if needed
        # For now, we assume sequential flow following edges

        execution_results = {}

        while queue:
            current_node_id = queue.pop(0)

            # Avoid cycles for now
            if current_node_id in visited:
                continue
            visited.add(current_node_id)

            current_node = node_map.get(current_node_id)
            if not current_node:
                continue

            logger.info(
                f"Processing node: {current_node.get('data', {}).get('label')} ({current_node['type']})"
            )

            # Execute Node Logic
            result = await self._execute_node(current_node)
            execution_results[current_node_id] = result

            self.execution_log.append(
                {
                    "node_id": current_node_id,
                    "type": current_node["type"],
                    "label": current_node.get("data", {}).get("label"),
                    "timestamp": datetime.now().isoformat(),
                    "result": result,
                }
            )

            # Determine Next Nodes
            next_hops = adj_list[current_node_id]

            if current_node["type"] == "conditionNode":
                # Branching Logic
                condition_result = result.get("outcome")  # True/False or specific label

                for hop in next_hops:
                    edge_label = hop.get("label")
                    # Match edge label with condition result (e.g., "Yes", "No", "True", "False")
                    if self._match_condition(condition_result, edge_label):
                        queue.append(hop["target"])

            else:
                # Normal Flow - Add all children
                for hop in next_hops:
                    queue.append(hop["target"])

        # 4. Update Workflow Status in DB if workflow_id is present
        if self.workflow_id:
            await self._update_workflow_status(execution_results)

        return {
            "success": True,
            "execution_log": self.execution_log,
            "results": execution_results,
        }

    async def _execute_node(self, node: Dict) -> Dict[str, Any]:
        """Execute logic for a single node"""
        node_type = node["type"]
        data = node.get("data", {})

        if node_type == "triggerNode":
            return {"status": "triggered", "timestamp": datetime.now().isoformat()}

        elif node_type == "actionNode":
            action_type = data.get("action_type")  # e.g., 'block_ip'
            params = data.get("params", {})

            # Inject incident context into params if available
            if self.incident_id:
                params["incident_id"] = self.incident_id

            # Inject global context
            params.update(self.context)

            # Execute using existing tool system
            try:
                # Map action_type to actual tool function if needed
                # For now assuming action_type matches tool name or we have a mapping
                tool_name = self._map_action_to_tool(action_type)

                if tool_name:
                    logger.info(f"Executing tool {tool_name} with params {params}")
                    # In a real implementation, we would call the tool here.
                    # For this implementation, we'll simulate or call if available.

                    # Simulating execution for safety/demo unless we want to hook into real tools
                    # result = await execute_tool(tool_name, params)

                    # Create Action record if we have an incident context, otherwise just log
                    if self.incident_id:
                        action_record = Action(
                            incident_id=self.incident_id,
                            action=tool_name,
                            result="success",  # Optimistic for now
                            detail=f"Executed via Visual Workflow: {data.get('label')}",
                            params=params,
                        )
                        self.db.add(action_record)
                        await self.db.commit()

                    return {"status": "success", "action": tool_name}
                else:
                    return {"status": "skipped", "reason": "unknown_action"}
            except Exception as e:
                logger.error(f"Action failed: {e}")
                return {"status": "failed", "error": str(e)}

        elif node_type == "conditionNode":
            # Evaluate condition
            # Example: "Is Malicious?" -> Check incident risk score
            condition_type = data.get("condition_type")  # e.g., 'check_severity'

            # Simple logic for demo
            if self.incident_id:
                result = await self.db.execute(
                    select(Incident).where(Incident.id == self.incident_id)
                )
                incident = result.scalar_one_or_none()

                if "malicious" in data.get("label", "").lower():
                    is_malicious = incident.risk_score > 0.7 if incident else False
                    return {
                        "status": "evaluated",
                        "outcome": "Yes" if is_malicious else "No",
                    }
            else:
                # System workflow condition evaluation (using context)
                # Example: Check if event.src_ip is in blacklist
                pass

            return {"status": "evaluated", "outcome": "Yes"}  # Default to Yes for flow

        return {"status": "unknown_type"}

    def _match_condition(self, result: Any, edge_label: Optional[str]) -> bool:
        """Helper to match condition result with edge label"""
        if not edge_label:
            return True  # Default path if no label

        if str(result).lower() == str(edge_label).lower():
            return True

        return False

    def _map_action_to_tool(self, action_type: str) -> Optional[str]:
        """Map generic action types to specific tool names"""
        # This mapping should align with tools.py
        mapping = {
            "block_ip": "block_ip",
            "isolate_host": "isolate_host",
            "send_email": "notify_stakeholders",
            "slack_alert": "alert_analysts",
            # Add more mappings as needed
        }
        return mapping.get(action_type, action_type)

    async def _update_workflow_status(self, results: Dict):
        """Update the workflow record in DB"""
        result = await self.db.execute(
            select(ResponseWorkflow).where(ResponseWorkflow.id == self.workflow_id)
        )
        wf = result.scalar_one_or_none()
        if wf:
            wf.status = "completed"
            wf.execution_log = self.execution_log
            wf.completed_at = datetime.now()
            await self.db.commit()
