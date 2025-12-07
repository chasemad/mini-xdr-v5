"""
Investigation Result Tracking for LangChain Tools

This module provides automatic tracking of tool executions as investigation results.
When AI agents execute tools (block_ip, isolate_host, etc.), this decorator
automatically creates InvestigationResult records in the database.

Usage:
    from .investigation_tracking import track_investigation, set_investigation_context

    # Before orchestration
    set_investigation_context(incident_id=123)

    # Decorate tool implementation
    @track_investigation("block_ip", "network")
    async def _block_ip_impl(ip_address: str, ...):
        # Tool logic here
        return json.dumps({"success": True, ...})
"""

import json
import logging
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Thread-safe context variable for current incident
_current_incident_id: ContextVar[Optional[int]] = ContextVar(
    "incident_id", default=None
)


def set_investigation_context(incident_id: int) -> None:
    """
    Set the current incident context for tool execution tracking.

    This should be called before orchestration begins to ensure all tool
    executions are properly associated with the incident.

    Args:
        incident_id: The ID of the incident being investigated
    """
    _current_incident_id.set(incident_id)
    logger.info(f"Investigation context set for incident {incident_id}")


def clear_investigation_context() -> None:
    """Clear the investigation context after orchestration completes."""
    _current_incident_id.set(None)


def get_investigation_context() -> Optional[int]:
    """Get the current incident ID from context."""
    return _current_incident_id.get()


def track_investigation(tool_name: str, tool_category: str = "investigation"):
    """
    Decorator to automatically track tool executions as investigation results.

    This decorator:
    - Measures execution time
    - Captures parameters, results, and errors
    - Persists to investigation_results table
    - Handles async tool functions

    Args:
        tool_name: Name of the tool (e.g., "block_ip", "isolate_host")
        tool_category: Category of the tool (e.g., "network", "endpoint", "investigation")

    Example:
        @track_investigation("block_ip", "network")
        async def _block_ip_impl(ip_address: str, duration_seconds: int = 3600):
            # Tool implementation
            return json.dumps({"success": True, "target": ip_address})
    """

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            incident_id = _current_incident_id.get()

            # Track execution timing
            start_time = datetime.now(timezone.utc)
            status = "running"
            error_message = None
            result_data = {}
            result_json = ""

            try:
                # Execute the tool
                result_json = await func(*args, **kwargs)

                # Parse result
                if isinstance(result_json, str):
                    try:
                        result_data = json.loads(result_json)
                    except json.JSONDecodeError:
                        result_data = {"raw_result": result_json}
                else:
                    result_data = result_json

                # Determine status from result
                if isinstance(result_data, dict):
                    if result_data.get("success") is False:
                        status = "failed"
                        error_message = result_data.get("error", "Unknown error")
                    else:
                        status = "completed"
                else:
                    status = "completed"

            except Exception as e:
                status = "failed"
                error_message = str(e)
                result_data = {"error": str(e), "error_type": type(e).__name__}
                result_json = json.dumps(result_data)
                logger.error(f"Tool {tool_name} failed: {e}", exc_info=True)
                # Re-raise to maintain original behavior
                raise

            finally:
                end_time = datetime.now(timezone.utc)
                execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

                # Persist to database if we have incident context
                if incident_id:
                    try:
                        await _persist_tool_execution(
                            incident_id=incident_id,
                            tool_name=tool_name,
                            tool_category=tool_category,
                            parameters=_extract_parameters(args, kwargs),
                            results=result_data,
                            status=status,
                            started_at=start_time,
                            completed_at=end_time,
                            execution_time_ms=execution_time_ms,
                            error_message=error_message,
                        )
                    except Exception as persist_error:
                        # Don't fail the tool if tracking fails
                        logger.error(
                            f"Failed to persist investigation result for {tool_name}: {persist_error}",
                            exc_info=True,
                        )
                else:
                    logger.warning(
                        f"Tool {tool_name} executed without incident context - "
                        "investigation result not tracked"
                    )

            return result_json

        return wrapper

    return decorator


def _extract_parameters(args: tuple, kwargs: dict) -> Dict[str, Any]:
    """Extract parameters from function arguments for storage."""
    params = {}

    # Add keyword arguments
    for key, value in kwargs.items():
        # Convert non-JSON-serializable types to strings
        if isinstance(value, (str, int, float, bool, list, dict, type(None))):
            params[key] = value
        else:
            params[key] = str(value)

    # Add positional arguments with generic names
    for i, arg in enumerate(args):
        key = f"arg_{i}"
        if isinstance(arg, (str, int, float, bool, list, dict, type(None))):
            params[key] = arg
        else:
            params[key] = str(arg)

    return params


async def _persist_tool_execution(
    incident_id: int,
    tool_name: str,
    tool_category: str,
    parameters: Dict[str, Any],
    results: Dict[str, Any],
    status: str,
    started_at: datetime,
    completed_at: datetime,
    execution_time_ms: int,
    error_message: Optional[str] = None,
) -> None:
    """
    Persist tool execution to the investigation_results table.

    Args:
        incident_id: ID of the incident being investigated
        tool_name: Name of the tool executed
        tool_category: Category of the tool
        parameters: Input parameters to the tool
        results: Output results from the tool
        status: Execution status (running, completed, failed)
        started_at: When execution started
        completed_at: When execution completed
        execution_time_ms: Execution duration in milliseconds
        error_message: Error message if execution failed
    """
    from ..db import AsyncSessionLocal
    from ..models import InvestigationResult

    # Generate unique investigation ID
    timestamp = int(started_at.timestamp())
    investigation_id = f"inv_{incident_id}_{tool_name}_{timestamp}"

    # Extract findings and IOCs from results
    findings_count = _extract_findings_count(results)
    iocs_discovered = _extract_iocs(results)

    # Infer severity and confidence
    severity = _infer_severity(tool_name, results, status)
    confidence_score = _extract_confidence(results)

    try:
        async with AsyncSessionLocal() as db:
            investigation = InvestigationResult(
                investigation_id=investigation_id,
                incident_id=incident_id,
                tool_name=tool_name,
                tool_category=tool_category,
                status=status,
                started_at=started_at,
                completed_at=completed_at,
                execution_time_ms=execution_time_ms,
                parameters=parameters,
                results=results,
                findings_count=findings_count,
                iocs_discovered=iocs_discovered,
                severity=severity,
                confidence_score=confidence_score,
                triggered_by="langchain_agent",
                auto_triggered=True,
                error_message=error_message,
                retry_count=0,
                exported=False,
            )

            db.add(investigation)
            await db.commit()
            await db.refresh(investigation)

            logger.info(
                f"Investigation result created: {investigation_id} "
                f"(tool={tool_name}, status={status}, duration={execution_time_ms}ms)"
            )

    except Exception as e:
        logger.error(f"Failed to persist investigation result: {e}", exc_info=True)
        raise


def _extract_findings_count(results: Dict[str, Any]) -> int:
    """Extract the number of findings from tool results."""
    if not isinstance(results, dict):
        return 0

    # Check for findings array
    findings = results.get("findings", [])
    if isinstance(findings, list):
        return len(findings)

    # Check for threats/alerts/anomalies
    for key in ["threats", "alerts", "anomalies", "matches"]:
        if key in results and isinstance(results[key], list):
            return len(results[key])

    return 0


def _extract_iocs(results: Dict[str, Any]) -> Optional[Dict[str, list]]:
    """Extract indicators of compromise from tool results."""
    if not isinstance(results, dict):
        return None

    iocs = {}

    # Check for IOCs in various formats
    if "iocs" in results:
        iocs = results["iocs"]
    elif "iocs_discovered" in results:
        iocs = results["iocs_discovered"]
    elif "indicators" in results:
        iocs = results["indicators"]

    # Extract specific IOC types from results
    ioc_fields = {
        "ip_addresses": ["ip", "ips", "ip_address", "ip_addresses", "target"],
        "domains": ["domain", "domains", "hostname", "hostnames"],
        "urls": ["url", "urls"],
        "hashes": ["hash", "hashes", "md5", "sha256"],
        "files": ["file", "files", "file_path", "file_paths"],
    }

    for ioc_type, field_names in ioc_fields.items():
        for field in field_names:
            if field in results and results[field]:
                if ioc_type not in iocs:
                    iocs[ioc_type] = []
                # Ensure it's a list
                value = results[field]
                if isinstance(value, list):
                    iocs[ioc_type].extend(value)
                else:
                    iocs[ioc_type].append(value)

    return iocs if iocs else None


def _infer_severity(tool_name: str, results: Dict[str, Any], status: str) -> str:
    """
    Infer severity based on tool name, results, and execution status.

    Priority:
    1. Explicit severity in results
    2. Derived from tool name
    3. Default based on status
    """
    # Check if results contain explicit severity
    if isinstance(results, dict) and "severity" in results:
        return results["severity"]

    # Failed executions are high severity
    if status == "failed":
        return "high"

    # Critical tools (emergency response)
    critical_tools = [
        "isolate_host",
        "disable_user",
        "emergency_backup",
        "system_recovery",
        "kill_process",
    ]
    if tool_name in critical_tools:
        return "critical"

    # High severity tools (active containment)
    high_tools = [
        "block_ip",
        "revoke_sessions",
        "dns_sinkhole",
        "network_segmentation",
        "malware_removal",
    ]
    if tool_name in high_tools:
        return "high"

    # Medium severity tools (investigation/monitoring)
    medium_tools = [
        "threat_intel_lookup",
        "collect_evidence",
        "capture_traffic",
        "behavior_analysis",
        "threat_hunting",
        "analyze_logs",
    ]
    if tool_name in medium_tools:
        return "medium"

    # Default to medium for unknown tools
    return "medium"


def _extract_confidence(results: Dict[str, Any]) -> float:
    """Extract confidence score from tool results."""
    if not isinstance(results, dict):
        return 0.85  # Default confidence

    # Check various confidence fields
    confidence_fields = [
        "confidence",
        "confidence_score",
        "certainty",
        "probability",
        "likelihood",
    ]

    for field in confidence_fields:
        if field in results:
            value = results[field]
            if isinstance(value, (int, float)):
                # Normalize to 0-1 range if needed
                if value > 1:
                    value = value / 100.0
                return max(0.0, min(1.0, value))

    # Default confidence based on success
    if results.get("success") is True:
        return 0.85
    elif results.get("success") is False:
        return 0.5

    return 0.75


__all__ = [
    "track_investigation",
    "set_investigation_context",
    "clear_investigation_context",
    "get_investigation_context",
]
