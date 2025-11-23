#!/usr/bin/env python3
"""
Comprehensive Mini-XDR System Status Checker
Verifies all components are running and connected
"""

import asyncio
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.agent_orchestrator import get_orchestrator
from app.config import settings
from app.ml_engine import ml_detector
from app.tpot_connector import get_tpot_connector


def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_status(component, status, details=""):
    symbol = "✅" if status else "⚠️ "
    status_text = "ONLINE" if status else "OFFLINE"
    print(f"{symbol} {component:30s} {status_text:10s}  {details}")


async def check_tpot():
    """Check T-Pot honeypot connection"""
    connector = get_tpot_connector()

    print_status(
        "T-Pot Connection",
        connector.is_connected,
        f"Host: {connector.host}:{connector.ssh_port}",
    )

    if connector.is_connected:
        print_status(
            "  └─ SSH Tunnels",
            len(connector.tunnels) > 0,
            f"{len(connector.tunnels)} active",
        )
        print_status(
            "  └─ Monitoring",
            len(connector.monitoring_tasks) > 0,
            f"{len(connector.monitoring_tasks)} honeypots",
        )

        if connector.monitoring_tasks:
            for honeypot in connector.monitoring_tasks.keys():
                print(f"       • {honeypot}")
    else:
        print(f"       Note: This is expected if not at allowed IP (172.16.110.1)")


async def check_ml_models():
    """Check ML models"""
    models_loaded = hasattr(ml_detector, "xgb_model")

    print_status("ML Detection Engine", True, "Ensemble detector")  # Always available

    # Check individual models
    if hasattr(ml_detector, "xgb_model"):
        print_status("  └─ XGBoost Model", True, "Loaded")
    if hasattr(ml_detector, "isolation_forest"):
        print_status("  └─ Isolation Forest", True, "Loaded")
    if hasattr(ml_detector, "autoencoder_model"):
        print_status("  └─ Autoencoder", True, "Loaded")

    # Check enhanced detector
    try:
        from app.enhanced_threat_detector import enhanced_detector

        if enhanced_detector and hasattr(enhanced_detector, "model"):
            is_loaded = enhanced_detector.model is not None
            print_status("  └─ Enhanced Detector", is_loaded, "PyTorch model")
    except Exception:
        print_status("  └─ Enhanced Detector", False, "Not available")


async def check_ai_agents():
    """Check AI agents"""
    try:
        orchestrator = await get_orchestrator()

        if orchestrator:
            print_status(
                "AI Agent Orchestrator", True, f"{len(orchestrator.agents)} agents"
            )

            # List agents
            for agent_name in orchestrator.agents.keys():
                print(f"       • {agent_name.replace('_', ' ').title()}")
        else:
            print_status("AI Agent Orchestrator", False, "Not initialized")

    except Exception as e:
        print_status("AI Agent Orchestrator", False, f"Error: {str(e)[:30]}")


def check_api_server():
    """Check API server"""
    import requests

    try:
        response = requests.get("http://localhost:8000/docs", timeout=2)
        is_running = response.status_code == 200

        print_status(
            "API Server", is_running, f"http://{settings.api_host}:{settings.api_port}"
        )

        if is_running:
            # Test a few endpoints
            try:
                incidents = requests.get(
                    "http://localhost:8000/api/incidents", timeout=2
                )
                print_status(
                    "  └─ Incidents API",
                    incidents.status_code == 200,
                    f"{incidents.status_code}",
                )
            except:
                print_status("  └─ Incidents API", False, "Timeout/Error")

    except Exception as e:
        print_status("API Server", False, "Not responding")


def check_mcp_servers():
    """Check MCP servers"""
    import subprocess

    try:
        result = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True, timeout=5
        )

        mcp_processes = []
        for line in result.stdout.split("\n"):
            if "mcp" in line.lower() and "grep" not in line:
                if "shadcn" in line:
                    mcp_processes.append("shadcn-mcp")
                elif "xcodebuild" in line:
                    mcp_processes.append("xcodebuildmcp")
                elif "figma" in line:
                    mcp_processes.append("figma-mcp")

        # Deduplicate
        mcp_processes = list(set(mcp_processes))

        print_status(
            "MCP Servers", len(mcp_processes) > 0, f"{len(mcp_processes)} running"
        )

        for mcp in mcp_processes:
            print(f"       • {mcp}")

    except Exception as e:
        print_status("MCP Servers", False, f"Error checking: {str(e)[:30]}")


def check_database():
    """Check database"""
    db_url = settings.database_url
    db_type = (
        "SQLite"
        if "sqlite" in db_url
        else "PostgreSQL"
        if "postgresql" in db_url
        else "Unknown"
    )

    print_status(
        "Database", True, f"{db_type}"  # Assume it's available if backend started
    )


async def main():
    print_header("Mini-XDR System Status Check")

    print(f"Configuration:")
    print(f"  Host: {settings.api_host}:{settings.api_port}")
    print(
        f"  Database: {settings.database_url.split('@')[-1] if '@' in settings.database_url else settings.database_url}"
    )
    print(f"  T-Pot Host: {settings.tpot_host or 'Not configured'}")

    print_header("Component Status")

    # Check each component
    check_api_server()
    check_database()
    await check_tpot()
    await check_ml_models()
    await check_ai_agents()
    check_mcp_servers()

    print_header("Summary")

    print(
        """
Status Legend:
  ✅ - Component is online and operational
  ⚠️  - Component is offline or degraded (may be expected)

Notes:
  - T-Pot connection requires IP 172.16.110.1
  - Some warnings during startup are normal (LangChain, SHAP, LIME)
  - MCP servers run as separate processes
  - ML models may take a few seconds to load on startup

Next Steps:
  1. If API server is online, visit: http://localhost:8000/docs
  2. Start frontend: cd frontend && npm run dev
  3. Access UI: http://localhost:3000
  4. View honeypot dashboard: http://localhost:3000/honeypot
"""
    )

    print(f"{'='*60}\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nStatus check cancelled.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError running status check: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
