import os
import sys

# Add backend directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("Verifying imports...")
try:
    from app.orchestrator.workflow import LANGGRAPH_AVAILABLE, StateGraph

    print(f"Workflow imports: OK (LANGGRAPH_AVAILABLE={LANGGRAPH_AVAILABLE})")
except Exception as e:
    print(f"Workflow imports FAILED: {e}")

try:
    from app.agents.containment_agent import ContainmentAgent

    print("ContainmentAgent imports: OK")
except Exception as e:
    print(f"ContainmentAgent imports FAILED: {e}")

try:
    from app.agents.nlp_analyzer import NaturalLanguageThreatAnalyzer

    print("NLPAnalyzer imports: OK")
except Exception as e:
    print(f"NLPAnalyzer imports FAILED: {e}")
