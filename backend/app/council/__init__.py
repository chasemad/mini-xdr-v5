"""
Council of Models - GenAI Agents

This package contains the Deep GenAI layer that provides:
- Gemini 3: Deep reasoning and verification
- Grok: Real-time external threat intelligence
- OpenAI: Precise remediation and code generation
"""

from .gemini_judge import gemini_judge_node
from .grok_intel import grok_intel_node
from .openai_remediation import openai_remediation_node

__all__ = [
    "gemini_judge_node",
    "grok_intel_node",
    "openai_remediation_node",
]
